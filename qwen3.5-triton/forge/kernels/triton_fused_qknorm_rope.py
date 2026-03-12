"""
Fused QK-Norm + Partial RoPE  - for Qwen3.5-27B attention layers.

Replaces per attention layer:
  - triton_rmsnorm(q)  - 1 kernel
  - triton_rmsnorm(k)  - 1 kernel
  - apply_rotary_pos_emb(q, k, cos, sin)  - ~10 kernels
    (slice×4, rotate_half×2, mul×4, add×2, cat×2)

With 2 Triton kernel launches (one for Q, one for K).

Over 16 attention layers: saves ~192 kernel executions per token.

Architecture specifics (Qwen3.5-27B):
  - head_dim = 256
  - rotary_dim = 64 (partial_rotary_factor=0.25)
  - rotate_half format: split at dim/2, negate+swap
  - RMSNorm uses (1+weight) scaling
  - cos/sin are duplicated: cos[0:32] == cos[32:64]
"""
import torch
import triton
import triton.language as tl


# Qwen3.5-27B attention constants
_HEAD_DIM = 256
_HALF_ROT = 32    # rotary_dim // 2 = 64 // 2
_ROTARY_DIM = 64
# Pad 192 (= 256-64) to next power of 2 for tl.arange
_PASS_DIM_PADDED = 256


@triton.jit
def _fused_qknorm_rope_kernel(
    X_ptr,          # Input Q or K: [B, num_heads, 1, head_dim]
    W_ptr,          # Norm weight: [head_dim]
    Cos_ptr,        # cos values: [rotary_dim] (only first HALF_ROT used)
    Sin_ptr,        # sin values: [rotary_dim] (only first HALF_ROT used)
    Out_ptr,        # Output: same shape as X
    HEAD_DIM: tl.constexpr,
    HALF_ROT: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    PASS_DIM_PADDED: tl.constexpr,
    eps: tl.constexpr,
    stride_xb, stride_xh, stride_xd,
    stride_ob, stride_oh, stride_od,
):
    """Fused RMSNorm(1+w) + partial RoPE for one head.

    Each program handles one (batch, head) pair:
      1. Load full head vector [HEAD_DIM=256]
      2. Compute RMSNorm with (1+weight) scaling in FP32
      3. Apply rotate_half RoPE to first ROTARY_DIM=64 dims
      4. Pass through remaining 192 dims unchanged
      5. Store result as BF16
    """
    head_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    x_base = X_ptr + batch_id * stride_xb + head_id * stride_xh
    o_base = Out_ptr + batch_id * stride_ob + head_id * stride_oh

    # === Load head vector in three groups ===
    # Group 1: first half of rotary [0..31]
    rot1_offs = tl.arange(0, HALF_ROT)
    rot1 = tl.load(x_base + rot1_offs * stride_xd).to(tl.float32)

    # Group 2: second half of rotary [32..63]
    rot2_offs = HALF_ROT + tl.arange(0, HALF_ROT)
    rot2 = tl.load(x_base + rot2_offs * stride_xd).to(tl.float32)

    # Group 3: pass-through [64..255] (padded to PASS_DIM_PADDED)
    pass_offs = ROTARY_DIM + tl.arange(0, PASS_DIM_PADDED)
    pass_mask = pass_offs < HEAD_DIM
    x_pass = tl.load(x_base + pass_offs * stride_xd, mask=pass_mask, other=0.0).to(tl.float32)

    # === RMSNorm across all HEAD_DIM dims ===
    # Masked-out values are 0, so they don't contribute to sum
    sq_sum = tl.sum(rot1 * rot1) + tl.sum(rot2 * rot2) + tl.sum(x_pass * x_pass)
    var = sq_sum / HEAD_DIM
    rrms = tl.rsqrt(var + eps)

    # Load norm weights in same groups
    w_rot1 = tl.load(W_ptr + rot1_offs).to(tl.float32)
    w_rot2 = tl.load(W_ptr + rot2_offs).to(tl.float32)
    w_pass = tl.load(W_ptr + pass_offs, mask=pass_mask, other=0.0).to(tl.float32)

    # Apply norm: x * rrms * (1 + w)  - Qwen3.5 style
    rot1_n = rot1 * rrms * (1.0 + w_rot1)
    rot2_n = rot2 * rrms * (1.0 + w_rot2)
    x_pass_n = x_pass * rrms * (1.0 + w_pass)

    # === Partial RoPE on first ROTARY_DIM dims ===
    # HF rotate_half: [-x2, x1] where x1=first_half, x2=second_half
    # result = x * cos + rotate_half(x) * sin
    # result[0:32]  = rot1_n * cos - rot2_n * sin
    # result[32:64] = rot2_n * cos + rot1_n * sin
    # Note: cos[0:32] == cos[32:64] in HF, so we only load first 32
    cos_val = tl.load(Cos_ptr + tl.arange(0, HALF_ROT)).to(tl.float32)
    sin_val = tl.load(Sin_ptr + tl.arange(0, HALF_ROT)).to(tl.float32)

    out_rot1 = rot1_n * cos_val - rot2_n * sin_val
    out_rot2 = rot2_n * cos_val + rot1_n * sin_val

    # === Store all three groups ===
    tl.store(o_base + rot1_offs * stride_od, out_rot1.to(tl.bfloat16))
    tl.store(o_base + rot2_offs * stride_od, out_rot2.to(tl.bfloat16))
    tl.store(o_base + pass_offs * stride_od, x_pass_n.to(tl.bfloat16), mask=pass_mask)


def fused_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused QK-Norm + partial RoPE in a single Triton kernel.

    Combines RMSNorm(1+weight) with partial RoPE application.
    Handles Qwen3.5's rotate_half format and partial_rotary_factor=0.25.

    Args:
        x: [B, num_heads, S, head_dim=256]  - Q or K tensor (S must be 1)
        weight: [head_dim=256]  - RMSNorm weight
        cos: [B, S, rotary_dim=64]  - cosine position embeddings
        sin: [B, S, rotary_dim=64]  - sine position embeddings
        eps: RMSNorm epsilon

    Returns:
        [B, num_heads, S, head_dim=256]  - normalized and rotated tensor
    """
    B, num_heads, S, head_dim = x.shape
    assert S == 1, "Fused QKnorm+RoPE only supports decode (S=1)"

    out = torch.empty_like(x, dtype=torch.bfloat16)

    # Extract cos/sin values for this position [rotary_dim=64]
    # cos shape: [B, S, rotary_dim]  - squeeze to get the values
    cos_flat = cos.reshape(-1)  # [64]
    sin_flat = sin.reshape(-1)  # [64]

    grid = (num_heads, B)

    _fused_qknorm_rope_kernel[grid](
        x, weight, cos_flat, sin_flat, out,
        _HEAD_DIM, _HALF_ROT, _ROTARY_DIM, _PASS_DIM_PADDED,
        eps,
        x.stride(0), x.stride(1), x.stride(3),   # skip stride(2) since S=1
        out.stride(0), out.stride(1), out.stride(3),
    )

    return out
