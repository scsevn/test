"""
Fused DeltaNet Post-Projection + Recurrent State Update  - 2 kernels -> 1.

Combines triton_deltanet_prep.py (post_proj) and triton_deltanet_recurrent.py
(recurrent step) into a single kernel launch per v_head.

Eliminates per DeltaNet layer:
  - 48 extra kernel launches (one post_proj + one recurrent -> one fused)
  - ~73KB intermediate HBM traffic (48 heads x 5 tensors written then read back)

Over 48 DeltaNet layers: saves 2,304 kernel launches/token and ~3.5MB HBM traffic.

The kernel body is: post_proj math in registers -> recurrent tiled state update.
No intermediate Q/K/V/gate/beta ever touches HBM.

Grid: (NUM_V_HEADS, B) = (48, 1) for decode.

V5 fixes:
  - Fix 3a: K/Q scratch buffers  - store normalized k/q once, avoid redundant
    reloads + re-normalization in tile loops. Saves 2 loads+divides per tile.
  - Fix 3b: Fused RMSNormGated output  - apply rmsnorm(o)*silu(z) in-kernel
    before storing output, eliminating 48 separate rmsnorm_gated launches/layer.
"""
import torch
import triton
import triton.language as tl


# Constants matching Qwen3.5-27B DeltaNet config
_NUM_K_HEADS = 16
_NUM_V_HEADS = 48
_HEAD_DIM = 128
_KEY_DIM = _NUM_K_HEADS * _HEAD_DIM  # 2048
_V_PER_K = _NUM_V_HEADS // _NUM_K_HEADS  # 3
_INV_SQRT_DK = _HEAD_DIM ** -0.5  # 0.08838...
_L2_EPS = 1e-6


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 16}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 8}, num_warps=8, num_stages=1),
    ],
    key=["Dk", "Dv"],
)
@triton.jit
def _fused_postproj_recurrent_kernel(
    # Post-proj inputs
    QKV_ptr,       # [CONV_DIM] conv1d output (after SiLU)
    Alpha_ptr,     # [NUM_V_HEADS] projected alpha
    BetaRaw_ptr,   # [NUM_V_HEADS] projected beta
    NegAExp_ptr,   # [NUM_V_HEADS] pre-computed -exp(A_log)
    DtBias_ptr,    # [NUM_V_HEADS] pre-computed dt_bias (float32)
    # Scratch buffers for normalized Q/K (Fix 3a)
    K_scratch_ptr, # [B, NUM_V_HEADS, Dk] normalized k scratch
    Q_scratch_ptr, # [B, NUM_V_HEADS, Dk] normalized q scratch
    # State [B, num_heads, Dk, Dv]
    S_ptr,
    # Output [B, num_heads, Dv]
    O_ptr,
    # RMSNormGated params (Fix 3b)  - set FUSE_NORM=False to skip
    NormW_ptr,     # [Dv] norm weight (per v_head output dim)
    Z_ptr,         # [B, NUM_V_HEADS, Dv] gate values for SiLU
    norm_eps,
    # Strides for Z
    stride_zb, stride_zh, stride_zd,
    # Dimensions
    Dk: tl.constexpr, Dv: tl.constexpr,
    # Post-proj constants
    KEY_DIM: tl.constexpr,
    V_PER_K: tl.constexpr,
    INV_SQRT_DK: tl.constexpr,
    L2_EPS: tl.constexpr,
    # Strides for state
    stride_sb, stride_sh, stride_sdk, stride_sdv,
    # Strides for QKV_ptr (per batch, flat)
    stride_qkv_b,
    # Strides for alpha/beta (per batch)
    stride_ab, stride_bb,
    # Strides for output
    stride_ob, stride_oh, stride_od,
    # Strides for scratch
    stride_ks_b, stride_ks_h, stride_ks_d,
    stride_qs_b, stride_qs_h, stride_qs_d,
    # Feature flag
    FUSE_NORM: tl.constexpr,
    # Tile sizes
    BLOCK_DV: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    """Fused post-projection + recurrent step for one (v_head, batch) pair.

    Phase 1 (post-proj): Compute q, k, v, gate, beta in registers from raw inputs.
                          Store normalized k/q to scratch (Fix 3a).
    Phase 2 (recurrent): Tiled state decay, delta rule update, output query.
                          Load k/q from scratch instead of re-normalizing.
    Phase 3 (optional):   Apply RMSNormGated to output in-kernel (Fix 3b).
    """
    h = tl.program_id(0)   # v_head index [0..47]
    bid = tl.program_id(1)  # batch index
    k_head = h // V_PER_K   # k_head index [0..15]

    # ===================================================================
    # PHASE 1: Post-projection (all in registers, no HBM write)
    # ===================================================================
    offs_d = tl.arange(0, BLOCK_DV)  # [0..127] (BLOCK_DV == Dv == 128)
    dv_mask = offs_d < Dv

    # Load Q, K from conv output (indexed by k_head)
    qkv_base = QKV_ptr + bid * stride_qkv_b
    q_offset = k_head * Dk
    k_offset = KEY_DIM + k_head * Dk
    v_offset = KEY_DIM + KEY_DIM + h * Dv

    q_raw = tl.load(qkv_base + q_offset + offs_d, mask=dv_mask, other=0.0).to(tl.float32)
    k_raw = tl.load(qkv_base + k_offset + offs_d, mask=dv_mask, other=0.0).to(tl.float32)
    v = tl.load(qkv_base + v_offset + offs_d, mask=dv_mask, other=0.0).to(tl.float32)

    # L2 normalize Q
    q_sq_sum = tl.sum(q_raw * q_raw, axis=0)
    q_norm = tl.sqrt(q_sq_sum)
    q_norm = tl.maximum(q_norm, L2_EPS)
    q = (q_raw / q_norm) * INV_SQRT_DK  # normalize + scale in one step

    # L2 normalize K
    k_sq_sum = tl.sum(k_raw * k_raw, axis=0)
    k_norm = tl.sqrt(k_sq_sum)
    k_norm = tl.maximum(k_norm, L2_EPS)
    k = k_raw / k_norm

    # Fix 3a: Store normalized k and q to scratch buffers (write once, read many)
    # These are tiny: 128 * 2B = 256 bytes each, stays in L1/L2.
    ks_base = K_scratch_ptr + bid * stride_ks_b + h * stride_ks_h
    qs_base = Q_scratch_ptr + bid * stride_qs_b + h * stride_qs_h
    tl.store(ks_base + offs_d * stride_ks_d, k.to(tl.bfloat16), mask=dv_mask)
    tl.store(qs_base + offs_d * stride_qs_d, q.to(tl.bfloat16), mask=dv_mask)

    # Compute gate: g = neg_A_exp * softplus(alpha + dt_bias)
    alpha_h = tl.load(Alpha_ptr + bid * stride_ab + h).to(tl.float32)
    neg_a_exp_h = tl.load(NegAExp_ptr + h).to(tl.float32)
    dt_bias_h = tl.load(DtBias_ptr + h).to(tl.float32)
    sp_input = alpha_h + dt_bias_h
    sp = tl.where(sp_input > 20.0, sp_input, tl.log(1.0 + tl.exp(sp_input)))
    gate = neg_a_exp_h * sp  # this is log-space gate (negative)

    # Compute beta: sigmoid(beta_raw)
    beta_raw_h = tl.load(BetaRaw_ptr + bid * stride_bb + h).to(tl.float32)
    beta = tl.sigmoid(beta_raw_h)

    # Now q, k, v, gate, beta are all in registers. No HBM write (except scratch).

    # ===================================================================
    # PHASE 2: Recurrent state update (tiled, with scratch-based k/q loads)
    # ===================================================================
    decay = tl.exp(gate)

    s_base = S_ptr + bid * stride_sb + h * stride_sh

    # === Pass 1: Decay state + accumulate S^T @ k ===
    accumulated = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    for r_start in range(0, Dk, BLOCK_DK):
        dk_offs = r_start + tl.arange(0, BLOCK_DK)
        dk_mask = dk_offs < Dk

        # Load state tile [BLOCK_DK, BLOCK_DV]
        s_ptrs = s_base + dk_offs[:, None] * stride_sdk + offs_d[None, :] * stride_sdv
        s_mask = dk_mask[:, None] & dv_mask[None, :]
        s_tile = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

        # Decay
        s_tile = s_tile * decay

        # Fix 3a: Load normalized k from scratch (no re-normalization needed)
        k_tile = tl.load(ks_base + dk_offs * stride_ks_d, mask=dk_mask, other=0.0).to(tl.float32)

        # Accumulate S^T @ k
        accumulated += tl.sum(s_tile * k_tile[:, None], axis=0)

        # Store decayed state
        tl.store(s_ptrs, s_tile.to(tl.bfloat16), mask=s_mask)

    # delta = beta * (v - S^T @ k)
    delta = beta * (v - accumulated)

    # === Pass 2: State update + compute S^T @ q ===
    output = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    for r_start in range(0, Dk, BLOCK_DK):
        dk_offs = r_start + tl.arange(0, BLOCK_DK)
        dk_mask = dk_offs < Dk

        # Reload decayed state (should be in L2 cache)
        s_ptrs = s_base + dk_offs[:, None] * stride_sdk + offs_d[None, :] * stride_sdv
        s_mask = dk_mask[:, None] & dv_mask[None, :]
        s_tile = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

        # Rank-1 update: S += outer(k, delta)
        # Fix 3a: Load normalized k from scratch
        k_tile = tl.load(ks_base + dk_offs * stride_ks_d, mask=dk_mask, other=0.0).to(tl.float32)
        s_tile += k_tile[:, None] * delta[None, :]

        # Store updated state
        tl.store(s_ptrs, s_tile.to(tl.bfloat16), mask=s_mask)

        # Accumulate output: S^T @ q
        # Fix 3a: Load normalized q from scratch
        q_tile = tl.load(qs_base + dk_offs * stride_qs_d, mask=dk_mask, other=0.0).to(tl.float32)
        output += tl.sum(s_tile * q_tile[:, None], axis=0)

    # ===================================================================
    # PHASE 3 (optional): Fused RMSNormGated output (Fix 3b)
    # ===================================================================
    if FUSE_NORM:
        # RMSNorm on output
        var = tl.sum(output * output, axis=0) / Dv
        rrms = tl.rsqrt(var + norm_eps)
        norm_w = tl.load(NormW_ptr + offs_d, mask=dv_mask, other=0.0).to(tl.float32)
        o_normed = output * rrms * norm_w

        # SiLU(z): z * sigmoid(z)
        z_val = tl.load(
            Z_ptr + bid * stride_zb + h * stride_zh + offs_d * stride_zd,
            mask=dv_mask, other=0.0,
        ).to(tl.float32)
        z_silu = z_val * tl.sigmoid(z_val)

        # Fused multiply
        final_output = (o_normed * z_silu).to(tl.bfloat16)
    else:
        final_output = output.to(tl.bfloat16)

    # Store output [Dv]
    o_ptrs = O_ptr + bid * stride_ob + h * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, final_output, mask=dv_mask)


def fused_postproj_recurrent(
    qkv_conv: torch.Tensor,
    alpha: torch.Tensor,
    beta_raw: torch.Tensor,
    neg_A_exp: torch.Tensor,
    dt_bias_f: torch.Tensor,
    state: torch.Tensor,
    norm_weight: torch.Tensor = None,
    z: torch.Tensor = None,
    norm_eps: float = 1e-6,
) -> tuple:
    """Fused post-projection + recurrent step for DeltaNet decode.

    Combines deltanet_post_proj() and deltanet_recurrent_step() into a single
    kernel launch per v_head. Intermediate Q/K/V/gate/beta never touch HBM.

    V5: Optionally fuses RMSNormGated output (pass norm_weight and z to enable).

    Args:
        qkv_conv: [B, CONV_DIM] conv1d output after SiLU
        alpha: [B, NUM_V_HEADS] projected alpha
        beta_raw: [B, NUM_V_HEADS] projected beta
        neg_A_exp: [NUM_V_HEADS] pre-computed -exp(A_log) (float32)
        dt_bias_f: [NUM_V_HEADS] pre-computed dt_bias (float32)
        state: [B, NUM_V_HEADS, Dk, Dv] recurrent state (MODIFIED IN-PLACE)
        norm_weight: [Dv] optional RMSNorm weight for fused output norm
        z: [B, NUM_V_HEADS, Dv] optional gate values for SiLU in fused norm
        norm_eps: epsilon for RMSNorm

    Returns:
        output: [B, NUM_V_HEADS, Dv] attention output (BF16)
        state: same tensor, updated in-place
    """
    B = qkv_conv.shape[0]
    num_heads = _NUM_V_HEADS
    Dk = _HEAD_DIM
    Dv = _HEAD_DIM

    output = torch.empty(B, num_heads, Dv, device=qkv_conv.device, dtype=torch.bfloat16)

    # Fix 3a: Allocate scratch buffers for normalized k/q
    # Tiny: B * 48 * 128 * 2B = 12KB per batch element
    k_scratch = torch.empty(B, num_heads, Dk, device=qkv_conv.device, dtype=torch.bfloat16)
    q_scratch = torch.empty(B, num_heads, Dk, device=qkv_conv.device, dtype=torch.bfloat16)

    # Fix 3b: Determine if we should fuse norm
    fuse_norm = norm_weight is not None and z is not None

    # Dummy tensors for when norm is not fused (never accessed due to FUSE_NORM=False)
    if not fuse_norm:
        norm_weight = qkv_conv  # dummy, not accessed
        z = qkv_conv  # dummy, not accessed
        z_stride_b = 0
        z_stride_h = 0
        z_stride_d = 0
    else:
        z_stride_b = z.stride(0)
        z_stride_h = z.stride(1)
        z_stride_d = z.stride(2)

    grid = (num_heads, B)

    _fused_postproj_recurrent_kernel[grid](
        qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias_f,
        k_scratch, q_scratch,
        state, output,
        norm_weight, z, norm_eps,
        z_stride_b, z_stride_h, z_stride_d,
        Dk, Dv,
        _KEY_DIM, _V_PER_K, _INV_SQRT_DK, _L2_EPS,
        # State strides
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        # QKV stride (per batch)
        qkv_conv.stride(0),
        # Alpha/beta strides (per batch)
        alpha.stride(0), beta_raw.stride(0),
        # Output strides
        output.stride(0), output.stride(1), output.stride(2),
        # Scratch strides
        k_scratch.stride(0), k_scratch.stride(1), k_scratch.stride(2),
        q_scratch.stride(0), q_scratch.stride(1), q_scratch.stride(2),
        # Feature flag
        FUSE_NORM=fuse_norm,
    )

    return output, state
