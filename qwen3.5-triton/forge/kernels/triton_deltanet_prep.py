"""
Fused DeltaNet Post-Projection Processing  - 17 ops → 1 kernel per layer.

After projecting QKV/alpha/beta and running conv1d, there are ~17 small
PyTorch operations per DeltaNet layer:
  - repeat_interleave Q (16→48 heads): 1 op
  - repeat_interleave K (16→48 heads): 1 op
  - F.normalize Q (norm, clamp, div): ~3 ops
  - F.normalize K: ~3 ops
  - scale Q by 1/sqrt(dk): 1 op
  - gate: float() + reshape + softplus(alpha+dt_bias) + neg_A_exp*: ~5 ops
  - beta: float() + reshape + sigmoid: ~3 ops

This kernel fuses ALL of them into a single launch.
Grid: (NUM_V_HEADS,) = (48,)  - one block per v_head.

Over 48 DeltaNet layers: saves 768 kernel launches per token.
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


@triton.jit
def _deltanet_post_proj_kernel(
    # Inputs
    QKV_ptr,       # [CONV_DIM] conv1d output (after SiLU)
    Alpha_ptr,     # [NUM_V_HEADS] projected alpha
    BetaRaw_ptr,   # [NUM_V_HEADS] projected beta
    NegAExp_ptr,   # [NUM_V_HEADS] pre-computed -exp(A_log)
    DtBias_ptr,    # [NUM_V_HEADS] pre-computed dt_bias (float32)
    # Outputs
    Q_ptr,         # [NUM_V_HEADS, HEAD_DIM]
    K_ptr,         # [NUM_V_HEADS, HEAD_DIM]
    V_ptr,         # [NUM_V_HEADS, HEAD_DIM]
    Gate_ptr,      # [NUM_V_HEADS] float32
    Beta_ptr,      # [NUM_V_HEADS] float32
    # Constants
    KEY_DIM: tl.constexpr,
    V_PER_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INV_SQRT_DK: tl.constexpr,
    L2_EPS: tl.constexpr,
):
    """Fused post-projection processing for one v_head.

    Per v_head h:
      1. Map h → k_head (h // V_PER_K)
      2. Load Q[k_head], K[k_head] from conv output (shared across V_PER_K v_heads)
      3. Load V[h] from conv output
      4. L2 normalize Q and K
      5. Scale Q by 1/sqrt(dk)
      6. Compute gate = -exp(A_log) * softplus(alpha + dt_bias)
      7. Compute beta = sigmoid(beta_raw)
    """
    h = tl.program_id(0)  # v_head index [0..47]
    k_head = h // V_PER_K  # k_head index [0..15]

    offs = tl.arange(0, HEAD_DIM)  # [0..127]

    # === Load Q, K from conv output (indexed by k_head) ===
    q_offset = k_head * HEAD_DIM
    k_offset = KEY_DIM + k_head * HEAD_DIM
    v_offset = KEY_DIM + KEY_DIM + h * HEAD_DIM

    q_raw = tl.load(QKV_ptr + q_offset + offs).to(tl.float32)
    k_raw = tl.load(QKV_ptr + k_offset + offs).to(tl.float32)
    v = tl.load(QKV_ptr + v_offset + offs)  # keep BF16, stored directly

    # === L2 normalize Q ===
    q_sq_sum = tl.sum(q_raw * q_raw, axis=0)
    q_norm = tl.sqrt(q_sq_sum)
    q_norm = tl.maximum(q_norm, L2_EPS)
    q = q_raw / q_norm

    # === L2 normalize K ===
    k_sq_sum = tl.sum(k_raw * k_raw, axis=0)
    k_norm = tl.sqrt(k_sq_sum)
    k_norm = tl.maximum(k_norm, L2_EPS)
    k = k_raw / k_norm

    # === Scale Q by 1/sqrt(dk) ===
    q = q * INV_SQRT_DK

    # === Compute gate: g = neg_A_exp * softplus(alpha + dt_bias) ===
    alpha_h = tl.load(Alpha_ptr + h).to(tl.float32)
    neg_a_exp_h = tl.load(NegAExp_ptr + h).to(tl.float32)
    dt_bias_h = tl.load(DtBias_ptr + h).to(tl.float32)

    sp_input = alpha_h + dt_bias_h
    # Numerically stable softplus: log(1 + exp(x)), clamped for large x
    sp = tl.where(sp_input > 20.0, sp_input, tl.log(1.0 + tl.exp(sp_input)))
    gate_h = neg_a_exp_h * sp

    # === Compute beta: sigmoid(beta_raw) ===
    beta_raw_h = tl.load(BetaRaw_ptr + h).to(tl.float32)
    beta_h = tl.sigmoid(beta_raw_h)

    # === Store outputs ===
    out_base = h * HEAD_DIM
    tl.store(Q_ptr + out_base + offs, q.to(tl.bfloat16))
    tl.store(K_ptr + out_base + offs, k.to(tl.bfloat16))
    tl.store(V_ptr + out_base + offs, v)  # already BF16
    tl.store(Gate_ptr + h, gate_h)   # float32 for recurrent step
    tl.store(Beta_ptr + h, beta_h)   # float32 for recurrent step


def deltanet_post_proj(
    qkv_conv: torch.Tensor,
    alpha: torch.Tensor,
    beta_raw: torch.Tensor,
    neg_A_exp: torch.Tensor,
    dt_bias_f: torch.Tensor,
) -> tuple:
    """Fused post-projection processing for DeltaNet decode (B=1).

    Takes raw conv1d output + projected alpha/beta, produces ready-to-use
    Q, K, V (expanded, normalized, scaled) + gate + beta for recurrent step.

    Args:
        qkv_conv: [B, CONV_DIM] conv1d output after SiLU (B must be 1)
        alpha: [B, NUM_V_HEADS] projected alpha
        beta_raw: [B, NUM_V_HEADS] projected beta
        neg_A_exp: [NUM_V_HEADS] pre-computed -exp(A_log) (float32)
        dt_bias_f: [NUM_V_HEADS] pre-computed dt_bias (float32)

    Returns:
        q: [B, NUM_V_HEADS, HEAD_DIM]  - L2-normed, scaled, expanded
        k: [B, NUM_V_HEADS, HEAD_DIM]  - L2-normed, expanded
        v: [B, NUM_V_HEADS, HEAD_DIM]  - raw from conv output
        gate: [B, NUM_V_HEADS]  - float32 gate values
        beta: [B, NUM_V_HEADS]  - float32 beta values
    """
    B = qkv_conv.shape[0]
    device = qkv_conv.device

    q = torch.empty(B, _NUM_V_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.empty(B, _NUM_V_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.empty(B, _NUM_V_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16)
    gate = torch.empty(B, _NUM_V_HEADS, device=device, dtype=torch.float32)
    beta = torch.empty(B, _NUM_V_HEADS, device=device, dtype=torch.float32)

    # Process each batch element (typically B=1 for decode)
    for b in range(B):
        _deltanet_post_proj_kernel[(_NUM_V_HEADS,)](
            qkv_conv[b],
            alpha[b],
            beta_raw[b],
            neg_A_exp,
            dt_bias_f,
            q[b],
            k[b],
            v[b],
            gate[b],
            beta[b],
            KEY_DIM=_KEY_DIM,
            V_PER_K=_V_PER_K,
            HEAD_DIM=_HEAD_DIM,
            INV_SQRT_DK=_INV_SQRT_DK,
            L2_EPS=_L2_EPS,
        )

    return q, k, v, gate, beta
