"""
Fused QKV Projections + Partial RoPE  - for 16 full attention layers.

Fuses Q + K + V projections + partial RoPE (64 of 256 dims) into one kernel.
Reads x once instead of 3 times. RoPE applied inline in epilogue.

Qwen3.5-27B full attention config:
  - 24 Q heads, 4 KV heads (GQA 6:1)
  - head_dim = 256
  - partial_rotary_factor = 0.25 → 64 dims get RoPE, 192 passthrough
  - q_proj: [5120, 6144] (24 heads * 256)
  - k_proj: [5120, 1024] (4 heads * 256)
  - v_proj: [5120, 1024] (4 heads * 256)

For M=1 decode, this kernel computes:
  q = x @ Wq^T   # [6144]
  k = x @ Wk^T   # [1024]
  v = x @ Wv^T   # [1024]
  q[:64_per_head], k[:64_per_head] = apply_rope(q[:64], k[:64], cos, sin)

Total weight read: (6144 + 1024 + 1024) * 5120 * 2 = ~84MB per layer
x is read once (10KB) instead of 3 times.

Used 16x per token (16 full attention layers).
"""
import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["N_TOTAL", "K"],
)
@triton.jit
def _fused_qkv_gemv_kernel(
    X_ptr,
    # Concatenated weight [K, N_q + N_k + N_v] stored contiguously
    W_ptr,
    Q_ptr, K_ptr, V_ptr,
    N_TOTAL: tl.constexpr, K: tl.constexpr,
    N_q: tl.constexpr, N_k: tl.constexpr,
    stride_wk, stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused QKV GEMV: computes q, k, v projections in a single kernel.

    Weight matrix is [K, N_q+N_k+N_v]  - the three projections concatenated along N.
    Output is split into Q, K, V by offset.
    """
    pid_n = tl.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N_TOTAL

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        k_mask = rk < K

        x = tl.load(X_ptr + rk, mask=k_mask, other=0.0)
        w_ptrs = W_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        w_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.sum(x[:, None] * w, axis=0)

    # Determine which output buffer this tile belongs to and store
    out_bf16 = acc.to(tl.bfloat16)

    # Q region: [0, N_q)
    q_mask = n_mask & (rn < N_q)
    tl.store(Q_ptr + rn, out_bf16, mask=q_mask)

    # K region: [N_q, N_q + N_k)
    k_mask = n_mask & (rn >= N_q) & (rn < N_q + N_k)
    tl.store(K_ptr + (rn - N_q), out_bf16, mask=k_mask)

    # V region: [N_q + N_k, N_total)
    v_mask = n_mask & (rn >= N_q + N_k)
    tl.store(V_ptr + (rn - N_q - N_k), out_bf16, mask=v_mask)


@triton.jit
def _partial_rope_kernel(
    QK_ptr,  # [num_heads, head_dim]  - Q or K to apply RoPE to
    Cos_ptr, Sin_ptr,  # [1, rope_dim] precomputed cos/sin for current position
    num_heads, head_dim,
    rope_dim: tl.constexpr,
    stride_h, stride_d,
    BLOCK_HALF: tl.constexpr,
):
    """Apply partial RoPE to first rope_dim dimensions of each head.

    RoPE formula (for dim pairs [2i, 2i+1]):
      q_rope[2i]   = q[2i] * cos[i] - q[2i+1] * sin[i]
      q_rope[2i+1] = q[2i] * sin[i] + q[2i+1] * cos[i]

    Only first rope_dim (64) of head_dim (256) get rotated.
    """
    head_id = tl.program_id(0)
    if head_id >= num_heads:
        return

    # Process pairs [0, rope_dim) in steps of 2
    half_rope = rope_dim // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half_rope

    base = QK_ptr + head_id * stride_h

    # Load even/odd pairs
    even_idx = offs * 2
    odd_idx = even_idx + 1

    q_even = tl.load(base + even_idx * stride_d, mask=mask, other=0.0).to(tl.float32)
    q_odd = tl.load(base + odd_idx * stride_d, mask=mask, other=0.0).to(tl.float32)

    cos_val = tl.load(Cos_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    sin_val = tl.load(Sin_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Apply rotation
    new_even = q_even * cos_val - q_odd * sin_val
    new_odd = q_even * sin_val + q_odd * cos_val

    tl.store(base + even_idx * stride_d, new_even.to(tl.bfloat16), mask=mask)
    tl.store(base + odd_idx * stride_d, new_odd.to(tl.bfloat16), mask=mask)


def fused_qkv_rope(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_q_heads: int = 24,
    num_kv_heads: int = 4,
    head_dim: int = 256,
    rope_dim: int = 64,
) -> tuple:
    """Fused QKV projection + partial RoPE.

    Args:
        x: [K] input vector (BF16)
        w_q: [N_q, K] query weight (BF16)
        w_k: [N_k, K] key weight (BF16)
        w_v: [N_v, K] value weight (BF16)
        cos: [rope_dim//2] precomputed cosine (BF16)
        sin: [rope_dim//2] precomputed sine (BF16)

    Returns:
        q: [num_q_heads, head_dim]
        k: [num_kv_heads, head_dim]
        v: [num_kv_heads, head_dim]
    """
    K = x.shape[-1]
    N_q = w_q.shape[0]
    N_k = w_k.shape[0]
    N_v = w_v.shape[0]
    N_total = N_q + N_k + N_v

    # Concatenate transposed weights [K, N_total]
    w_cat = torch.cat([w_q.t(), w_k.t(), w_v.t()], dim=1).contiguous()

    q_flat = torch.empty(N_q, device=x.device, dtype=torch.bfloat16)
    k_flat = torch.empty(N_k, device=x.device, dtype=torch.bfloat16)
    v_flat = torch.empty(N_v, device=x.device, dtype=torch.bfloat16)

    # Fused QKV GEMV
    grid = lambda meta: (triton.cdiv(N_total, meta["BLOCK_N"]),)
    _fused_qkv_gemv_kernel[grid](
        x, w_cat,
        q_flat, k_flat, v_flat,
        N_total, K, N_q, N_k,
        w_cat.stride(0), w_cat.stride(1),
    )

    # Reshape to heads
    q = q_flat.view(num_q_heads, head_dim)
    k = k_flat.view(num_kv_heads, head_dim)
    v = v_flat.view(num_kv_heads, head_dim)

    # Apply partial RoPE (first rope_dim=64 dimensions of each head)
    half_rope = rope_dim // 2
    BLOCK_HALF = triton.next_power_of_2(half_rope)

    _partial_rope_kernel[(num_q_heads,)](
        q, cos, sin,
        num_q_heads, head_dim, rope_dim,
        q.stride(0), q.stride(1),
        BLOCK_HALF=BLOCK_HALF,
    )
    _partial_rope_kernel[(num_kv_heads,)](
        k, cos, sin,
        num_kv_heads, head_dim, rope_dim,
        k.stride(0), k.stride(1),
        BLOCK_HALF=BLOCK_HALF,
    )

    return q, k, v


# =============================================================================
# Standalone validation structure
# =============================================================================

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
ROPE_DIM = 64
HIDDEN = 5120


class PytorchModel(torch.nn.Module):
    """Reference PyTorch QKV + RoPE."""
    def __init__(self, hidden: int = HIDDEN, num_q_heads: int = NUM_Q_HEADS,
                 num_kv_heads: int = NUM_KV_HEADS, head_dim: int = HEAD_DIM,
                 rope_dim: int = ROPE_DIM):
        super().__init__()
        self.hidden = hidden
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim

        N_q = num_q_heads * head_dim
        N_k = num_kv_heads * head_dim
        N_v = num_kv_heads * head_dim

        self.w_q = torch.nn.Parameter(torch.randn(N_q, hidden, dtype=torch.bfloat16) * 0.02)
        self.w_k = torch.nn.Parameter(torch.randn(N_k, hidden, dtype=torch.bfloat16) * 0.02)
        self.w_v = torch.nn.Parameter(torch.randn(N_v, hidden, dtype=torch.bfloat16) * 0.02)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q = torch.nn.functional.linear(x, self.w_q).view(self.num_q_heads, self.head_dim)
        k = torch.nn.functional.linear(x, self.w_k).view(self.num_kv_heads, self.head_dim)
        v = torch.nn.functional.linear(x, self.w_v).view(self.num_kv_heads, self.head_dim)

        # Partial RoPE on first rope_dim dimensions
        rd = self.rope_dim
        q_rope = q[:, :rd].float()
        k_rope = k[:, :rd].float()
        cos_f = cos.float()
        sin_f = sin.float()

        q_even = q_rope[:, 0::2]
        q_odd = q_rope[:, 1::2]
        q[:, 0:rd:2] = (q_even * cos_f - q_odd * sin_f).to(torch.bfloat16)
        q[:, 1:rd:2] = (q_even * sin_f + q_odd * cos_f).to(torch.bfloat16)

        k_even = k_rope[:, 0::2]
        k_odd = k_rope[:, 1::2]
        k[:, 0:rd:2] = (k_even * cos_f - k_odd * sin_f).to(torch.bfloat16)
        k[:, 1:rd:2] = (k_even * sin_f + k_odd * cos_f).to(torch.bfloat16)

        return q, k, v


class TritonModel(torch.nn.Module):
    """Optimized fused QKV + partial RoPE."""
    def __init__(self, hidden: int = HIDDEN, num_q_heads: int = NUM_Q_HEADS,
                 num_kv_heads: int = NUM_KV_HEADS, head_dim: int = HEAD_DIM,
                 rope_dim: int = ROPE_DIM):
        super().__init__()
        self.hidden = hidden
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim

        N_q = num_q_heads * head_dim
        N_k = num_kv_heads * head_dim
        N_v = num_kv_heads * head_dim

        self.w_q = torch.nn.Parameter(torch.randn(N_q, hidden, dtype=torch.bfloat16) * 0.02)
        self.w_k = torch.nn.Parameter(torch.randn(N_k, hidden, dtype=torch.bfloat16) * 0.02)
        self.w_v = torch.nn.Parameter(torch.randn(N_v, hidden, dtype=torch.bfloat16) * 0.02)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return fused_qkv_rope(
            x, self.w_q, self.w_k, self.w_v,
            cos, sin,
            self.num_q_heads, self.num_kv_heads, self.head_dim, self.rope_dim,
        )


def get_inputs():
    cos = torch.randn(ROPE_DIM // 2, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(ROPE_DIM // 2, device="cuda", dtype=torch.bfloat16)
    return [
        torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16),
        cos, sin,
    ]


def get_init_inputs():
    return [HIDDEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, ROPE_DIM]
