"""
BF16 GEMV with Split-K  - the critical path kernel (65% of decode time).

For decode (M=1), all linear projections are GEMV: y[1,N] = x[1,K] @ W[K,N].
This is memory-bound: we need to read the entire weight matrix (K*N*2 bytes).

Split-K strategy: partition K dimension across multiple thread blocks to
increase SM utilization. Each block computes a partial sum over K_PER_SPLIT
elements, then a reduce kernel sums the partials.

Target: 416 blocks for B200's 208 SMs (2 waves).

Key GEMV shapes in Qwen3.5-27B:
  - MLP gate/up_proj: [5120, 17408]  - dominates time
  - MLP down_proj: [17408, 5120]
  - DeltaNet in_proj: [5120, 10240+]
  - Attn q_proj: [5120, 12288] (gated, doubled)
  - lm_head: [5120, 248320]  - 2.4GB, 1x per token

Adapted from int4_gemm.py split-K strategy (no dequant, just BF16 dot products).
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


# =============================================================================
# Standard BF16 GEMV (no split-K)  - used when N is large enough
# =============================================================================

@triton.autotune(
    configs=[
        # High-occupancy for memory-bound GEMV (M=1)
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
            num_warps=2, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
            num_warps=2, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
            num_warps=4, num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_SIZE_M": 8},
            num_warps=8, num_stages=4,
        ),
        # B200-specific: wider tiles exploiting 256KB SRAM/SM
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 512, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
            num_warps=8, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 512, "GROUP_SIZE_M": 8},
            num_warps=8, num_stages=2,
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def _bf16_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """BF16 GEMM: Y[M,N] = X[M,K] @ W[K,N].

    Grouped tile ordering for L2 cache locality.
    Uses tl.dot for BF16 tensor core acceleration.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped tile ordering for L2 locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
        x_mask = (rm[:, None] < M) & (rk[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        w_mask = (rk[:, None] < K) & n_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # BF16 tensor core dot product
        acc += tl.dot(x, w)

    y = acc.to(tl.bfloat16)
    y_ptrs = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    y_mask = (rm[:, None] < M) & n_mask[None, :]
    tl.store(y_ptrs, y, mask=y_mask)


# =============================================================================
# Split-K BF16 GEMV  - increases block count for better SM utilization
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
            num_warps=2, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
            num_warps=2, num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
            num_warps=4, num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_SIZE_M": 8},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_SIZE_M": 8},
            num_warps=8, num_stages=4,
        ),
    ],
    key=["N", "K_PER_SPLIT"],
)
@triton.jit
def _bf16_gemm_splitk_kernel(
    X_ptr, W_ptr, Partial_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_sk, stride_pm, stride_pn,
    K_PER_SPLIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Split-K BF16 GEMM: each block processes K_PER_SPLIT of K dimension.

    Stores FP32 partial results. A separate reduce kernel sums across SPLIT_K.
    Grid: (num_m_blocks * num_n_blocks * SPLIT_K,)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_mn = num_pid_m * num_pid_n

    split_k_id = pid // num_pid_mn
    pid_mn = pid % num_pid_mn

    # Grouped tile ordering
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N

    k_start_base = split_k_id * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, K_PER_SPLIT, BLOCK_K):
        k_start = k_start_base + k_offset
        rk = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
        x_mask = (rm[:, None] < M) & (rk[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        w_mask = (rk[:, None] < K) & n_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    # Store FP32 partials
    p_ptrs = Partial_ptr + split_k_id * stride_sk + rm[:, None] * stride_pm + rn[None, :] * stride_pn
    p_mask = (rm[:, None] < M) & n_mask[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


@triton.jit
def _splitk_reduce_kernel(
    Partial_ptr, Y_ptr,
    M, N, stride_sk, stride_pm, stride_pn,
    stride_ym, stride_yn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum FP32 partial results across SPLIT_K splits, store as BF16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for sk in range(SPLIT_K):
        ptrs = Partial_ptr + sk * stride_sk + rm[:, None] * stride_pm + rn[None, :] * stride_pn
        acc += tl.load(ptrs, mask=mask, other=0.0)

    y_ptrs = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask)


# =============================================================================
# BF16 GEMV for decode (M=1)  - vector-matrix product, memory-bound
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def _bf16_gemv_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_wk, stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """BF16 GEMV: y[N] = x[K] @ W[K,N] for M=1 decode.

    Each program handles one BLOCK_N tile of the output.
    Accumulates across K in BLOCK_K steps.
    """
    pid_n = tl.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        k_mask = rk < K

        # Load x vector [BLOCK_K]
        x = tl.load(X_ptr + rk, mask=k_mask, other=0.0)

        # Load W tile [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        w_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Vector-matrix: sum x[k] * w[k, n] over k
        acc += tl.sum(x[:, None] * w, axis=0)

    tl.store(Y_ptr + rn, acc.to(tl.bfloat16), mask=n_mask)


# =============================================================================
# Split-K GEMV for decode  - more blocks for better SM utilization
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=2),
    ],
    key=["N", "K_PER_SPLIT"],
)
@triton.jit
def _bf16_gemv_splitk_kernel(
    X_ptr, W_ptr, Partial_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_wk, stride_wn,
    stride_psk, stride_pn,
    K_PER_SPLIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Split-K BF16 GEMV: each block processes K_PER_SPLIT of K dimension."""
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)

    split_k_id = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N

    k_start_base = split_k_id * K_PER_SPLIT
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_offset in range(0, K_PER_SPLIT, BLOCK_K):
        k_start = k_start_base + k_offset
        rk = k_start + tl.arange(0, BLOCK_K)
        k_mask = rk < K

        x = tl.load(X_ptr + rk, mask=k_mask, other=0.0)
        w_ptrs = W_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        w_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.sum(x[:, None] * w, axis=0)

    # Store FP32 partials [SPLIT_K, N]
    p_ptrs = Partial_ptr + split_k_id * stride_psk + rn * stride_pn
    tl.store(p_ptrs, acc, mask=n_mask)


@triton.jit
def _gemv_splitk_reduce_kernel(
    Partial_ptr, Y_ptr,
    N, stride_psk, stride_pn,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum FP32 partial results across SPLIT_K splits for GEMV, store as BF16."""
    pid_n = tl.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for sk in range(SPLIT_K):
        ptrs = Partial_ptr + sk * stride_psk + rn * stride_pn
        acc += tl.load(ptrs, mask=n_mask, other=0.0)

    tl.store(Y_ptr + rn, acc.to(tl.bfloat16), mask=n_mask)


# =============================================================================
# Dispatch logic
# =============================================================================

# Target ~416 blocks for B200 (208 SMs × 2 waves)
_TARGET_BLOCKS = 416
_MIN_K_PER_SPLIT = 256


def _compute_split_k(M: int, N: int, K: int, block_n: int = 128) -> int:
    """Determine optimal split-K for SM utilization on B200."""
    num_n_blocks = triton.cdiv(N, block_n)
    if M > 4:
        # For batched GEMM, N-tiles alone may fill SMs
        num_mn_blocks = triton.cdiv(M, 16) * num_n_blocks
        if num_mn_blocks >= _TARGET_BLOCKS:
            return 1
    elif num_n_blocks >= _TARGET_BLOCKS:
        return 1

    split_k = max(1, (_TARGET_BLOCKS + num_n_blocks - 1) // num_n_blocks)
    max_sk = K // _MIN_K_PER_SPLIT
    split_k = min(split_k, max_sk)

    # Ensure K divides evenly into splits aligned to BLOCK_K=128
    while split_k > 1 and K % (split_k * 128) != 0:
        split_k -= 1
    return max(1, split_k)


def bf16_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    N: int,
    K: int,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """BF16 linear forward: Y = X @ W^T. Uses split-K for small N.

    Args:
        x: [M, K] or [K] input activations (BF16)
        weight: [N, K] weight matrix (BF16)  - stored row-major (N outer)
        N: output features
        K: input features
        bias: optional [N] bias

    Returns:
        [M, N] or [N] output (BF16)
    """
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)
    M = x.shape[0]

    # Weight is [N, K], kernel expects [K, N]  - use transposed strides
    # weight.t() is [K, N] with strides (1, K) for row-major [N, K]
    w_t = weight.t().contiguous()

    if M == 1:
        # Use specialized GEMV kernel
        return _bf16_gemv_dispatch(x.squeeze(0), w_t, N, K, bias, squeeze)

    split_k = _compute_split_k(M, N, K)

    if split_k <= 1:
        y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        _bf16_gemm_kernel[grid](
            x, w_t, y,
            M, N, K,
            x.stride(0), x.stride(1),
            w_t.stride(0), w_t.stride(1),
            y.stride(0), y.stride(1),
        )
    else:
        k_per_split = K // split_k
        partial = torch.empty(split_k, M, N, device=x.device, dtype=torch.float32)
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]) * split_k,
        )
        _bf16_gemm_splitk_kernel[grid](
            x, w_t, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            w_t.stride(0), w_t.stride(1),
            partial.stride(0), partial.stride(1), partial.stride(2),
            K_PER_SPLIT=k_per_split,
            SPLIT_K=split_k,
        )
        y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
        reduce_grid = (triton.cdiv(M, 16), triton.cdiv(N, 128))
        _splitk_reduce_kernel[reduce_grid](
            partial, y,
            M, N,
            partial.stride(0), partial.stride(1), partial.stride(2),
            y.stride(0), y.stride(1),
            SPLIT_K=split_k,
            BLOCK_M=16,
            BLOCK_N=128,
        )

    if bias is not None:
        y = y + bias.unsqueeze(0)
    if squeeze:
        y = y.squeeze(0)
    return y


def _bf16_gemv_dispatch(
    x: torch.Tensor,
    w_t: torch.Tensor,
    N: int,
    K: int,
    bias: torch.Tensor,
    squeeze: bool,
) -> torch.Tensor:
    """Dispatch GEMV with optional split-K for small N."""
    split_k = _compute_split_k(1, N, K)

    if split_k <= 1:
        y = torch.empty(N, device=x.device, dtype=torch.bfloat16)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
        _bf16_gemv_kernel[grid](
            x, w_t, y,
            N, K,
            w_t.stride(0), w_t.stride(1),
        )
    else:
        k_per_split = K // split_k
        partial = torch.empty(split_k, N, device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]) * split_k,)
        _bf16_gemv_splitk_kernel[grid](
            x, w_t, partial,
            N, K,
            w_t.stride(0), w_t.stride(1),
            partial.stride(0), partial.stride(1),
            K_PER_SPLIT=k_per_split,
            SPLIT_K=split_k,
        )
        y = torch.empty(N, device=x.device, dtype=torch.bfloat16)
        reduce_grid = (triton.cdiv(N, 128),)
        _gemv_splitk_reduce_kernel[reduce_grid](
            partial, y,
            N,
            partial.stride(0), partial.stride(1),
            SPLIT_K=split_k,
            BLOCK_N=128,
        )

    if bias is not None:
        y = y + bias
    if not squeeze:
        y = y.unsqueeze(0)
    return y


# =============================================================================
# Standalone validation structure
# =============================================================================

DEFAULT_K = 5120
DEFAULT_N = 17408  # MLP intermediate_size

WEIGHT_SEED = 42


class PytorchModel(nn.Module):
    """Reference BF16 Linear (torch.matmul)."""
    def __init__(self, K: int = DEFAULT_K, N: int = DEFAULT_N):
        super().__init__()
        self.K = K
        self.N = N
        torch.manual_seed(WEIGHT_SEED)
        self.weight = nn.Parameter(torch.randn(N, K, dtype=torch.bfloat16) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class TritonModel(nn.Module):
    """Optimized BF16 GEMV with split-K for B200."""
    def __init__(self, K: int = DEFAULT_K, N: int = DEFAULT_N):
        super().__init__()
        self.K = K
        self.N = N
        torch.manual_seed(WEIGHT_SEED)
        self.weight = nn.Parameter(torch.randn(N, K, dtype=torch.bfloat16) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.K)
        y = bf16_linear_forward(x_2d, self.weight, self.N, self.K)
        return y.reshape(*orig_shape[:-1], self.N)


def get_inputs():
    return [torch.randn(1, 1, DEFAULT_K, device="cuda", dtype=torch.bfloat16)]


def get_init_inputs():
    return [DEFAULT_K, DEFAULT_N]
