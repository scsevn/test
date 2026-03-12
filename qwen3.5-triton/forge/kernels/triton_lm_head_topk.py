"""
Fused lm_head GEMV + Streaming Argmax  - never materializes full logits.

For Qwen3.5-27B: vocab_size = 248,320, hidden = 5120.
lm_head weight = [248320, 5120] = 2.4GB in BF16.

Without fusion: GEMV produces [248320] logits → argmax scans all 248K floats.
With fusion: each tile computes local max during GEMV, then a small reduce
kernel finds the global argmax. Never writes 248K logits to HBM.

Saves: 248320 * 2 = ~500KB HBM write + read.

For CUDA graph compatibility: the graph shape is deterministic (always produces
exactly 1 token index), unlike top-k which varies.

Grid: cdiv(248320, BLOCK_N) = ~1940 blocks at BLOCK_N=128.
With split-K: ~3880 blocks (good utilization for B200's 208 SMs).

V5: Added batched variants for B>1 inference.
  - fused_lm_head_argmax_batched: dynamic allocation, B>1
  - fused_lm_head_argmax_static_batched: CUDA-graph safe, B>1
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def _lm_head_local_max_kernel(
    X_ptr, W_ptr,
    LocalMax_ptr, LocalIdx_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_wk, stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMV + local argmax per tile.

    Each program computes BLOCK_N output values and finds their local max.
    Stores (max_value, max_index) per tile for reduce kernel.
    """
    pid_n = tl.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        k_mask = rk < K

        x = tl.load(X_ptr + rk, mask=k_mask, other=0.0)
        w_ptrs = W_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        w_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.sum(x[:, None] * w, axis=0)

    # Find local max within this BLOCK_N tile
    # Use -inf for masked positions
    acc_masked = tl.where(n_mask, acc, float('-inf'))
    local_max = tl.max(acc_masked, axis=0)

    # Find the index of the local max
    is_max = (acc_masked == local_max)
    # Get the global index  - use the first matching position
    local_idx = tl.min(tl.where(is_max, rn, N), axis=0)

    # Store per-tile max and index
    tl.store(LocalMax_ptr + pid_n, local_max)
    tl.store(LocalIdx_ptr + pid_n, local_idx)


@triton.jit
def _global_argmax_kernel(
    LocalMax_ptr, LocalIdx_ptr, Result_ptr,
    num_tiles,
    BLOCK_T: tl.constexpr,
):
    """Reduce local maxes to find global argmax.

    Single program scans all tile results.
    """
    best_val = tl.full((), float('-inf'), dtype=tl.float32)
    best_idx = tl.full((), 0, dtype=tl.int64)

    for t_start in range(0, num_tiles, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < num_tiles

        vals = tl.load(LocalMax_ptr + offs, mask=mask, other=float('-inf'))
        idxs = tl.load(LocalIdx_ptr + offs, mask=mask, other=0)

        # Find max in this chunk
        chunk_max = tl.max(vals, axis=0)
        if chunk_max > best_val:
            is_best = (vals == chunk_max)
            best_idx = tl.min(tl.where(is_best, idxs, 2**31), axis=0)
            best_val = chunk_max

    tl.store(Result_ptr, best_idx)


def fused_lm_head_argmax(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Fused lm_head GEMV + argmax without materializing full logits.

    Args:
        x: [K] hidden state (BF16)
        weight: [N, K] lm_head weight (BF16) where N=vocab_size

    Returns:
        [1] token index (int64)
    """
    N, K = weight.shape
    w_t = weight.t().contiguous()  # [K, N]

    # Phase 1: GEMV with local max per tile
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)

    # Estimate num tiles for allocation
    max_block_n = 512
    max_tiles = triton.cdiv(N, 128)  # worst case

    local_max = torch.empty(max_tiles, device=x.device, dtype=torch.float32)
    local_idx = torch.empty(max_tiles, device=x.device, dtype=torch.int64)

    _lm_head_local_max_kernel[grid](
        x, w_t,
        local_max, local_idx,
        N, K,
        w_t.stride(0), w_t.stride(1),
    )

    # Phase 2: Global argmax reduce
    # Get actual number of tiles used (depends on autotune BLOCK_N)
    num_tiles = max_tiles  # conservative upper bound
    result = torch.empty(1, device=x.device, dtype=torch.int64)

    BLOCK_T = triton.next_power_of_2(min(num_tiles, 2048))
    _global_argmax_kernel[(1,)](
        local_max, local_idx, result,
        num_tiles,
        BLOCK_T=BLOCK_T,
    )

    return result


def fused_lm_head_argmax_static(
    x: torch.Tensor,
    w_t: torch.Tensor,
    local_max: torch.Tensor,
    local_idx: torch.Tensor,
    result: torch.Tensor,
) -> torch.Tensor:
    """CUDA-graph-safe fused lm_head GEMV + argmax. No allocations inside.

    All buffers must be pre-allocated before CUDA graph capture.

    Args:
        x: [K] hidden state (BF16)
        w_t: [K, N] pre-transposed lm_head weight (BF16)
        local_max: [max_tiles] pre-allocated float32 buffer
        local_idx: [max_tiles] pre-allocated int64 buffer
        result: [1] pre-allocated int64 output

    Returns:
        result tensor (same as input, written in-place)
    """
    K, N = w_t.shape
    max_tiles = local_max.shape[0]

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)

    _lm_head_local_max_kernel[grid](
        x, w_t,
        local_max, local_idx,
        N, K,
        w_t.stride(0), w_t.stride(1),
    )

    BLOCK_T = triton.next_power_of_2(min(max_tiles, 2048))
    _global_argmax_kernel[(1,)](
        local_max, local_idx, result,
        max_tiles,
        BLOCK_T=BLOCK_T,
    )

    return result


# =============================================================================
# Batched variants (V5)  - B>1 inference support
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def _lm_head_local_max_batched_kernel(
    X_ptr, W_ptr,
    LocalMax_ptr, LocalIdx_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_xb,          # stride between batch elements in X
    stride_wk, stride_wn,
    stride_lm_b,        # stride between batch elements in LocalMax
    stride_li_b,        # stride between batch elements in LocalIdx
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched GEMV + local argmax per (tile, batch) pair.

    Grid: (num_tiles, B)
    Each program computes BLOCK_N output values for one batch element.
    """
    pid_n = tl.program_id(0)   # tile index
    bid = tl.program_id(1)     # batch index

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base = X_ptr + bid * stride_xb

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        k_mask = rk < K

        x = tl.load(x_base + rk, mask=k_mask, other=0.0)
        w_ptrs = W_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        w_mask = k_mask[:, None] & n_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.sum(x[:, None] * w, axis=0)

    acc_masked = tl.where(n_mask, acc, float('-inf'))
    local_max = tl.max(acc_masked, axis=0)

    is_max = (acc_masked == local_max)
    local_idx = tl.min(tl.where(is_max, rn, N), axis=0)

    tl.store(LocalMax_ptr + bid * stride_lm_b + pid_n, local_max)
    tl.store(LocalIdx_ptr + bid * stride_li_b + pid_n, local_idx)


@triton.jit
def _global_argmax_batched_kernel(
    LocalMax_ptr, LocalIdx_ptr, Result_ptr,
    num_tiles,
    stride_lm_b,   # stride between batch elements in LocalMax
    stride_li_b,   # stride between batch elements in LocalIdx
    BLOCK_T: tl.constexpr,
):
    """Batched reduce: one program per batch element.

    Grid: (B,)
    """
    bid = tl.program_id(0)

    best_val = tl.full((), float('-inf'), dtype=tl.float32)
    best_idx = tl.full((), 0, dtype=tl.int64)

    lm_base = LocalMax_ptr + bid * stride_lm_b
    li_base = LocalIdx_ptr + bid * stride_li_b

    for t_start in range(0, num_tiles, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < num_tiles

        vals = tl.load(lm_base + offs, mask=mask, other=float('-inf'))
        idxs = tl.load(li_base + offs, mask=mask, other=0)

        chunk_max = tl.max(vals, axis=0)
        if chunk_max > best_val:
            is_best = (vals == chunk_max)
            best_idx = tl.min(tl.where(is_best, idxs, 2**31), axis=0)
            best_val = chunk_max

    tl.store(Result_ptr + bid, best_idx)


def fused_lm_head_argmax_batched(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Fused lm_head GEMV + argmax for batched inputs (B>1).

    Args:
        x: [B, K] hidden states (BF16)
        weight: [N, K] lm_head weight (BF16) where N=vocab_size

    Returns:
        [B] token indices (int64)
    """
    B, K = x.shape
    N = weight.shape[0]
    w_t = weight.t().contiguous()  # [K, N]

    max_tiles = triton.cdiv(N, 128)

    local_max = torch.empty(B, max_tiles, device=x.device, dtype=torch.float32)
    local_idx = torch.empty(B, max_tiles, device=x.device, dtype=torch.int64)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), B)

    _lm_head_local_max_batched_kernel[grid](
        x, w_t,
        local_max, local_idx,
        N, K,
        x.stride(0),
        w_t.stride(0), w_t.stride(1),
        local_max.stride(0), local_idx.stride(0),
    )

    result = torch.empty(B, device=x.device, dtype=torch.int64)

    BLOCK_T = triton.next_power_of_2(min(max_tiles, 2048))
    _global_argmax_batched_kernel[(B,)](
        local_max, local_idx, result,
        max_tiles,
        local_max.stride(0), local_idx.stride(0),
        BLOCK_T=BLOCK_T,
    )

    return result


def fused_lm_head_argmax_static_batched(
    x: torch.Tensor,
    w_t: torch.Tensor,
    local_max: torch.Tensor,
    local_idx: torch.Tensor,
    result: torch.Tensor,
    B: int,
) -> torch.Tensor:
    """CUDA-graph-safe fused lm_head GEMV + argmax for batched inputs.

    All buffers must be pre-allocated for max batch size before capture.

    Args:
        x: [B, K] hidden states (BF16)
        w_t: [K, N] pre-transposed lm_head weight (BF16)
        local_max: [max_B, max_tiles] pre-allocated float32 buffer
        local_idx: [max_B, max_tiles] pre-allocated int64 buffer
        result: [max_B] pre-allocated int64 output
        B: actual batch size (must be <= max_B)

    Returns:
        result tensor (same as input, written in-place)
    """
    K, N = w_t.shape
    max_tiles = local_max.shape[1]

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), B)

    _lm_head_local_max_batched_kernel[grid](
        x, w_t,
        local_max, local_idx,
        N, K,
        x.stride(0),
        w_t.stride(0), w_t.stride(1),
        local_max.stride(0), local_idx.stride(0),
    )

    BLOCK_T = triton.next_power_of_2(min(max_tiles, 2048))
    _global_argmax_batched_kernel[(B,)](
        local_max, local_idx, result,
        max_tiles,
        local_max.stride(0), local_idx.stride(0),
        BLOCK_T=BLOCK_T,
    )

    return result


def fused_lm_head_logits(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Standard lm_head GEMV that materializes logits (for top-p/top-k sampling).

    Falls back to bf16_linear_forward from triton_bf16_gemv.
    """
    from forge.kernels.triton_bf16_gemv import bf16_linear_forward
    N, K = weight.shape
    return bf16_linear_forward(x, weight, N, K)


# =============================================================================
# Standalone validation structure
# =============================================================================

VOCAB_SIZE = 248320
HIDDEN = 5120
WEIGHT_SEED = 42


class PytorchModel(torch.nn.Module):
    """Reference PyTorch lm_head + argmax."""
    def __init__(self, hidden: int = HIDDEN, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size
        torch.manual_seed(WEIGHT_SEED)
        self.weight = torch.nn.Parameter(
            torch.randn(vocab_size, hidden, dtype=torch.bfloat16) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.nn.functional.linear(x, self.weight)
        return logits.argmax(dim=-1)


class TritonModel(torch.nn.Module):
    """Optimized fused lm_head + streaming argmax."""
    def __init__(self, hidden: int = HIDDEN, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size
        torch.manual_seed(WEIGHT_SEED)
        self.weight = torch.nn.Parameter(
            torch.randn(vocab_size, hidden, dtype=torch.bfloat16) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_lm_head_argmax(x.squeeze(), self.weight)


def get_inputs():
    return [torch.randn(1, HIDDEN, device="cuda", dtype=torch.bfloat16)]


def get_init_inputs():
    return [HIDDEN, VOCAB_SIZE]
