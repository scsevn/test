"""
Fused Residual-Add + RMSNorm  - two ops in one kernel launch.

Replaces the common pattern at every layer boundary:
  hidden = residual + x          # kernel 1: elementwise add
  normed = rmsnorm(hidden, w)    # kernel 2: norm

With a single Triton kernel that:
  1. Loads residual + x → computes sum in FP32
  2. Computes RMSNorm on the sum
  3. Stores BOTH: normed output AND new residual

Used 64x per token (once per layer, between attention and MLP blocks).
Eliminates 64 separate residual-add kernel launches per token.

B200: each row is independent, grid=(M,). For decode M=1 → 1 block.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_residual_rmsnorm_kernel(
    R_ptr, X_ptr, W_ptr, Normed_ptr, NewR_ptr,
    stride_r, stride_x, stride_n, stride_nr,
    D, eps,
    BLOCK_D: tl.constexpr,
    ADD_ONE_TO_WEIGHT: tl.constexpr,
):
    """Fused: new_r = residual + x; normed = rmsnorm(new_r, weight).

    Outputs both the un-normed sum (next residual) and the normed version.
    All computation in FP32, stores BF16.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    r = tl.load(R_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if ADD_ONE_TO_WEIGHT:
        w = w + 1.0

    # Residual add in FP32
    h = r + x

    # RMSNorm on the sum
    var = tl.sum(h * h, axis=0) / D
    rrms = tl.rsqrt(var + eps)
    normed = (h * rrms * w).to(tl.bfloat16)

    # Store normed output (for next block's input)
    tl.store(Normed_ptr + row * stride_n + offs, normed, mask=mask)
    # Store new residual (the raw sum, used as next residual connection)
    tl.store(NewR_ptr + row * stride_nr + offs, h.to(tl.bfloat16), mask=mask)


def fused_residual_rmsnorm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_one_to_weight: bool = True,
) -> tuple:
    """Fused residual-add + RMSNorm in a single Triton kernel.

    Computes: new_residual = residual + x
              normed = rmsnorm(new_residual, weight, eps)

    Args:
        residual: [..., D] tensor
        x: [..., D] tensor to add
        weight: [D] norm weight
        eps: epsilon
        add_one_to_weight: True for Qwen3.5's (1+weight) scaling

    Returns:
        (new_residual, normed)  - both [..., D] in BF16
    """
    orig_shape = residual.shape
    D = orig_shape[-1]
    r_2d = residual.reshape(-1, D)
    x_2d = x.reshape(-1, D)
    M = r_2d.shape[0]

    normed = torch.empty_like(r_2d, dtype=torch.bfloat16)
    new_r = torch.empty_like(r_2d, dtype=torch.bfloat16)

    BLOCK_D = triton.next_power_of_2(D)
    _fused_residual_rmsnorm_kernel[(M,)](
        r_2d, x_2d, weight, normed, new_r,
        r_2d.stride(0), x_2d.stride(0), normed.stride(0), new_r.stride(0),
        D, eps,
        BLOCK_D=BLOCK_D,
        ADD_ONE_TO_WEIGHT=add_one_to_weight,
    )

    return new_r.reshape(orig_shape), normed.reshape(orig_shape)
