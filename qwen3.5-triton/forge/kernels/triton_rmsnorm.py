"""
Triton RMSNorm kernel  - single kernel replaces PyTorch's 7-op RMSNorm.

Replaces: cast -> pow -> mean -> rsqrt -> mul -> mul -> cast
With: single Triton kernel (load + var + rsqrt + scale + store)

Used 128x per token in Qwen3.5-27B:
  - input_layernorm: 64 layers
  - post_attention_layernorm: 64 layers

Ported from qwen-benchmark, adapted for hidden_size=5120 (BLOCK_N=8192).

B200: 208 SMs, each row is 1 program → grid=(M,) where M=batch*seq.
For decode M=1, only 1 SM used  - but this kernel is launch-bound, not compute-bound.
The win is eliminating 6 kernel launches → 1.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, N, eps,
    BLOCK_N: tl.constexpr,
    ADD_ONE_TO_WEIGHT: tl.constexpr,
):
    """Fused RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * scale.

    Qwen3_5RMSNorm uses scale = (1 + weight) with weight init to zeros.
    Standard RMSNorm uses scale = weight with weight init to ones.
    ADD_ONE_TO_WEIGHT selects the Qwen3.5 variant.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Qwen3_5RMSNorm: scale = (1 + weight), standard: scale = weight
    if ADD_ONE_TO_WEIGHT:
        w = w + 1.0

    # Variance (mean of squares) + rsqrt
    var = tl.sum(x * x, axis=0) / N
    rrms = tl.rsqrt(var + eps)

    # Normalize and scale
    y = (x * rrms * w).to(tl.bfloat16)
    tl.store(Y_ptr + row * stride_x + offs, y, mask=mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_one_to_weight: bool = True,
) -> torch.Tensor:
    """Fused RMSNorm using single Triton kernel (replaces 7 PyTorch ops).

    Args:
        x: [..., hidden_size] input tensor (BF16)
        weight: [hidden_size] norm weight (BF16)
        eps: epsilon for numerical stability
        add_one_to_weight: if True, uses (1 + weight) scaling (Qwen3.5 variant).
                          Qwen3_5RMSNorm initializes weight=zeros and uses (1+w).

    Returns:
        [..., hidden_size] normalized tensor (BF16)
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d, dtype=torch.bfloat16)
    x_bf16 = x_2d.to(torch.bfloat16) if x_2d.dtype != torch.bfloat16 else x_2d
    w_bf16 = weight.to(torch.bfloat16) if weight.dtype != torch.bfloat16 else weight
    BLOCK_N = triton.next_power_of_2(N)
    _triton_rmsnorm_kernel[(M,)](
        x_bf16, w_bf16, y, x_bf16.stride(0), N, eps,
        BLOCK_N=BLOCK_N,
        ADD_ONE_TO_WEIGHT=add_one_to_weight,
    )
    return y.reshape(orig_shape)


# ============================================================================
# Standalone validation structure
# ============================================================================

class PytorchModel(torch.nn.Module):
    """Reference PyTorch RMSNorm."""
    def __init__(self, hidden_size: int = 5120, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        var = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(var + self.eps)
        return (x_normed * self.weight.float()).to(torch.bfloat16)


class TritonModel(torch.nn.Module):
    """Optimized RMSNorm using single Triton kernel."""
    def __init__(self, hidden_size: int = 5120, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rmsnorm(x, self.weight, self.eps)


def get_inputs():
    return [torch.randn(1, 1, 5120, device="cuda", dtype=torch.bfloat16)]


def get_init_inputs():
    return [5120, 1e-6]
