"""
Triton RMSNormGated kernel  - fuses rmsnorm(hidden) * silu(gate) into 1 kernel.

Replaces ~10 PyTorch ops:
  norm_hidden = rmsnorm(hidden)  # 7 ops
  gate_act = silu(gate)          # 2 ops (sigmoid + mul)
  output = norm_hidden * gate_act # 1 op

Used 48x per token (DeltaNet output gate normalization in all 48 DeltaNet layers).

Data flow: load hidden + gate → variance → rsqrt → scale → silu(gate) → multiply → store

B200: Each row is independent → grid=(M,). BLOCK_N=8192 for hidden_size=5120.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rmsnorm_gated_kernel(
    H_ptr, G_ptr, W_ptr, Y_ptr,
    stride_h, stride_g, stride_y,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    """Fused RMSNorm * SiLU(gate): y = rmsnorm(h) * silu(g)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # Load hidden state and gate
    h = tl.load(H_ptr + row * stride_h + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + row * stride_g + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm on hidden
    var = tl.sum(h * h, axis=0) / N
    rrms = tl.rsqrt(var + eps)
    h_normed = h * rrms * w

    # SiLU(gate) = gate * sigmoid(gate)
    g_sigmoid = tl.sigmoid(g)
    g_silu = g * g_sigmoid

    # Fused multiply
    y = (h_normed * g_silu).to(tl.bfloat16)
    tl.store(Y_ptr + row * stride_y + offs, y, mask=mask)


def triton_rmsnorm_gated(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm * SiLU(gate) in a single Triton kernel.

    Args:
        hidden: [..., D] hidden state (BF16)
        gate: [..., D] gate tensor (BF16)
        weight: [D] norm weight (BF16)
        eps: epsilon

    Returns:
        [..., D] = rmsnorm(hidden, weight) * silu(gate) (BF16)
    """
    orig_shape = hidden.shape
    D = orig_shape[-1]
    h_2d = hidden.reshape(-1, D)
    g_2d = gate.reshape(-1, D)
    M = h_2d.shape[0]

    y = torch.empty_like(h_2d, dtype=torch.bfloat16)
    h_bf16 = h_2d.to(torch.bfloat16) if h_2d.dtype != torch.bfloat16 else h_2d
    g_bf16 = g_2d.to(torch.bfloat16) if g_2d.dtype != torch.bfloat16 else g_2d
    w_bf16 = weight.to(torch.bfloat16) if weight.dtype != torch.bfloat16 else weight

    BLOCK_N = triton.next_power_of_2(D)
    _triton_rmsnorm_gated_kernel[(M,)](
        h_bf16, g_bf16, w_bf16, y,
        h_bf16.stride(0), g_bf16.stride(0), y.stride(0),
        D, eps, BLOCK_N=BLOCK_N,
    )
    return y.reshape(orig_shape)


# ============================================================================
# Standalone validation structure
# ============================================================================

class PytorchModel(torch.nn.Module):
    """Reference PyTorch RMSNormGated."""
    def __init__(self, hidden_size: int = 5120, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h_f32 = hidden.float()
        var = h_f32.pow(2).mean(-1, keepdim=True)
        h_normed = h_f32 * torch.rsqrt(var + self.eps) * self.weight.float()
        g_silu = gate.float() * torch.sigmoid(gate.float())
        return (h_normed * g_silu).to(torch.bfloat16)


class TritonModel(torch.nn.Module):
    """Optimized RMSNormGated using single Triton kernel."""
    def __init__(self, hidden_size: int = 5120, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return triton_rmsnorm_gated(hidden, gate, self.weight, self.eps)


def get_inputs():
    return [
        torch.randn(1, 1, 5120, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 1, 5120, device="cuda", dtype=torch.bfloat16),
    ]


def get_init_inputs():
    return [5120, 1e-6]
