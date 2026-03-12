"""
Fused SiLU Gate MLP  - fuses silu(gate_proj(x)) * up_proj(x) elementwise.

For M=1 decode, this kernel takes the gate and up projection outputs
and fuses SiLU activation with element-wise multiply, eliminating the
need to materialize intermediate tensors.

Full fusion would do dual-GEMV + SiLU + multiply in one kernel, but
benchmarking shows separate GEMVs + fused elementwise is faster because:
- GEMVs benefit from split-K (different block counts)
- Elementwise is launch-bound, not compute-bound

This kernel handles the elementwise part:
  output[i] = silu(gate[i]) * up[i]
            = gate[i] * sigmoid(gate[i]) * up[i]

Saves ~70KB HBM traffic per layer by not writing+reading the intermediate
silu result (17408 * 2 bytes * 2 = 69.6KB).

Used 64x per token (every layer has MLP).
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_N": 512}, num_warps=2),
    ],
    key=["N"],
)
@triton.jit
def _silu_mul_kernel(
    Gate_ptr, Up_ptr, Out_ptr,
    N,
    stride_gate, stride_up, stride_out,
    BLOCK_N: tl.constexpr,
):
    """Fused SiLU(gate) * up: out = gate * sigmoid(gate) * up."""
    pid = tl.program_id(0)
    row = tl.program_id(1)

    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    gate = tl.load(Gate_ptr + row * stride_gate + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up_ptr + row * stride_up + offs, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) * up
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up

    tl.store(Out_ptr + row * stride_out + offs, out.to(tl.bfloat16), mask=mask)


def fused_silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Fused SiLU(gate) * up in a single Triton kernel.

    Args:
        gate: [..., N] gate projection output (BF16)
        up: [..., N] up projection output (BF16)

    Returns:
        [..., N] = silu(gate) * up (BF16)
    """
    orig_shape = gate.shape
    N = orig_shape[-1]
    gate_2d = gate.reshape(-1, N)
    up_2d = up.reshape(-1, N)
    M = gate_2d.shape[0]

    out = torch.empty_like(gate_2d, dtype=torch.bfloat16)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), M)

    _silu_mul_kernel[grid](
        gate_2d, up_2d, out,
        N,
        gate_2d.stride(0), up_2d.stride(0), out.stride(0),
    )

    return out.reshape(orig_shape)


# =============================================================================
# Fused dual-GEMV + SiLU + multiply variant (experimental)
# For cases where gate_proj and up_proj share the same input x
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def _fused_gate_up_silu_gemv_kernel(
    X_ptr, Wg_ptr, Wu_ptr, Y_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_wgk, stride_wgn,
    stride_wuk, stride_wun,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused gate_proj + up_proj + SiLU + multiply for M=1 GEMV.

    Each program computes one BLOCK_N tile:
      gate_tile = x @ Wg[:, n_tile]
      up_tile = x @ Wu[:, n_tile]
      out_tile = silu(gate_tile) * up_tile

    Reads x once per N-tile (x lives in L2 after first read).
    """
    pid_n = tl.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        k_mask = rk < K

        x = tl.load(X_ptr + rk, mask=k_mask, other=0.0)

        wg_ptrs = Wg_ptr + rk[:, None] * stride_wgk + rn[None, :] * stride_wgn
        wg_mask = k_mask[:, None] & n_mask[None, :]
        wg = tl.load(wg_ptrs, mask=wg_mask, other=0.0)

        wu_ptrs = Wu_ptr + rk[:, None] * stride_wuk + rn[None, :] * stride_wun
        wu = tl.load(wu_ptrs, mask=wg_mask, other=0.0)

        acc_gate += tl.sum(x[:, None] * wg, axis=0)
        acc_up += tl.sum(x[:, None] * wu, axis=0)

    # Fused SiLU(gate) * up
    silu_gate = acc_gate * tl.sigmoid(acc_gate)
    out = silu_gate * acc_up

    tl.store(Y_ptr + rn, out.to(tl.bfloat16), mask=n_mask)


def fused_gate_up_silu_gemv(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> torch.Tensor:
    """Fused gate+up projection + SiLU + multiply for M=1 decode.

    Args:
        x: [K] input vector (BF16)
        w_gate: [N, K] gate projection weight (BF16)
        w_up: [N, K] up projection weight (BF16)

    Returns:
        [N] = silu(x @ Wg^T) * (x @ Wu^T) (BF16)
    """
    N, K = w_gate.shape
    wg_t = w_gate.t().contiguous()
    wu_t = w_up.t().contiguous()

    y = torch.empty(N, device=x.device, dtype=torch.bfloat16)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)

    _fused_gate_up_silu_gemv_kernel[grid](
        x, wg_t, wu_t, y,
        N, K,
        wg_t.stride(0), wg_t.stride(1),
        wu_t.stride(0), wu_t.stride(1),
    )

    return y


# =============================================================================
# Standalone validation structure
# =============================================================================

DEFAULT_K = 5120
DEFAULT_N = 17408  # intermediate_size


class PytorchModel(torch.nn.Module):
    """Reference PyTorch SiLU MLP gate."""
    def __init__(self, K: int = DEFAULT_K, N: int = DEFAULT_N):
        super().__init__()
        self.K = K
        self.N = N

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


class TritonModel(torch.nn.Module):
    """Optimized fused SiLU * multiply."""
    def __init__(self, K: int = DEFAULT_K, N: int = DEFAULT_N):
        super().__init__()
        self.K = K
        self.N = N

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return fused_silu_mul(gate, up)


def get_inputs():
    return [
        torch.randn(1, DEFAULT_N, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, DEFAULT_N, device="cuda", dtype=torch.bfloat16),
    ]


def get_init_inputs():
    return [DEFAULT_K, DEFAULT_N]


# =============================================================================
# Fused sigmoid(gate) * attn_output  - for attention output gating
# Replaces 4 PyTorch ops (float + sigmoid + to + mul) with 1 Triton kernel.
# Used 16x per token (once per attention layer).
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_N": 4096}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _sigmoid_gate_mul_kernel(
    A_ptr, G_ptr, Out_ptr,
    N,
    stride_a, stride_g, stride_out,
    BLOCK_N: tl.constexpr,
):
    """Fused: out = attn_output * sigmoid(gate). All in FP32, store BF16."""
    row = tl.program_id(1)
    pid_n = tl.program_id(0)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    a = tl.load(A_ptr + row * stride_a + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + row * stride_g + offs, mask=mask, other=0.0).to(tl.float32)
    out = a * tl.sigmoid(g)
    tl.store(Out_ptr + row * stride_out + offs, out.to(tl.bfloat16), mask=mask)


def fused_sigmoid_mul(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fused sigmoid(gate) * attn_output in a single Triton kernel.

    Replaces: attn_output * torch.sigmoid(gate.float()).to(attn_output.dtype)

    Args:
        attn_output: [..., D] attention output (BF16)
        gate: [..., D] gate tensor (BF16)

    Returns:
        [..., D] = attn_output * sigmoid(gate) (BF16)
    """
    orig_shape = attn_output.shape
    N = orig_shape[-1]
    a_2d = attn_output.reshape(-1, N)
    g_2d = gate.reshape(-1, N)
    M = a_2d.shape[0]

    out = torch.empty_like(a_2d, dtype=torch.bfloat16)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), M)

    _sigmoid_gate_mul_kernel[grid](
        a_2d, g_2d, out,
        N,
        a_2d.stride(0), g_2d.stride(0), out.stride(0),
    )

    return out.reshape(orig_shape)
