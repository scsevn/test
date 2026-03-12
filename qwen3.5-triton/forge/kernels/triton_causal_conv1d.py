"""
Fused Causal Conv1d Update  - depthwise conv1d + SiLU for DeltaNet decode.

DeltaNet uses depthwise conv1d with kernel_size=4 on projected QKV.
During decode: shift state window, insert new input, dot product + SiLU activation.

Replaces:
  - torch.roll on conv state
  - state[:, :, -1] = new_input
  - F.conv1d or manual dot product
  - F.silu activation
With: single fused Triton kernel

V5: Added multi-channel variant (Fix 3c).
  Original: 1 channel/block → 10240 blocks = 49 waves on 208 SMs.
  Multi-channel: CHANNELS_PER_BLOCK channels/block → fewer blocks, better wave efficiency.
  With CPB=8: 1280 blocks = 6 waves. Each block does 8x the work (32 multiply-adds).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_update_kernel(
    # Input: new value to insert [B, D]
    X_ptr,
    # Conv state: [B, D, kernel_size]  - ring buffer
    State_ptr,
    # Conv weight: [D, kernel_size]
    Weight_ptr,
    # Optional bias: [D]
    Bias_ptr,
    # Output: [B, D]
    Y_ptr,
    # Dimensions
    D, kernel_size,
    # Strides
    stride_xb, stride_xd,
    stride_sb, stride_sd, stride_sk,
    stride_wd, stride_wk,
    stride_yb, stride_yd,
    # Config
    HAS_BIAS: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    """Fused causal conv1d state update + convolution + optional SiLU.

    Per channel:
      1. Shift state left by 1 (discard oldest)
      2. Insert new input at position kernel_size-1
      3. Dot product: sum(state * weight)
      4. Optional: apply SiLU activation
    """
    pid_d = tl.program_id(0)  # channel index
    batch_id = tl.program_id(1)

    if pid_d >= D:
        return

    # Load new input for this channel
    x_val = tl.load(X_ptr + batch_id * stride_xb + pid_d * stride_xd).to(tl.float32)

    # Shift state left by 1 and insert new value
    # State is [B, D, kernel_size]  - we process one (batch, channel) slice
    state_base = State_ptr + batch_id * stride_sb + pid_d * stride_sd

    # Shift: state[i] = state[i+1] for i in 0..kernel_size-2, state[kernel_size-1] = x
    # We do this as: load all, shift in registers, store back
    # For kernel_size=4, this is just 4 loads + 4 stores
    acc = tl.zeros((), dtype=tl.float32)
    weight_base = Weight_ptr + pid_d * stride_wd

    # Process kernel_size positions
    # After shift: pos 0 gets old pos 1, pos 1 gets old pos 2, etc.
    # Final pos gets x_val
    for i in range(KERNEL_SIZE):
        if i < KERNEL_SIZE - 1:
            # Load from next position (shift left)
            s_val = tl.load(state_base + (i + 1) * stride_sk).to(tl.float32)
        else:
            # Last position gets new input
            s_val = x_val

        # Store shifted state
        tl.store(state_base + i * stride_sk, s_val.to(tl.bfloat16))

        # Accumulate convolution
        w_val = tl.load(weight_base + i * stride_wk).to(tl.float32)
        acc += s_val * w_val

    # Add bias
    if HAS_BIAS:
        bias_val = tl.load(Bias_ptr + pid_d).to(tl.float32)
        acc += bias_val

    # Apply SiLU: x * sigmoid(x)
    if APPLY_SILU:
        acc = acc * tl.sigmoid(acc)

    tl.store(Y_ptr + batch_id * stride_yb + pid_d * stride_yd, acc.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({"CHANNELS_PER_BLOCK": 1}, num_warps=1, num_stages=1),
        triton.Config({"CHANNELS_PER_BLOCK": 4}, num_warps=2, num_stages=1),
        triton.Config({"CHANNELS_PER_BLOCK": 8}, num_warps=4, num_stages=1),
        triton.Config({"CHANNELS_PER_BLOCK": 16}, num_warps=4, num_stages=1),
    ],
    key=["D", "KERNEL_SIZE"],
)
@triton.jit
def _causal_conv1d_update_multichannel_kernel(
    # Input: new value to insert [B, D]
    X_ptr,
    # Conv state: [B, D, kernel_size]  - ring buffer
    State_ptr,
    # Conv weight: [D, kernel_size]
    Weight_ptr,
    # Optional bias: [D]
    Bias_ptr,
    # Output: [B, D]
    Y_ptr,
    # Dimensions
    D, kernel_size,
    # Strides
    stride_xb, stride_xd,
    stride_sb, stride_sd, stride_sk,
    stride_wd, stride_wk,
    stride_yb, stride_yd,
    # Config
    HAS_BIAS: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    """Multi-channel fused causal conv1d (Fix 3c).

    Each block processes CHANNELS_PER_BLOCK channels instead of 1.
    Reduces grid from D blocks to D/CPB blocks (e.g., 10240→1280 with CPB=8).
    Each channel is still independent  - no cross-channel interaction.
    """
    pid = tl.program_id(0)    # block index
    batch_id = tl.program_id(1)

    # Channel indices for this block
    ch_offs = pid * CHANNELS_PER_BLOCK + tl.arange(0, CHANNELS_PER_BLOCK)
    ch_mask = ch_offs < D

    # Load new inputs for all channels in this block
    x_vals = tl.load(
        X_ptr + batch_id * stride_xb + ch_offs * stride_xd,
        mask=ch_mask, other=0.0,
    ).to(tl.float32)

    # Process each kernel position
    acc = tl.zeros((CHANNELS_PER_BLOCK,), dtype=tl.float32)

    for i in range(KERNEL_SIZE):
        if i < KERNEL_SIZE - 1:
            # Load from next position (shift left)
            s_vals = tl.load(
                State_ptr + batch_id * stride_sb + ch_offs * stride_sd + (i + 1) * stride_sk,
                mask=ch_mask, other=0.0,
            ).to(tl.float32)
        else:
            # Last position gets new input
            s_vals = x_vals

        # Store shifted state
        tl.store(
            State_ptr + batch_id * stride_sb + ch_offs * stride_sd + i * stride_sk,
            s_vals.to(tl.bfloat16),
            mask=ch_mask,
        )

        # Load weights and accumulate
        w_vals = tl.load(
            Weight_ptr + ch_offs * stride_wd + i * stride_wk,
            mask=ch_mask, other=0.0,
        ).to(tl.float32)
        acc += s_vals * w_vals

    # Add bias
    if HAS_BIAS:
        bias_vals = tl.load(Bias_ptr + ch_offs, mask=ch_mask, other=0.0).to(tl.float32)
        acc += bias_vals

    # Apply SiLU
    if APPLY_SILU:
        acc = acc * tl.sigmoid(acc)

    tl.store(
        Y_ptr + batch_id * stride_yb + ch_offs * stride_yd,
        acc.to(tl.bfloat16),
        mask=ch_mask,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    apply_silu: bool = True,
) -> tuple:
    """Fused causal conv1d state update + convolution + SiLU.

    Args:
        x: [B, D] new input to insert
        conv_state: [B, D, kernel_size] rolling state buffer (MODIFIED IN-PLACE)
        weight: [D, kernel_size] conv weights
        bias: optional [D] bias
        apply_silu: whether to apply SiLU activation

    Returns:
        y: [B, D] convolution output
        conv_state: same tensor, updated in-place
    """
    B, D = x.shape
    kernel_size = weight.shape[1]

    y = torch.empty_like(x, dtype=torch.bfloat16)

    grid = (D, B)

    _causal_conv1d_update_kernel[grid](
        x, conv_state, weight, bias if bias is not None else x,  # dummy for None
        y,
        D, kernel_size,
        x.stride(0), x.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None,
        APPLY_SILU=apply_silu,
        KERNEL_SIZE=kernel_size,
    )

    return y, conv_state


def causal_conv1d_update_multichannel(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    apply_silu: bool = True,
) -> tuple:
    """Multi-channel fused causal conv1d (Fix 3c).

    Same API as causal_conv1d_update but processes multiple channels per block.
    Reduces grid from D to D/CPB blocks for better wave efficiency on B200.

    Args:
        x: [B, D] new input to insert
        conv_state: [B, D, kernel_size] rolling state buffer (MODIFIED IN-PLACE)
        weight: [D, kernel_size] conv weights
        bias: optional [D] bias
        apply_silu: whether to apply SiLU activation

    Returns:
        y: [B, D] convolution output
        conv_state: same tensor, updated in-place
    """
    B, D = x.shape
    kernel_size = weight.shape[1]

    y = torch.empty_like(x, dtype=torch.bfloat16)

    # Grid: autotune picks CHANNELS_PER_BLOCK, we use max possible for grid size
    # The actual CPB is determined by autotune, but grid must use worst case (CPB=1)
    # Triton autotune handles this: grid lambda gets the meta dict
    grid = lambda meta: (triton.cdiv(D, meta["CHANNELS_PER_BLOCK"]), B)

    _causal_conv1d_update_multichannel_kernel[grid](
        x, conv_state, weight, bias if bias is not None else x,
        y,
        D, kernel_size,
        x.stride(0), x.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None,
        APPLY_SILU=apply_silu,
        KERNEL_SIZE=kernel_size,
    )

    return y, conv_state


# =============================================================================
# Standalone validation structure
# =============================================================================

DEFAULT_D = 10240  # DeltaNet total conv channels: Q(2048)+K(2048)+V(6144)
DEFAULT_KERNEL_SIZE = 4


class PytorchModel(torch.nn.Module):
    """Reference PyTorch causal conv1d update."""
    def __init__(self, D: int = DEFAULT_D, kernel_size: int = DEFAULT_KERNEL_SIZE):
        super().__init__()
        self.D = D
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(torch.randn(D, kernel_size, dtype=torch.bfloat16) * 0.02)
        self.register_buffer("conv_state", torch.zeros(1, D, kernel_size, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shift state left
        self.conv_state[:, :, :-1] = self.conv_state[:, :, 1:].clone()
        self.conv_state[:, :, -1] = x
        # Convolution: sum(state * weight, dim=-1)
        y = (self.conv_state.float() * self.weight.float().unsqueeze(0)).sum(dim=-1)
        # SiLU
        y = y * torch.sigmoid(y)
        return y.to(torch.bfloat16)


class TritonModel(torch.nn.Module):
    """Optimized causal conv1d using fused Triton kernel."""
    def __init__(self, D: int = DEFAULT_D, kernel_size: int = DEFAULT_KERNEL_SIZE):
        super().__init__()
        self.D = D
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(torch.randn(D, kernel_size, dtype=torch.bfloat16) * 0.02)
        self.register_buffer("conv_state", torch.zeros(1, D, kernel_size, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, self.conv_state = causal_conv1d_update(
            x, self.conv_state, self.weight, apply_silu=True,
        )
        return y


def get_inputs():
    return [torch.randn(1, DEFAULT_D, device="cuda", dtype=torch.bfloat16)]


def get_init_inputs():
    return [DEFAULT_D, DEFAULT_KERNEL_SIZE]
