import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _deltanet_recurrent_kernel(
    Q_ptr, K_ptr, V_ptr, Beta_ptr, G_ptr,
    S_ptr,
    O_ptr,

    Dk: tl.constexpr,
    Dv: tl.constexpr,

    stride_sb, stride_sh, stride_sdk, stride_sdv,

    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,

    stride_ob, stride_oh, stride_od,

    stride_betab, stride_betah,
    stride_gb, stride_gh,

    BLOCK_DV: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    """
    One program handles one (batch, head).
    State layout: [B, H, Dk, Dv]
    """

    head_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    beta = tl.load(
        Beta_ptr + batch_id * stride_betab + head_id * stride_betah
    ).to(tl.float32)

    g_raw = tl.load(
        G_ptr + batch_id * stride_gb + head_id * stride_gh
    ).to(tl.float32)

    # Stable -softplus(g)
    # softplus(x) = log(1 + exp(x))
    # stable form:
    # softplus(x) = max(x, 0) + log(1 + exp(-abs(x)))
    g_abs = tl.abs(g_raw)
    g_pos = tl.maximum(g_raw, 0.0)
    softplus_g = g_pos + tl.log(1.0 + tl.exp(-g_abs))
    g = -softplus_g

    decay = tl.exp(g)  # in (0, 1)

    dv_offs = tl.arange(0, BLOCK_DV)
    dv_mask = dv_offs < Dv

    q_base = Q_ptr + batch_id * stride_qb + head_id * stride_qh
    k_base = K_ptr + batch_id * stride_kb + head_id * stride_kh
    s_base = S_ptr + batch_id * stride_sb + head_id * stride_sh

    v = tl.load(
        V_ptr + batch_id * stride_vb + head_id * stride_vh + dv_offs * stride_vd,
        mask=dv_mask,
        other=0.0,
    ).to(tl.float32)

    accumulated = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    # Pass 1: decay state + accumulate S^T @ k
    for r_start in range(0, Dk, BLOCK_DK):
        dk_offs = r_start + tl.arange(0, BLOCK_DK)
        dk_mask = dk_offs < Dk

        s_ptrs = s_base + dk_offs[:, None] * stride_sdk + dv_offs[None, :] * stride_sdv
        s_mask = dk_mask[:, None] & dv_mask[None, :]

        s_tile = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        s_tile = s_tile * decay

        k_tile = tl.load(
            k_base + dk_offs * stride_kd,
            mask=dk_mask,
            other=0.0,
        ).to(tl.float32)

        accumulated += tl.sum(s_tile * k_tile[:, None], axis=0)

        tl.store(s_ptrs, s_tile.to(tl.bfloat16), mask=s_mask)

    delta = beta * (v - accumulated)

    output = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    # Pass 2: rank-1 update + output
    for r_start in range(0, Dk, BLOCK_DK):
        dk_offs = r_start + tl.arange(0, BLOCK_DK)
        dk_mask = dk_offs < Dk

        s_ptrs = s_base + dk_offs[:, None] * stride_sdk + dv_offs[None, :] * stride_sdv
        s_mask = dk_mask[:, None] & dv_mask[None, :]

        s_tile = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

        k_tile = tl.load(
            k_base + dk_offs * stride_kd,
            mask=dk_mask,
            other=0.0,
        ).to(tl.float32)

        s_tile += k_tile[:, None] * delta[None, :]

        tl.store(s_ptrs, s_tile.to(tl.bfloat16), mask=s_mask)

        q_tile = tl.load(
            q_base + dk_offs * stride_qd,
            mask=dk_mask,
            other=0.0,
        ).to(tl.float32)

        output += tl.sum(s_tile * q_tile[:, None], axis=0)

    o_ptrs = O_ptr + batch_id * stride_ob + head_id * stride_oh + dv_offs * stride_od
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=dv_mask)


def deltanet_recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
):
    """
    q:    [B, H, Dk] bf16
    k:    [B, H, Dk] bf16
    v:    [B, H, Dv] bf16
    beta: [B, H]     bf16/float32
    gate: [B, H]     bf16/float32
    state:[B, H, Dk, Dv] bf16, updated in-place
    """
    B, num_heads, Dk = q.shape
    Dv = v.shape[-1]

    assert state.shape == (B, num_heads, Dk, Dv)
    assert Dv == 128, "当前 baseline 固定 BLOCK_DV=128，只支持 Dv=128"
    assert q.is_cuda and k.is_cuda and v.is_cuda and beta.is_cuda and gate.is_cuda and state.is_cuda

    output = torch.empty(
        B, num_heads, Dv,
        device=q.device,
        dtype=torch.bfloat16,
    )

    grid = (num_heads, B)

    # 固定一个安全 config，先保证 correctness
    _deltanet_recurrent_kernel[grid](
        q, k, v, beta, gate,
        state,
        output,

        Dk, Dv,

        state.stride(0), state.stride(1), state.stride(2), state.stride(3),

        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),

        output.stride(0), output.stride(1), output.stride(2),

        beta.stride(0), beta.stride(1),
        gate.stride(0), gate.stride(1),

        BLOCK_DV=128,
        BLOCK_DK=16,
        num_warps=4,
        num_stages=1,
    )

    return output, state


def deltanet_recurrent_step_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
):
    # reference uses the same gate transform: decay = exp(-softplus(g))
    decay = torch.exp(-F.softplus(gate.float())).unsqueeze(-1).unsqueeze(-1)

    state_f = state.float() * decay
    stk = torch.einsum("bhkd,bhk->bhd", state_f, k.float())
    residual = v.float() - stk
    delta = beta.float().unsqueeze(-1) * residual
    state_f = state_f + torch.einsum("bhk,bhd->bhkd", k.float(), delta)
    output = torch.einsum("bhkd,bhk->bhd", state_f, q.float())

    return output.to(torch.bfloat16), state_f.to(torch.bfloat16)


NUM_HEADS = 48
DK = 128
DV = 128