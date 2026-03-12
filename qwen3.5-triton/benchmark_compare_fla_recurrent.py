import torch
import torch.nn.functional as F

from forge.kernels.triton_deltanet_recurrent import (
    deltanet_recurrent_step,
    NUM_HEADS,
    DK,
    DV,
)

# 这里要求你的环境里已经能 import fla
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule


def make_inputs(device="cuda", batch_size=1):
    B = batch_size

    q = torch.randn(B, NUM_HEADS, DK, device=device, dtype=torch.float32).mul_(0.5).to(torch.bfloat16)
    k = torch.randn(B, NUM_HEADS, DK, device=device, dtype=torch.float32).mul_(0.5).to(torch.bfloat16)
    v = torch.randn(B, NUM_HEADS, DV, device=device, dtype=torch.float32).mul_(0.5).to(torch.bfloat16)

    beta = torch.rand(B, NUM_HEADS, device=device, dtype=torch.float32).to(torch.bfloat16)
    gate = torch.randn(B, NUM_HEADS, device=device, dtype=torch.float32).mul_(0.02).to(torch.bfloat16)

    state = torch.zeros(B, NUM_HEADS, DK, DV, device=device, dtype=torch.bfloat16)
    return q, k, v, beta, gate, state


def fla_recurrent_step_from_raw_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate_raw: torch.Tensor,
    state: torch.Tensor,
):
    """
    把你当前实现的一步输入，包装成 FLA fused_recurrent 的 T=1 调用。
    q/k:   [B, H, DK]
    v:     [B, H, DV]
    beta:  [B, H]
    gate:  [B, H]  raw gate
    state: [B, H, DK, DV]
    """
    # FLA 的 g 是 log-decay，不是 raw gate
    g_fla = -F.softplus(gate_raw.float())   # [B, H]

    q_fla = q.unsqueeze(1).contiguous()         # [B, 1, H, DK]
    k_fla = k.unsqueeze(1).contiguous()         # [B, 1, H, DK]
    v_fla = v.unsqueeze(1).contiguous()         # [B, 1, H, DV]
    beta_fla = beta.unsqueeze(1).contiguous()   # [B, 1, H]
    g_fla = g_fla.unsqueeze(1).contiguous()     # [B, 1, H]

    # 为了和你当前 kernel 对齐，scale 必须设成 1.0
    o, final_state = fused_recurrent_gated_delta_rule(
        q=q_fla,
        k=k_fla,
        v=v_fla,
        g=g_fla,
        beta=beta_fla,
        scale=1.0,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )
    # o: [B, 1, H, DV] -> [B, H, DV]
    return o[:, 0].contiguous(), final_state


def tensor_stats(name, x):
    x_f = x.float()
    has_nan = torch.isnan(x_f).any().item()
    has_inf = torch.isinf(x_f).any().item()
    max_abs = x_f.abs().max().item()
    print(f"{name}: has_nan={has_nan}, has_inf={has_inf}, max_abs={max_abs:.6f}")


@torch.no_grad()
def check_correctness():
    print("=== Correctness: current vs FLA fused recurrent ===")
    torch.manual_seed(0)

    q, k, v, beta, gate, state0 = make_inputs(device="cuda", batch_size=1)

    state_cur = state0.clone()
    out_cur, state_cur = deltanet_recurrent_step(q, k, v, beta, gate, state_cur)

    state_fla = state0.clone()
    out_fla, state_fla = fla_recurrent_step_from_raw_gate(q, k, v, beta, gate, state_fla)

    tensor_stats("out_current", out_cur)
    tensor_stats("state_current", state_cur)
    tensor_stats("out_fla", out_fla)
    tensor_stats("state_fla", state_fla)

    out_diff = (out_cur.float() - out_fla.float()).abs().max().item()
    state_diff = (state_cur.float() - state_fla.float()).abs().max().item()

    print(f"max |out_current - out_fla|   = {out_diff:.6f}")
    print(f"max |state_current - state_fla| = {state_diff:.6f}")


@torch.no_grad()
def benchmark_current(num_warmup=100, num_iters=5000):
    q, k, v, beta, gate, state = make_inputs(device="cuda", batch_size=1)

    for _ in range(num_warmup):
        _, state = deltanet_recurrent_step(q, k, v, beta, gate, state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _, state = deltanet_recurrent_step(q, k, v, beta, gate, state)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_us = total_ms * 1000.0 / num_iters
    return total_ms, avg_us


@torch.no_grad()
def benchmark_fla(num_warmup=100, num_iters=5000):
    q, k, v, beta, gate, state = make_inputs(device="cuda", batch_size=1)

    # 预先把 T=1 版本准备好，避免把 unsqueeze 的开销放进循环里
    g_fla = -F.softplus(gate.float())
    q_fla = q.unsqueeze(1).contiguous()
    k_fla = k.unsqueeze(1).contiguous()
    v_fla = v.unsqueeze(1).contiguous()
    beta_fla = beta.unsqueeze(1).contiguous()
    g_fla = g_fla.unsqueeze(1).contiguous()

    for _ in range(num_warmup):
        _, state = fused_recurrent_gated_delta_rule(
            q=q_fla,
            k=k_fla,
            v=v_fla,
            g=g_fla,
            beta=beta_fla,
            scale=1.0,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _, state = fused_recurrent_gated_delta_rule(
            q=q_fla,
            k=k_fla,
            v=v_fla,
            g=g_fla,
            beta=beta_fla,
            scale=1.0,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
        )
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_us = total_ms * 1000.0 / num_iters
    return total_ms, avg_us


def main():
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Config: B=1, H={NUM_HEADS}, DK={DK}, DV={DV}")

    check_correctness()

    print("\n=== Benchmark: current recurrent vs FLA fused recurrent ===")
    total_ms_cur, avg_us_cur = benchmark_current()
    total_ms_fla, avg_us_fla = benchmark_fla()

    print(f"current total   = {total_ms_cur:.3f} ms")
    print(f"current avg     = {avg_us_cur:.3f} us/step")

    print(f"fla total       = {total_ms_fla:.3f} ms")
    print(f"fla avg         = {avg_us_fla:.3f} us/step")

    print(f"speedup (current / fla) = {avg_us_cur / avg_us_fla:.3f}x")
    print(f"speedup (fla / current) = {avg_us_fla / avg_us_cur:.3f}x")


if __name__ == "__main__":
    main()