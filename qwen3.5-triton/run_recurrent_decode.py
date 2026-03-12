import torch

from forge.kernels.triton_deltanet_recurrent import (
    deltanet_recurrent_step,
    deltanet_recurrent_step_pytorch,
    NUM_HEADS,
    DK,
    DV,
)


def make_inputs(device="cuda", batch_size=1):
    B = batch_size

    q = torch.randn(B, NUM_HEADS, DK, device=device, dtype=torch.float32).mul_(0.5).to(torch.bfloat16)
    k = torch.randn(B, NUM_HEADS, DK, device=device, dtype=torch.float32).mul_(0.5).to(torch.bfloat16)
    v = torch.randn(B, NUM_HEADS, DV, device=device, dtype=torch.float32).mul_(0.5).to(torch.bfloat16)

    beta = torch.rand(B, NUM_HEADS, device=device, dtype=torch.float32).to(torch.bfloat16)
    gate = torch.randn(B, NUM_HEADS, device=device, dtype=torch.float32).mul_(0.02).to(torch.bfloat16)

    state = torch.zeros(B, NUM_HEADS, DK, DV, device=device, dtype=torch.bfloat16)
    return q, k, v, beta, gate, state


def tensor_stats(name, x):
    x_f = x.float()
    has_nan = torch.isnan(x_f).any().item()
    has_inf = torch.isinf(x_f).any().item()
    max_abs = x_f.abs().max().item()
    print(f"{name}: has_nan={has_nan}, has_inf={has_inf}, max_abs={max_abs:.6f}")


def check_correctness():
    print("=== Correctness Check ===")
    torch.manual_seed(0)

    q, k, v, beta, gate, state0 = make_inputs(device="cuda", batch_size=1)

    state_triton = state0.clone()
    out_triton, state_triton = deltanet_recurrent_step(q, k, v, beta, gate, state_triton)

    state_ref = state0.clone()
    out_ref, state_ref = deltanet_recurrent_step_pytorch(q, k, v, beta, gate, state_ref)

    tensor_stats("out_triton", out_triton)
    tensor_stats("state_triton", state_triton)
    tensor_stats("out_ref", out_ref)
    tensor_stats("state_ref", state_ref)

    if (
        torch.isnan(out_triton.float()).any()
        or torch.isnan(state_triton.float()).any()
        or torch.isnan(out_ref.float()).any()
        or torch.isnan(state_ref.float()).any()
    ):
        print("Correctness: FAIL (NaN detected)")
        return False

    if (
        torch.isinf(out_triton.float()).any()
        or torch.isinf(state_triton.float()).any()
        or torch.isinf(out_ref.float()).any()
        or torch.isinf(state_ref.float()).any()
    ):
        print("Correctness: FAIL (Inf detected)")
        return False

    out_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    state_diff = (state_triton.float() - state_ref.float()).abs().max().item()

    print(f"max |output_triton - output_ref| = {out_diff:.6f}")
    print(f"max |state_triton  - state_ref | = {state_diff:.6f}")

    passed = (out_diff < 5e-2 and state_diff < 5e-2)
    if passed:
        print("Correctness: PASS")
    else:
        print("Correctness: WARNING (difference may be too large)")
    return passed


def benchmark_one_impl(name, step_fn, num_warmup=100, num_iters=5000):
    print(f"\n=== Benchmark: {name} ===")
    torch.manual_seed(0)

    q, k, v, beta, gate, state = make_inputs(device="cuda", batch_size=1)

    # warmup
    for _ in range(num_warmup):
        _, state = step_fn(q, k, v, beta, gate, state)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _, state = step_fn(q, k, v, beta, gate, state)
    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_us = total_ms * 1000.0 / num_iters
    head_steps_per_s = (num_iters * NUM_HEADS) / (total_ms / 1000.0)

    print(f"num_iters = {num_iters}")
    print(f"total time = {total_ms:.3f} ms")
    print(f"avg latency per decode step = {avg_us:.3f} us")
    print(f"throughput = {head_steps_per_s:.2f} head-steps/s")

    return {
        "name": name,
        "total_ms": total_ms,
        "avg_us": avg_us,
        "throughput": head_steps_per_s,
    }


def benchmark_compare(num_warmup=100, num_iters=5000):
    print("\n=== Triton vs PyTorch Benchmark Comparison ===")

    triton_result = benchmark_one_impl(
        name="Triton recurrent kernel",
        step_fn=deltanet_recurrent_step,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    pytorch_result = benchmark_one_impl(
        name="PyTorch reference",
        step_fn=deltanet_recurrent_step_pytorch,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    speedup = pytorch_result["avg_us"] / triton_result["avg_us"]

    print("\n=== Summary ===")
    print(f"Triton avg latency : {triton_result['avg_us']:.3f} us/step")
    print(f"PyTorch avg latency: {pytorch_result['avg_us']:.3f} us/step")
    print(f"Speedup (PyTorch / Triton): {speedup:.2f}x")


def main():
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Config: B=1, H={NUM_HEADS}, DK={DK}, DV={DV}")

    passed = check_correctness()
    if not passed:
        print("\nCorrectness 未通过，benchmark 结果仅供参考。")

    benchmark_compare(num_warmup=100, num_iters=5000)


if __name__ == "__main__":
    main()