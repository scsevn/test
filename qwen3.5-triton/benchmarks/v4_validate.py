"""
V4 Validation & Benchmark  - Correctness + Performance on B200.

Three validation stages:
  1. Kernel correctness: fused_postproj_recurrent vs separate kernels (cosine > 0.9999)
  2. E2E correctness: patched v4 vs unpatched HF (identical greedy tokens)
  3. State drift: 1000 decode steps, compare accumulated recurrent state
  4. Benchmark: 3 warmup + 20 timed runs, report tok/s and ITL

Run via Modal:
    modal run benchmarks/v4_validate.py
    modal run benchmarks/v4_validate.py::main --correctness-only
    modal run benchmarks/v4_validate.py::main --benchmark-only
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import modal
from modal_config import (
    app, forge_llm_image, model_cache, results_volume, hf_secret,
    MODEL_CACHE_PATH, RESULTS_PATH, GPU_CONFIG, MODEL_ID,
)

PROMPT = "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves."
MAX_NEW_TOKENS = 256
NUM_RUNS = 20
WARMUP_RUNS = 3


# =========================================================================
# Stage 1: Kernel-level correctness
# =========================================================================
@app.function(
    image=forge_llm_image,
    gpu=GPU_CONFIG["optimized"]["gpu"],
    timeout=GPU_CONFIG["optimized"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[hf_secret],
)
def validate_fused_kernel():
    """Compare fused_postproj_recurrent vs separate deltanet_post_proj + deltanet_recurrent_step."""
    import torch
    import torch.nn.functional as F

    from forge.kernels.triton_deltanet_prep import deltanet_post_proj
    from forge.kernels.triton_deltanet_recurrent import deltanet_recurrent_step
    from forge.kernels.triton_deltanet_fused import fused_postproj_recurrent
    from forge.kernels.triton_lm_head_topk import fused_lm_head_argmax, fused_lm_head_argmax_static
    import triton

    device = "cuda"
    torch.manual_seed(42)

    NUM_K_HEADS = 16
    NUM_V_HEADS = 48
    HEAD_DIM = 128
    KEY_DIM = NUM_K_HEADS * HEAD_DIM  # 2048
    VALUE_DIM = NUM_V_HEADS * HEAD_DIM  # 6144
    CONV_DIM = KEY_DIM + KEY_DIM + VALUE_DIM  # 10240

    print("=" * 60)
    print("STAGE 1: Kernel-level correctness")
    print("=" * 60)

    # --- Test 1a: Fused post_proj + recurrent vs separate ---
    print("\n[1a] fused_postproj_recurrent vs separate kernels")

    num_steps = 100
    max_cosine_diff = 0.0
    max_state_diff = 0.0

    for step in range(num_steps):
        B = 1
        qkv_conv = torch.randn(B, CONV_DIM, device=device, dtype=torch.bfloat16)
        alpha = torch.randn(B, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
        beta_raw = torch.randn(B, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
        neg_A_exp = -torch.rand(NUM_V_HEADS, device=device, dtype=torch.float32).exp()
        dt_bias_f = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32) * 0.1

        # Two identical initial states
        state_sep = torch.randn(B, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.bfloat16)
        state_fused = state_sep.clone()

        # Separate path
        q, k, v, gate, beta = deltanet_post_proj(qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias_f)
        out_sep, state_sep = deltanet_recurrent_step(q, k, v, beta, gate, state_sep)

        # Fused path
        out_fused, state_fused = fused_postproj_recurrent(
            qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias_f, state_fused,
        )

        # Compare outputs
        cos_sim = F.cosine_similarity(
            out_sep.flatten().float(),
            out_fused.flatten().float(),
            dim=0,
        ).item()

        state_cos = F.cosine_similarity(
            state_sep.flatten().float(),
            state_fused.flatten().float(),
            dim=0,
        ).item()

        max_cosine_diff = max(max_cosine_diff, 1.0 - cos_sim)
        max_state_diff = max(max_state_diff, 1.0 - state_cos)

        if step < 3 or step == num_steps - 1:
            print(f"  step {step:3d}: output cos={cos_sim:.6f}, state cos={state_cos:.6f}")

    output_pass = (1.0 - max_cosine_diff) > 0.9999
    state_pass = (1.0 - max_state_diff) > 0.9999
    print(f"  Worst output cosine: {1.0 - max_cosine_diff:.6f} {'PASS' if output_pass else 'FAIL'}")
    print(f"  Worst state cosine:  {1.0 - max_state_diff:.6f} {'PASS' if state_pass else 'FAIL'}")

    # --- Test 1b: State drift over many steps ---
    print(f"\n[1b] State drift: {1000} accumulated steps")

    state_sep = torch.zeros(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.bfloat16)
    state_fused = state_sep.clone()

    torch.manual_seed(123)
    for step in range(1000):
        qkv_conv = torch.randn(1, CONV_DIM, device=device, dtype=torch.bfloat16)
        alpha = torch.randn(1, NUM_V_HEADS, device=device, dtype=torch.bfloat16)
        beta_raw = torch.randn(1, NUM_V_HEADS, device=device, dtype=torch.bfloat16)

        # Separate
        q, k, v, gate, beta = deltanet_post_proj(qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias_f)
        out_sep, state_sep = deltanet_recurrent_step(q, k, v, beta, gate, state_sep)

        # Fused
        out_fused, state_fused = fused_postproj_recurrent(
            qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias_f, state_fused,
        )

        if step % 250 == 0 or step == 999:
            s_cos = F.cosine_similarity(
                state_sep.flatten().float(), state_fused.flatten().float(), dim=0
            ).item()
            o_cos = F.cosine_similarity(
                out_sep.flatten().float(), out_fused.flatten().float(), dim=0
            ).item()
            print(f"  step {step:4d}: output cos={o_cos:.6f}, state cos={s_cos:.6f}")

    final_state_cos = F.cosine_similarity(
        state_sep.flatten().float(), state_fused.flatten().float(), dim=0
    ).item()
    drift_pass = final_state_cos > 0.999
    print(f"  Final state cosine after 1000 steps: {final_state_cos:.6f} {'PASS' if drift_pass else 'FAIL'}")

    # --- Test 1c: Fused lm_head argmax (static) vs F.linear + argmax ---
    print(f"\n[1c] fused_lm_head_argmax_static vs F.linear + argmax")

    VOCAB = 248320
    HIDDEN = 5120
    torch.manual_seed(42)
    lm_weight = torch.randn(VOCAB, HIDDEN, device=device, dtype=torch.bfloat16) * 0.02
    lm_w_t = lm_weight.t().contiguous()
    max_tiles = triton.cdiv(VOCAB, 128)
    local_max = torch.empty(max_tiles, device=device, dtype=torch.float32)
    local_idx = torch.empty(max_tiles, device=device, dtype=torch.int64)
    result_buf = torch.empty(1, device=device, dtype=torch.int64)

    lm_match = 0
    lm_total = 20
    for i in range(lm_total):
        x = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)

        # Reference
        logits = F.linear(x.unsqueeze(0), lm_weight)
        ref_idx = logits.argmax(dim=-1).item()

        # Fused static
        fused_lm_head_argmax_static(x, lm_w_t, local_max, local_idx, result_buf)
        fused_idx = result_buf.item()

        match = ref_idx == fused_idx
        lm_match += int(match)
        if i < 5 or not match:
            print(f"  sample {i}: ref={ref_idx}, fused={fused_idx} {'MATCH' if match else 'MISMATCH'}")

    lm_pass = lm_match == lm_total
    print(f"  Matches: {lm_match}/{lm_total} {'PASS' if lm_pass else 'FAIL'}")

    results = {
        "fused_kernel_output_cosine": 1.0 - max_cosine_diff,
        "fused_kernel_state_cosine": 1.0 - max_state_diff,
        "state_drift_1000_steps": final_state_cos,
        "lm_head_argmax_match": f"{lm_match}/{lm_total}",
        "all_pass": output_pass and state_pass and drift_pass and lm_pass,
    }
    print(f"\nStage 1 result: {'ALL PASS' if results['all_pass'] else 'SOME FAILURES'}")
    return results


# =========================================================================
# Stage 2: E2E correctness (v4 patched vs unpatched HF)
# =========================================================================
@app.function(
    image=forge_llm_image,
    gpu=GPU_CONFIG["optimized"]["gpu"],
    timeout=GPU_CONFIG["optimized"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[hf_secret],
)
def validate_e2e():
    """Compare v4 patched greedy output vs unpatched HF generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from forge.llm.patch_qwen35 import load_model, patch_model
    from forge.llm.cache import HybridCache
    from forge.llm.generate import generate_eager

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    print("=" * 60)
    print("STAGE 2: E2E correctness (v4 vs unpatched)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    # --- Unpatched baseline ---
    print("\nLoading unpatched model...")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
        cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )
    model_base.requires_grad_(False)

    with torch.no_grad():
        base_output = model_base.generate(
            input_ids, max_new_tokens=64, do_sample=False, use_cache=True,
        )
    base_tokens = base_output[0].tolist()
    base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
    print(f"Baseline ({len(base_tokens)} tokens): {base_text[:200]}...")

    del model_base
    torch.cuda.empty_cache()

    # --- V4 patched ---
    print("\nLoading and patching model (v4)...")
    model = load_model(model_id=MODEL_ID, cache_dir=MODEL_CACHE_PATH,
                        device=device, hf_token=hf_token)
    cache = HybridCache(batch_size=1, max_cache_len=512, device=device)
    model = patch_model(model, cache)

    patched_ids, timing = generate_eager(
        model, input_ids, max_new_tokens=64, cache=cache, device=device,
    )
    patched_tokens = patched_ids[0].tolist()
    patched_text = tokenizer.decode(patched_ids[0], skip_special_tokens=True)
    print(f"Patched  ({len(patched_tokens)} tokens): {patched_text[:200]}...")

    # Compare
    exact_match = base_tokens == patched_tokens
    print(f"\nExact token match: {exact_match}")

    first_diff = None
    min_len = min(len(base_tokens), len(patched_tokens))
    matching = 0
    for i in range(min_len):
        if base_tokens[i] == patched_tokens[i]:
            matching += 1
        elif first_diff is None:
            first_diff = i

    if first_diff is not None:
        prompt_len = input_ids.shape[1]
        print(f"First mismatch at token index {first_diff} (decode step {first_diff - prompt_len})")
        print(f"  Base:    ...{base_tokens[max(0,first_diff-2):first_diff+3]}")
        print(f"  Patched: ...{patched_tokens[max(0,first_diff-2):first_diff+3]}")

    match_pct = matching / min_len * 100 if min_len > 0 else 0
    print(f"Token match rate: {matching}/{min_len} ({match_pct:.1f}%)")

    del model
    torch.cuda.empty_cache()

    results = {
        "exact_match": exact_match,
        "token_match_rate": round(match_pct, 1),
        "first_diff_index": first_diff,
        "base_len": len(base_tokens),
        "patched_len": len(patched_tokens),
        "base_text": base_text[:500],
        "patched_text": patched_text[:500],
    }
    print(f"\nStage 2 result: {'PASS' if exact_match else f'PARTIAL ({match_pct:.1f}% match)'}")
    return results


# =========================================================================
# Stage 3: Benchmark (eager + CUDA graph)
# =========================================================================
@app.function(
    image=forge_llm_image,
    gpu=GPU_CONFIG["optimized"]["gpu"],
    timeout=GPU_CONFIG["optimized"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results_volume},
    secrets=[hf_secret],
)
def run_v4_benchmark():
    """Full v4 benchmark: eager + CUDA graph, 20 timed runs."""
    import torch
    from transformers import AutoTokenizer

    from forge.llm.patch_qwen35 import load_model, patch_model, verify_patch
    from forge.llm.cache import HybridCache
    from forge.llm.generate import benchmark_decode, save_results

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    print("=" * 60)
    print("STAGE 3: V4 Benchmark")
    print("=" * 60)

    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"VRAM: {total_mem / (1024**3):.1f}GB")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )

    t0 = time.time()
    model = load_model(
        model_id=MODEL_ID, cache_dir=MODEL_CACHE_PATH,
        device=device, hf_token=hf_token,
    )
    load_time = time.time() - t0

    cache = HybridCache(batch_size=1, max_cache_len=MAX_NEW_TOKENS + 256, device=device)
    print(f"Cache allocated: {cache.memory_mb():.1f}MB")

    t0 = time.time()
    model = patch_model(model, cache)
    patch_time = time.time() - t0
    print(f"Patching took {patch_time:.1f}s")

    assert verify_patch(model), "Patch verification failed!"
    vram_after = torch.cuda.memory_allocated() / (1024**3)
    print(f"VRAM after patching: {vram_after:.1f}GB")

    # --- Eager benchmark ---
    print(f"\n{'='*60}")
    print("EAGER BENCHMARK (no CUDA graph)")
    print(f"{'='*60}")

    eager_results = benchmark_decode(
        model, tokenizer, prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS, num_runs=NUM_RUNS,
        warmup_runs=WARMUP_RUNS, use_cuda_graph=False,
        cache=cache, device=device,
    )
    eager_results["method"] = "triton_v4_eager"

    # --- CUDA Graph benchmark ---
    print(f"\n{'='*60}")
    print("CUDA GRAPH BENCHMARK")
    print(f"{'='*60}")

    graph_results = benchmark_decode(
        model, tokenizer, prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS, num_runs=NUM_RUNS,
        warmup_runs=WARMUP_RUNS, use_cuda_graph=True,
        cache=cache, device=device,
    )
    graph_results["method"] = "triton_v4_cuda_graph"

    # --- Load v3 baseline for comparison ---
    v3_path = f"{RESULTS_PATH}/optimized_results.json"
    v3_baseline = None
    if os.path.exists(v3_path):
        with open(v3_path) as f:
            v3_baseline = json.load(f)

    hf_path = f"{RESULTS_PATH}/baseline_results.json"
    hf_baseline = None
    if os.path.exists(hf_path):
        with open(hf_path) as f:
            hf_baseline = json.load(f)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"V4 RESULTS - {MODEL_ID} on {torch.cuda.get_device_name()}")
    print(f"{'='*60}")

    e_tok = eager_results['decode_tok_s']['mean']
    e_itl = eager_results['mean_itl_ms']['mean']
    g_tok = graph_results['decode_tok_s']['mean']
    g_itl = graph_results['mean_itl_ms']['mean']

    print(f"\nEager:      {e_tok:.1f} tok/s, ITL={e_itl:.2f}ms")
    print(f"CUDA Graph: {g_tok:.1f} tok/s, ITL={g_itl:.2f}ms")

    if v3_baseline and v3_baseline.get("cuda_graph"):
        v3_tok = v3_baseline["cuda_graph"]["decode_tok_s"]["mean"]
        v3_itl = v3_baseline["cuda_graph"]["mean_itl_ms"]["mean"]
        improvement = (g_tok - v3_tok) / v3_tok * 100
        itl_saved = v3_itl - g_itl
        print(f"\nv3 baseline: {v3_tok:.1f} tok/s, ITL={v3_itl:.2f}ms")
        print(f"v4 vs v3:    {improvement:+.1f}% ({itl_saved:+.2f}ms ITL)")

    if hf_baseline:
        hf_tok = hf_baseline["decode_tok_s"]["mean"]
        speedup = g_tok / hf_tok
        print(f"HF baseline: {hf_tok:.1f} tok/s")
        print(f"v4 vs HF:    {speedup:.2f}x")

    print(f"\nPeak VRAM: {graph_results['peak_vram_gb']:.1f}GB")
    print(f"{'='*60}")

    # Save
    combined = {
        "version": "v4",
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(),
        "eager": eager_results,
        "cuda_graph": graph_results,
        "v3_baseline": v3_baseline,
        "hf_baseline": hf_baseline,
        "load_time_s": round(load_time, 1),
        "patch_time_s": round(patch_time, 1),
        "vram_gb": round(vram_after, 1),
        "optimizations": [
            "fused_postproj_recurrent (2 kernels -> 1, saves 2304 launches/token)",
            "inter_layer_residual_norm_chaining (63 HBM round-trips eliminated)",
            "fused_lm_head_argmax_static (CUDA-graph-safe, no 248K logit materialization)",
        ],
    }

    save_results(combined, f"{RESULTS_PATH}/v4_results.json")
    results_volume.commit()

    return combined


# =========================================================================
# Entrypoint
# =========================================================================
@app.local_entrypoint()
def main(
    correctness_only: bool = False,
    benchmark_only: bool = False,
):
    """Run v4 validation and benchmark.

    Flags:
        --correctness-only: Skip benchmark, only run correctness tests
        --benchmark-only: Skip correctness, only run benchmark
    """
    results = {}

    if not benchmark_only:
        # Stage 1: Kernel correctness
        print("\n" + "=" * 60)
        print("LAUNCHING STAGE 1: Kernel-level correctness")
        print("=" * 60)
        kernel_results = validate_fused_kernel.remote()
        results["kernel_correctness"] = kernel_results
        print(f"\nKernel correctness: {'PASS' if kernel_results['all_pass'] else 'FAIL'}")

        # Stage 2: E2E correctness
        print("\n" + "=" * 60)
        print("LAUNCHING STAGE 2: E2E correctness")
        print("=" * 60)
        e2e_results = validate_e2e.remote()
        results["e2e_correctness"] = e2e_results
        print(f"\nE2E correctness: {'PASS' if e2e_results['exact_match'] else 'PARTIAL'}")

    if not correctness_only:
        # Stage 3: Benchmark
        print("\n" + "=" * 60)
        print("LAUNCHING STAGE 3: V4 Benchmark")
        print("=" * 60)
        bench_results = run_v4_benchmark.remote()
        results["benchmark"] = bench_results

        if bench_results.get("cuda_graph"):
            tok_s = bench_results["cuda_graph"]["decode_tok_s"]["mean"]
            itl = bench_results["cuda_graph"]["mean_itl_ms"]["mean"]
            print(f"\nFinal: {tok_s:.1f} tok/s, ITL={itl:.2f}ms (CUDA graph)")

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("V4 VALIDATION SUMMARY")
    print("=" * 60)

    if "kernel_correctness" in results:
        kc = results["kernel_correctness"]
        print(f"  Kernel correctness:  {'PASS' if kc['all_pass'] else 'FAIL'}")
        print(f"    Output cosine: {kc['fused_kernel_output_cosine']:.6f}")
        print(f"    State drift (1000 steps): {kc['state_drift_1000_steps']:.6f}")
        print(f"    lm_head argmax: {kc['lm_head_argmax_match']}")

    if "e2e_correctness" in results:
        e2e = results["e2e_correctness"]
        match_str = "EXACT" if e2e["exact_match"] else f"{e2e['token_match_rate']}%"
        print(f"  E2E token match:     {match_str}")

    if "benchmark" in results and results["benchmark"].get("cuda_graph"):
        b = results["benchmark"]
        tok_s = b["cuda_graph"]["decode_tok_s"]["mean"]
        itl = b["cuda_graph"]["mean_itl_ms"]["mean"]
        vram = b.get("vram_gb", "?")
        print(f"  CUDA Graph decode:   {tok_s:.1f} tok/s, ITL={itl:.2f}ms, VRAM={vram}GB")

    print("=" * 60)
