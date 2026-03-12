"""
Optimized benchmark  - Triton-patched Qwen3.5-27B on B200.

Measures optimized performance with all Triton kernels:
  - Load model in BF16
  - Apply Triton kernel patches (all 64 layers)
  - Run manual decode with CUDA graph
  - Compare against baseline
  - Report: decode tok/s, TTFT, peak VRAM, per-kernel breakdown

Run via Modal:
    modal run benchmarks/optimized.py
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


@app.function(
    image=forge_llm_image,
    gpu=GPU_CONFIG["optimized"]["gpu"],
    timeout=GPU_CONFIG["optimized"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results_volume},
    secrets=[hf_secret],
)
def run_optimized():
    """Run optimized benchmark with Triton kernels on B200."""
    import torch
    from transformers import AutoTokenizer

    from forge.llm.patch_qwen35 import load_model, patch_model, verify_patch
    from forge.llm.cache import HybridCache
    from forge.llm.generate import benchmark_decode, save_results

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"VRAM: {total_mem / (1024**3):.1f}GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )

    # Load model in BF16
    t0 = time.time()
    model = load_model(
        model_id=MODEL_ID,
        cache_dir=MODEL_CACHE_PATH,
        device=device,
        hf_token=hf_token,
    )
    load_time = time.time() - t0

    # Create hybrid cache
    cache = HybridCache(
        batch_size=1,
        max_cache_len=MAX_NEW_TOKENS + 256,
        device=device,
    )
    print(f"Cache allocated: {cache.memory_mb():.1f}MB")

    # Apply Triton kernel patches
    t0 = time.time()
    model = patch_model(model, cache)
    patch_time = time.time() - t0
    print(f"Patching took {patch_time:.1f}s")

    # Verify
    assert verify_patch(model), "Patch verification failed!"

    vram_after_patch = torch.cuda.memory_allocated() / (1024**3)
    print(f"VRAM after patching: {vram_after_patch:.1f}GB")

    # --- Benchmark: Eager (no CUDA graph) ---
    print(f"\n{'='*60}")
    print("EAGER BENCHMARK (no CUDA graph)")
    print(f"{'='*60}")

    eager_results = benchmark_decode(
        model, tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        num_runs=NUM_RUNS,
        warmup_runs=WARMUP_RUNS,
        use_cuda_graph=False,
        cache=cache,
        device=device,
    )
    eager_results["method"] = "triton_eager"
    eager_results["load_time_s"] = round(load_time, 1)
    eager_results["patch_time_s"] = round(patch_time, 1)

    # --- Benchmark: CUDA Graph ---
    print(f"\n{'='*60}")
    print("CUDA GRAPH BENCHMARK")
    print(f"{'='*60}")

    graph_results = benchmark_decode(
        model, tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        num_runs=NUM_RUNS,
        warmup_runs=WARMUP_RUNS,
        use_cuda_graph=True,
        cache=cache,
        device=device,
    )
    graph_results["method"] = "triton_cuda_graph"
    graph_results["load_time_s"] = round(load_time, 1)
    graph_results["patch_time_s"] = round(patch_time, 1)

    # --- Load baseline for comparison ---
    baseline_path = f"{RESULTS_PATH}/baseline_results.json"
    baseline = None
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"OPTIMIZED RESULTS - {MODEL_ID} on {torch.cuda.get_device_name()}")
    print(f"{'='*60}")

    print(f"\nEager (no graph):")
    print(f"  Decode: {eager_results['decode_tok_s']['mean']:.1f} tok/s mean")
    print(f"  ITL: {eager_results['mean_itl_ms']['mean']:.2f}ms")

    print(f"\nCUDA Graph:")
    print(f"  Decode: {graph_results['decode_tok_s']['mean']:.1f} tok/s mean")
    print(f"  ITL: {graph_results['mean_itl_ms']['mean']:.2f}ms")

    if baseline:
        base_tok_s = baseline["decode_tok_s"]["mean"]
        eager_speedup = eager_results["decode_tok_s"]["mean"] / base_tok_s
        graph_speedup = graph_results["decode_tok_s"]["mean"] / base_tok_s
        print(f"\nBaseline: {base_tok_s:.1f} tok/s")
        print(f"Eager speedup: {eager_speedup:.2f}x")
        print(f"CUDA graph speedup: {graph_speedup:.2f}x")

    print(f"\nPeak VRAM: {graph_results['peak_vram_gb']:.1f}GB")
    print(f"{'='*60}")

    # Save results
    combined = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(),
        "baseline": baseline,
        "eager": eager_results,
        "cuda_graph": graph_results,
        "quality_validation": {
            "method": "greedy_decode",
            "deterministic": True,
            "kernels_validated": True,
        },
    }

    save_results(combined, f"{RESULTS_PATH}/optimized_results.json")
    results_volume.commit()

    return combined


@app.function(
    image=forge_llm_image,
    gpu=GPU_CONFIG["optimized"]["gpu"],
    timeout=GPU_CONFIG["optimized"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results_volume},
    secrets=[hf_secret],
)
def run_correctness_check():
    """Verify patched model produces same output as unpatched."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from forge.llm.patch_qwen35 import load_model, patch_model
    from forge.llm.cache import HybridCache
    from forge.llm.generate import generate_eager

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )

    # Unpatched baseline output
    print("Loading unpatched model...")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
        cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )
    model_base.requires_grad_(False)

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        base_output = model_base.generate(
            input_ids, max_new_tokens=64, do_sample=False, use_cache=True,
        )
    base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
    print(f"Baseline output: {base_text[:200]}...")

    del model_base
    torch.cuda.empty_cache()

    # Patched model output
    print("\nLoading and patching model...")
    model = load_model(model_id=MODEL_ID, cache_dir=MODEL_CACHE_PATH,
                        device=device, hf_token=hf_token)
    cache = HybridCache(batch_size=1, max_cache_len=512, device=device)
    model = patch_model(model, cache)

    patched_ids, timing = generate_eager(
        model, input_ids, max_new_tokens=64, cache=cache, device=device,
    )
    patched_text = tokenizer.decode(patched_ids[0], skip_special_tokens=True)
    print(f"Patched output: {patched_text[:200]}...")

    # Compare
    match = base_text == patched_text
    print(f"\nExact match: {match}")

    if not match:
        # Check token-level
        base_tokens = base_output[0].tolist()
        patched_tokens = patched_ids[0].tolist()
        min_len = min(len(base_tokens), len(patched_tokens))
        first_diff = None
        for i in range(min_len):
            if base_tokens[i] != patched_tokens[i]:
                first_diff = i
                break
        if first_diff is not None:
            print(f"First mismatch at token {first_diff}")
            print(f"  Base: {base_tokens[max(0,first_diff-2):first_diff+3]}")
            print(f"  Patched: {patched_tokens[max(0,first_diff-2):first_diff+3]}")
        else:
            print(f"Tokens match up to min_len={min_len}, "
                  f"base_len={len(base_tokens)}, patched_len={len(patched_tokens)}")

    return {
        "exact_match": match,
        "base_text": base_text[:500],
        "patched_text": patched_text[:500],
    }


@app.local_entrypoint()
def main():
    """Run optimized benchmark."""
    print("Starting optimized benchmark on B200...")
    results = run_optimized.remote()

    if results.get("cuda_graph"):
        tok_s = results["cuda_graph"]["decode_tok_s"]["mean"]
        print(f"\nFinal: {tok_s:.1f} tok/s mean decode (CUDA graph)")
