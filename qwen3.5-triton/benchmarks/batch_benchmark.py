"""
V5 Batch Benchmark  - B=1,2,4,8,16 throughput on B200.

Tests batch inference scaling where weight reads are amortized across
all batch elements. Expected: ~linear aggregate throughput scaling up to
the compute-bound crossover point.

B=1:  94.4 tok/s (read 54GB weights for 1 token)
B=8:  ~750 tok/s (read 54GB weights for 8 tokens  - amortized)
B=16: ~1200 tok/s (approaching compute-bound regime)

Reports:
  - Aggregate tok/s (total tokens across all users)
  - Per-user tok/s (individual user throughput)
  - Per-user ITL (latency per user)
  - Peak VRAM per batch size
  - Scaling efficiency (actual vs linear)

Also validates correctness: B=1 and B>1 must produce identical per-user
tokens for the same prompt (greedy decoding is deterministic).

Run via Modal:
    modal run benchmarks/batch_benchmark.py
    modal run benchmarks/batch_benchmark.py::main --correctness-only
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
MAX_NEW_TOKENS = 128  # Shorter for batch benchmark (8x more tokens total at B=8)
BATCH_SIZES = [1, 2, 4, 8, 16]
NUM_RUNS = 10
WARMUP_RUNS = 2


@app.function(
    image=forge_llm_image,
    gpu=GPU_CONFIG["optimized"]["gpu"],
    timeout=GPU_CONFIG["optimized"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results_volume},
    secrets=[hf_secret],
)
def run_batch_benchmark():
    """Benchmark batched inference at B=1,2,4,8,16."""
    import torch
    from transformers import AutoTokenizer

    from forge.llm.patch_qwen35 import load_model, patch_model, verify_patch
    from forge.llm.cache import HybridCache
    from forge.llm.generate import (
        benchmark_decode, benchmark_decode_batched, save_results,
    )

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"VRAM: {total_mem / (1024**3):.1f}GB")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )

    t0 = time.time()
    model = load_model(
        model_id=MODEL_ID,
        cache_dir=MODEL_CACHE_PATH,
        device=device,
        hf_token=hf_token,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Patch model with B=1 cache first (will create per-run caches for batched)
    cache_b1 = HybridCache(batch_size=1, max_cache_len=MAX_NEW_TOKENS + 256, device=device)
    model = patch_model(model, cache_b1)
    assert verify_patch(model), "Patch verification failed!"
    vram_base = torch.cuda.memory_allocated() / (1024**3)
    print(f"VRAM after patching: {vram_base:.1f}GB")

    all_results = {}

    # --- B=1 baseline (uses standard generate_cuda_graph) ---
    print(f"\n{'='*60}")
    print(f"BATCH SIZE = 1 (baseline)")
    print(f"{'='*60}")

    b1_results = benchmark_decode(
        model, tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        num_runs=NUM_RUNS,
        warmup_runs=WARMUP_RUNS,
        use_cuda_graph=True,
        cache=cache_b1,
        device=device,
    )
    b1_tok_s = b1_results["decode_tok_s"]["mean"]
    b1_itl = b1_results["mean_itl_ms"]["mean"]

    all_results["B=1"] = {
        "batch_size": 1,
        "aggregate_tok_s": b1_tok_s,
        "per_user_tok_s": b1_tok_s,
        "per_user_itl_ms": b1_itl,
        "peak_vram_gb": b1_results["peak_vram_gb"],
        "scaling_efficiency": 1.0,
    }

    # --- B>1 benchmarks ---
    # patch_model binds the cache into layer closures, so we must
    # re-patch the model for each batch size.
    for B in BATCH_SIZES:
        if B == 1:
            continue

        print(f"\n{'='*60}")
        print(f"BATCH SIZE = {B}")
        print(f"{'='*60}")

        torch.cuda.reset_peak_memory_stats()

        # Re-patch model with B-sized cache
        cache_b = HybridCache(batch_size=B, max_cache_len=MAX_NEW_TOKENS + 256, device=device)
        model = patch_model(model, cache_b)
        print(f"  Re-patched with B={B} cache: {cache_b.memory_mb():.1f}MB")

        def make_cache_b(batch_size, _cache=cache_b):
            """Return same cache (reset inside generate)."""
            return _cache

        results = benchmark_decode_batched(
            model, tokenizer,
            prompt=PROMPT,
            batch_size=B,
            max_new_tokens=MAX_NEW_TOKENS,
            num_runs=NUM_RUNS,
            warmup_runs=WARMUP_RUNS,
            use_cuda_graph=True,
            cache_factory=make_cache_b,
            device=device,
        )

        agg_tok_s = results["aggregate_tok_s"]["mean"]
        per_user_itl = results["per_user_itl_ms"]["mean"]
        ideal_agg = b1_tok_s * B
        scaling_eff = agg_tok_s / ideal_agg if ideal_agg > 0 else 0

        all_results[f"B={B}"] = {
            "batch_size": B,
            "aggregate_tok_s": agg_tok_s,
            "per_user_tok_s": results["per_user_tok_s"]["mean"],
            "per_user_itl_ms": per_user_itl,
            "peak_vram_gb": results["peak_vram_gb"],
            "scaling_efficiency": round(scaling_eff, 3),
        }

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"BATCH SCALING RESULTS  - {MODEL_ID} on {torch.cuda.get_device_name()}")
    print(f"{'='*80}")
    print(f"\n{'B':>4}  {'Agg tok/s':>12}  {'Per-user tok/s':>15}  {'Per-user ITL':>13}  {'VRAM (GB)':>10}  {'Scaling':>8}")
    print(f"{'':>4}  {'':>12}  {'':>15}  {'(ms)':>13}  {'':>10}  {'Eff.':>8}")
    print(f"{'-'*80}")

    for key in sorted(all_results.keys(), key=lambda k: all_results[k]["batch_size"]):
        r = all_results[key]
        B = r["batch_size"]
        print(f"{B:>4}  {r['aggregate_tok_s']:>12.1f}  {r['per_user_tok_s']:>15.1f}  "
              f"{r['per_user_itl_ms']:>13.2f}  {r['peak_vram_gb']:>10.1f}  "
              f"{r['scaling_efficiency']:>7.1%}")

    print(f"\nLinear scaling baseline: B=1 = {b1_tok_s:.1f} tok/s")
    print(f"{'='*80}")

    # Save
    combined = {
        "version": "v5_batch",
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(),
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_len": tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1],
        "b1_baseline_tok_s": b1_tok_s,
        "batch_results": all_results,
    }

    save_results(combined, f"{RESULTS_PATH}/v5_batch_results.json")
    results_volume.commit()

    return combined


@app.function(
    image=forge_llm_image,
    gpu=GPU_CONFIG["optimized"]["gpu"],
    timeout=GPU_CONFIG["optimized"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[hf_secret],
)
def validate_batch_correctness():
    """Verify B=1 and B>1 produce identical per-user tokens."""
    import torch
    from transformers import AutoTokenizer

    from forge.llm.patch_qwen35 import load_model, patch_model
    from forge.llm.cache import HybridCache
    from forge.llm.generate import generate_eager, generate_eager_batched

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )

    model = load_model(model_id=MODEL_ID, cache_dir=MODEL_CACHE_PATH,
                        device=device, hf_token=hf_token)

    print("=" * 60)
    print("BATCH CORRECTNESS VALIDATION")
    print("=" * 60)

    # --- B=1 reference ---
    cache_b1 = HybridCache(batch_size=1, max_cache_len=256, device=device)
    model = patch_model(model, cache_b1)

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    b1_ids, _ = generate_eager(
        model, input_ids, max_new_tokens=32, cache=cache_b1, device=device,
    )
    b1_tokens = b1_ids[0].tolist()
    prompt_len = input_ids.shape[1]
    b1_gen = b1_tokens[prompt_len:]
    print(f"B=1 generated {len(b1_gen)} tokens: {b1_gen[:10]}...")

    # --- B=4 batched ---
    B = 4
    cache_b4 = HybridCache(batch_size=B, max_cache_len=256, device=device)
    # Re-patch with new cache
    model = patch_model(model, cache_b4)

    input_ids_batch = input_ids.expand(B, -1).contiguous()
    b4_ids, _ = generate_eager_batched(
        model, input_ids_batch, max_new_tokens=32, cache=cache_b4, device=device,
    )

    # Compare each user in the batch to B=1 reference
    all_match = True
    for u in range(B):
        user_tokens = b4_ids[u].tolist()
        user_gen = user_tokens[prompt_len:][:len(b1_gen)]

        match = user_gen == b1_gen
        if not match:
            first_diff = None
            for i in range(min(len(user_gen), len(b1_gen))):
                if user_gen[i] != b1_gen[i]:
                    first_diff = i
                    break
            print(f"  User {u}: MISMATCH at token {first_diff}")
            print(f"    B=1:  {b1_gen[max(0,first_diff-2):first_diff+3]}")
            print(f"    B={B}: {user_gen[max(0,first_diff-2):first_diff+3]}")
            all_match = False
        else:
            print(f"  User {u}: MATCH ({len(user_gen)} tokens)")

    print(f"\nAll users match B=1: {'PASS' if all_match else 'FAIL'}")

    return {
        "all_match": all_match,
        "batch_size": B,
        "tokens_compared": len(b1_gen),
    }


@app.local_entrypoint()
def main(correctness_only: bool = False):
    """Run batch benchmark on B200."""
    if correctness_only:
        print("Running batch correctness validation...")
        results = validate_batch_correctness.remote()
        print(f"\nCorrectness: {'PASS' if results['all_match'] else 'FAIL'}")
        return

    print("Starting V5 batch benchmark on B200...")
    results = run_batch_benchmark.remote()

    if results.get("batch_results"):
        print(f"\nSummary:")
        for key, r in sorted(results["batch_results"].items(),
                              key=lambda x: x[1]["batch_size"]):
            print(f"  {key}: {r['aggregate_tok_s']:.1f} agg tok/s, "
                  f"{r['per_user_itl_ms']:.2f}ms ITL/user, "
                  f"{r['scaling_efficiency']:.0%} efficiency")
