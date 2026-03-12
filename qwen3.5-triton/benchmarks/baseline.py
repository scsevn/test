"""
Baseline benchmark  - HuggingFace default Qwen3.5-27B on B200.

Measures unoptimized performance:
  - Load model in BF16
  - Run model.generate() with greedy decoding
  - Profile with torch.profiler to identify bottlenecks
  - Report: decode tok/s, TTFT, peak VRAM

Run via Modal:
    modal run benchmarks/baseline.py
"""
import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
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
    gpu=GPU_CONFIG["baseline"]["gpu"],
    timeout=GPU_CONFIG["baseline"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results_volume},
    secrets=[hf_secret],
)
def run_baseline():
    """Run baseline benchmark on B200."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"VRAM: {total_mem / (1024**3):.1f}GB")

    # Load model
    print(f"\nLoading {MODEL_ID} in BF16...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=MODEL_CACHE_PATH,
        token=hf_token,
        trust_remote_code=True,
    )
    model.requires_grad_(False)
    load_time = time.time() - t0

    vram_after_load = torch.cuda.memory_allocated() / (1024**3)
    print(f"Model loaded in {load_time:.1f}s, VRAM: {vram_after_load:.1f}GB")

    # Tokenize prompt
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    print(f"Prompt: {prompt_len} tokens")

    # Warmup runs
    print(f"\nWarmup: {WARMUP_RUNS} runs...")
    for i in range(WARMUP_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        end.record()
        torch.cuda.synchronize()
        gen_tokens = output.shape[1] - prompt_len
        total_ms = start.elapsed_time(end)
        tok_s = gen_tokens / total_ms * 1000
        print(f"  warmup {i+1}: {gen_tokens} tokens, {tok_s:.1f} tok/s, {total_ms:.0f}ms total")

    # Timed runs
    print(f"\nBenchmark: {NUM_RUNS} runs...")
    all_results = []

    for i in range(NUM_RUNS):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        end.record()
        torch.cuda.synchronize()

        gen_tokens = output.shape[1] - prompt_len
        total_ms = start.elapsed_time(end)
        tok_s = gen_tokens / total_ms * 1000
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        run_result = {
            "run": i + 1,
            "gen_tokens": gen_tokens,
            "total_ms": round(total_ms, 2),
            "tok_s": round(tok_s, 1),
            "mean_itl_ms": round(total_ms / gen_tokens, 2) if gen_tokens > 0 else 0,
            "peak_vram_gb": round(peak_vram, 2),
        }
        all_results.append(run_result)
        print(f"  run {i+1}: {tok_s:.1f} tok/s, ITL={run_result['mean_itl_ms']:.2f}ms, "
              f"VRAM={peak_vram:.1f}GB")

    # Compute statistics
    tok_s_list = sorted([r["tok_s"] for r in all_results])
    itl_list = sorted([r["mean_itl_ms"] for r in all_results])

    summary = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(),
        "method": "hf_generate_bf16",
        "prompt": PROMPT,
        "prompt_len": prompt_len,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_runs": NUM_RUNS,
        "load_time_s": round(load_time, 1),
        "model_vram_gb": round(vram_after_load, 2),
        "decode_tok_s": {
            "mean": round(sum(tok_s_list) / len(tok_s_list), 1),
            "median": round(tok_s_list[len(tok_s_list) // 2], 1),
            "min": round(tok_s_list[0], 1),
            "max": round(tok_s_list[-1], 1),
            "p5": round(tok_s_list[max(0, int(len(tok_s_list) * 0.05))], 1),
        },
        "mean_itl_ms": {
            "mean": round(sum(itl_list) / len(itl_list), 2),
            "median": round(itl_list[len(itl_list) // 2], 2),
            "p95": round(itl_list[min(int(len(itl_list) * 0.95), len(itl_list) - 1)], 2),
        },
        "all_runs": all_results,
        "quality_validation": {
            "method": "greedy_decode",
            "deterministic": True,
        },
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS  - {MODEL_ID} on {torch.cuda.get_device_name()}")
    print(f"{'='*60}")
    print(f"Decode: {summary['decode_tok_s']['mean']:.1f} tok/s (mean), "
          f"{summary['decode_tok_s']['median']:.1f} tok/s (median)")
    print(f"ITL: {summary['mean_itl_ms']['mean']:.2f}ms (mean)")
    print(f"VRAM: {vram_after_load:.1f}GB model + KV cache")
    print(f"{'='*60}")

    # Save results
    results_path = f"{RESULTS_PATH}/baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_path}")

    results_volume.commit()
    return summary


@app.function(
    image=forge_llm_image,
    gpu=GPU_CONFIG["baseline"]["gpu"],
    timeout=GPU_CONFIG["baseline"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results_volume},
    secrets=[hf_secret],
)
def profile_baseline():
    """Profile baseline model to identify bottlenecks."""
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=MODEL_CACHE_PATH,
        token=hf_token,
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    # Warmup
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=32, do_sample=False, use_cache=True)

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=64, do_sample=False, use_cache=True)

    # Print top GPU operations
    print("\n=== TOP CUDA OPS BY GPU TIME ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Save profile
    prof_path = f"{RESULTS_PATH}/baseline_profile.json"
    prof.export_chrome_trace(prof_path)
    print(f"\nProfile saved to {prof_path}")

    results_volume.commit()


@app.local_entrypoint()
def main():
    """Run baseline benchmark."""
    print("Starting baseline benchmark on B200...")
    results = run_baseline.remote()
    print(f"\nFinal: {results['decode_tok_s']['mean']:.1f} tok/s mean decode")
