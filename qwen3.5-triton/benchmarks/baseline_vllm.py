"""
vLLM baseline benchmark  - Qwen3.5-27B on B200.

Measures the state-of-the-art production inference engine performance:
  - vLLM with BF16, tensor_parallel=1
  - Same prompt, same max_tokens for fair comparison
  - Reports decode tok/s, TTFT, peak VRAM

Run via Modal:
    modal run benchmarks/baseline_vllm.py
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import modal
from modal_config import (
    app, model_cache, results_volume, hf_secret,
    MODEL_CACHE_PATH, RESULTS_PATH, MODEL_ID,
)

PROMPT = "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves."
MAX_NEW_TOKENS = 256
NUM_RUNS = 20
WARMUP_RUNS = 3

# vLLM image (separate deps)
_project_root = Path(__file__).parent.parent
_ignore_patterns = ["__pycache__", ".git", "*.pyc", "results"]

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "vllm>=0.8.0",
        "torch>=2.6.0",
        "transformers>=4.57.0",
        "huggingface_hub>=0.20.0",
    )
    .add_local_dir(str(_project_root), remote_path="/root", ignore=_ignore_patterns)
)


@app.function(
    image=vllm_image,
    gpu="B200",
    timeout=1800,
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results_volume},
    secrets=[hf_secret],
)
def run_vllm_benchmark():
    """Run vLLM benchmark on B200."""
    import torch
    from vllm import LLM, SamplingParams

    hf_token = os.environ.get("HF_TOKEN")

    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"VRAM: {total_mem / (1024**3):.1f}GB")

    # Initialize vLLM engine
    print(f"\nLoading {MODEL_ID} with vLLM...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=1024,
        download_dir=MODEL_CACHE_PATH,
        trust_remote_code=True,
        enforce_eager=False,  # Allow CUDA graph
    )
    load_time = time.time() - t0
    print(f"vLLM loaded in {load_time:.1f}s")

    # Greedy sampling (deterministic, same as our benchmark)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    # Warmup
    print(f"\nWarmup: {WARMUP_RUNS} runs...")
    for i in range(WARMUP_RUNS):
        t0 = time.time()
        outputs = llm.generate([PROMPT], sampling_params)
        elapsed = time.time() - t0
        gen_tokens = len(outputs[0].outputs[0].token_ids)
        tok_s = gen_tokens / elapsed
        print(f"  warmup {i+1}: {gen_tokens} tokens, {tok_s:.1f} tok/s, {elapsed*1000:.0f}ms")

    # Timed runs  - measure per-request latency (single request = decode-bound)
    print(f"\nBenchmark: {NUM_RUNS} runs (single request = decode-dominated)...")
    all_results = []

    for i in range(NUM_RUNS):
        torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        outputs = llm.generate([PROMPT], sampling_params)
        elapsed = time.time() - t0

        gen_tokens = len(outputs[0].outputs[0].token_ids)
        tok_s = gen_tokens / elapsed
        mean_itl = (elapsed * 1000) / gen_tokens if gen_tokens > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        run_result = {
            "run": i + 1,
            "gen_tokens": gen_tokens,
            "total_ms": round(elapsed * 1000, 2),
            "tok_s": round(tok_s, 1),
            "mean_itl_ms": round(mean_itl, 2),
            "peak_vram_gb": round(peak_vram, 2),
        }
        all_results.append(run_result)
        print(f"  run {i+1}: {tok_s:.1f} tok/s, ITL={mean_itl:.2f}ms, VRAM={peak_vram:.1f}GB")

    # Statistics
    tok_s_list = sorted([r["tok_s"] for r in all_results])
    itl_list = sorted([r["mean_itl_ms"] for r in all_results])

    summary = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(),
        "method": "vllm_bf16",
        "vllm_version": None,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_runs": NUM_RUNS,
        "load_time_s": round(load_time, 1),
        "decode_tok_s": {
            "mean": round(sum(tok_s_list) / len(tok_s_list), 1),
            "median": round(tok_s_list[len(tok_s_list) // 2], 1),
            "min": round(tok_s_list[0], 1),
            "max": round(tok_s_list[-1], 1),
        },
        "mean_itl_ms": {
            "mean": round(sum(itl_list) / len(itl_list), 2),
            "median": round(itl_list[len(itl_list) // 2], 2),
            "p95": round(itl_list[min(int(len(itl_list) * 0.95), len(itl_list) - 1)], 2),
        },
        "peak_vram_gb": round(max(r["peak_vram_gb"] for r in all_results), 2),
        "all_runs": all_results,
    }

    # Get vLLM version
    try:
        import vllm
        summary["vllm_version"] = vllm.__version__
    except Exception:
        pass

    # Print
    print(f"\n{'='*60}")
    print(f"vLLM RESULTS  - {MODEL_ID} on {torch.cuda.get_device_name()}")
    print(f"{'='*60}")
    print(f"Decode: {summary['decode_tok_s']['mean']:.1f} tok/s (mean)")
    print(f"ITL: {summary['mean_itl_ms']['mean']:.2f}ms")
    print(f"Peak VRAM: {summary['peak_vram_gb']:.1f}GB")
    print(f"vLLM version: {summary['vllm_version']}")
    print(f"{'='*60}")

    # Save
    results_path = f"{RESULTS_PATH}/vllm_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")

    results_volume.commit()
    return summary


@app.local_entrypoint()
def main():
    """Run vLLM baseline benchmark."""
    print("Starting vLLM benchmark on B200...")
    results = run_vllm_benchmark.remote()
    print(f"\nvLLM: {results['decode_tok_s']['mean']:.1f} tok/s mean decode")
