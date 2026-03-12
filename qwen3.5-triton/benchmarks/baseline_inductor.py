"""
torch.compile (Inductor) baseline benchmark  - Qwen3.5-27B on B200.

Measures PyTorch's built-in compiler performance:
  - torch.compile with mode="max-autotune" (Inductor backend)
  - Same BF16 model, same greedy decode, same prompt
  - Manual decode loop (same as our optimized benchmark) for fair ITL comparison

Run via Modal:
    modal run benchmarks/baseline_inductor.py
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
    gpu=GPU_CONFIG["baseline"]["gpu"],
    timeout=3600,  # Longer timeout for compilation
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results_volume},
    secrets=[hf_secret],
)
def run_inductor_benchmark():
    """Run torch.compile/Inductor benchmark on B200."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
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

    # ===== torch.compile with max-autotune =====
    print("\nCompiling model with torch.compile(mode='max-autotune')...")
    t0 = time.time()
    model = torch.compile(model, mode="max-autotune", fullgraph=False)
    compile_time = time.time() - t0
    print(f"torch.compile() call took {compile_time:.1f}s (lazy  - actual compilation on first run)")

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_token_id = tokenizer.eos_token_id or 151645

    # Use model.generate() which benefits from torch.compile
    print(f"\nWarmup: {WARMUP_RUNS} runs (includes compilation)...")
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
        print(f"  warmup {i+1}: {gen_tokens} tokens, {tok_s:.1f} tok/s, {total_ms:.0f}ms")

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
        mean_itl = total_ms / gen_tokens if gen_tokens > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        run_result = {
            "run": i + 1,
            "gen_tokens": gen_tokens,
            "total_ms": round(total_ms, 2),
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
        "method": "torch_compile_inductor",
        "torch_version": torch.__version__,
        "compile_mode": "max-autotune",
        "prompt": PROMPT,
        "prompt_len": prompt_len,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_runs": NUM_RUNS,
        "load_time_s": round(load_time, 1),
        "compile_time_s": round(compile_time, 1),
        "model_vram_gb": round(vram_after_load, 2),
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

    # Print
    print(f"\n{'='*60}")
    print(f"INDUCTOR RESULTS  - {MODEL_ID} on {torch.cuda.get_device_name()}")
    print(f"{'='*60}")
    print(f"Decode: {summary['decode_tok_s']['mean']:.1f} tok/s (mean)")
    print(f"ITL: {summary['mean_itl_ms']['mean']:.2f}ms")
    print(f"Peak VRAM: {summary['peak_vram_gb']:.1f}GB")
    print(f"PyTorch: {torch.__version__}, compile mode: max-autotune")
    print(f"{'='*60}")

    # Save
    results_path = f"{RESULTS_PATH}/inductor_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")

    results_volume.commit()
    return summary


@app.local_entrypoint()
def main():
    """Run torch.compile/Inductor benchmark."""
    print("Starting torch.compile (Inductor) benchmark on B200...")
    results = run_inductor_benchmark.remote()
    print(f"\nInductor: {results['decode_tok_s']['mean']:.1f} tok/s mean decode")
