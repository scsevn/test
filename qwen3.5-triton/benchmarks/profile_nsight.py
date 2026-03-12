"""
V5 Profiling  - torch.profiler decode step analysis on B200.

Runs a single-batch decode (no CUDA graph) with torch.profiler to get
exact per-kernel timing breakdown. Validates our wall-clock estimates:
  - GEMV (cuBLAS): ~8.2ms (~77% of ITL)
  - Non-GEMV (recurrent, conv1d, SDPA, norms): ~2.4ms

Outputs:
  1. Top-50 kernels by CUDA time (table)
  2. Chrome trace JSON for visualization (load in chrome://tracing)
  3. Summary: GEMV vs non-GEMV breakdown

Run via Modal:
    modal run benchmarks/profile_nsight.py
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
PROFILE_STEPS = 10  # Number of decode steps to profile (for stable averages)


@app.function(
    image=forge_llm_image,
    gpu=GPU_CONFIG["optimized"]["gpu"],
    timeout=GPU_CONFIG["optimized"]["timeout"],
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results_volume},
    secrets=[hf_secret],
)
def run_profile():
    """Profile decode step with torch.profiler  - no CUDA graph."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    from transformers import AutoTokenizer

    from forge.llm.patch_qwen35 import load_model, patch_model, verify_patch
    from forge.llm.cache import HybridCache

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda"

    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"VRAM: {total_mem / (1024**3):.1f}GB")

    # Load tokenizer + model
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

    # Create cache and patch
    cache = HybridCache(
        batch_size=1,
        max_cache_len=512,
        device=device,
    )
    model = patch_model(model, cache)
    assert verify_patch(model), "Patch verification failed!"
    print(f"VRAM after patching: {torch.cuda.memory_allocated() / (1024**3):.1f}GB")

    # --- Prefill ---
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    batch_size, prompt_len = input_ids.shape
    print(f"Prompt: {prompt_len} tokens")

    cache_position = torch.arange(prompt_len, device=device)
    position_ids = cache_position.unsqueeze(0)

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        for layer in model.model.layers:
            hidden = layer(
                hidden,
                cache_position=cache_position,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=True,
            )
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)

    next_token = logits[:, -1:, :].argmax(dim=-1)
    cur_pos = prompt_len
    print("Prefill complete")

    # --- Warmup decode (trigger autotune) ---
    print("Warming up decode (5 steps)...")
    for _ in range(5):
        cache_position = torch.tensor([cur_pos], device=device)
        position_ids = cache_position.unsqueeze(0)
        with torch.no_grad():
            hidden = model.model.embed_tokens(next_token)
            position_embeddings = model.model.rotary_emb(hidden, position_ids)
            x_normed_next = None
            for layer in model.model.layers:
                result = layer(
                    hidden,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=True,
                    x_normed_input=x_normed_next,
                )
                if isinstance(result, tuple) and len(result) == 2:
                    hidden, x_normed_next = result
                else:
                    hidden = result
                    x_normed_next = None
            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        cur_pos += 1

    # --- Profiled decode ---
    print(f"\nProfiling {PROFILE_STEPS} decode steps...")
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for step in range(PROFILE_STEPS):
            cache_position = torch.tensor([cur_pos], device=device)
            position_ids = cache_position.unsqueeze(0)
            with torch.no_grad():
                hidden = model.model.embed_tokens(next_token)
                position_embeddings = model.model.rotary_emb(hidden, position_ids)
                x_normed_next = None
                for layer in model.model.layers:
                    result = layer(
                        hidden,
                        cache_position=cache_position,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        use_cache=True,
                        x_normed_input=x_normed_next,
                    )
                    if isinstance(result, tuple) and len(result) == 2:
                        hidden, x_normed_next = result
                    else:
                        hidden = result
                        x_normed_next = None
                hidden = model.model.norm(hidden)
                logits = model.lm_head(hidden)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            cur_pos += 1

    torch.cuda.synchronize()

    # --- Print results ---
    print("\n" + "=" * 100)
    print(f"TOP 50 KERNELS BY CUDA TIME ({PROFILE_STEPS} decode steps)")
    print("=" * 100)
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=50,
    )
    print(table)

    # --- Per-step averages ---
    print("\n" + "=" * 100)
    print("PER-STEP AVERAGES (divided by {})".format(PROFILE_STEPS))
    print("=" * 100)

    key_avg = prof.key_averages()

    # Debug: print available attributes on first event
    if key_avg:
        sample = key_avg[0]
        cuda_attrs = [a for a in dir(sample) if 'cuda' in a.lower() or 'device' in a.lower()]
        print(f"\n  [DEBUG] Available CUDA/device attrs: {cuda_attrs}")
        # Try to find the right attribute
        for attr in ['self_cuda_time_total', 'self_device_time_total', 'cuda_time_total', 'device_time_total']:
            val = getattr(sample, attr, 'MISSING')
            if val != 'MISSING':
                print(f"  [DEBUG] {attr} = {val}")

    # Categorize kernels
    total_cuda_us = 0
    gemv_us = 0
    triton_us = 0
    sdpa_us = 0
    other_us = 0

    def _get_cuda_time(evt):
        """Get CUDA time from event, trying multiple attribute names."""
        for attr in ['self_cuda_time_total', 'self_device_time_total', 'cuda_time_total', 'device_time_total']:
            val = getattr(evt, attr, None)
            if val is not None and val > 0:
                return val
        return 0

    categories = {}
    for evt in key_avg:
        cuda_us = _get_cuda_time(evt)
        if cuda_us <= 0:
            continue
        total_cuda_us += cuda_us
        name = evt.key

        # Categorize by kernel name patterns
        name_lower = name.lower()
        if "nvjet" in name_lower or "gemm" in name_lower or "aten::mm" in name_lower:
            cat = "GEMV/cuBLAS"
        elif ("_fused_" in name_lower or "_triton_" in name_lower or
              "_causal_conv1d" in name_lower or "_silu_mul" in name_lower or
              "_sigmoid_gate" in name_lower or "_fused_qknorm" in name_lower or
              "_lm_head_" in name_lower or "postproj_recurrent" in name_lower):
            cat = "Triton/Fused kernels"
        elif "flash" in name_lower or "sdpa" in name_lower:
            cat = "SDPA/FlashAttn"
        elif "index_copy" in name_lower or "index_select" in name_lower:
            cat = "KV cache ops"
        elif "cuLaunch" in name_lower:
            cat = "Launch overhead"
        else:
            cat = "Other"

        if cat not in categories:
            categories[cat] = {"total_us": 0, "kernels": []}
        categories[cat]["total_us"] += cuda_us
        categories[cat]["kernels"].append((name, cuda_us))

    per_step_total_ms = total_cuda_us / PROFILE_STEPS / 1000.0

    print(f"\nTotal CUDA time per step: {per_step_total_ms:.2f}ms")
    print(f"\nBreakdown by category:")
    print(f"  {'Category':<25} {'Total (ms)':<12} {'Per-step (ms)':<14} {'% of total':<10}")
    print(f"  {'-'*60}")
    for cat, data in sorted(categories.items(), key=lambda x: -x[1]["total_us"]):
        total_ms = data["total_us"] / 1000.0
        per_step_ms = total_ms / PROFILE_STEPS
        pct = data["total_us"] / total_cuda_us * 100 if total_cuda_us > 0 else 0
        print(f"  {cat:<25} {total_ms:<12.2f} {per_step_ms:<14.3f} {pct:<10.1f}%")
        # Top 5 kernels in this category
        top_kernels = sorted(data["kernels"], key=lambda x: -x[1])[:5]
        for name, us in top_kernels:
            k_ms = us / 1000.0 / PROFILE_STEPS
            print(f"    {name[:60]:<62} {k_ms:.3f}ms/step")

    # --- Export Chrome trace ---
    trace_path = f"{RESULTS_PATH}/v5_profile.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    print("  Open in chrome://tracing or Perfetto UI")

    # --- Save summary ---
    summary = {
        "version": "v5",
        "profile_steps": PROFILE_STEPS,
        "gpu": torch.cuda.get_device_name(),
        "per_step_total_ms": round(per_step_total_ms, 3),
        "categories": {
            cat: {
                "total_ms": round(data["total_us"] / 1000.0, 3),
                "per_step_ms": round(data["total_us"] / 1000.0 / PROFILE_STEPS, 3),
                "pct": round(data["total_us"] / total_cuda_us * 100, 1) if total_cuda_us > 0 else 0,
                "top_kernels": [
                    {"name": name, "per_step_ms": round(us / 1000.0 / PROFILE_STEPS, 4)}
                    for name, us in sorted(data["kernels"], key=lambda x: -x[1])[:10]
                ],
            }
            for cat, data in sorted(categories.items(), key=lambda x: -x[1]["total_us"])
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
    }

    summary_path = f"{RESULTS_PATH}/v5_profile_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    results_volume.commit()
    return summary


@app.local_entrypoint()
def main():
    """Run profiling on B200."""
    print("Starting V5 profiling on B200...")
    results = run_profile.remote()

    print(f"\nPer-step decode: {results['per_step_total_ms']:.2f}ms")
    if "categories" in results:
        for cat, data in results["categories"].items():
            print(f"  {cat}: {data['per_step_ms']:.3f}ms ({data['pct']:.1f}%)")
