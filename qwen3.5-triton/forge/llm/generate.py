"""
Manual decode loop with CUDA graph capture for Qwen3.5-27B.

Dense model = static computation graph during decode (unlike MoE).
CUDA graph captures the entire decode step and replays it, eliminating
~640 kernel launches per token = ~3ms overhead.

Pattern:
  1. Prefill: run normally (variable seq_len)
  2. Warmup: 5 eager decode steps (triggers all Triton autotune)
  3. Capture: record one decode step as CUDA graph
  4. Replay: replay graph for remaining tokens

Expected gain: +40-50% throughput from eliminating launch overhead.
"""
import time
import json
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import triton

from forge.kernels.triton_lm_head_topk import (
    fused_lm_head_argmax_static,
    fused_lm_head_argmax_batched,
    fused_lm_head_argmax_static_batched,
)


def generate_eager(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    eos_token_id: int = 151645,
    device: str = "cuda",
    cache=None,
) -> tuple:
    """Manual autoregressive generation WITHOUT CUDA graph.

    Uses CUDA event timing for accurate measurement.
    Falls back to model.generate() style but with manual loop for timing.

    Returns (output_ids, timing_dict).
    """
    batch_size, prompt_len = input_ids.shape

    # Reset cache for new sequence
    if cache is not None:
        cache.reset()

    # --- Prefill ---
    cache_position = torch.arange(prompt_len, device=device)
    position_ids = cache_position.unsqueeze(0)
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)

    prefill_start.record()
    with torch.no_grad():
        # For patched model: pass through all layers
        hidden = model.model.embed_tokens(input_ids)

        # Compute position embeddings (RoPE) for attention layers
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        for layer_idx, layer in enumerate(model.model.layers):
            hidden = layer(
                hidden,
                cache_position=cache_position,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=True,
            )
            # Handle tuple returns
            if isinstance(hidden, tuple):
                hidden = hidden[0]

        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)

    next_token = logits[:, -1:, :].argmax(dim=-1)
    prefill_end.record()

    generated_ids = [next_token]
    cur_pos = prompt_len

    # --- Decode loop ---
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)

    decode_start.record()
    decode_tokens = 0

    for step in range(max_new_tokens - 1):
        cache_position = torch.tensor([cur_pos], device=device)
        position_ids = cache_position.unsqueeze(0)

        with torch.no_grad():
            hidden = model.model.embed_tokens(next_token)

            # Compute position embeddings for this decode step
            position_embeddings = model.model.rotary_emb(hidden, position_ids)

            x_normed_next = None
            for layer_idx, layer in enumerate(model.model.layers):
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
        generated_ids.append(next_token)
        cur_pos += 1
        decode_tokens += 1

        # Check EOS every 8 tokens
        if decode_tokens % 8 == 0:
            recent = torch.cat(generated_ids[-8:], dim=-1)
            if (recent == eos_token_id).any().item():
                break

    decode_end.record()
    torch.cuda.synchronize()

    prefill_ms = prefill_start.elapsed_time(prefill_end)
    decode_ms = decode_start.elapsed_time(decode_end)
    decode_tok_s = (decode_tokens / decode_ms * 1000) if decode_ms > 0 else 0.0
    mean_itl_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0.0

    all_ids = torch.cat([input_ids] + generated_ids, dim=-1)

    # Trim at first EOS
    gen_portion = all_ids[0, prompt_len:]
    eos_mask = (gen_portion == eos_token_id)
    if eos_mask.any():
        first_eos = eos_mask.nonzero(as_tuple=False)[0].item()
        all_ids = all_ids[:, :prompt_len + first_eos]

    timing = {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tokens": decode_tokens,
        "decode_tok_s": round(decode_tok_s, 1),
        "mean_itl_ms": round(mean_itl_ms, 2),
        "prompt_len": prompt_len,
        "method": "eager",
    }

    return all_ids, timing


def generate_cuda_graph(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    eos_token_id: int = 151645,
    device: str = "cuda",
    cache=None,
    num_warmup: int = 5,
) -> tuple:
    """Manual generation with CUDA graph for decode phase.

    Captures the ENTIRE decode step as a single CUDA graph.
    Dense model has static computation graph  - perfect for CUDA graph.

    Returns (output_ids, timing_dict).
    """
    batch_size, prompt_len = input_ids.shape

    if cache is not None:
        cache.reset()

    # --- Prefill (eager) ---
    cache_position = torch.arange(prompt_len, device=device)
    position_ids = cache_position.unsqueeze(0)
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)

    prefill_start.record()
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
    prefill_end.record()

    generated_ids = [next_token.clone()]
    cur_pos = prompt_len

    # --- Warmup decode (eager, triggers Triton autotune) ---
    static_token = next_token.clone()
    static_cache_pos = torch.tensor([cur_pos], device=device, dtype=torch.long)

    # Static position_ids for graph capture
    static_pos_ids = torch.tensor([[cur_pos]], device=device, dtype=torch.long)

    # Pre-allocate fused lm_head buffers (CUDA graph safe  - no allocations in decode)
    lm_w_t = model.lm_head.weight.t().contiguous()  # [K, N] = [5120, 248320]
    _N_vocab = model.lm_head.weight.shape[0]
    _max_tiles = triton.cdiv(_N_vocab, 128)  # worst case tile count
    lm_local_max = torch.empty(_max_tiles, device=device, dtype=torch.float32)
    lm_local_idx = torch.empty(_max_tiles, device=device, dtype=torch.int64)
    lm_result = torch.empty(1, device=device, dtype=torch.int64)

    def _decode_step():
        """Single decode step  - must use static tensors for graph capture."""
        h = model.model.embed_tokens(static_token)
        pos_emb = model.model.rotary_emb(h, static_pos_ids)
        x_normed_next = None
        for layer in model.model.layers:
            result = layer(
                h,
                cache_position=static_cache_pos,
                position_ids=static_pos_ids,
                position_embeddings=pos_emb,
                use_cache=True,
                x_normed_input=x_normed_next,
            )
            if isinstance(result, tuple) and len(result) == 2:
                h, x_normed_next = result
            else:
                h = result
                x_normed_next = None
        h = model.model.norm(h)
        # Fused lm_head + argmax: no 248K logit materialization
        h_flat = h.squeeze()  # [5120]
        fused_lm_head_argmax_static(h_flat, lm_w_t, lm_local_max, lm_local_idx, lm_result)
        return lm_result.view(1, 1)

    for _ in range(num_warmup):
        with torch.no_grad():
            warmup_next = _decode_step()
        generated_ids.append(warmup_next.clone())
        static_token.copy_(warmup_next)
        cur_pos += 1
        static_cache_pos.fill_(cur_pos)
        static_pos_ids.fill_(cur_pos)

    # --- CUDA Graph Capture ---
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            static_next_token = _decode_step()

    generated_ids.append(static_next_token.clone())
    cur_pos += 1

    # --- Decode with graph replay ---
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)

    decode_start.record()
    decode_tokens = 0

    remaining = max_new_tokens - num_warmup - 2
    for step in range(remaining):
        static_token.copy_(static_next_token)
        static_cache_pos.fill_(cur_pos)
        static_pos_ids.fill_(cur_pos)

        graph.replay()

        generated_ids.append(static_next_token.clone())
        cur_pos += 1
        decode_tokens += 1

        # Check EOS every 8 tokens
        if decode_tokens % 8 == 0:
            recent = torch.cat(generated_ids[-8:], dim=-1)
            if (recent == eos_token_id).any().item():
                break

    decode_end.record()
    torch.cuda.synchronize()

    prefill_ms = prefill_start.elapsed_time(prefill_end)
    decode_ms = decode_start.elapsed_time(decode_end)
    decode_tok_s = (decode_tokens / decode_ms * 1000) if decode_ms > 0 else 0.0
    mean_itl_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0.0

    all_ids = torch.cat([input_ids] + generated_ids, dim=-1)

    gen_portion = all_ids[0, prompt_len:]
    eos_mask = (gen_portion == eos_token_id)
    if eos_mask.any():
        first_eos = eos_mask.nonzero(as_tuple=False)[0].item()
        all_ids = all_ids[:, :prompt_len + first_eos]

    timing = {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tokens": decode_tokens,
        "decode_tok_s": round(decode_tok_s, 1),
        "mean_itl_ms": round(mean_itl_ms, 2),
        "prompt_len": prompt_len,
        "method": "cuda_graph",
        "warmup_tokens": num_warmup,
    }

    return all_ids, timing


def benchmark_decode(
    model: nn.Module,
    tokenizer,
    prompt: str = "Explain the theory of general relativity in detail.",
    max_new_tokens: int = 256,
    num_runs: int = 20,
    warmup_runs: int = 3,
    use_cuda_graph: bool = True,
    cache=None,
    device: str = "cuda",
) -> dict:
    """Run full decode benchmark with statistics.

    Args:
        model: patched or unpatched model
        tokenizer: HF tokenizer
        prompt: test prompt
        max_new_tokens: tokens to generate per run
        num_runs: timed runs
        warmup_runs: untimed warmup
        use_cuda_graph: enable CUDA graph capture

    Returns:
        dict with mean/median/p95/p99 statistics
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_fn = generate_cuda_graph if use_cuda_graph else generate_eager

    # Warmup runs
    print(f"Warmup: {warmup_runs} runs...")
    for i in range(warmup_runs):
        _, timing = gen_fn(model, input_ids, max_new_tokens=max_new_tokens,
                           cache=cache, device=device)
        print(f"  warmup {i+1}: {timing['decode_tok_s']:.1f} tok/s")

    # Timed runs
    print(f"Benchmark: {num_runs} runs...")
    all_timings = []
    for i in range(num_runs):
        _, timing = gen_fn(model, input_ids, max_new_tokens=max_new_tokens,
                           cache=cache, device=device)
        all_timings.append(timing)
        print(f"  run {i+1}: {timing['decode_tok_s']:.1f} tok/s, ITL={timing['mean_itl_ms']:.2f}ms")

    # Compute statistics
    tok_s_list = [t["decode_tok_s"] for t in all_timings]
    itl_list = [t["mean_itl_ms"] for t in all_timings]
    prefill_list = [t["prefill_ms"] for t in all_timings]

    tok_s_sorted = sorted(tok_s_list)
    itl_sorted = sorted(itl_list)

    results = {
        "model": "Qwen3.5-27B",
        "method": "cuda_graph" if use_cuda_graph else "eager",
        "prompt_len": input_ids.shape[1],
        "max_new_tokens": max_new_tokens,
        "num_runs": num_runs,
        "decode_tok_s": {
            "mean": round(sum(tok_s_list) / len(tok_s_list), 1),
            "median": round(tok_s_sorted[len(tok_s_sorted) // 2], 1),
            "p95": round(tok_s_sorted[int(len(tok_s_sorted) * 0.05)], 1),  # p95 = 5th percentile of tok/s (worst)
            "p99": round(tok_s_sorted[int(len(tok_s_sorted) * 0.01)], 1),
            "min": round(min(tok_s_list), 1),
            "max": round(max(tok_s_list), 1),
        },
        "mean_itl_ms": {
            "mean": round(sum(itl_list) / len(itl_list), 2),
            "median": round(itl_sorted[len(itl_sorted) // 2], 2),
            "p95": round(itl_sorted[int(len(itl_sorted) * 0.95)], 2),
            "p99": round(itl_sorted[min(int(len(itl_sorted) * 0.99), len(itl_sorted) - 1)], 2),
        },
        "prefill_ms": {
            "mean": round(sum(prefill_list) / len(prefill_list), 2),
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
        "all_runs": all_timings,
    }

    return results


def generate_eager_batched(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    eos_token_id: int = 151645,
    device: str = "cuda",
    cache=None,
) -> tuple:
    """Batched autoregressive generation WITHOUT CUDA graph.

    All B sequences are decoded in lockstep (same number of steps).
    Per-sequence EOS tracking: sequences that hit EOS get masked but
    continue decoding (padded) until all sequences finish or max_new_tokens.

    Args:
        input_ids: [B, prompt_len]  - all prompts must have same length
        cache: HybridCache initialized with batch_size=B

    Returns (output_ids, timing_dict).
    """
    batch_size, prompt_len = input_ids.shape

    if cache is not None:
        cache.reset()

    # --- Prefill ---
    cache_position = torch.arange(prompt_len, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)

    prefill_start.record()
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

    next_token = logits[:, -1:, :].argmax(dim=-1)  # [B, 1]
    prefill_end.record()

    generated_ids = [next_token]
    cur_pos = prompt_len

    # Per-sequence EOS tracking
    eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # --- Decode loop ---
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)

    decode_start.record()
    decode_tokens = 0

    for step in range(max_new_tokens - 1):
        cache_position = torch.tensor([cur_pos], device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

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

        next_token = logits[:, -1:, :].argmax(dim=-1)  # [B, 1]

        # Mask EOS'd sequences to pad token
        next_token = torch.where(
            eos_reached.unsqueeze(1),
            torch.full_like(next_token, eos_token_id),
            next_token,
        )

        generated_ids.append(next_token)
        cur_pos += 1
        decode_tokens += 1

        # Check EOS every 8 tokens
        if decode_tokens % 8 == 0:
            recent = torch.cat(generated_ids[-8:], dim=-1)  # [B, 8]
            eos_reached |= (recent == eos_token_id).any(dim=-1)
            if eos_reached.all().item():
                break

    decode_end.record()
    torch.cuda.synchronize()

    prefill_ms = prefill_start.elapsed_time(prefill_end)
    decode_ms = decode_start.elapsed_time(decode_end)
    total_decode_tokens = decode_tokens * batch_size
    aggregate_tok_s = (total_decode_tokens / decode_ms * 1000) if decode_ms > 0 else 0.0
    per_user_itl_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0.0

    all_ids = torch.cat([input_ids] + generated_ids, dim=-1)  # [B, prompt_len + decode_tokens + 1]

    timing = {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tokens_per_user": decode_tokens,
        "total_decode_tokens": total_decode_tokens,
        "batch_size": batch_size,
        "aggregate_tok_s": round(aggregate_tok_s, 1),
        "per_user_tok_s": round(aggregate_tok_s / batch_size, 1),
        "per_user_itl_ms": round(per_user_itl_ms, 2),
        "prompt_len": prompt_len,
        "method": "eager_batched",
    }

    return all_ids, timing


def generate_cuda_graph_batched(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    eos_token_id: int = 151645,
    device: str = "cuda",
    cache=None,
    num_warmup: int = 5,
) -> tuple:
    """Batched generation with CUDA graph for decode phase.

    All B sequences decoded in lockstep. CUDA graph captures one batched
    decode step and replays it. Uses fused_lm_head_argmax_static_batched
    for allocation-free lm_head.

    Args:
        input_ids: [B, prompt_len]  - all prompts must have same length
        cache: HybridCache initialized with batch_size=B

    Returns (output_ids, timing_dict).
    """
    batch_size, prompt_len = input_ids.shape

    if cache is not None:
        cache.reset()

    # --- Prefill (eager) ---
    cache_position = torch.arange(prompt_len, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)

    prefill_start.record()
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

    next_token = logits[:, -1:, :].argmax(dim=-1)  # [B, 1]
    prefill_end.record()

    generated_ids = [next_token.clone()]
    cur_pos = prompt_len

    # --- Static tensors for graph capture ---
    static_token = next_token.clone()  # [B, 1]
    static_cache_pos = torch.tensor([cur_pos], device=device, dtype=torch.long)
    static_pos_ids = torch.full((batch_size, 1), cur_pos, device=device, dtype=torch.long)

    # Pre-allocate fused lm_head buffers for batched inference
    lm_w_t = model.lm_head.weight.t().contiguous()  # [K, N]
    _N_vocab = model.lm_head.weight.shape[0]
    _max_tiles = triton.cdiv(_N_vocab, 128)
    lm_local_max = torch.empty(batch_size, _max_tiles, device=device, dtype=torch.float32)
    lm_local_idx = torch.empty(batch_size, _max_tiles, device=device, dtype=torch.int64)
    lm_result = torch.empty(batch_size, device=device, dtype=torch.int64)

    def _decode_step():
        """Single batched decode step  - uses static tensors for graph capture."""
        h = model.model.embed_tokens(static_token)
        pos_emb = model.model.rotary_emb(h, static_pos_ids)
        x_normed_next = None
        for layer in model.model.layers:
            result = layer(
                h,
                cache_position=static_cache_pos,
                position_ids=static_pos_ids,
                position_embeddings=pos_emb,
                use_cache=True,
                x_normed_input=x_normed_next,
            )
            if isinstance(result, tuple) and len(result) == 2:
                h, x_normed_next = result
            else:
                h = result
                x_normed_next = None
        h = model.model.norm(h)
        # Fused batched lm_head + argmax
        h_flat = h.squeeze(1)  # [B, 5120]
        fused_lm_head_argmax_static_batched(
            h_flat, lm_w_t, lm_local_max, lm_local_idx, lm_result, batch_size,
        )
        return lm_result.view(batch_size, 1)

    # --- Warmup decode ---
    for _ in range(num_warmup):
        with torch.no_grad():
            warmup_next = _decode_step()
        generated_ids.append(warmup_next.clone())
        static_token.copy_(warmup_next)
        cur_pos += 1
        static_cache_pos.fill_(cur_pos)
        static_pos_ids.fill_(cur_pos)

    # --- CUDA Graph Capture ---
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            static_next_token = _decode_step()

    generated_ids.append(static_next_token.clone())
    cur_pos += 1

    # Per-sequence EOS tracking
    eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # --- Decode with graph replay ---
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)

    decode_start.record()
    decode_tokens = 0

    remaining = max_new_tokens - num_warmup - 2
    for step in range(remaining):
        static_token.copy_(static_next_token)
        static_cache_pos.fill_(cur_pos)
        static_pos_ids.fill_(cur_pos)

        graph.replay()

        generated_ids.append(static_next_token.clone())
        cur_pos += 1
        decode_tokens += 1

        # Check EOS every 8 tokens
        if decode_tokens % 8 == 0:
            recent = torch.cat(generated_ids[-8:], dim=-1)  # [B, 8]
            eos_reached |= (recent == eos_token_id).any(dim=-1)
            if eos_reached.all().item():
                break

    decode_end.record()
    torch.cuda.synchronize()

    prefill_ms = prefill_start.elapsed_time(prefill_end)
    decode_ms = decode_start.elapsed_time(decode_end)
    total_decode_tokens = decode_tokens * batch_size
    aggregate_tok_s = (total_decode_tokens / decode_ms * 1000) if decode_ms > 0 else 0.0
    per_user_itl_ms = decode_ms / decode_tokens if decode_tokens > 0 else 0.0

    all_ids = torch.cat([input_ids] + generated_ids, dim=-1)  # [B, total_len]

    timing = {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "decode_tokens_per_user": decode_tokens,
        "total_decode_tokens": total_decode_tokens,
        "batch_size": batch_size,
        "aggregate_tok_s": round(aggregate_tok_s, 1),
        "per_user_tok_s": round(aggregate_tok_s / batch_size, 1),
        "per_user_itl_ms": round(per_user_itl_ms, 2),
        "prompt_len": prompt_len,
        "method": "cuda_graph_batched",
        "warmup_tokens": num_warmup,
    }

    return all_ids, timing


def benchmark_decode_batched(
    model: nn.Module,
    tokenizer,
    prompt: str = "Explain the theory of general relativity in detail.",
    batch_size: int = 8,
    max_new_tokens: int = 256,
    num_runs: int = 10,
    warmup_runs: int = 2,
    use_cuda_graph: bool = True,
    cache_factory=None,
    device: str = "cuda",
) -> dict:
    """Run batched decode benchmark.

    Args:
        model: patched model
        tokenizer: HF tokenizer
        prompt: test prompt (replicated B times)
        batch_size: number of concurrent sequences
        cache_factory: callable(batch_size) -> HybridCache
        use_cuda_graph: enable CUDA graph capture

    Returns:
        dict with aggregate tok/s, per-user tok/s, per-user ITL
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_ids = input_ids.expand(batch_size, -1).contiguous()  # [B, prompt_len]

    gen_fn = generate_cuda_graph_batched if use_cuda_graph else generate_eager_batched

    # Warmup
    print(f"Warmup: {warmup_runs} runs (B={batch_size})...")
    for i in range(warmup_runs):
        cache = cache_factory(batch_size)
        _, timing = gen_fn(model, input_ids, max_new_tokens=max_new_tokens,
                           cache=cache, device=device)
        print(f"  warmup {i+1}: {timing['aggregate_tok_s']:.1f} agg tok/s, "
              f"{timing['per_user_itl_ms']:.2f}ms ITL/user")

    # Timed runs
    print(f"Benchmark: {num_runs} runs (B={batch_size})...")
    all_timings = []
    for i in range(num_runs):
        cache = cache_factory(batch_size)
        _, timing = gen_fn(model, input_ids, max_new_tokens=max_new_tokens,
                           cache=cache, device=device)
        all_timings.append(timing)
        print(f"  run {i+1}: {timing['aggregate_tok_s']:.1f} agg tok/s, "
              f"{timing['per_user_itl_ms']:.2f}ms ITL/user")

    agg_list = [t["aggregate_tok_s"] for t in all_timings]
    per_user_list = [t["per_user_tok_s"] for t in all_timings]
    itl_list = [t["per_user_itl_ms"] for t in all_timings]
    prefill_list = [t["prefill_ms"] for t in all_timings]

    agg_sorted = sorted(agg_list)
    itl_sorted = sorted(itl_list)

    results = {
        "model": "Qwen3.5-27B",
        "method": "cuda_graph_batched" if use_cuda_graph else "eager_batched",
        "batch_size": batch_size,
        "prompt_len": input_ids.shape[1],
        "max_new_tokens": max_new_tokens,
        "num_runs": num_runs,
        "aggregate_tok_s": {
            "mean": round(sum(agg_list) / len(agg_list), 1),
            "median": round(agg_sorted[len(agg_sorted) // 2], 1),
            "min": round(min(agg_list), 1),
            "max": round(max(agg_list), 1),
        },
        "per_user_tok_s": {
            "mean": round(sum(per_user_list) / len(per_user_list), 1),
        },
        "per_user_itl_ms": {
            "mean": round(sum(itl_list) / len(itl_list), 2),
            "median": round(itl_sorted[len(itl_sorted) // 2], 2),
            "p95": round(itl_sorted[int(len(itl_sorted) * 0.95)], 2),
        },
        "prefill_ms": {
            "mean": round(sum(prefill_list) / len(prefill_list), 2),
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
        "all_runs": all_timings,
    }

    return results


def save_results(results: dict, path: str):
    """Save benchmark results as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
