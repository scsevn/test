# Triton Kernels for Qwen3.5-27B on NVIDIA B200

Hand-written Triton kernels that accelerate Qwen3.5-27B inference to **92.5 tok/s** single-user
and **724 tok/s** at batch size 16 on a single NVIDIA B200. Pure BF16, no quantization,
no external inference engines.

## Results

Measured on NVIDIA B200 (208 SMs, 8 TB/s HBM3e, 192GB).

### Single-User Decode (B=1)

| Method | tok/s | ITL | VRAM | Speedup |
|--------|-------|-----|------|---------|
| HuggingFace generate (BF16) | 29.5 | 33.9ms | 56GB | 1.0x |
| torch.compile (max-autotune) | 18.1 | 55.1ms | 50.3GB | 0.6x |
| **This repo (eager)** | **42.2** | **23.7ms** | **50.5GB** | **1.4x** |
| **This repo (CUDA graph)** | **92.5** | **10.81ms** | **50.5GB** | **3.1x** |

torch.compile is slower than HuggingFace defaults because DeltaNet's recurrent state
updates cause graph breaks that prevent effective compilation.

### Batched Inference (B>1)

Weight reads dominate decode time (~80%). Batching amortizes the 54GB weight read across
all users, yielding near-linear aggregate throughput scaling.

| Batch | Aggregate tok/s | Per-User ITL | VRAM | Scaling Efficiency |
|-------|----------------|--------------|------|--------------------|
| B=1 | 92.2 | 10.85ms | 52.8GB | 100% |
| B=2 | 173.1 | 11.55ms | 53.1GB | 93.9% |
| B=4 | 305.1 | 13.11ms | 53.0GB | 82.7% |
| B=8 | 501.9 | 15.94ms | 53.5GB | 68.0% |
| B=16 | 723.9 | 22.10ms | 54.3GB | 49.1% |

Per-user latency stays flat through B=2, degrades gently through B=8, and hits the
compute-bound regime at B=16 where cuBLAS transitions from memory-bound to compute-bound GEMMs.

### Profiler Breakdown (B=1, per decode step)

| Category | Time | % of Step |
|----------|------|-----------|
| cuBLAS (aten::mm) | 8.59ms | 80.4% |
| Fused DeltaNet recurrent | 0.56ms | 5.2% |
| Fused residual + RMSNorm | 0.42ms | 3.9% |
| Causal conv1d | 0.37ms | 3.5% |
| Kernel launch overhead | 0.27ms | 2.5% |
| FlashAttention (SDPA) | 0.19ms | 1.8% |
| SiLU MLP | 0.11ms | 1.1% |
| KV cache ops | 0.10ms | 0.9% |
| RMSNorm gated | 0.09ms | 0.8% |

The theoretical floor for this model is 6.7ms (54GB / 8 TB/s), which gives ~149 tok/s.
The gap between measured ITL (10.81ms) and the floor is the non-GEMV kernel overhead (2.2ms)
plus cuBLAS not achieving perfect bandwidth utilization.

## Architecture

Qwen3.5-27B is a 64-layer hybrid model:
- **48 DeltaNet layers** (75%): Gated DeltaNet linear attention with recurrent state
- **16 GQA layers** (25%): Standard grouped-query attention (24 Q heads, 4 KV heads, head_dim=256)
- **MLP**: SiLU-gated, intermediate_size=17408
- **Vocab**: 248,320 tokens
- **BF16 size**: ~54GB

The layer pattern repeats as `[DeltaNet, DeltaNet, DeltaNet, GQA] x 16`.

## Kernels

All kernels are written in Triton with `@triton.autotune` configs tuned for B200.

### Core Kernels

| Kernel | File | Purpose | Calls/Token |
|--------|------|---------|-------------|
| Fused DeltaNet Recurrent | `forge/kernels/triton_deltanet_fused.py` | Post-projection + recurrent state update in one kernel | 48 |
| DeltaNet Recurrent (standalone) | `forge/kernels/triton_deltanet_recurrent.py` | Tiled state update: decay, delta rule, rank-1 update, output | 48 |
| DeltaNet Post-Projection | `forge/kernels/triton_deltanet_prep.py` | L2-norm, scaling, gating, beta computation | 48 |
| Causal Conv1d | `forge/kernels/triton_causal_conv1d.py` | Fused shift + insert + dot product + SiLU | 48 |
| RMSNorm | `forge/kernels/triton_rmsnorm.py` | Fused 7-op RMSNorm | 128 |
| RMSNorm Gated | `forge/kernels/triton_rmsnorm_gated.py` | RMSNorm * SiLU(gate) for DeltaNet output | 48 |
| Fused Residual + RMSNorm | `forge/kernels/triton_fused_residual_norm.py` | Residual add + RMSNorm in one pass | 63 |
| SiLU MLP | `forge/kernels/triton_silu_mlp.py` | Fused SiLU(gate) * up | 64 |
| QK-Norm + RoPE | `forge/kernels/triton_fused_qknorm_rope.py` | Fused normalization + partial rotary embedding | 16 |
| LM Head + Argmax | `forge/kernels/triton_lm_head_topk.py` | Fused GEMV + streaming argmax (no 248K logit materialization) | 1 |
| BF16 GEMV | `forge/kernels/triton_bf16_gemv.py` | Split-K GEMV (reference, cuBLAS used in practice) | 0 |

### What Makes the DeltaNet Kernel Special

The fused DeltaNet kernel (`triton_deltanet_fused.py`) merges the post-projection and recurrent
steps into a single kernel launch per head. DeltaNet's recurrent state is a [128, 128] matrix
(32KB in BF16) that fits entirely in B200's 256KB SRAM per SM.

The kernel:
1. Splits QKV from the conv1d output
2. L2-normalizes Q and K
3. Computes gating (sigmoid of learned parameter + dt_bias)
4. Computes beta (softplus of raw beta, scaled by alpha)
5. Applies exponential decay to state: `S *= exp(gate)`
6. Runs the delta rule: `delta = beta * (v - S^T @ k)`
7. Updates state: `S += outer(k, delta)`
8. Computes output: `o = S^T @ q`

All in one kernel, with tiled [BLOCK_DK, BLOCK_DV] loads for the state matrix.
This eliminates 2,304 kernel launches per token (48 heads x 2 kernels x 24 tiles)
and removes 3.5MB of intermediate HBM traffic.

## Integration

### Model Patching

The `forge/llm/patch_qwen35.py` module monkey-patches all 64 layers of the HuggingFace model:

```python
from forge.llm.patch_qwen35 import load_model, patch_model
from forge.llm.cache import HybridCache

model = load_model(model_id="Qwen/Qwen3.5-27B", device="cuda")
cache = HybridCache(batch_size=1, max_cache_len=512, device="cuda")
model = patch_model(model, cache)
```

Key optimizations in the patcher:
- **Weight pre-concatenation**: Merges 4 DeltaNet projections into 1 GEMV, 2 MLP projections into 1, 3 attention projections into 1
- **Weight deduplication**: Original parameters replaced with views into concatenated weights (saves ~30GB)
- **Inter-layer chaining**: Fuses the residual-add of one layer's output with the next layer's input RMSNorm

### Generation

```python
from forge.llm.generate import generate_cuda_graph

output_ids, timing = generate_cuda_graph(
    model, input_ids,
    max_new_tokens=256,
    cache=cache,
    device="cuda",
)
```

The CUDA graph path captures the entire decode step (all 64 layers + lm_head + argmax)
into a single graph, eliminating kernel launch overhead entirely.

### Batched Generation

```python
from forge.llm.generate import generate_cuda_graph_batched
from forge.llm.cache import HybridCache

cache = HybridCache(batch_size=8, max_cache_len=512, device="cuda")
model = patch_model(model, cache)

# input_ids: [B, seq_len]
output_ids, timing = generate_cuda_graph_batched(
    model, input_ids,
    max_new_tokens=256,
    cache=cache,
    device="cuda",
)
```

All sequences decode in lockstep with per-sequence EOS tracking.

## Optimization History

| Version | Eager tok/s | CUDA Graph tok/s | ITL | Key Changes |
|---------|-------------|------------------|-----|-------------|
| v1 | 24.2 | 58.7 | 17.0ms | Initial Triton kernels |
| v2 | 34.0 | 71.7 | 13.96ms | Weight pre-concatenation + dedup |
| v2+ | 38.1 | 86.9 | 11.50ms | Tiled DeltaNet recurrent |
| v3 | 42.2 | 92.5 | 10.81ms | Deep fusion (residual+norm, QKnorm+RoPE) |
| v4 | 42.2 | 94.4 | 10.60ms | Fused post-proj+recurrent, lm_head+argmax |

## File Structure

```
forge/
  kernels/                    # Triton kernel implementations
    triton_deltanet_fused.py  # Fused post-proj + recurrent (flagship)
    triton_deltanet_recurrent.py
    triton_deltanet_prep.py
    triton_causal_conv1d.py
    triton_rmsnorm.py
    triton_rmsnorm_gated.py
    triton_fused_residual_norm.py
    triton_silu_mlp.py
    triton_fused_qknorm_rope.py
    triton_lm_head_topk.py
    triton_bf16_gemv.py
  llm/
    patch_qwen35.py           # Model patcher (monkey-patches 64 layers)
    cache.py                  # Hybrid cache (DeltaNet state + KV cache)
    generate.py               # Decode loops (eager, CUDA graph, batched)
benchmarks/
  optimized.py                # Full benchmark (eager + CUDA graph)
  batch_benchmark.py          # Batched throughput (B=1..16)
  profile_nsight.py           # torch.profiler decode step analysis
  baseline.py                 # HuggingFace generate baseline
  baseline_inductor.py        # torch.compile baseline
  v4_validate.py              # Kernel correctness + E2E validation
```

## Requirements

- Python 3.11+
- PyTorch 2.6+
- Triton 3.1+
- Transformers 4.57+ (Qwen3.5 support)
- NVIDIA GPU with BF16 support (developed on B200)

```bash
pip install -r requirements.txt
```

## Hardware Notes

Kernel autotune configs are tuned for NVIDIA B200 (Blackwell, sm_100, 208 SMs).
The kernels will run on other GPUs but may need re-tuning. Key parameters that
change across GPUs:
- Number of SMs (affects optimal grid sizes and split-K factors)
- SRAM per SM (affects tile sizes for the recurrent kernel)
- HBM bandwidth (affects the memory-bound / compute-bound crossover for batching)
- L2 cache size (affects tile reuse strategies)

## Key Lessons

1. **cuBLAS beats custom Triton GEMV for M=1**: For single-token decode, cuBLAS achieves ~77% of peak bandwidth. Our custom split-K GEMV was ~4x slower. We use F.linear (cuBLAS) for all projections.

2. **DeltaNet state fits in SRAM**: The [128, 128] recurrent state (32KB) fits in B200's 256KB SRAM per SM, making the recurrent kernel compute-bound rather than memory-bound.

3. **Weight reads dominate**: At 80% of decode time, the only way to significantly increase single-user throughput beyond ~95 tok/s is quantization (INT8/FP8). Batching amortizes the same weight reads across multiple users.

4. **torch.compile fails on DeltaNet**: The recurrent state updates and dynamic control flow in DeltaNet cause graph breaks that prevent torch.compile from being effective. Manual Triton kernels are the only viable optimization path for this architecture.

5. **CUDA graphs give 2.2x speedup**: Capturing the entire decode step eliminates ~640 kernel launches per token, turning a 23.7ms eager step into 10.81ms.

## License

MIT
