# Optimization Log: Qwen3.5-27B Dense on B200

## Overview

Maximize decode throughput for Qwen3.5-27B (Dense, hybrid DeltaNet + GQA)
on NVIDIA B200 using pure Triton kernels. No quantization, no vLLM.

Baseline target: ~45-55 tok/s (HuggingFace BF16 default)
Optimized target: 85-110 tok/s (1.7-2.2x speedup)

---

## Hardware: NVIDIA B200

- 208 SMs, 8 TB/s HBM3e bandwidth, 192GB HBM3e
- 96MB L2 cache, 256KB SRAM per SM
- BF16 tensor cores: 2,250 TFLOPS
- Theoretical decode floor: 54GB / 8TB/s = 6.7ms = ~149 tok/s
- Perfect wave sizes: 208 / 416 / 624 blocks

---

## Kernel 1: Triton RMSNorm

**File**: `forge/kernels/triton_rmsnorm.py`
**Usage**: 128x per token (2 norms x 64 layers)
**Strategy**: Single kernel replaces 7 PyTorch ops (cast, pow, mean, rsqrt, mul, mul, cast)
**BLOCK_N**: 8192 (next power-of-2 for hidden_size=5120)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Kernel launches per norm | 7 | 1 | -6 |
| HBM round-trips | 7 | 1 | -6 |

**Status**: Ported from qwen-benchmark. Awaiting B200 validation.

---

## Kernel 2: Triton RMSNormGated

**File**: `forge/kernels/triton_rmsnorm_gated.py`
**Usage**: 48x per token (DeltaNet output gate normalization)
**Strategy**: Fuses rmsnorm(hidden) * silu(gate) into 1 kernel (~10 ops -> 1)

**Status**: New kernel. Awaiting B200 validation.

---

## Kernel 3: BF16 GEMV with Split-K (CRITICAL, 65% of decode time)

**File**: `forge/kernels/triton_bf16_gemv.py`
**Usage**: Every linear projection in every layer
**Strategy**: Split-K for B200 SM utilization, target 416 blocks (2 waves)

Key shapes:
- MLP gate/up: [5120, 17408], 178MB per layer
- MLP down: [17408, 5120], 178MB per layer
- lm_head: [5120, 248320], 2.4GB, 1x per token

**Variants**:
1. Standard GEMM (for M > 1 prefill)
2. Split-K GEMM (for small N with M=1)
3. Specialized GEMV (M=1, no tl.dot, uses vector-matrix multiply)
4. Split-K GEMV (M=1, more blocks)

**Status**: Implemented with B200 autotune configs. Awaiting benchmark vs cuBLAS.

---

## Kernel 4: Fused DeltaNet Recurrent (FLAGSHIP)

**File**: `forge/kernels/triton_deltanet_recurrent.py`
**Usage**: 48x per token (all DeltaNet layers)
**Strategy**: Fuses entire recurrent step into 1 kernel per head

State per head: [128, 128] = 32KB BF16, fits in B200's 256KB SRAM.

**Math**:
```
S *= exp(g)              # gate decay
residual = v - S^T @ k   # delta rule
delta = beta * residual
S += outer(k, delta)     # rank-1 update
o = S^T @ q              # output
```

**Two-pass design**:
- Pass 1: Load S, apply decay, accumulate S^T@k
- Pass 2: Compute delta, update S, accumulate S^T@q

**Variants**:
1. V1: Row-by-row processing (register-friendly), IMPLEMENTED
2. V2: Tiled processing (for larger states), IMPLEMENTED
3. V3: Multi-head per block (TODO)

**Correctness**: Must verify 1000 decode steps without state drift.

**Status**: Two variants implemented. Awaiting correctness validation.

---

## Kernel 5: Fused Causal Conv1d Update

**File**: `forge/kernels/triton_causal_conv1d.py`
**Usage**: 48x per token (DeltaNet conv on projected QKV)
**Strategy**: Shift + insert + dot product + SiLU in one kernel

Grid: (8192 channels, B) per layer = 8192 blocks, ~40 waves on B200.

**Status**: Implemented. Awaiting validation.

---

## Kernel 6: Fused SiLU Gate MLP

**File**: `forge/kernels/triton_silu_mlp.py`
**Usage**: 64x per token (every layer)
**Strategy**: Fuses SiLU(gate) * up elementwise (saves 70KB HBM per layer)

**Variants**:
1. Elementwise fusion only (separate GEMVs + fused activation), IMPLEMENTED
2. Fully fused dual-GEMV + SiLU (experimental), IMPLEMENTED

**Status**: Both variants implemented. Must benchmark A vs B.

---

## Kernel 7: Fused QKV + Partial RoPE

**File**: `forge/kernels/triton_fused_qkv_rope.py`
**Usage**: 16x per token (full attention layers only)
**Strategy**: Single GEMV for concatenated Q+K+V + inline partial RoPE

Reads x once (10KB) instead of 3 times. RoPE on first 64 of 256 dims.

**Status**: Implemented. Awaiting validation.

---

## Kernel 8: Fused lm_head + Streaming Argmax

**File**: `forge/kernels/triton_lm_head_topk.py`
**Usage**: 1x per token
**Strategy**: Never materializes full 248K logits vector.
Each GEMV tile computes local max, reduce kernel finds global argmax.

Saves ~500KB HBM write+read per token. CUDA graph compatible (deterministic shape).

**Status**: Implemented. Awaiting validation.

---

## Integration

**Model patcher**: `forge/llm/patch_qwen35.py`
- Monkey-patches all 64 layers
- DeltaNet forward: fused proj + conv1d + recurrent + norm_gated + output
- Attention forward: fused norm + GEMV QKV + SDPA + output

**Cache**: `forge/llm/cache.py`
- HybridCache: DeltaNet (conv_state + recurrent_state) + KV cache
- Total: ~72MB DeltaNet + KV cache

**Generation**: `forge/llm/generate.py`
- Eager decode loop with CUDA event timing
- CUDA graph capture after warmup
- Eliminates ~640 kernel launches per token

---

## Expected Gains

| Source | Expected Speedup | Notes |
|--------|-----------------|-------|
| CUDA graph | +40-50% | Eliminates 3ms of 7ms launch overhead |
| Fused DeltaNet recurrent | +10-15% | 1 kernel vs 5+ per layer |
| Fused RMSNorm/Gated | +5-8% | 128 fusion points per token |
| Fused SiLU MLP | +3-5% | Eliminates intermediate writes |
| BF16 GEMV split-K | +5-10% | Better SM utilization vs cuBLAS |

---

## Next Steps

1. [ ] Run baseline benchmark on B200
2. [ ] Validate all 8 kernels (correctness + cosine > 0.9999)
3. [ ] Run optimized benchmark
4. [ ] Evolve top 3 kernels
5. [ ] DeltaNet 1000-step drift test
6. [ ] E2E correctness check (patched vs unpatched greedy decode)
7. [ ] Profile optimized model, identify remaining bottlenecks
8. [ ] Iterate: evolve kernels, re-benchmark
