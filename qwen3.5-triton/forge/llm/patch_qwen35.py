"""
Qwen3.5-27B Dense Model Patcher (v3: Deep Fusion)

v3 optimizations over v2:
  1. Weight deduplication: replace original params with views into concatenated weights
     Saves ~30GB VRAM (81GB → ~54GB), reduces HBM controller pressure
  2. Fused QK-norm + partial RoPE: replaces ~18 PyTorch ops with 2 Triton launches
     per attention layer. Saves ~192 kernel executions over 16 layers.
  3. Fused sigmoid gate multiply: replaces 4 PyTorch ops with 1 Triton kernel
     per attention layer. Saves 48 kernel executions.

v2 optimizations (retained):
  1. Weight pre-concatenation: 4→1 GEMV for DeltaNet proj, 2→1 for MLP, 3→1 for attention
  2. Fused residual+RMSNorm: eliminates 64 separate add kernels per token
  3. Fused DeltaNet post-projection: 17 ops → 1 Triton kernel per layer
  4. Pre-computed constants: -exp(A_log), dt_bias.float() computed once at patch time

Kernel launch count per token:
  v1: ~1250 → v2: ~450 → v3: ~210 (83% reduction from v1)

Monkey-patches layer.forward() for all 64 layers of Qwen3.5-27B:
  - 48 DeltaNet layers (Qwen3_5GatedDeltaNet) -> fused_deltanet_forward()
  - 16 full attention layers (Qwen3_5Attention) -> fused_attention_forward()

Actual Qwen3.5-27B architecture (from model inspection):

  DeltaNet layer (linear_attn: Qwen3_5GatedDeltaNet):
    in_proj_qkv: [10240, 5120] -- fused Q(2048)+K(2048)+V(6144)
    in_proj_a:   [48, 5120]    -- alpha (num_v_heads=48)
    in_proj_b:   [48, 5120]    -- beta (num_v_heads=48)
    in_proj_z:   [6144, 5120]  -- gate z (value_dim=6144)
    A_log:       [48]          -- log-space decay
    dt_bias:     [48]          -- delta-time bias
    conv1d:      Conv1d(10240, 1, 4, groups=10240) -- depthwise
    norm:        RMSNormGated([128]) -- per head_v_dim
    out_proj:    [5120, 6144]  -- output

  Attention layer (self_attn: Qwen3_5Attention):
    q_proj:      [12288, 5120] -- 48*256 (24 heads * 256 * 2 for gating)
    k_proj:      [1024, 5120]  -- 4 KV heads * 256
    v_proj:      [1024, 5120]  -- 4 KV heads * 256
    o_proj:      [5120, 6144]  -- hidden_size x (24 * 256)
    q_norm:      [256]         -- per head dim
    k_norm:      [256]         -- per head dim

  Both layer types share:
    mlp.gate_proj: [17408, 5120]
    mlp.up_proj:   [17408, 5120]
    mlp.down_proj: [5120, 17408]
    input_layernorm: [5120]
    post_attention_layernorm: [5120]
"""
import gc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.kernels.triton_rmsnorm import triton_rmsnorm
from forge.kernels.triton_rmsnorm_gated import triton_rmsnorm_gated
from forge.kernels.triton_fused_residual_norm import fused_residual_rmsnorm
from forge.kernels.triton_deltanet_recurrent import deltanet_recurrent_step
from forge.kernels.triton_deltanet_prep import deltanet_post_proj
from forge.kernels.triton_deltanet_fused import fused_postproj_recurrent
from forge.kernels.triton_causal_conv1d import causal_conv1d_update
from forge.kernels.triton_silu_mlp import fused_silu_mul, fused_sigmoid_mul
from forge.kernels.triton_fused_qknorm_rope import fused_qknorm_rope
from forge.llm.cache import HybridCache


# Layer pattern: [DeltaNet, DeltaNet, DeltaNet, Attention] x 16
DELTANET_PATTERN = [True, True, True, False] * 16  # True = DeltaNet

# DeltaNet dimension constants (from model config)
NUM_K_HEADS = 16
HEAD_K_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM  # 2048
NUM_V_HEADS = 48
HEAD_V_DIM = 128
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM  # 6144
CONV_DIM = KEY_DIM + KEY_DIM + VALUE_DIM  # 10240 (Q+K+V channels for conv1d)

# Full attention constants
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
Q_OUT_DIM = NUM_Q_HEADS * HEAD_DIM  # 6144
KV_OUT_DIM = NUM_KV_HEADS * HEAD_DIM  # 1024
Q_PROJ_DIM = 12288  # 2 * Q_OUT_DIM (includes gate for output gating)

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408

# Pre-computed constants
INV_SQRT_DK = HEAD_K_DIM ** -0.5  # 1/sqrt(128)

# Split indices for pre-concatenated DeltaNet projection weights
# w_delta_proj = cat(w_qkv[10240], w_a[48], w_b[48], w_z[6144]) = [16480, 5120]
DELTA_SPLIT_QKV = CONV_DIM              # 10240
DELTA_SPLIT_A = DELTA_SPLIT_QKV + NUM_V_HEADS  # 10288
DELTA_SPLIT_B = DELTA_SPLIT_A + NUM_V_HEADS    # 10336
DELTA_PROJ_DIM = DELTA_SPLIT_B + VALUE_DIM     # 16480

# Split indices for pre-concatenated attention QKV weights
# w_attn_qkv = cat(q_proj[12288], k_proj[1024], v_proj[1024]) = [14336, 5120]
ATTN_SPLIT_Q = Q_PROJ_DIM               # 12288
ATTN_SPLIT_K = ATTN_SPLIT_Q + KV_OUT_DIM  # 13312
ATTN_QKV_DIM = ATTN_SPLIT_K + KV_OUT_DIM  # 14336


def is_deltanet_layer(layer_idx: int) -> bool:
    return DELTANET_PATTERN[layer_idx]


def load_model(
    model_id: str = "Qwen/Qwen3.5-27B",
    cache_dir: str = "/cache/models",
    device: str = "cuda",
    hf_token: Optional[str] = None,
) -> nn.Module:
    """Load Qwen3.5-27B in BF16 on device."""
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_id} in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    vram_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"Model loaded: {vram_gb:.1f}GB VRAM")

    return model


def patch_model(
    model: nn.Module,
    cache: HybridCache,
) -> nn.Module:
    """Patch all 64 layers with fused Triton kernels (v2: aggressive fusion)."""
    config = model.config
    hidden_size = config.hidden_size
    eps = getattr(config, 'rms_norm_eps', 1e-6)
    intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers

    print(f"Patching {num_layers} layers (v3: weight dedup + fused QKnorm+RoPE + sigmoid gate)...")

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]

        # Inter-layer chaining: pass next layer's input_layernorm weight
        # so we can fuse the final residual-add with the next layer's norm
        next_norm_w = None
        if layer_idx < num_layers - 1:
            next_norm_w = model.model.layers[layer_idx + 1].input_layernorm.weight

        if is_deltanet_layer(layer_idx):
            fwd = _create_fused_deltanet_forward(
                layer, layer_idx, cache,
                hidden_size=hidden_size,
                eps=eps,
                intermediate_size=intermediate_size,
                next_layer_norm_w=next_norm_w,
            )
        else:
            fwd = _create_fused_attention_forward(
                layer, layer_idx, cache,
                hidden_size=hidden_size,
                eps=eps,
                intermediate_size=intermediate_size,
                next_layer_norm_w=next_norm_w,
            )

        layer.forward = fwd

    _patch_final_norm(model, eps)

    # Free old weight tensors that were replaced by views into concatenated buffers
    gc.collect()
    torch.cuda.empty_cache()

    vram_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"Patching complete. Cache: {cache.memory_mb():.1f}MB, VRAM: {vram_gb:.1f}GB")
    return model


def _create_fused_deltanet_forward(
    layer: nn.Module,
    layer_idx: int,
    cache: HybridCache,
    hidden_size: int,
    eps: float,
    intermediate_size: int,
    next_layer_norm_w=None,
):
    """Create fused forward for a Qwen3_5GatedDeltaNet layer.

    v2 optimizations:
    - Pre-concatenated projection weights: 4 GEMVs → 1
    - Pre-concatenated MLP weights: 2 GEMVs → 1
    - Fused residual+RMSNorm between attention and MLP
    - Pre-computed -exp(A_log) and dt_bias.float()
    """
    # Norm weights
    input_ln_w = layer.input_layernorm.weight
    post_ln_w = layer.post_attention_layernorm.weight

    attn = layer.linear_attn  # Qwen3_5GatedDeltaNet

    # ===== PRE-CONCATENATE DELTANET PROJECTION WEIGHTS + DEDUP =====
    # Instead of 4 separate F.linear calls, do 1 with concatenated weights
    # w_qkv[10240,5120] + w_a[48,5120] + w_b[48,5120] + w_z[6144,5120] = [16480,5120]
    w_delta_proj = torch.cat([
        attn.in_proj_qkv.weight,   # [10240, 5120]
        attn.in_proj_a.weight,     # [48, 5120]
        attn.in_proj_b.weight,     # [48, 5120]
        attn.in_proj_z.weight,     # [6144, 5120]
    ], dim=0).contiguous()  # [16480, 5120]
    # Replace original params with views into concatenated (shares storage, frees originals)
    attn.in_proj_qkv.weight.data = w_delta_proj[:DELTA_SPLIT_QKV]
    attn.in_proj_a.weight.data = w_delta_proj[DELTA_SPLIT_QKV:DELTA_SPLIT_A]
    attn.in_proj_b.weight.data = w_delta_proj[DELTA_SPLIT_A:DELTA_SPLIT_B]
    attn.in_proj_z.weight.data = w_delta_proj[DELTA_SPLIT_B:]

    # ===== PRE-COMPUTE CONSTANTS =====
    # These are model parameters that never change during inference
    neg_A_exp = (-attn.A_log.float().exp())  # [48]  - negate + exp once
    dt_bias_f = attn.dt_bias.float()         # [48]  - cast once

    # Conv1d weights
    conv1d = attn.conv1d
    conv_w = conv1d.weight.squeeze(1)  # [10240, 1, 4] -> [10240, 4]
    conv_b = getattr(conv1d, 'bias', None)

    # Output norm and projection
    norm_w = attn.norm.weight            # [128] (head_v_dim)
    w_out = attn.out_proj.weight         # [5120, 6144]

    # ===== PRE-CONCATENATE MLP WEIGHTS + DEDUP =====
    # gate_proj[17408,5120] + up_proj[17408,5120] = [34816,5120]
    mlp = layer.mlp
    w_mlp_gate_up = torch.cat([
        mlp.gate_proj.weight,   # [17408, 5120]
        mlp.up_proj.weight,     # [17408, 5120]
    ], dim=0).contiguous()  # [34816, 5120]
    # Dedup: replace originals with views
    mlp.gate_proj.weight.data = w_mlp_gate_up[:INTERMEDIATE_SIZE]
    mlp.up_proj.weight.data = w_mlp_gate_up[INTERMEDIATE_SIZE:]
    w_mlp_down = mlp.down_proj.weight  # [5120, 17408]

    def fused_forward(
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        x_normed_input=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if q_len > 1:
            # ====== PREFILL PATH  - fall back to HuggingFace ======
            return layer.__class__.forward(
                layer, hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # ============================================================
        # DECODE PATH (q_len == 1)  - fused kernels
        # ============================================================
        residual = hidden_states

        # Use pre-computed norm from previous layer if available
        if x_normed_input is not None:
            x_normed = x_normed_input
        else:
            x_normed = triton_rmsnorm(hidden_states, input_ln_w, eps, add_one_to_weight=True)
        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for all DeltaNet projections =====
        # One cuBLAS call instead of 4: reads x_2d once, 16480 output elements
        proj_all = F.linear(x_2d, w_delta_proj)  # [M, 16480]
        qkv = proj_all[:, :DELTA_SPLIT_QKV]           # [M, 10240]
        alpha = proj_all[:, DELTA_SPLIT_QKV:DELTA_SPLIT_A]  # [M, 48]
        beta_raw = proj_all[:, DELTA_SPLIT_A:DELTA_SPLIT_B]  # [M, 48]
        z = proj_all[:, DELTA_SPLIT_B:]                # [M, 6144]

        dn_cache = cache.get_deltanet_cache(layer_idx)

        # Fused causal conv1d update (shift+insert+conv+silu)
        qkv_flat = qkv.reshape(bsz, CONV_DIM)
        qkv_conv, _ = causal_conv1d_update(
            qkv_flat, dn_cache.conv_state, conv_w, conv_b, apply_silu=True,
        )

        # ===== FUSED POST-PROJ + RECURRENT: 2 kernels → 1 =====
        # Post-proj (L2 norm, scale, gate, beta) + recurrent (decay, delta,
        # state update, output query) in a single kernel launch per v_head.
        # Intermediate Q/K/V/gate/beta never touch HBM.
        attn_output, dn_cache.recurrent_state = fused_postproj_recurrent(
            qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias_f,
            dn_cache.recurrent_state,
        )

        # RMSNormGated per-head
        attn_flat = attn_output.reshape(-1, HEAD_V_DIM)
        z_flat = z.reshape(-1, HEAD_V_DIM)
        attn_normed = triton_rmsnorm_gated(attn_flat, z_flat, norm_w, eps)
        attn_output = attn_normed.reshape(bsz, 1, VALUE_DIM)

        # Output projection
        o_2d = attn_output.reshape(-1, VALUE_DIM)
        attn_output = F.linear(o_2d, w_out).reshape(bsz, q_len, hidden_size)

        # ===== FUSED RESIDUAL + RMSNORM =====
        # Replaces: hidden = residual + attn_output; normed = rmsnorm(hidden)
        # One kernel instead of two
        hidden_states, x_normed = fused_residual_rmsnorm(
            residual, attn_output, post_ln_w, eps, add_one_to_weight=True,
        )

        # ============================================================
        # MLP block
        # ============================================================
        # hidden_states is already the new residual (from fused kernel)

        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for gate+up projections =====
        # One cuBLAS call instead of 2: reads x_2d once, 34816 output elements
        gate_up = F.linear(x_2d, w_mlp_gate_up)  # [M, 34816]
        gate_out = gate_up[:, :INTERMEDIATE_SIZE]
        up_out = gate_up[:, INTERMEDIATE_SIZE:]

        mlp_mid = fused_silu_mul(gate_out, up_out)

        down_out = F.linear(
            mlp_mid.reshape(-1, intermediate_size), w_mlp_down,
        ).reshape(bsz, q_len, hidden_size)

        # Inter-layer chaining: fuse residual-add + next layer's input norm
        if next_layer_norm_w is not None and q_len == 1:
            hidden_states, x_normed_next = fused_residual_rmsnorm(
                hidden_states, down_out, next_layer_norm_w, eps, add_one_to_weight=True,
            )
            return hidden_states, x_normed_next
        else:
            hidden_states = hidden_states + down_out
            return hidden_states

    return fused_forward


def _create_fused_attention_forward(
    layer: nn.Module,
    layer_idx: int,
    cache: HybridCache,
    hidden_size: int,
    eps: float,
    intermediate_size: int,
    next_layer_norm_w=None,
):
    """Create fused forward for a Qwen3_5Attention layer.

    v3 optimizations:
    - Weight dedup: original params replaced with views into concatenated
    - Fused QK-norm + partial RoPE: 2 Triton launches instead of ~18 PyTorch ops
    - Fused sigmoid gate multiply: 1 Triton kernel instead of 4 PyTorch ops
    v2 retained:
    - Pre-concatenated QKV weights: 3 GEMVs → 1
    - Pre-concatenated MLP weights: 2 GEMVs → 1
    - Fused residual+RMSNorm between attention and MLP
    """
    input_ln_w = layer.input_layernorm.weight
    post_ln_w = layer.post_attention_layernorm.weight
    self_attn = layer.self_attn
    mlp = layer.mlp

    # ===== PRE-CONCATENATE ATTENTION QKV WEIGHTS + DEDUP =====
    # q_proj[12288,5120] + k_proj[1024,5120] + v_proj[1024,5120] = [14336,5120]
    w_attn_qkv = torch.cat([
        self_attn.q_proj.weight,  # [12288, 5120]
        self_attn.k_proj.weight,  # [1024, 5120]
        self_attn.v_proj.weight,  # [1024, 5120]
    ], dim=0).contiguous()  # [14336, 5120]
    # Dedup: replace originals with views
    self_attn.q_proj.weight.data = w_attn_qkv[:ATTN_SPLIT_Q]
    self_attn.k_proj.weight.data = w_attn_qkv[ATTN_SPLIT_Q:ATTN_SPLIT_K]
    self_attn.v_proj.weight.data = w_attn_qkv[ATTN_SPLIT_K:]

    w_o = self_attn.o_proj.weight  # [5120, 6144]

    # Pre-extract norm weights for fused QKnorm+RoPE
    q_norm_w = self_attn.q_norm.weight if hasattr(self_attn, 'q_norm') and self_attn.q_norm is not None else None
    k_norm_w = self_attn.k_norm.weight if hasattr(self_attn, 'k_norm') and self_attn.k_norm is not None else None

    # ===== PRE-CONCATENATE MLP WEIGHTS + DEDUP =====
    w_mlp_gate_up = torch.cat([
        mlp.gate_proj.weight,   # [17408, 5120]
        mlp.up_proj.weight,     # [17408, 5120]
    ], dim=0).contiguous()  # [34816, 5120]
    # Dedup: replace originals with views
    mlp.gate_proj.weight.data = w_mlp_gate_up[:INTERMEDIATE_SIZE]
    mlp.up_proj.weight.data = w_mlp_gate_up[INTERMEDIATE_SIZE:]
    w_mlp_down = mlp.down_proj.weight  # [5120, 17408]

    def fused_forward(
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        x_normed_input=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # ============================================================
        # Attention block
        # ============================================================
        residual = hidden_states

        # Use pre-computed norm from previous layer if available
        if x_normed_input is not None and q_len == 1:
            x_normed = x_normed_input
        else:
            x_normed = triton_rmsnorm(hidden_states, input_ln_w, eps, add_one_to_weight=True)
        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for Q+K+V =====
        qkv_all = F.linear(x_2d, w_attn_qkv)  # [M, 14336]
        q_full = qkv_all[:, :ATTN_SPLIT_Q]        # [M, 12288]
        k = qkv_all[:, ATTN_SPLIT_Q:ATTN_SPLIT_K]  # [M, 1024]
        v = qkv_all[:, ATTN_SPLIT_K:]               # [M, 1024]

        # q_proj output is 12288 = num_q_heads * head_dim * 2 (interleaved Q + gate)
        q_full_heads = q_full.view(bsz, q_len, NUM_Q_HEADS, HEAD_DIM * 2)
        q, q_gate = q_full_heads.chunk(2, dim=-1)
        q_gate = q_gate.reshape(bsz, q_len, -1)

        # Reshape Q/K/V to [B, num_heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # RoPE position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
        elif hasattr(self_attn, 'rotary_emb'):
            cos, sin = self_attn.rotary_emb(v, position_ids)
        else:
            cos, sin = None, None

        if q_len == 1 and cos is not None and q_norm_w is not None:
            # ===== FUSED QK-NORM + PARTIAL ROPE (decode only) =====
            # Replaces: 2x triton_rmsnorm + apply_rotary_pos_emb (~18 ops → 2 launches)
            q = fused_qknorm_rope(q, q_norm_w, cos, sin, eps)
            k = fused_qknorm_rope(k, k_norm_w, cos, sin, eps)
        else:
            # Prefill fallback: use separate ops
            if q_norm_w is not None:
                q = triton_rmsnorm(q, q_norm_w, eps, add_one_to_weight=True)
            if k_norm_w is not None:
                k = triton_rmsnorm(k, k_norm_w, eps, add_one_to_weight=True)
            if cos is not None:
                try:
                    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
                except ImportError:
                    try:
                        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
                    except ImportError:
                        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache
        if cache_position is not None:
            cache.update_kv(layer_idx, k, v, cache_position)
            k_full, v_full = cache.get_kv_for_attention(layer_idx)
        elif past_key_values is not None:
            cache_kwargs = {}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            k_full, v_full = past_key_values.update(k, v, layer_idx, cache_kwargs)
        else:
            k_full, v_full = k, v

        # SDPA (native GQA)
        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :k_full.shape[-2]]

        attn_output = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=causal_mask,
            dropout_p=0.0,
            enable_gqa=(NUM_KV_HEADS != NUM_Q_HEADS),
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, Q_OUT_DIM)

        # ===== FUSED SIGMOID GATE MULTIPLY =====
        # Replaces: attn_output * sigmoid(q_gate.float()).to(bf16)  - 4 ops → 1 kernel
        attn_output = fused_sigmoid_mul(attn_output, q_gate)

        # Output projection
        o_2d = attn_output.reshape(-1, Q_OUT_DIM)
        attn_output = F.linear(o_2d, w_o).reshape(bsz, q_len, hidden_size)

        # ===== FUSED RESIDUAL + RMSNORM =====
        hidden_states, x_normed = fused_residual_rmsnorm(
            residual, attn_output, post_ln_w, eps, add_one_to_weight=True,
        )

        # ============================================================
        # MLP block
        # ============================================================
        x_2d = x_normed.reshape(-1, hidden_size)

        # ===== SINGLE GEMV for gate+up =====
        gate_up = F.linear(x_2d, w_mlp_gate_up)  # [M, 34816]
        gate_out = gate_up[:, :INTERMEDIATE_SIZE]
        up_out = gate_up[:, INTERMEDIATE_SIZE:]

        mlp_mid = fused_silu_mul(gate_out, up_out)

        down_out = F.linear(
            mlp_mid.reshape(-1, intermediate_size), w_mlp_down,
        ).reshape(bsz, q_len, hidden_size)

        # Inter-layer chaining: fuse residual-add + next layer's input norm
        if next_layer_norm_w is not None and q_len == 1:
            hidden_states, x_normed_next = fused_residual_rmsnorm(
                hidden_states, down_out, next_layer_norm_w, eps, add_one_to_weight=True,
            )
            return hidden_states, x_normed_next
        else:
            hidden_states = hidden_states + down_out
            return hidden_states

    return fused_forward


def _patch_final_norm(model: nn.Module, eps: float):
    """Patch final RMSNorm to use Triton kernel."""
    final_norm = model.model.norm
    final_norm_w = final_norm.weight

    def fused_norm_forward(hidden_states):
        return triton_rmsnorm(hidden_states, final_norm_w, eps, add_one_to_weight=True)

    final_norm.forward = fused_norm_forward


def verify_patch(model: nn.Module) -> bool:
    """Verify that all layers have been patched."""
    num_layers = model.config.num_hidden_layers
    patched = 0
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        fwd = layer.forward
        if callable(fwd) and hasattr(fwd, '__closure__') and fwd.__closure__ is not None:
            patched += 1

    print(f"Patch verification: {patched}/{num_layers} layers patched")
    return patched == num_layers
