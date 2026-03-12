"""
Inspect Qwen3.5-27B model architecture  - module tree + parameter shapes.

Loads the model and prints:
  1. Key config attributes
  2. Module tree for layer 0 (DeltaNet/linear_attn) and layer 3 (attention)
  3. Parameter names and shapes for layers 0 and 3
  4. Detailed inspection of linear_attn submodules

Run via Modal:
    modal run benchmarks/inspect_model.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import modal
from modal_config import (
    app, forge_llm_image, model_cache, hf_secret,
    MODEL_CACHE_PATH, MODEL_ID,
)


@app.function(
    image=forge_llm_image,
    gpu="B200",
    timeout=1200,
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[hf_secret],
)
def inspect_model():
    """Inspect Qwen3.5-27B architecture in detail."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    hf_token = os.environ.get("HF_TOKEN")

    # ── 1. Config inspection ──────────────────────────────────────────
    print("=" * 70)
    print("SECTION 1: MODEL CONFIG")
    print("=" * 70)
    config = AutoConfig.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_PATH, token=hf_token, trust_remote_code=True,
    )
    # Print all config attributes
    for k, v in sorted(vars(config).items()):
        if not k.startswith("_"):
            print(f"  {k}: {v}")

    # ── 2. Load model ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LOADING MODEL...")
    print("=" * 70)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=MODEL_CACHE_PATH,
        token=hf_token,
        trust_remote_code=True,
    )
    model.requires_grad_(False)
    print(f"Model class: {type(model).__name__}")
    print(f"Model module: {type(model).__module__}")

    # ── 3. Module tree for layers 0 and 3 ─────────────────────────────
    for layer_idx in [0, 3]:
        print("\n" + "=" * 70)
        print(f"SECTION 2: MODULE TREE  - layer {layer_idx}")
        print("=" * 70)

        layer = model.model.layers[layer_idx]
        print(f"Layer class: {type(layer).__name__} ({type(layer).__module__})")

        def print_tree(module, prefix="", name=""):
            line = f"{prefix}{name}: {type(module).__name__}"
            # Show extra info for leaf modules
            child_list = list(module.named_children())
            if not child_list:
                # It's a leaf  - show shape info if it has weight
                extras = []
                if hasattr(module, "weight"):
                    extras.append(f"weight={tuple(module.weight.shape)}")
                if hasattr(module, "bias") and module.bias is not None:
                    extras.append(f"bias={tuple(module.bias.shape)}")
                if extras:
                    line += f"  [{', '.join(extras)}]"
            print(line)
            for i, (child_name, child_mod) in enumerate(child_list):
                is_last = (i == len(child_list) - 1)
                connector = "└─ " if is_last else "├─ "
                child_prefix = prefix + ("   " if is_last else "│  ")
                print_tree(child_mod, prefix=prefix + connector, name=child_name)

        print_tree(layer, name=f"layers.{layer_idx}")

    # ── 4. Parameter names + shapes for layers 0 and 3 ────────────────
    for layer_idx in [0, 3]:
        print("\n" + "=" * 70)
        print(f"SECTION 3: PARAMETERS  - layer {layer_idx}")
        print("=" * 70)

        layer = model.model.layers[layer_idx]
        for name, param in layer.named_parameters():
            print(f"  {name:60s}  {str(tuple(param.shape)):30s}  {param.dtype}")

        # Also show buffers (some may be registered as buffers, not params)
        print(f"\n  --- Buffers for layer {layer_idx} ---")
        for name, buf in layer.named_buffers():
            print(f"  {name:60s}  {str(tuple(buf.shape)):30s}  {buf.dtype}")

    # ── 5. Detailed linear_attn inspection (layer 0) ──────────────────
    print("\n" + "=" * 70)
    print("SECTION 4: DETAILED linear_attn INSPECTION (layer 0)")
    print("=" * 70)

    layer0 = model.model.layers[0]

    # Check what attention-related submodules exist
    for attr_name in ["linear_attn", "self_attn", "attn"]:
        if hasattr(layer0, attr_name):
            attn_mod = getattr(layer0, attr_name)
            print(f"\n  layer0.{attr_name} exists: {type(attn_mod).__name__}")
            print(f"    module path: {type(attn_mod).__module__}")

            # Print all attributes (not just children)
            print(f"\n    --- All attributes of {attr_name} ---")
            for k in sorted(dir(attn_mod)):
                if k.startswith("_"):
                    continue
                try:
                    v = getattr(attn_mod, k)
                    if callable(v) and not isinstance(v, torch.nn.Module):
                        continue
                    if isinstance(v, torch.nn.Module):
                        print(f"      {k}: {type(v).__name__}")
                    elif isinstance(v, (torch.Tensor, torch.nn.Parameter)):
                        print(f"      {k}: Tensor {tuple(v.shape)} {v.dtype}")
                    elif isinstance(v, (int, float, bool, str, type(None))):
                        print(f"      {k}: {v}")
                except Exception:
                    pass

    # Specifically inspect the submodules we care about
    target_attrs = [
        "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
        "A_log", "dt_bias", "conv1d", "norm", "out_proj",
    ]
    attn_mod = getattr(layer0, "linear_attn", None) or getattr(layer0, "self_attn", None)
    if attn_mod is not None:
        print(f"\n  --- Targeted submodule inspection on {type(attn_mod).__name__} ---")
        for attr in target_attrs:
            if hasattr(attn_mod, attr):
                v = getattr(attn_mod, attr)
                if isinstance(v, torch.nn.Module):
                    print(f"    {attr}: {type(v).__name__}")
                    for pn, pp in v.named_parameters():
                        print(f"      .{pn}: {tuple(pp.shape)} {pp.dtype}")
                    for bn, bb in v.named_buffers():
                        print(f"      .{bn} (buffer): {tuple(bb.shape)} {bb.dtype}")
                elif isinstance(v, (torch.Tensor, torch.nn.Parameter)):
                    print(f"    {attr}: Tensor {tuple(v.shape)} {v.dtype}")
                else:
                    print(f"    {attr}: {type(v).__name__} = {v}")
            else:
                print(f"    {attr}: NOT FOUND")

    # ── 6. Layer type pattern across all layers ───────────────────────
    print("\n" + "=" * 70)
    print("SECTION 5: LAYER TYPE PATTERN (all 64 layers)")
    print("=" * 70)
    for i, layer in enumerate(model.model.layers):
        has_linear = hasattr(layer, "linear_attn")
        has_self = hasattr(layer, "self_attn")
        linear_type = type(getattr(layer, "linear_attn", None)).__name__ if has_linear else "---"
        self_type = type(getattr(layer, "self_attn", None)).__name__ if has_self else "---"
        # Which one has actual parameters?
        linear_params = sum(1 for _ in getattr(layer, "linear_attn", torch.nn.Module()).parameters()) if has_linear else 0
        self_params = sum(1 for _ in getattr(layer, "self_attn", torch.nn.Module()).parameters()) if has_self else 0
        print(f"  layer {i:2d}: linear_attn={linear_type}({linear_params} params)  "
              f"self_attn={self_type}({self_params} params)")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


@app.local_entrypoint()
def main():
    """Run model inspection."""
    print("Inspecting Qwen3.5-27B architecture on B200...")
    inspect_model.remote()
    print("\nDone.")
