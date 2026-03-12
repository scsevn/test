"""
Cache management for Qwen3.5-27B hybrid DeltaNet + GQA model.

Two cache types:
  1. DeltaNet cache (48 layers): conv_state + recurrent_state
  2. KV cache (16 layers): standard static KV cache for full attention

DeltaNet cache per layer:
  - conv_state: [B, conv_dim, kernel_size]  - rolling buffer for causal conv1d
  - recurrent_state: [B, num_v_heads, Dk, Dv]  - linear attention state
    After Q/K expansion via repeat_interleave, state is per-v-head:
    [B, 48, 128, 128] = 48 heads x 128 x 128 = 1.5MB per layer (BF16)

Total DeltaNet state: 48 layers x 1.5MB = ~72MB (fits in B200's 96MB L2)

KV cache: uses pre-allocated static tensors for the 16 full attention layers.
"""
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeltaNetLayerCache:
    """Cache for a single DeltaNet layer."""
    conv_state: torch.Tensor     # [B, conv_dim, kernel_size]
    recurrent_state: torch.Tensor  # [B, num_heads, Dk, Dv]


class DeltaNetCache:
    """Cache for all 48 DeltaNet layers."""

    def __init__(
        self,
        batch_size: int,
        num_deltanet_layers: int,
        conv_dim: int,
        kernel_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.batch_size = batch_size
        self.num_layers = num_deltanet_layers
        self.device = device
        self.dtype = dtype
        self.layers = []

        for _ in range(num_deltanet_layers):
            conv_state = torch.zeros(
                batch_size, conv_dim, kernel_size,
                device=device, dtype=dtype,
            )
            # Recurrent state: [B, num_v_heads, Dk, Dv]
            # After Q/K repeat_interleave, the recurrent function operates
            # on num_v_heads independent heads, each with [Dk, Dv] state.
            # For Qwen3.5-27B: [B, 48, 128, 128]
            recurrent_state = torch.zeros(
                batch_size, num_value_heads, key_head_dim, value_head_dim,
                device=device, dtype=dtype,
            )
            self.layers.append(DeltaNetLayerCache(
                conv_state=conv_state,
                recurrent_state=recurrent_state,
            ))

    def get_layer(self, deltanet_layer_idx: int) -> DeltaNetLayerCache:
        """Get cache for a specific DeltaNet layer (0-indexed within DeltaNet layers only)."""
        return self.layers[deltanet_layer_idx]

    def reset(self):
        """Zero out all caches (for new sequence)."""
        for layer in self.layers:
            layer.conv_state.zero_()
            layer.recurrent_state.zero_()

    def memory_bytes(self) -> int:
        """Total memory usage in bytes."""
        total = 0
        for layer in self.layers:
            total += layer.conv_state.nelement() * layer.conv_state.element_size()
            total += layer.recurrent_state.nelement() * layer.recurrent_state.element_size()
        return total


class HybridCache:
    """Combined cache for the full Qwen3.5-27B hybrid model.

    Maps global layer indices to the correct cache type:
      - Layers 0,1,2 → DeltaNet cache index 0,1,2
      - Layer 3 → KV cache index 0
      - Layers 4,5,6 → DeltaNet cache index 3,4,5
      - Layer 7 → KV cache index 1
      - ...pattern repeats

    Layer pattern: [DeltaNet, DeltaNet, DeltaNet, Attention] x 16 blocks
    """

    def __init__(
        self,
        batch_size: int = 1,
        max_cache_len: int = 4096,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        # DeltaNet config
        num_deltanet_layers: int = 48,
        conv_dim: int = 10240,      # Q(2048)+K(2048)+V(6144) channels for conv1d
        kernel_size: int = 4,
        num_key_heads: int = 16,
        num_value_heads: int = 48,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        # Attention config
        num_attention_layers: int = 16,
        num_q_heads: int = 24,
        num_kv_heads: int = 4,
        head_dim: int = 256,
    ):
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # DeltaNet cache
        self.deltanet = DeltaNetCache(
            batch_size=batch_size,
            num_deltanet_layers=num_deltanet_layers,
            conv_dim=conv_dim,
            kernel_size=kernel_size,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            device=device,
            dtype=dtype,
        )

        # KV cache for full attention layers
        # Pre-allocate static KV cache tensors
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.kv_cache_k = torch.zeros(
            num_attention_layers, batch_size, num_kv_heads, max_cache_len, head_dim,
            device=device, dtype=dtype,
        )
        self.kv_cache_v = torch.zeros(
            num_attention_layers, batch_size, num_kv_heads, max_cache_len, head_dim,
            device=device, dtype=dtype,
        )
        self.kv_seq_len = 0

        # Build layer index mapping
        # Layer pattern: [D, D, D, A] x 16
        self._layer_type = []  # 'deltanet' or 'attention'
        self._deltanet_idx = []  # maps global layer to deltanet cache index
        self._attention_idx = []  # maps global layer to attention cache index
        d_idx = 0
        a_idx = 0
        for block in range(16):
            for pos in range(4):
                if pos < 3:
                    self._layer_type.append('deltanet')
                    self._deltanet_idx.append(d_idx)
                    self._attention_idx.append(-1)
                    d_idx += 1
                else:
                    self._layer_type.append('attention')
                    self._deltanet_idx.append(-1)
                    self._attention_idx.append(a_idx)
                    a_idx += 1

    def is_deltanet(self, layer_idx: int) -> bool:
        return self._layer_type[layer_idx] == 'deltanet'

    def get_deltanet_cache(self, layer_idx: int) -> DeltaNetLayerCache:
        """Get DeltaNet cache for a global layer index."""
        d_idx = self._deltanet_idx[layer_idx]
        return self.deltanet.get_layer(d_idx)

    def get_kv_cache(self, layer_idx: int) -> tuple:
        """Get KV cache tensors for an attention layer.

        Returns (k_cache, v_cache) sliced to current seq_len.
        """
        a_idx = self._attention_idx[layer_idx]
        return self.kv_cache_k[a_idx], self.kv_cache_v[a_idx]

    def update_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, cache_position: torch.Tensor):
        """Update KV cache at position(s).

        CUDA graph compatible  - uses index_copy_, no CPU-GPU sync.

        Args:
            k: [B, num_kv_heads, seq_len, head_dim]
            v: [B, num_kv_heads, seq_len, head_dim]
            cache_position: [seq_len] tensor of position indices on device
        """
        a_idx = self._attention_idx[layer_idx]
        # index_copy_ along dim=2 (sequence dimension)
        # Works with both single-token decode and multi-token prefill
        self.kv_cache_k[a_idx].index_copy_(2, cache_position, k)
        self.kv_cache_v[a_idx].index_copy_(2, cache_position, v)

    def get_kv_for_attention(self, layer_idx: int) -> tuple:
        """Get K, V for attention computation.

        Returns full pre-allocated cache (CUDA graph compatible).
        Zero-padded positions have negligible impact on attention output.

        Returns:
            k: [B, num_kv_heads, max_cache_len, head_dim]
            v: [B, num_kv_heads, max_cache_len, head_dim]
        """
        a_idx = self._attention_idx[layer_idx]
        return self.kv_cache_k[a_idx], self.kv_cache_v[a_idx]

    def reset(self):
        """Reset all caches for new sequence."""
        self.deltanet.reset()
        self.kv_cache_k.zero_()
        self.kv_cache_v.zero_()
        self.kv_seq_len = 0

    def memory_bytes(self) -> int:
        """Total memory usage in bytes."""
        dn_bytes = self.deltanet.memory_bytes()
        kv_bytes = (
            self.kv_cache_k.nelement() * self.kv_cache_k.element_size() +
            self.kv_cache_v.nelement() * self.kv_cache_v.element_size()
        )
        return dn_bytes + kv_bytes

    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 * 1024)
