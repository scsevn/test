"""
Qwen3.5-27B  - Shared Benchmark Configuration

App, container images, volumes, GPU configs for benchmarking
Triton kernel optimization on Qwen3.5-27B Dense (BF16) on B200.

GPU: NVIDIA B200 (Blackwell)  - 208 SMs, 8 TB/s BW, 192GB HBM3e, sm_100
"""
import modal
from pathlib import Path

# Modal app
app = modal.App("qwen35-27b-triton")

# Persistent volumes
model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("benchmark-results", create_if_missing=True)

# HuggingFace secret for model access
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Volume mount paths
MODEL_CACHE_PATH = "/cache/models"
RESULTS_PATH = "/results"

# Project root for mounting source into containers
_project_root = Path(__file__).parent
_ignore_patterns = ["__pycache__", ".git", "*.pyc", "results"]

# Triton + CUDA dev image for B200 (Blackwell, sm_100)
forge_llm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch>=2.6.0",
        "triton>=3.1.0",
        "transformers>=4.57.0",
        "accelerate>=0.25.0",
        "huggingface_hub>=0.20.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
    )
    .add_local_dir(str(_project_root), remote_path="/root", ignore=_ignore_patterns)
)

# GPU configurations
GPU_CONFIG = {
    "baseline": {"gpu": "B200", "timeout": 1800},
    "optimized": {"gpu": "B200", "timeout": 1800},
}

# Model constants
MODEL_ID = "Qwen/Qwen3.5-27B"
HIDDEN_SIZE = 5120
NUM_LAYERS = 64
NUM_ATTENTION_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
INTERMEDIATE_SIZE = 17408
VOCAB_SIZE = 248320
LINEAR_NUM_KEY_HEADS = 16
LINEAR_NUM_VALUE_HEADS = 48
LINEAR_KEY_HEAD_DIM = 128
LINEAR_VALUE_HEAD_DIM = 128
KEY_DIM = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM       # 2048
VALUE_DIM = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM # 6144
CONV_DIM = KEY_DIM + KEY_DIM + VALUE_DIM                   # 10240
DV_PER_K_HEAD = (LINEAR_NUM_VALUE_HEADS // LINEAR_NUM_KEY_HEADS) * LINEAR_VALUE_HEAD_DIM  # 384
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10_000_000.0
PARTIAL_ROTARY_FACTOR = 0.25  # 64 of 256 dims
