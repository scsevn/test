import subprocess
from pathlib import Path
import modal

APP_NAME = "qwen35-triton-b200"
PROJECT_ROOT = Path(__file__).resolve().parent
AI_INFRA_ROOT = PROJECT_ROOT.parent
FLA_ROOT = AI_INFRA_ROOT / "flash-linear-attention"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path="/root/qwen3.5-triton",
        copy=True,
    )
    .add_local_dir(
        str(FLA_ROOT),
        remote_path="/root/flash-linear-attention",
        copy=True,
    )
    .env({
        "PYTHONPATH": "/root/qwen3.5-triton:/root/flash-linear-attention"
    })
    .workdir("/root/qwen3.5-triton")
    .run_commands(
        "python -V",
        "pip install -U pip setuptools wheel",
        "pip install -r requirements.txt",
        "pip install einops",
        "pip install -e /root/flash-linear-attention",
    )
)

def _run_cmd(cmd: list[str]):
    print(f"\n[Modal] Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60,
)
def run_validate():
    _run_cmd(["python", "benchmarks/v4_validate.py"])


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60,
)
def run_baseline():
    _run_cmd(["python", "benchmarks/baseline.py"])


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60,
)
def run_optimized():
    _run_cmd(["python", "benchmarks/optimized.py"])


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60,
)
def run_recurrent():
    _run_cmd(["python", "run_recurrent_decode.py"])


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60,
)
def run_compare_fla():
    _run_cmd(["python", "benchmark_compare_fla_recurrent.py"])


@app.local_entrypoint()
def main(which: str = "recurrent"):
    if which == "validate":
        run_validate.remote()
    elif which == "baseline":
        run_baseline.remote()
    elif which == "optimized":
        run_optimized.remote()
    elif which == "recurrent":
        run_recurrent.remote()
    elif which == "compare_fla":
        run_compare_fla.remote()
    else:
        raise ValueError(
            f"Unknown option: {which}. Choose from validate / baseline / optimized / recurrent / compare_fla"
        )