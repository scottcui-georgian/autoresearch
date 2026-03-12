"""
Minimal Modal integration for autoresearch.

`autoresearch_runtime/prepare.py` runs once on a CPU container to populate the shared cache volume.
`autoresearch_runtime/train.py` runs on a single L40S GPU and returns the raw logs plus parsed metrics.
"""

from __future__ import annotations

import codecs
import os
import selectors
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "autoresearch"
LOCAL_REPO_DIR = Path(__file__).resolve().parent
LOCAL_RUNTIME_DIR = LOCAL_REPO_DIR / "autoresearch_runtime"
REMOTE_RUNTIME_DIR = "/root/autoresearch_runtime"
VOLUME_ROOT = "/cache-home"
MODAL_TIMEOUT_PREPARE_SECONDS = 60 * 60
MODAL_TIMEOUT_TRAIN_SECONDS = 30 * 60
SUMMARY_FIELDS = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(f"{APP_NAME}-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(LOCAL_RUNTIME_DIR), gpu="L40S")
    .add_local_file(LOCAL_RUNTIME_DIR / "prepare.py", remote_path=f"{REMOTE_RUNTIME_DIR}/prepare.py")
    .add_local_file(LOCAL_RUNTIME_DIR / "train.py", remote_path=f"{REMOTE_RUNTIME_DIR}/train.py")
)


def _run_python(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a repo script inside the Modal container while streaming and collecting its output."""
    env = os.environ.copy()
    env["HOME"] = VOLUME_ROOT
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        ["python", *args],
        cwd=REMOTE_RUNTIME_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    decoders = {
        "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
    }
    outputs = {"stdout": [], "stderr": []}
    writers = {"stdout": sys.stdout, "stderr": sys.stderr}

    while selector.get_map():
        for key, _ in selector.select():
            chunk = os.read(key.fileobj.fileno(), 4096)
            stream_name = key.data
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            text = decoders[stream_name].decode(chunk)
            if text:
                outputs[stream_name].append(text)
                writers[stream_name].write(text)
                writers[stream_name].flush()

    for stream_name, decoder in decoders.items():
        tail = decoder.decode(b"", final=True)
        if tail:
            outputs[stream_name].append(tail)
            writers[stream_name].write(tail)
            writers[stream_name].flush()

    return subprocess.CompletedProcess(
        args=["python", *args],
        returncode=proc.wait(),
        stdout="".join(outputs["stdout"]),
        stderr="".join(outputs["stderr"]),
    )


def _parse_training_summary(stdout: str) -> dict[str, str]:
    """Extract the final summary block so local tooling can parse run results reliably."""
    summary: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in SUMMARY_FIELDS:
            summary[key] = value.strip()
    return summary


def _tail(text: str, max_lines: int = 50) -> str:
    """Keep only the tail of large logs so crash output stays readable."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _validate_gpu() -> dict[str, Any]:
    """Fail fast unless Modal exposed exactly one CUDA GPU to the training job."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal GPU container.")
    device_count = torch.cuda.device_count()
    if device_count != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {device_count}.")
    return {
        "device_name": torch.cuda.get_device_name(0),
        "device_count": device_count,
        "capability": torch.cuda.get_device_capability(0),
    }


@app.function(
    image=image,
    cpu=8,
    timeout=MODAL_TIMEOUT_PREPARE_SECONDS,
    volumes={VOLUME_ROOT: cache_volume},
)
def prepare_remote(num_shards: int = 10, download_workers: int = 8) -> dict[str, Any]:
    """Populate the shared Modal cache volume with data shards and tokenizer artifacts."""
    cache_volume.reload()
    proc = _run_python(
        [
            "prepare.py",
            "--num-shards",
            str(num_shards),
            "--download-workers",
            str(download_workers),
        ]
    )
    cache_volume.commit()
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=MODAL_TIMEOUT_TRAIN_SECONDS,
    volumes={VOLUME_ROOT: cache_volume},
)
def train_remote() -> dict[str, Any]:
    """Run one training experiment on Modal and return logs plus parsed summary fields."""
    cache_volume.reload()
    gpu_info = _validate_gpu()
    proc = _run_python(["train.py"])
    cache_volume.commit()
    summary = _parse_training_summary(proc.stdout)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "summary": summary,
        "gpu": gpu_info,
    }


@app.local_entrypoint()
def main(action: str, num_shards: int = 10, download_workers: int = 8) -> None:
    """Expose a tiny local CLI so callers do not need to know Modal's Python API."""
    if action == "prepare":
        with modal.enable_output() as output_manager:
            output_manager.set_quiet_mode(True)
            result = prepare_remote.remote(
                num_shards=num_shards,
                download_workers=download_workers,
            )
        if result["stdout"]:
            sys.stdout.write(result["stdout"])
            if not result["stdout"].endswith("\n"):
                sys.stdout.write("\n")
        if result["stderr"]:
            sys.stderr.write(result["stderr"])
            if not result["stderr"].endswith("\n"):
                sys.stderr.write("\n")
        if result["returncode"] != 0:
            raise SystemExit(result["returncode"])
        return

    if action == "train":
        with modal.enable_output() as output_manager:
            output_manager.set_quiet_mode(True)
            result = train_remote.remote()
        summary = result["summary"]
        if result["returncode"] != 0 or "val_bpb" not in summary:
            tail = _tail(result["stdout"] + ("\n" if result["stdout"] and result["stderr"] else "") + result["stderr"])
            if tail:
                sys.stderr.write("\n--- remote tail ---\n")
                sys.stderr.write(tail)
                if not tail.endswith("\n"):
                    sys.stderr.write("\n")
            raise SystemExit(result["returncode"] or 1)
        return

    raise SystemExit(f"Unsupported action: {action}")
