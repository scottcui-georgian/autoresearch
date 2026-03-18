"""
Modal backend for the NanoGPT task.

This file is task-owned. The shared `autoresearch` package does not know anything
about Modal or this task's runtime layout.
"""

from __future__ import annotations

import codecs
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

TASK_ROOT = Path(__file__).resolve().parent.parent
REMOTE_TASK_DIR = "/root/task"
VOLUME_ROOT = "/cache-home"

APP_NAME = "autoresearch-nanogpt"
RUNTIME_PROJECT_DIR = TASK_ROOT / ".runner" / "modal"
RUNNER_FILES = ["workspace/train.py", "workspace/prepare.py"]
ENTRYPOINTS = {
    "train": {"file": "workspace/train.py", "gpu": "L40S", "timeout": 1800},
    "prepare": {"file": "workspace/prepare.py", "cpu": 8, "timeout": 3600},
}

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(f"{APP_NAME}-cache", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").uv_sync(
    uv_project_dir=str(RUNTIME_PROJECT_DIR), gpu="L40S"
)
for relative_path in RUNNER_FILES:
    image = image.add_local_file(
        TASK_ROOT / relative_path,
        remote_path=f"{REMOTE_TASK_DIR}/{relative_path}",
    )


def _run_python(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a task script inside the Modal container while streaming and collecting output."""
    env = os.environ.copy()
    env["HOME"] = VOLUME_ROOT
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        ["python", *args],
        cwd=REMOTE_TASK_DIR,
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
    outputs: dict[str, list[str]] = {"stdout": [], "stderr": []}
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


def _tail(text: str, max_lines: int = 50) -> str:
    """Keep only the tail of large logs so crash output stays readable."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _extra_args_from_env() -> list[str]:
    """Read forwarded action args from the local runner wrapper."""
    raw = os.environ.get("AUTORESEARCH_MODAL_ACTION_ARGS", "[]")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid AUTORESEARCH_MODAL_ACTION_ARGS payload.") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError("AUTORESEARCH_MODAL_ACTION_ARGS must be a JSON array of strings.")
    return value


def _quiet_mode_from_env() -> bool:
    """Read whether Modal progress output should be suppressed locally."""
    return os.environ.get("AUTORESEARCH_MODAL_QUIET", "1") != "0"


def _validate_gpu() -> dict[str, Any]:
    """Fail fast unless Modal exposed exactly one CUDA GPU."""
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
    timeout=3600,
    volumes={VOLUME_ROOT: cache_volume},
)
def cpu_remote(entrypoint_file: str, extra_args: list[str] | None = None) -> dict[str, Any]:
    """Run a CPU-only entrypoint on Modal."""
    cache_volume.reload()
    proc = _run_python([entrypoint_file, *(extra_args or [])])
    cache_volume.commit()
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
    volumes={VOLUME_ROOT: cache_volume},
)
def gpu_remote(entrypoint_file: str, extra_args: list[str] | None = None) -> dict[str, Any]:
    """Run a GPU entrypoint on Modal."""
    cache_volume.reload()
    gpu_info = _validate_gpu()
    proc = _run_python([entrypoint_file, *(extra_args or [])])
    cache_volume.commit()
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "gpu": gpu_info,
    }


def _write_result(result: dict[str, Any]) -> None:
    """Write stdout/stderr from a remote result to local streams."""
    if result["stdout"]:
        sys.stdout.write(result["stdout"])
        if not result["stdout"].endswith("\n"):
            sys.stdout.write("\n")
    if result["stderr"]:
        sys.stderr.write(result["stderr"])
        if not result["stderr"].endswith("\n"):
            sys.stderr.write("\n")


@app.local_entrypoint()
def main(action: str) -> None:
    """Dispatch to the correct remote function based on the action name."""
    if action not in ENTRYPOINTS:
        valid = ", ".join(sorted(ENTRYPOINTS))
        raise SystemExit(f"Unknown action '{action}'. Valid actions: {valid}")

    spec = ENTRYPOINTS[action]
    extra_args = _extra_args_from_env()
    quiet_mode = _quiet_mode_from_env()

    with modal.enable_output() as output_manager:
        output_manager.set_quiet_mode(quiet_mode)
        if "gpu" in spec:
            result = gpu_remote.remote(spec["file"], extra_args)
        else:
            result = cpu_remote.remote(spec["file"], extra_args)

    _write_result(result)

    if result["returncode"] != 0:
        tail = _tail(
            result["stdout"]
            + ("\n" if result["stdout"] and result["stderr"] else "")
            + result["stderr"]
        )
        if tail:
            sys.stderr.write("\n--- remote tail ---\n")
            sys.stderr.write(tail)
            if not tail.endswith("\n"):
                sys.stderr.write("\n")
        raise SystemExit(result["returncode"])
