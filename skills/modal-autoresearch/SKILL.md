---
name: modal-autoresearch
description: Use when working on this repo's Modal execution path, including `modal_app.py`, `modal_runner.py`, `autoresearch_runtime/` execution, remote data prep/training runs, Modal volumes, GPU/image setup, or debugging why local-agent plus remote-GPU experiments are failing.
---

# Modal Autoresearch

## Overview

This repo keeps the research agent local and runs expensive work on Modal. The stable contract is:

- `autoresearch_runtime/prepare.py` runs remotely on Modal CPU once to populate a persistent cache volume.
- `autoresearch_runtime/train.py` runs remotely on a single Modal GPU.
- the agent stays inside `autoresearch_runtime/`, edits `train.py`, and uses `python modal_runner.py train`.

## Use This Skill When

- changing [`modal_app.py`](../../modal_app.py) or [`modal_runner.py`](../../modal_runner.py)
- debugging Modal image, volume, GPU, or log-streaming issues
- changing the remote GPU type or Python version
- explaining how local paths map to remote container paths
- adjusting the agent instructions for Modal-backed runs

## Ground Truth

- Local repo root is the source of truth for Modal infra and git operations.
- Local runtime project root is `autoresearch_runtime/`.
- Remote runtime code lives at `/root/autoresearch_runtime`.
- Persistent state is mounted at `/cache-home`, and subprocesses run with `HOME=/cache-home`.
- Because `autoresearch_runtime/prepare.py` and `autoresearch_runtime/train.py` use `os.path.expanduser("~")`, their cache resolves to `/cache-home/.cache/autoresearch` remotely without patching those files.
- The image currently uses `modal.Image.debian_slim(...)` plus `uv_sync(...)`. This worked for `torch==2.9.1+cu128` and the current `kernels` stack on Modal.

## Commands

One-time remote prep:

```bash
python modal_runner.py prepare
```

Single training run:

```bash
python modal_runner.py train
```

Agent-loop form inside `autoresearch_runtime/`:

```bash
python modal_runner.py train > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## Volume Rules

- Mounting means a Modal Volume appears inside the container filesystem at a chosen path.
- `cache_volume.commit()` persists writes made by the current container.
- `cache_volume.reload()` refreshes the current container's view so it can see commits from other containers.
- `python modal_runner.py prepare --num-shards N` is additive. Existing shards stay, missing shards are added, and the tokenizer is reused if already present.
- Re-running prepare does not reset the volume.

## Image And GPU Rules

- `debian_slim` does not include a full CUDA toolkit by default, but Modal GPU workers provide NVIDIA drivers and the CUDA Driver API.
- Many libraries still work because they bundle CUDA user-space dependencies; this repo's current stack does.
- If an image setup step needs GPU-aware compilation or detection, attach a GPU to that setup step instead of assuming the whole image build runs on a GPU.
- The current image uses Python 3.12 and runs `uv_sync(..., gpu="L40S")` to keep build-time behavior closer to runtime.
- The training function must stay hard-pinned to one GPU. Do not expose GPU count as a user parameter.

## Current Guardrails

- Remote training is pinned to a single `L40S`.
- Runtime validation asserts CUDA is available and exactly one GPU is visible.
- `autoresearch_runtime/train.py` computes MFU with an `L40S` reference instead of the previous H100-only constant.
- The validated baseline uses gradient accumulation steps of 4 and completes successfully on `L40S`.

## Logging Behavior

- `autoresearch_runtime/train.py` emits per-step progress while running.
- The Modal wrapper streams subprocess stdout and stderr live instead of buffering until completion.
- The final summary block remains unchanged so the research agent can parse `val_bpb`, `peak_vram_mb`, and the rest from `run.log`.

## Known Failure Modes

- `Expected /root/autoresearch_runtime/pyproject.toml to exist`
  Cause: `uv_sync(uv_project_dir=...)` was pointed at a remote path instead of the local runtime project path.
  Fix: pass the local runtime directory to `uv_sync`, then add runtime source files separately.

- `cannot mount volume on non-empty path: "/root/.cache"`
  Cause: the volume was mounted onto a path that already existed in the base image.
  Fix: mount onto an empty path such as `/cache-home` and set `HOME=/cache-home`.

- no live training logs
  Cause: subprocess output was captured and replayed only after the remote function returned.
  Fix: stream subprocess stdout and stderr incrementally.

- OOM on `L40S`
  Cause: upstream defaults were tuned for larger GPUs.
  Fix: lower `DEVICE_BATCH_SIZE` first and preserve `TOTAL_BATCH_SIZE` by increasing gradient accumulation.

## Editing Guidance

- Prefer changing Modal infra only when the execution substrate changes; keep research changes inside `autoresearch_runtime/train.py`.
- If Modal behavior changes, verify against current Modal docs before patching the wrapper.
- When debugging, separate environment failures from capacity failures:
  - environment failure: imports, CUDA visibility, kernels loading, build/setup issues
  - capacity failure: OOM, slow compile, too-large microbatch
