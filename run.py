#!/usr/bin/env python3
"""
Task-level entrypoint for NanoGPT experiments.

Usage:
    python run.py train > run.log 2>&1
    python run.py prepare --num-shards 10
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
MODAL_RUNNER = TASK_ROOT / ".runner" / "modal_runner.py"


def _require_cmd(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"`{name}` is not on PATH.")
    return path


def main() -> int:
    python_path = sys.executable or _require_cmd("python3")
    cmd = [python_path, str(MODAL_RUNNER), *sys.argv[1:]]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
