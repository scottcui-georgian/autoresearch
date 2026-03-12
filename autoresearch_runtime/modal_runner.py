"""
Agent-facing wrapper that preserves `python modal_runner.py ...` inside autoresearch_runtime/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent
ROOT_MODAL_RUNNER = RUNTIME_DIR.parent / "modal_runner.py"


def main() -> int:
    """Delegate runtime-visible commands to the repo-root Modal runner."""
    cmd = [sys.executable, str(ROOT_MODAL_RUNNER), *sys.argv[1:]]
    proc = subprocess.run(cmd, cwd=RUNTIME_DIR.parent, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
