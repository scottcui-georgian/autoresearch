"""Task-owned local wrapper for running NanoGPT actions on Modal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent.parent
MODAL_APP = TASK_ROOT / ".runner" / "modal_app.py"


def _require_cmd(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"`{name}` is not on PATH.")
    return path


def run_modal(action: str, extra_args: list[str] | None = None, *, quiet: bool = True) -> int:
    """Invoke the task-owned Modal app with cwd set to the task root."""
    modal_path = _require_cmd("modal")
    cmd = [modal_path, "run"]
    if quiet:
        cmd.append("-q")
    cmd.extend([str(MODAL_APP), "--action", action])
    env = os.environ.copy()
    env["AUTORESEARCH_MODAL_ACTION_ARGS"] = json.dumps(extra_args or [])
    env["AUTORESEARCH_MODAL_QUIET"] = "1" if quiet else "0"
    proc = subprocess.run(cmd, cwd=TASK_ROOT, env=env, check=False)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nanogpt experiments on Modal.")
    parser.add_argument("action", choices=("train", "prepare"))
    parser.add_argument("action_args", nargs=argparse.REMAINDER)
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reduce Modal progress noise when supported",
    )
    args = parser.parse_args()
    quiet = args.quiet if args.quiet is not None else (args.action != "train")
    return run_modal(args.action, args.action_args, quiet=quiet)


if __name__ == "__main__":
    sys.exit(main())
