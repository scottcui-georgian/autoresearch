"""
Thin local wrapper for Modal execution.

Examples:
    python modal_runner.py prepare --num-shards 10
    python modal_runner.py train > run.log 2>&1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODAL_APP = REPO_ROOT / "modal_app.py"


def _require_modal() -> str:
    """Resolve the Modal CLI once and fail with a clear install hint if it is missing."""
    modal_path = shutil.which("modal")
    if modal_path is None:
        raise SystemExit(
            "The `modal` CLI is not installed or not on PATH. "
            "Install it separately from this repo, for example with `uv tool install modal`."
        )
    return modal_path


def _run_modal(args: list[str], quiet: bool = True) -> int:
    """Invoke the Modal app quietly so run logs contain only experiment-relevant output."""
    modal_path = _require_modal()
    cmd = [modal_path, "run"]
    if quiet:
        cmd.append("-q")
    cmd.extend([str(MODAL_APP), *args])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return proc.returncode


def main() -> int:
    """Provide the stable local commands the research agent should use for prepare/train."""
    parser = argparse.ArgumentParser(description="Run autoresearch jobs on Modal.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Run one-time data preparation on Modal CPU.")
    prepare_parser.add_argument("--num-shards", type=int, default=10)
    prepare_parser.add_argument("--download-workers", type=int, default=8)

    subparsers.add_parser("train", help="Run a single training experiment on Modal L40S.")

    args = parser.parse_args()

    if args.command == "prepare":
        return _run_modal(
            [
                "--action",
                "prepare",
                "--num-shards",
                str(args.num_shards),
                "--download-workers",
                str(args.download_workers),
            ]
        )
    if args.command == "train":
        return _run_modal(["--action", "train"], quiet=False)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
