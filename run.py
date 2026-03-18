#!/usr/bin/env python3
"""
Task-specific runner for nanogpt experiments.

Usage:
    python run.py train > run.log 2>&1
    python run.py prepare --num-shards 10

Requires `autoresearch` and `modal` on PATH (install via `uv tool install`).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def _require_cmd(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"`{name}` is not on PATH. Install it with `uv tool install {name}`.")
    return path


def _get_modal_app_path() -> str:
    """Ask the autoresearch CLI where modal_app.py lives."""
    result = subprocess.run(
        [_require_cmd("autoresearch"), "modal-app-path"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"Failed to locate modal_app.py: {result.stderr.strip()}")
    return result.stdout.strip()


def run_modal(action: str, quiet: bool = True) -> int:
    modal_path = _require_cmd("modal")
    modal_app = _get_modal_app_path()
    cmd = [modal_path, "run"]
    if quiet:
        cmd.append("-q")
    cmd.extend([modal_app, "--action", action])
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nanogpt experiments on Modal.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run a single training experiment on Modal L40S.")

    prepare_parser = subparsers.add_parser("prepare", help="Run one-time data preparation on Modal CPU.")
    prepare_parser.add_argument("--num-shards", type=int, default=10)
    prepare_parser.add_argument("--download-workers", type=int, default=8)

    args = parser.parse_args()
    return run_modal(args.command, quiet=(args.command != "train"))


if __name__ == "__main__":
    sys.exit(main())
