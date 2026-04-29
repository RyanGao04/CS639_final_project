#!/usr/bin/env python3
"""Launch keyboard teleop demo collection for the E2E soccer policy."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "scripts" / "rl_workflow.py"
TRACE_DIR = ROOT / "tmp" / "rl_traces"
DEEPBOTS_WORLD = ROOT / "final_project" / "worlds" / "soccer_solo_deepbots.wbt"


def _default_controller_python():
    serl_python = os.environ.get("SERL_PYTHON")
    if serl_python:
        return serl_python

    local_serl_python = Path("/opt/anaconda3/envs/serl-cs639/bin/python")
    if local_serl_python.exists():
        return str(local_serl_python)

    return sys.executable


def _timestamp_name():
    return "teleop_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _print_command(cmd):
    print(" ".join(shlex.quote(str(part)) for part in cmd))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default=_timestamp_name(), help="Trace run name.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for rl_workflow.py.")
    parser.add_argument("--controller-python", default=_default_controller_python(), help="Python interpreter Webots uses for the deepbots controller.")
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--phase", type=int, default=3, help="Curriculum/reset phase. Phase 3 uses assignment-style starts.")
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--world", type=Path, default=DEEPBOTS_WORLD)
    parser.add_argument("--webots-bin", help="Path to Webots executable.")
    parser.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="realtime")
    parser.add_argument("--terminate-out-of-play", action="store_true")
    parser.add_argument("--no-launch-webots", action="store_true", help="Only print environment exports.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    cmd = [
        args.python,
        str(WORKFLOW),
        "deepbots-teleop",
        args.name,
        "--trace-dir",
        str(args.trace_dir),
        "--phase",
        str(args.phase),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--seed",
        str(args.seed),
        "--world",
        str(args.world),
        "--webots-mode",
        args.webots_mode,
        "--python",
        args.controller_python,
    ]
    if args.webots_bin:
        cmd.extend(["--webots-bin", args.webots_bin])
    if args.terminate_out_of_play:
        cmd.append("--terminate-out-of-play")
    if args.no_launch_webots:
        cmd.append("--no-launch-webots")
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"Teleop trace target: {args.trace_dir / (args.name + '_teleop.jsonl')}")
    print("Controls: W/Up forward, S/Down reverse, A/Left, D/Right, Space stop, R reset, Q quit.")
    _print_command(cmd)
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
