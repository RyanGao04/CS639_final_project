#!/usr/bin/env python3
"""Collect parallel Webots rollouts, then run offline SERL updates.

This is development scaffolding for speeding up RL iteration. It does not change
the submission controller. Each worker launches a separate headless Webots
process through ``rl_workflow.py deepbots-record`` and writes its own JSONL trace.
After collection, this script runs ``rl_workflow.py serl-train`` on the demo
traces plus the newly collected rollout traces.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "scripts" / "rl_workflow.py"
TRACE_DIR = ROOT / "tmp" / "rl_traces"
CHECKPOINT_DIR = ROOT / "tmp" / "rl_checkpoints"
DEFAULT_WORLD = ROOT / "final_project" / "worlds" / "soccer_solo_deepbots.wbt"
DEFAULT_RUNTIME_WEIGHTS = (
    ROOT / "final_project" / "controllers" / "robot_one_controller" / "e2e_rl_policy_weights.npz"
)
RESET_WORKER_ARGS = (
    ("reset_sampler", "--reset-sampler"),
    ("reset_ball_margin", "--reset-ball-margin"),
    ("reset_edge_band", "--reset-edge-band"),
    ("reset_edge_fraction", "--reset-edge-fraction"),
    ("reset_corner_fraction", "--reset-corner-fraction"),
    ("reset_behind_fraction", "--reset-behind-fraction"),
    ("reset_min_robot_dist", "--reset-min-robot-dist"),
)


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_name(value):
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or _timestamp()


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _python_path(value):
    if value:
        return str(Path(value).expanduser().resolve())
    return sys.executable


def _worker_command(args, worker_name, seed, weights_path):
    command = [
        _python_path(args.workflow_python),
        str(WORKFLOW),
        "deepbots-record",
        worker_name,
        "--trace-dir",
        str(args.trace_dir),
        "--policy",
        "actor",
        "--phase",
        str(args.phase),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--seed",
        str(seed),
        "--python",
        _python_path(args.controller_python),
        "--world",
        str(args.world),
        "--webots-mode",
        args.webots_mode,
        "--batch",
    ]
    if weights_path is not None:
        command.extend(["--weights", str(weights_path)])
    for attr, flag in RESET_WORKER_ARGS:
        value = getattr(args, attr, None)
        if value is not None:
            command.extend([flag, str(value)])
    if args.terminate_out_of_play:
        command.append("--terminate-out-of-play")
    if args.webots_bin:
        command.extend(["--webots-bin", str(args.webots_bin)])
    return command


def _terminate_process(process, grace_seconds=15.0):
    if process.poll() is not None:
        return
    process.terminate()
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    process.kill()
    process.wait(timeout=5)


def _collect_rollouts(args, iteration, weights_path):
    processes = []
    trace_paths = []
    log_paths = []
    prefix = f"{args.name}_iter{iteration:02d}"

    for worker_index in range(args.workers):
        worker_name = f"{prefix}_worker{worker_index:02d}"
        seed = args.seed + (iteration - 1) * args.workers + worker_index
        trace_path = args.trace_dir / f"{worker_name}_deepbots.jsonl"
        log_path = args.checkpoint_dir / f"{worker_name}.log"
        command = _worker_command(args, worker_name, seed, weights_path)
        trace_paths.append(trace_path)
        log_paths.append(log_path)

        print(f"[collect] worker={worker_index} seed={seed} trace={trace_path}")
        print(f"[collect] command: {' '.join(command)}")
        if args.dry_run:
            continue

        log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((process, log_handle, worker_index))

    if args.dry_run:
        return trace_paths

    deadline = time.monotonic() + args.worker_seconds
    try:
        while time.monotonic() < deadline:
            if processes and all(process.poll() is not None for process, _, _ in processes):
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[collect] interrupted; stopping workers")
    finally:
        for process, _, worker_index in processes:
            if process.poll() is None:
                print(f"[collect] stopping worker={worker_index}")
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, _, worker_index in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print(f"[collect] killing worker={worker_index}")
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
        for _, log_handle, _ in processes:
            log_handle.close()

    usable_traces = []
    for trace_path, log_path in zip(trace_paths, log_paths):
        if trace_path.exists() and trace_path.stat().st_size > 0:
            usable_traces.append(trace_path)
            continue
        print(f"[collect] warning: empty/missing trace {trace_path}; see {log_path}")
    return usable_traces


def _train_command(args, iteration, rollout_traces, output_path):
    traces = [str(path) for path in args.demo_traces] + [str(path) for path in rollout_traces]
    command = [
        _python_path(args.workflow_python),
        str(WORKFLOW),
        "serl-train",
        *traces,
        "--trace-dir",
        str(args.trace_dir),
        "--serl-root",
        str(args.serl_root),
        "--python",
        _python_path(args.controller_python),
        "--output",
        str(output_path),
        "--steps",
        str(args.offline_steps),
        "--batch-size",
        str(args.batch_size),
        "--utd-ratio",
        str(args.utd_ratio),
        "--lr",
        str(args.lr),
        "--gamma",
        str(args.gamma),
        "--seed",
        str(args.seed + 10000 + iteration),
        "--capacity",
        str(args.capacity),
        "--critic-ensemble-size",
        str(args.critic_ensemble_size),
        "--critic-subsample-size",
        str(args.critic_subsample_size),
        "--log-period",
        str(args.log_period),
    ]
    if args.success_only:
        command.append("--success-only")
    if args.dry_run:
        command.append("--dry-run")
    return command


def _run_offline_train(args, iteration, rollout_traces, output_path):
    command = _train_command(args, iteration, rollout_traces, output_path)
    print(f"[train] output={output_path}")
    print(f"[train] command: {' '.join(command)}")
    if args.dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def _existing_weight_path(args):
    if args.initial_weights:
        path = Path(args.initial_weights).expanduser().resolve()
        if not path.exists() and not args.dry_run:
            raise SystemExit(f"Initial weights not found: {path}")
        return path
    if DEFAULT_RUNTIME_WEIGHTS.exists():
        return DEFAULT_RUNTIME_WEIGHTS
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=f"parallel_serl_{_timestamp()}")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-seconds", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--demo-traces",
        nargs="*",
        type=Path,
        default=[TRACE_DIR / "*teleop*.jsonl"],
        help="Demo traces included in every offline training pass.",
    )
    parser.add_argument(
        "--controller-python",
        default=os.environ.get("SERL_PYTHON") or os.environ.get("WEBOTS_CONTROLLER_PYTHON") or sys.executable,
    )
    parser.add_argument("--workflow-python", default=sys.executable)
    parser.add_argument("--serl-root", type=Path, default=Path(os.environ.get("SERL_ROOT", "~/serl")).expanduser())
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--webots-bin", type=Path)
    parser.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="fast")
    parser.add_argument("--phase", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--terminate-out-of-play", action="store_true")
    parser.add_argument("--reset-sampler", choices=("central", "full_field"), default="full_field")
    parser.add_argument("--reset-ball-margin", type=float)
    parser.add_argument("--reset-edge-band", type=float)
    parser.add_argument("--reset-edge-fraction", type=float)
    parser.add_argument("--reset-corner-fraction", type=float)
    parser.add_argument("--reset-behind-fraction", type=float)
    parser.add_argument("--reset-min-robot-dist", type=float)
    parser.add_argument("--offline-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--utd-ratio", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--capacity", type=int, default=500000)
    parser.add_argument("--critic-ensemble-size", type=int, default=10)
    parser.add_argument("--critic-subsample-size", type=int, default=2)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--log-period", type=int, default=500)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.name = _normalize_name(args.name)
    args.trace_dir = _ensure_dir(args.trace_dir.expanduser().resolve())
    args.checkpoint_dir = _ensure_dir(args.checkpoint_dir.expanduser().resolve())
    args.world = args.world.expanduser().resolve()
    args.serl_root = args.serl_root.expanduser().resolve()
    args.demo_traces = [path.expanduser() for path in args.demo_traces]
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")

    weights_path = _existing_weight_path(args)
    final_output = None
    print(f"[setup] worktree={ROOT}")
    print(f"[setup] initial_weights={weights_path}")
    print(f"[setup] workers={args.workers} worker_seconds={args.worker_seconds}")

    for iteration in range(1, args.iterations + 1):
        rollout_traces = _collect_rollouts(args, iteration, weights_path)
        print(f"[collect] iteration={iteration} usable_traces={len(rollout_traces)}")
        if args.collect_only:
            continue
        if not rollout_traces and not args.dry_run:
            raise SystemExit("No rollout traces were collected; refusing to train.")

        if args.output and args.iterations == 1:
            output_path = args.output.expanduser().resolve()
        else:
            output_path = args.checkpoint_dir / f"{args.name}_iter{iteration:02d}_weights.npz"
        _run_offline_train(args, iteration, rollout_traces, output_path)
        weights_path = output_path
        final_output = output_path

    if final_output is not None:
        print(f"[done] final_weights={final_output}")
    else:
        print("[done] collection complete")


if __name__ == "__main__":
    raise SystemExit(main())
