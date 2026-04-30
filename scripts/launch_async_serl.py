#!/usr/bin/env python3
"""Launch one async SERL learner plus multiple headless Webots actor workers."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "scripts" / "rl_workflow.py"
LEARNER = ROOT / "scripts" / "async_serl_learner.py"
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
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe).strip("_") or _timestamp()


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _python_path(value):
    if value:
        return str(Path(value).expanduser().resolve())
    return sys.executable


def _learner_command(args):
    command = [
        _python_path(args.controller_python),
        str(LEARNER),
        "--bind",
        args.endpoint,
        "--output",
        str(args.output),
        "--updates",
        str(args.updates),
        "--pretrain-steps",
        str(args.pretrain_steps),
        "--min-online",
        str(args.min_online),
        "--batch-size",
        str(args.batch_size),
        "--utd-ratio",
        str(args.utd_ratio),
        "--demo-fraction",
        str(args.demo_fraction),
        "--lr",
        str(args.lr),
        "--gamma",
        str(args.gamma),
        "--seed",
        str(args.seed),
        "--capacity",
        str(args.capacity),
        "--critic-ensemble-size",
        str(args.critic_ensemble_size),
        "--critic-subsample-size",
        str(args.critic_subsample_size),
        "--log-period",
        str(args.log_period),
        "--export-period",
        str(args.export_period),
        "--drain-batch",
        str(args.drain_batch),
        "--updates-per-drain",
        str(args.updates_per_drain),
        "--reuse-batch-updates",
        str(args.reuse_batch_updates),
        "--serl-root",
        str(args.serl_root),
        "--demo-traces",
        *[str(path) for path in args.demo_traces],
    ]
    if args.seconds > 0:
        command.extend(["--seconds", str(args.seconds)])
    if args.success_only:
        command.append("--success-only")
    return command


def _worker_command(args, worker_index):
    worker_name = f"{args.name}_worker{worker_index:02d}"
    seed = args.seed + worker_index
    command = [
        _python_path(args.workflow_python),
        str(WORKFLOW),
        "deepbots-record",
        worker_name,
        "--policy",
        "async_actor",
        "--no-trace",
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
        "--weights",
        str(args.output),
    ]
    for attr, flag in RESET_WORKER_ARGS:
        value = getattr(args, attr, None)
        if value is not None:
            command.extend([flag, str(value)])
    if args.terminate_out_of_play:
        command.append("--terminate-out-of-play")
    if args.webots_bin:
        command.extend(["--webots-bin", str(args.webots_bin)])
    return command


def _terminate_group(process, name, grace_seconds=20.0):
    if process.poll() is not None:
        return
    print(f"[stop] {name}")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    print(f"[kill] {name}")
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _seed_output_weights(args):
    if args.output.exists() or args.initial_weights is None:
        return
    source = args.initial_weights.expanduser().resolve()
    if source.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, args.output)
        print(f"[setup] seeded async output from {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=f"async_serl_{_timestamp()}")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--output", type=Path, default=CHECKPOINT_DIR / "async_serl_latest_weights.npz")
    parser.add_argument("--initial-weights", type=Path, default=DEFAULT_RUNTIME_WEIGHTS)
    parser.add_argument(
        "--demo-traces",
        nargs="*",
        type=Path,
        default=[TRACE_DIR / "*teleop*.jsonl"],
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
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
    parser.add_argument("--updates", type=int, default=50000)
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--min-online", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--utd-ratio", type=int, default=16)
    parser.add_argument("--demo-fraction", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--capacity", type=int, default=500000)
    parser.add_argument("--critic-ensemble-size", type=int, default=10)
    parser.add_argument("--critic-subsample-size", type=int, default=2)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument("--export-period", type=int, default=500)
    parser.add_argument("--drain-batch", type=int, default=4096)
    parser.add_argument("--updates-per-drain", type=int, default=1)
    parser.add_argument("--reuse-batch-updates", type=int, default=1)
    parser.add_argument("--async-reload-every", type=int, default=250)
    parser.add_argument("--async-worker-max-steps", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.name = _normalize_name(args.name)
    args.output = args.output.expanduser().resolve()
    args.checkpoint_dir = _ensure_dir(args.checkpoint_dir.expanduser().resolve())
    args.serl_root = args.serl_root.expanduser().resolve()
    args.world = args.world.expanduser().resolve()
    args.demo_traces = [path.expanduser() for path in args.demo_traces]
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    if not args.dry_run:
        _seed_output_weights(args)

    learner_log_path = args.checkpoint_dir / f"{args.name}_learner.log"
    learner_command = _learner_command(args)
    print(f"[learner] log={learner_log_path}")
    print(f"[learner] command: {' '.join(learner_command)}")
    worker_commands = [_worker_command(args, index) for index in range(args.workers)]
    for index, command in enumerate(worker_commands):
        print(f"[worker {index}] command: {' '.join(command)}")
    if args.dry_run:
        return 0

    processes = []
    learner_env = os.environ.copy()
    learner_log = open(learner_log_path, "w", encoding="utf-8", buffering=1)
    learner = subprocess.Popen(
        learner_command,
        cwd=ROOT,
        env=learner_env,
        stdout=learner_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    processes.append(("learner", learner, learner_log))
    time.sleep(2.0)
    if learner.poll() is not None:
        print(f"[learner] exited during startup rc={learner.returncode}; not launching workers")
        learner_log.close()
        return learner.returncode

    for index, command in enumerate(worker_commands):
        worker_env = os.environ.copy()
        worker_env["RL_ASYNC_TRANSITION_PUSH"] = args.endpoint
        worker_env["RL_ASYNC_POLICY_WEIGHTS_PATH"] = str(args.output)
        worker_env["RL_ASYNC_WORKER_ID"] = f"{args.name}_{index:02d}"
        worker_env["RL_ASYNC_RELOAD_EVERY"] = str(args.async_reload_every)
        if args.async_worker_max_steps > 0:
            worker_env["RL_ASYNC_MAX_STEPS"] = str(args.async_worker_max_steps)
        log_path = args.checkpoint_dir / f"{args.name}_worker{index:02d}.log"
        log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=worker_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((f"worker{index:02d}", process, log_handle))
        print(f"[worker {index}] pid={process.pid} log={log_path}")

    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    try:
        while True:
            if learner.poll() is not None:
                print(f"[learner] exited rc={learner.returncode}")
                break
            if deadline is not None and time.monotonic() >= deadline:
                print("[main] wall-clock limit reached")
                break
            if all(process.poll() is not None for name, process, _ in processes if name.startswith("worker")):
                print("[main] all workers exited")
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("[main] interrupted")
    finally:
        for name, process, _ in processes:
            _terminate_group(process, name)
        for _, _, log_handle in processes:
            log_handle.close()

    print(f"[done] latest_weights={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
