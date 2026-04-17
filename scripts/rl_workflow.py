#!/usr/bin/env python3
"""Convenience CLI for collecting traces, training, and activating RL policies."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = ROOT / "final_project" / "controllers" / "robot_one_controller"
TRAIN_SCRIPT = CONTROLLER_DIR / "train_rl_actor_from_traces.py"
RUNTIME_WEIGHTS_PATH = CONTROLLER_DIR / "rl_policy_weights.npz"
TRACE_DIR = ROOT / "tmp" / "rl_traces"
CHECKPOINT_DIR = ROOT / "tmp" / "rl_checkpoints"


EXPECTED_SHAPES = {
    "w1": (64, 16),
    "b1": (64,),
    "w2": (64, 64),
    "b2": (64,),
    "w3": (2, 64),
    "b3": (2,),
}


def _timestamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _shell_quote(value):
    return json.dumps(str(value))


def _collect_trace_files(patterns, trace_dir):
    resolved = []
    if not patterns:
        patterns = [str(trace_dir / "*.jsonl")]

    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            matches = sorted(Path().glob(pattern)) if not Path(pattern).is_absolute() else sorted(Path(pattern).parent.glob(Path(pattern).name))
            resolved.extend(match.resolve() for match in matches if match.is_file())
            continue

        candidate = Path(pattern).expanduser()
        if candidate.is_dir():
            resolved.extend(sorted(path.resolve() for path in candidate.glob("*.jsonl")))
        elif candidate.is_file():
            resolved.append(candidate.resolve())

    deduped = []
    seen = set()
    for path in resolved:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _extract_arrays(state_dict):
    direct_keys = ("w1", "b1", "w2", "b2", "w3", "b3")
    if all(key in state_dict for key in direct_keys):
        arrays = {key: state_dict[key].detach().cpu().numpy() for key in direct_keys}
    else:
        candidate_sets = [
            {
                "w1": "actor.0.weight",
                "b1": "actor.0.bias",
                "w2": "actor.2.weight",
                "b2": "actor.2.bias",
                "w3": "actor.4.weight",
                "b3": "actor.4.bias",
            },
            {
                "w1": "net.0.weight",
                "b1": "net.0.bias",
                "w2": "net.2.weight",
                "b2": "net.2.bias",
                "w3": "net.4.weight",
                "b3": "net.4.bias",
            },
        ]
        arrays = None
        for candidate in candidate_sets:
            if all(key in state_dict for key in candidate.values()):
                arrays = {
                    target: state_dict[source].detach().cpu().numpy()
                    for target, source in candidate.items()
                }
                break
        if arrays is None:
            raise KeyError(f"Unsupported checkpoint keys: {sorted(state_dict.keys())}")

    for key, expected_shape in EXPECTED_SHAPES.items():
        array = np.asarray(arrays[key], dtype=np.float32)
        if array.shape != expected_shape:
            raise ValueError(f"{key} has shape {array.shape}, expected {expected_shape}")
        arrays[key] = array
    return arrays


def _load_checkpoint_arrays(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    return _extract_arrays(state_dict)


def _save_runtime_weights(arrays, output_path):
    _ensure_dir(output_path.parent)
    np.savez(output_path, **arrays)
    return output_path


def cmd_record(args):
    trace_dir = _ensure_dir(args.trace_dir)
    trace_name = _normalize_name(args.name or _timestamp())
    trace_path = trace_dir / f"{trace_name}.jsonl"

    env = os.environ.copy()
    env["RL_TRACE_PATH"] = str(trace_path)
    env["WEBOTS_LIVE_VISUALIZER"] = "1" if args.visualizer else "0"
    if args.weights:
        env["RL_POLICY_WEIGHTS_PATH"] = str(Path(args.weights).expanduser().resolve())

    print(f"Trace file: {trace_path}")
    print(f"WEBOTS_LIVE_VISUALIZER={env['WEBOTS_LIVE_VISUALIZER']}")
    if "RL_POLICY_WEIGHTS_PATH" in env:
        print(f"RL_POLICY_WEIGHTS_PATH={env['RL_POLICY_WEIGHTS_PATH']}")

    if args.command:
        print(f"Running command with trace env: {' '.join(args.command)}")
        completed = subprocess.run(args.command, cwd=ROOT, env=env)
        return completed.returncode

    print("\nRun Webots from this shell with the following environment:")
    print(f"export RL_TRACE_PATH={_shell_quote(trace_path)}")
    print(f"export WEBOTS_LIVE_VISUALIZER={env['WEBOTS_LIVE_VISUALIZER']}")
    if "RL_POLICY_WEIGHTS_PATH" in env:
        print(f"export RL_POLICY_WEIGHTS_PATH={_shell_quote(env['RL_POLICY_WEIGHTS_PATH'])}")
    return 0


def _run_train_subprocess(trace_files, output_path, args):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        *[str(path) for path in trace_files],
        "--output",
        str(output_path),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--gamma",
        str(args.gamma),
        "--device",
        args.device,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def cmd_train(args):
    trace_dir = _ensure_dir(args.trace_dir)
    checkpoint_dir = _ensure_dir(args.checkpoint_dir)
    trace_files = _collect_trace_files(args.traces, trace_dir)
    if not trace_files:
        raise SystemExit("No trace files found.")

    run_name = _normalize_name(args.name or _timestamp())
    output_path = Path(args.output).expanduser().resolve() if args.output else (checkpoint_dir / f"{run_name}.pt")
    _run_train_subprocess(trace_files, output_path, args)
    print(f"Checkpoint ready: {output_path}")
    return 0


def cmd_activate(args):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else RUNTIME_WEIGHTS_PATH
    arrays = _load_checkpoint_arrays(checkpoint_path)
    _save_runtime_weights(arrays, output_path)
    print(f"Activated RL weights: {output_path}")
    return 0


def cmd_train_activate(args):
    trace_dir = _ensure_dir(args.trace_dir)
    checkpoint_dir = _ensure_dir(args.checkpoint_dir)
    trace_files = _collect_trace_files(args.traces, trace_dir)
    if not trace_files:
        raise SystemExit("No trace files found.")

    run_name = _normalize_name(args.name or _timestamp())
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    _run_train_subprocess(trace_files, checkpoint_path, args)

    arrays = _load_checkpoint_arrays(checkpoint_path)
    runtime_path = Path(args.runtime_output).expanduser().resolve() if args.runtime_output else RUNTIME_WEIGHTS_PATH
    _save_runtime_weights(arrays, runtime_path)

    print(f"Checkpoint ready: {checkpoint_path}")
    print(f"Activated RL weights: {runtime_path}")
    return 0


def cmd_deactivate(args):
    runtime_path = Path(args.runtime_output).expanduser().resolve() if args.runtime_output else RUNTIME_WEIGHTS_PATH
    if runtime_path.exists():
        runtime_path.unlink()
        print(f"Removed active RL weights: {runtime_path}")
    else:
        print(f"No active RL weights file at: {runtime_path}")
    return 0


def cmd_status(args):
    del args
    trace_dir = _ensure_dir(TRACE_DIR)
    checkpoint_dir = _ensure_dir(CHECKPOINT_DIR)

    print(f"Project root: {ROOT}")
    print(f"Controller dir: {CONTROLLER_DIR}")
    print(f"Default runtime weights: {RUNTIME_WEIGHTS_PATH}")
    print(f"Runtime weights active: {'yes' if RUNTIME_WEIGHTS_PATH.exists() else 'no'}")

    recent_traces = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    recent_ckpts = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]

    print("\nRecent traces:")
    if recent_traces:
        for path in recent_traces:
            print(f"  {path.name}")
    else:
        print("  none")

    print("\nRecent checkpoints:")
    if recent_ckpts:
        for path in recent_ckpts:
            print(f"  {path.name}")
    else:
        print("  none")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Prepare RL trace env and optionally run a command under it.")
    record.add_argument("name", nargs="?", help="Trace run name.")
    record.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    record.add_argument("--visualizer", action="store_true", help="Keep the live visualizer enabled during recording.")
    record.add_argument("--weights", type=Path, help="Optional runtime weights file to use for this recording run.")
    record.add_argument("command", nargs=argparse.REMAINDER, help="Optional command to run after '--'.")
    record.set_defaults(func=cmd_record)

    for sub_name, help_text, func in (
        ("train", "Train an actor checkpoint from collected traces.", cmd_train),
        ("train-activate", "Train from traces and activate the resulting weights file.", cmd_train_activate),
    ):
        sub = subparsers.add_parser(sub_name, help=help_text)
        sub.add_argument("traces", nargs="*", help="Trace files, directories, or glob patterns. Default: tmp/rl_traces/*.jsonl")
        sub.add_argument("--name", help="Run name for the checkpoint.")
        sub.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
        sub.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
        sub.add_argument("--output", type=Path, help="Checkpoint output path. train only.")
        sub.add_argument("--runtime-output", type=Path, help="Runtime .npz output path. train-activate only.")
        sub.add_argument("--epochs", type=int, default=80)
        sub.add_argument("--batch-size", type=int, default=256)
        sub.add_argument("--lr", type=float, default=3e-4)
        sub.add_argument("--gamma", type=float, default=0.995)
        sub.add_argument("--device", default="cpu")
        sub.set_defaults(func=func)

    activate = subparsers.add_parser("activate", help="Convert a checkpoint into the runtime .npz weights file.")
    activate.add_argument("checkpoint", help="PyTorch checkpoint created by train_rl_actor_from_traces.py")
    activate.add_argument("--output", type=Path, help="Runtime .npz output path.")
    activate.set_defaults(func=cmd_activate)

    deactivate = subparsers.add_parser("deactivate", help="Remove the active runtime weights file.")
    deactivate.add_argument("--runtime-output", type=Path, help="Runtime .npz file to remove.")
    deactivate.set_defaults(func=cmd_deactivate)

    status = subparsers.add_parser("status", help="Show current RL workflow state.")
    status.set_defaults(func=cmd_status)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    command = getattr(args, "command", None)
    if command and command[:1] == ["--"]:
        args.command = command[1:]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
