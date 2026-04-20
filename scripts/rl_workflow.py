#!/usr/bin/env python3
"""Convenience CLI for collecting traces, training, and activating RL policies."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
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
DEFAULT_WORLD = ROOT / "final_project" / "worlds" / "soccer_solo.wbt"
DEEPBOTS_WORLD = ROOT / "final_project" / "worlds" / "soccer_solo_deepbots.wbt"
DEEPBOTS_CONTROLLER = ROOT / "final_project" / "controllers" / "rl_training_controller" / "rl_training_controller.py"


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


def _find_webots_binary(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_path = os.environ.get("WEBOTS_BIN")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    which_path = shutil.which("webots")
    if which_path:
        candidates.append(Path(which_path))

    candidates.extend(
        [
            Path("/Applications/Webots.app/Contents/MacOS/webots"),
            Path("/Applications/Webots.app"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_world_path(world):
    world_path = Path(world).expanduser()
    if not world_path.is_absolute():
        world_path = (ROOT / world_path).resolve()
    return world_path


def _webots_command(webots_bin, world_path, mode=None, batch=False):
    cmd = [str(webots_bin)]
    if batch:
        cmd.append("--batch")
    if mode:
        cmd.append(f"--mode={mode}")
    cmd.append(str(world_path))
    return cmd


def _launch_webots(args, env, world):
    webots_bin = _find_webots_binary(getattr(args, "webots_bin", None))
    if webots_bin is None:
        raise SystemExit(
            "Could not find a Webots executable. Pass --webots-bin or set WEBOTS_BIN."
        )

    world_path = _resolve_world_path(world)
    if not world_path.exists():
        raise SystemExit(f"World file not found: {world_path}")

    launch_cmd = _webots_command(
        webots_bin,
        world_path,
        mode=getattr(args, "webots_mode", None),
        batch=getattr(args, "batch", False),
    )
    print(f"Launching Webots: {' '.join(launch_cmd)}")
    if getattr(args, "dry_run", False):
        return 0
    completed = subprocess.run(launch_cmd, cwd=ROOT, env=env)
    return completed.returncode


def _print_env_exports(env, keys):
    print("\nRun Webots from this shell with the following environment:")
    for key in keys:
        if key in env:
            print(f"export {key}={_shell_quote(env[key])}")


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
        if args.dry_run:
            return 0
        completed = subprocess.run(args.command, cwd=ROOT, env=env)
        return completed.returncode

    if args.launch_webots:
        return _launch_webots(args, env, args.world)

    _print_env_exports(env, ("RL_TRACE_PATH", "WEBOTS_LIVE_VISUALIZER", "RL_POLICY_WEIGHTS_PATH"))
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


def _deepbots_trace_path(args, suffix="deepbots", run_name=None):
    trace_dir = _ensure_dir(args.trace_dir)
    trace_name = run_name or _normalize_name(args.name or _timestamp())
    return trace_dir / f"{trace_name}_{suffix}.jsonl"


def _base_deepbots_env(args, trace_path=None):
    env = os.environ.copy()
    env["RL_DEEPBOTS_PHASE"] = str(args.phase)
    env["RL_DEEPBOTS_MAX_STEPS"] = str(args.max_episode_steps)
    env["RL_DEEPBOTS_SEED"] = str(args.seed)
    if trace_path is not None:
        env["RL_DEEPBOTS_TRACE_PATH"] = str(trace_path)
        env["RL_TRACE_PATH"] = str(trace_path)
    if getattr(args, "weights", None):
        env["RL_POLICY_WEIGHTS_PATH"] = str(Path(args.weights).expanduser().resolve())
    return env


def cmd_deepbots_record(args):
    if args.policy == "sac" and not args.model:
        raise SystemExit("--policy sac requires --model pointing to an SB3 SAC .zip file.")

    run_name = _normalize_name(args.name or _timestamp())
    trace_path = _deepbots_trace_path(args, run_name=run_name)
    env = _base_deepbots_env(args, trace_path=trace_path)
    mode = "sac_eval" if args.policy == "sac" else args.policy
    env["RL_DEEPBOTS_MODE"] = mode
    if args.model:
        env["RL_DEEPBOTS_MODEL_PATH"] = str(Path(args.model).expanduser().resolve())

    print(f"Deepbots mode: {mode}")
    print(f"Trace file: {trace_path}")
    if "RL_POLICY_WEIGHTS_PATH" in env:
        print(f"RL_POLICY_WEIGHTS_PATH={env['RL_POLICY_WEIGHTS_PATH']}")
    if "RL_DEEPBOTS_MODEL_PATH" in env:
        print(f"RL_DEEPBOTS_MODEL_PATH={env['RL_DEEPBOTS_MODEL_PATH']}")

    if args.launch_webots:
        return _launch_webots(args, env, args.world)

    _print_env_exports(
        env,
        (
            "RL_DEEPBOTS_MODE",
            "RL_DEEPBOTS_TRACE_PATH",
            "RL_TRACE_PATH",
            "RL_DEEPBOTS_PHASE",
            "RL_DEEPBOTS_MAX_STEPS",
            "RL_DEEPBOTS_SEED",
            "RL_POLICY_WEIGHTS_PATH",
            "RL_DEEPBOTS_MODEL_PATH",
        ),
    )
    return 0


def cmd_deepbots_train(args):
    checkpoint_dir = _ensure_dir(args.checkpoint_dir)
    run_name = _normalize_name(args.name or _timestamp())
    trace_path = _deepbots_trace_path(args, run_name=run_name)
    model_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint_dir / f"{run_name}_sac.zip"
    )

    env = _base_deepbots_env(args, trace_path=trace_path)
    env["RL_DEEPBOTS_MODE"] = "sac_train"
    env["RL_DEEPBOTS_TOTAL_STEPS"] = str(args.timesteps)
    env["RL_DEEPBOTS_MODEL_PATH"] = str(model_path)
    env["RL_DEEPBOTS_LEARNING_STARTS"] = str(args.learning_starts)
    env["RL_DEEPBOTS_BATCH_SIZE"] = str(args.batch_size)
    env["RL_DEEPBOTS_LR"] = str(args.lr)
    env["RL_DEEPBOTS_GAMMA"] = str(args.gamma)
    if args.tensorboard_dir:
        env["RL_DEEPBOTS_TENSORBOARD_DIR"] = str(Path(args.tensorboard_dir).expanduser().resolve())

    if args.activate:
        runtime_path = (
            Path(args.runtime_output).expanduser().resolve()
            if args.runtime_output
            else RUNTIME_WEIGHTS_PATH
        )
        env["RL_DEEPBOTS_EXPORT_PATH"] = str(runtime_path)

    print(f"Deepbots mode: sac_train")
    print(f"SAC model output: {model_path}")
    print(f"Transition trace: {trace_path}")
    if "RL_DEEPBOTS_EXPORT_PATH" in env:
        print(f"Runtime export: {env['RL_DEEPBOTS_EXPORT_PATH']}")

    if args.launch_webots:
        return _launch_webots(args, env, args.world)

    _print_env_exports(
        env,
        (
            "RL_DEEPBOTS_MODE",
            "RL_DEEPBOTS_TRACE_PATH",
            "RL_TRACE_PATH",
            "RL_DEEPBOTS_PHASE",
            "RL_DEEPBOTS_MAX_STEPS",
            "RL_DEEPBOTS_SEED",
            "RL_DEEPBOTS_TOTAL_STEPS",
            "RL_DEEPBOTS_MODEL_PATH",
            "RL_DEEPBOTS_EXPORT_PATH",
            "RL_DEEPBOTS_LEARNING_STARTS",
            "RL_DEEPBOTS_BATCH_SIZE",
            "RL_DEEPBOTS_LR",
            "RL_DEEPBOTS_GAMMA",
            "RL_DEEPBOTS_TENSORBOARD_DIR",
        ),
    )
    return 0


def _extract_sb3_sac_arrays(model_path):
    try:
        import torch.nn as nn
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is required to activate an SB3 SAC model. "
            "Run: python scripts/rl_workflow.py install-rl-deps"
        ) from exc

    model = SAC.load(str(model_path))
    actor = model.policy.actor
    latent_pi = getattr(actor, "latent_pi", None)
    mu_layer = getattr(actor, "mu", None)
    if latent_pi is None or mu_layer is None:
        raise SystemExit("Unsupported SAC actor layout; expected latent_pi and mu modules.")

    linear_layers = [module for module in latent_pi.modules() if isinstance(module, nn.Linear)]
    if len(linear_layers) != 2 or not isinstance(mu_layer, nn.Linear):
        raise SystemExit("SAC actor must use exactly two hidden Linear layers plus mu output.")

    arrays = {
        "w1": linear_layers[0].weight.detach().cpu().numpy(),
        "b1": linear_layers[0].bias.detach().cpu().numpy(),
        "w2": linear_layers[1].weight.detach().cpu().numpy(),
        "b2": linear_layers[1].bias.detach().cpu().numpy(),
        "w3": mu_layer.weight.detach().cpu().numpy(),
        "b3": mu_layer.bias.detach().cpu().numpy(),
    }
    for key, expected_shape in EXPECTED_SHAPES.items():
        arrays[key] = np.asarray(arrays[key], dtype=np.float32)
        if arrays[key].shape != expected_shape:
            raise SystemExit(f"{key} has shape {arrays[key].shape}, expected {expected_shape}")
    return arrays


def cmd_activate_sb3(args):
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else RUNTIME_WEIGHTS_PATH
    arrays = _extract_sb3_sac_arrays(model_path)
    _save_runtime_weights(arrays, output_path)
    print(f"Activated SB3 SAC actor weights: {output_path}")
    return 0


def cmd_install_rl_deps(args):
    packages = ["gymnasium", "stable-baselines3", "tensorboard"]
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    print(f"Installing RL dependencies: {' '.join(packages)}")
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    completed = subprocess.run(cmd, cwd=ROOT)
    return completed.returncode


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
    print(f"Deepbots folder present: {'yes' if (ROOT / 'deepbots').exists() else 'no'}")
    print(f"Deepbots world present: {'yes' if DEEPBOTS_WORLD.exists() else 'no'}")
    print(f"Deepbots controller present: {'yes' if DEEPBOTS_CONTROLLER.exists() else 'no'}")

    recent_traces = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    recent_ckpts = sorted(
        list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.zip")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:5]

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
    record.add_argument("--world", default=str(DEFAULT_WORLD), help="Webots world to launch when auto-launch is enabled.")
    record.add_argument("--webots-bin", help="Path to the Webots executable.")
    record.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), help="Optional Webots run mode.")
    record.add_argument("--batch", action="store_true", help="Launch Webots in batch mode.")
    record.add_argument("--no-launch-webots", dest="launch_webots", action="store_false", help="Only print the trace environment instead of launching Webots.")
    record.add_argument("--dry-run", action="store_true", help="Print the launch command without running it.")
    record.set_defaults(launch_webots=True)
    record.set_defaults(func=cmd_record)

    deepbots_record = subparsers.add_parser("deepbots-record", help="Run the deepbots world with live policy actions and transition logging.")
    deepbots_record.add_argument("name", nargs="?", help="Trace run name.")
    deepbots_record.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    deepbots_record.add_argument("--policy", choices=("actor", "random", "sac"), default="actor", help="Policy source for live actions.")
    deepbots_record.add_argument("--weights", type=Path, help="Runtime .npz actor weights for --policy actor.")
    deepbots_record.add_argument("--model", type=Path, help="SB3 SAC .zip model for --policy sac.")
    deepbots_record.add_argument("--phase", type=int, default=2, help="Curriculum phase used by the deepbots reset sampler.")
    deepbots_record.add_argument("--max-episode-steps", type=int, default=1800)
    deepbots_record.add_argument("--seed", type=int, default=7)
    deepbots_record.add_argument("--world", default=str(DEEPBOTS_WORLD), help="Deepbots Webots world to launch.")
    deepbots_record.add_argument("--webots-bin", help="Path to the Webots executable.")
    deepbots_record.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="realtime", help="Webots run mode.")
    deepbots_record.add_argument("--batch", action="store_true", help="Launch Webots in batch mode.")
    deepbots_record.add_argument("--no-launch-webots", dest="launch_webots", action="store_false", help="Only print environment exports.")
    deepbots_record.add_argument("--dry-run", action="store_true", help="Print the launch command without running it.")
    deepbots_record.set_defaults(launch_webots=True)
    deepbots_record.set_defaults(func=cmd_deepbots_record)

    deepbots_train = subparsers.add_parser("deepbots-train", help="Train SAC inside Webots through the deepbots controller.")
    deepbots_train.add_argument("--name", help="Run name for the SAC model and trace.")
    deepbots_train.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    deepbots_train.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    deepbots_train.add_argument("--output", type=Path, help="SB3 SAC .zip output path.")
    deepbots_train.add_argument("--activate", action="store_true", help="Export the trained SAC deterministic actor to the runtime .npz weights.")
    deepbots_train.add_argument("--runtime-output", type=Path, help="Runtime .npz output path for --activate.")
    deepbots_train.add_argument("--timesteps", type=int, default=50000)
    deepbots_train.add_argument("--learning-starts", type=int, default=1000)
    deepbots_train.add_argument("--batch-size", type=int, default=256)
    deepbots_train.add_argument("--lr", type=float, default=3e-4)
    deepbots_train.add_argument("--gamma", type=float, default=0.995)
    deepbots_train.add_argument("--phase", type=int, default=2)
    deepbots_train.add_argument("--max-episode-steps", type=int, default=1800)
    deepbots_train.add_argument("--seed", type=int, default=7)
    deepbots_train.add_argument("--tensorboard-dir", type=Path)
    deepbots_train.add_argument("--world", default=str(DEEPBOTS_WORLD), help="Deepbots Webots world to launch.")
    deepbots_train.add_argument("--webots-bin", help="Path to the Webots executable.")
    deepbots_train.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="fast", help="Webots run mode.")
    deepbots_train.add_argument("--batch", action="store_true", default=True, help="Launch Webots in batch mode.")
    deepbots_train.add_argument("--no-batch", dest="batch", action="store_false", help="Show the Webots GUI during training.")
    deepbots_train.add_argument("--no-launch-webots", dest="launch_webots", action="store_false", help="Only print environment exports.")
    deepbots_train.add_argument("--dry-run", action="store_true", help="Print the launch command without running it.")
    deepbots_train.set_defaults(launch_webots=True)
    deepbots_train.set_defaults(func=cmd_deepbots_train)

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

    activate_sb3 = subparsers.add_parser("activate-sb3", help="Export and activate the deterministic actor from an SB3 SAC .zip model.")
    activate_sb3.add_argument("model", help="SB3 SAC .zip model created by deepbots-train.")
    activate_sb3.add_argument("--output", type=Path, help="Runtime .npz output path.")
    activate_sb3.set_defaults(func=cmd_activate_sb3)

    deactivate = subparsers.add_parser("deactivate", help="Remove the active runtime weights file.")
    deactivate.add_argument("--runtime-output", type=Path, help="Runtime .npz file to remove.")
    deactivate.set_defaults(func=cmd_deactivate)

    status = subparsers.add_parser("status", help="Show current RL workflow state.")
    status.set_defaults(func=cmd_status)

    install_deps = subparsers.add_parser("install-rl-deps", help="Install Python packages needed for deepbots SAC training.")
    install_deps.add_argument("--dry-run", action="store_true")
    install_deps.set_defaults(func=cmd_install_rl_deps)

    return parser


def main():
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if args.command == "record":
        if unknown[:1] == ["--"]:
            unknown = unknown[1:]
        args.command = unknown
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
