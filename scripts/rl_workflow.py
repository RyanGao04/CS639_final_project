#!/usr/bin/env python3
"""Convenience CLI for collecting traces, training, and activating RL policies."""

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = ROOT / "final_project" / "controllers" / "robot_one_controller"
TRAIN_SCRIPT = CONTROLLER_DIR / "train_rl_actor_from_traces.py"
SERL_TRAIN_SCRIPT = ROOT / "scripts" / "serl_e2e_train.py"
sys.path.insert(0, str(CONTROLLER_DIR))
try:
    from starter_controller import E2E_INPUT_DIM, EndToEndActorPolicy
except Exception:
    E2E_INPUT_DIM = 35

    class EndToEndActorPolicy:
        RUNTIME_WEIGHTS_FILENAME = "e2e_rl_policy_weights.npz"


RUNTIME_WEIGHTS_PATH = CONTROLLER_DIR / EndToEndActorPolicy.RUNTIME_WEIGHTS_FILENAME
TRACE_DIR = ROOT / "tmp" / "rl_traces"
CHECKPOINT_DIR = ROOT / "tmp" / "rl_checkpoints"
EVAL_DIR = ROOT / "tmp" / "rl_eval"
LOCALIZATION_METRICS_DIR = ROOT / "tmp" / "localization_metrics"
HIGH_INFO_LANDMARK_BINS = {"goal_structural", "2plus_mixed"}
DEFAULT_WORLD = ROOT / "final_project" / "worlds" / "soccer_solo.wbt"
DEEPBOTS_WORLD = ROOT / "final_project" / "worlds" / "soccer_solo_deepbots.wbt"
DEEPBOTS_CONTROLLER = ROOT / "final_project" / "controllers" / "rl_training_controller" / "rl_training_controller.py"
DEEPBOTS_DEFAULT_MAX_EPISODE_STEPS = 6000


EXPECTED_SHAPES = {
    "w1": (64, E2E_INPUT_DIM),
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
        cmd.extend(["--batch", "--no-rendering"])
    cmd.extend(["--stdout", "--stderr"])
    if mode:
        cmd.append(f"--mode={mode}")
    cmd.append(str(world_path))
    return cmd


def _ensure_deepbots_runtime_ini(env=None):
    controller_dir = DEEPBOTS_CONTROLLER.parent
    env = env or os.environ
    python_command = Path(env.get("WEBOTS_CONTROLLER_PYTHON", sys.executable)).expanduser().absolute()
    content = f"[python]\nCOMMAND = {python_command}\n"
    written = []
    for filename in ("runtime.ini", "config.ini"):
        path = controller_dir / filename
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def _launch_webots(args, env, world):
    webots_bin = _find_webots_binary(getattr(args, "webots_bin", None))
    if webots_bin is None:
        raise SystemExit(
            "Could not find a Webots executable. Pass --webots-bin or set WEBOTS_BIN."
        )

    world_path = _resolve_world_path(world)
    if not world_path.exists():
        raise SystemExit(f"World file not found: {world_path}")
    if "RL_DEEPBOTS_MODE" in env:
        runtime_paths = _ensure_deepbots_runtime_ini(env)
        print(f"Deepbots Python config: {', '.join(str(path) for path in runtime_paths)}")

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


def _trace_episode_key(path, row):
    info = row.get("info") or {}
    episode = row.get("episode", info.get("episode_index", 0))
    return str(path), episode


def _row_scored_correct_goal(row):
    info = row.get("info") or {}
    if info.get("correct_goal"):
        return True

    ball = row.get("actual_ball") or row.get("estimated_ball")
    if ball is None or len(ball) < 2:
        return False
    return float(ball[0]) > 4.5 and abs(float(ball[1])) <= 0.8


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
    import torch

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


def _sample_submission_ball_positions(args):
    if getattr(args, "fixed_ball_x", None) is not None or getattr(args, "fixed_ball_y", None) is not None:
        if args.fixed_ball_x is None or args.fixed_ball_y is None:
            raise SystemExit("--fixed-ball-x and --fixed-ball-y must be provided together.")
        return [
            (round(float(args.fixed_ball_x), 4), round(float(args.fixed_ball_y), 4))
            for _ in range(args.episodes)
        ]

    rng = np.random.default_rng(args.seed)
    positions = []
    max_attempts = max(200, args.episodes * 50)
    for _ in range(max_attempts):
        ball_x = float(rng.uniform(args.x_min, args.x_max))
        ball_y = float(rng.uniform(args.y_min, args.y_max))
        if math.hypot(ball_x + 1.0, ball_y) < args.min_robot_dist:
            continue
        positions.append((round(ball_x, 4), round(ball_y, 4)))
        if len(positions) >= args.episodes:
            break
    if len(positions) < args.episodes:
        raise SystemExit("Could not sample enough valid ball positions for submission-eval.")
    return positions


def cmd_submission_eval(args):
    eval_dir = _ensure_dir(args.eval_dir)
    trace_dir = _ensure_dir(args.trace_dir)
    run_name = _normalize_name(args.name or _timestamp())
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else eval_dir / f"{run_name}_submission_summary.json"
    )
    positions = _sample_submission_ball_positions(args)
    results = []

    for episode_index, (ball_x, ball_y) in enumerate(positions, start=1):
        trace_path = trace_dir / f"{run_name}_submission_ep{episode_index:02d}.jsonl"
        summary_path = eval_dir / f"{run_name}_submission_ep{episode_index:02d}.json"
        if trace_path.exists():
            trace_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

        env = os.environ.copy()
        env["RL_TRACE_PATH"] = str(trace_path)
        env["WEBOTS_LIVE_VISUALIZER"] = "1" if args.visualizer else "0"
        env["WEBOTS_SUBMISSION_BALL_X"] = str(ball_x)
        env["WEBOTS_SUBMISSION_BALL_Y"] = str(ball_y)
        env["WEBOTS_SUBMISSION_MAX_STEPS"] = str(args.max_steps)
        env["WEBOTS_SUBMISSION_SUMMARY_PATH"] = str(summary_path)
        env["WEBOTS_SUBMISSION_LABEL"] = f"{run_name}_ep{episode_index:02d}"
        if args.weights:
            env["RL_POLICY_WEIGHTS_PATH"] = str(Path(args.weights).expanduser().resolve())
        if args.residual_blend is not None:
            env["RL_RESIDUAL_BLEND"] = str(args.residual_blend)

        print(
            f"[submission-eval] episode {episode_index}/{args.episodes} "
            f"ball=({ball_x:.3f}, {ball_y:.3f})"
        )
        rc = _launch_webots(args, env, args.world)
        if rc != 0:
            result = {
                "episode": episode_index,
                "ball_init": [ball_x, ball_y],
                "outcome": "launch_error",
                "returncode": rc,
                "trace_path": str(trace_path),
            }
        elif not summary_path.exists():
            result = {
                "episode": episode_index,
                "ball_init": [ball_x, ball_y],
                "outcome": "missing_summary",
                "returncode": rc,
                "trace_path": str(trace_path),
            }
        else:
            with open(summary_path, "r", encoding="utf-8") as handle:
                result = json.load(handle)
            result["episode"] = episode_index
            result["returncode"] = rc
            result["trace_path"] = str(trace_path)
            result["summary_path"] = str(summary_path)

        results.append(result)
        print(
            "[submission-eval] "
            f"outcome={result['outcome']} "
            f"step={result.get('step', -1)} "
            f"got_to_ball={result.get('got_to_ball', False)}"
        )

    successes = sum(result.get("outcome") == "correct_goal" for result in results)
    wrong_goals = sum(result.get("outcome") == "wrong_goal" for result in results)
    timeouts = sum(result.get("outcome") == "timeout" for result in results)
    got_to_ball = sum(bool(result.get("got_to_ball")) for result in results)
    summary = {
        "run_name": run_name,
        "episodes_requested": args.episodes,
        "episodes_completed": len(results),
        "seed": args.seed,
        "world": str(_resolve_world_path(args.world)),
        "max_steps": args.max_steps,
        "successes": successes,
        "wrong_goals": wrong_goals,
        "timeouts": timeouts,
        "launch_errors": sum(
            result.get("outcome") in {"launch_error", "missing_summary"} for result in results
        ),
        "success_rate": successes / len(results) if results else 0.0,
        "get_to_ball_rate": got_to_ball / len(results) if results else 0.0,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "Submission eval: "
        f"success_rate={summary['success_rate']:.3f} "
        f"successes={successes}/{len(results)} "
        f"wrong={wrong_goals} "
        f"timeouts={timeouts} "
        f"get_to_ball={got_to_ball}/{len(results)}"
    )
    print(f"Summary: {output_path}")
    return 0 if summary["launch_errors"] == 0 else 1


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
        "--weighting",
        args.weighting,
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


def cmd_serl_train(args):
    traces = args.traces or [str(args.trace_dir / "*teleop*.jsonl")]
    trace_files = _collect_trace_files(traces, args.trace_dir)
    if not trace_files and not args.dry_run:
        raise SystemExit("No trace files found.")

    output_path = Path(args.output).expanduser().resolve() if args.output else RUNTIME_WEIGHTS_PATH
    python_bin = str(Path(args.python).expanduser()) if args.python else sys.executable
    cmd = [
        python_bin,
        str(SERL_TRAIN_SCRIPT),
        *[str(path) for path in (trace_files or traces)],
        "--serl-root",
        str(Path(args.serl_root).expanduser()),
        "--output",
        str(output_path),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--utd-ratio",
        str(args.utd_ratio),
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
    ]
    if args.success_only:
        cmd.append("--success-only")

    print(f"Running SERL E2E training: {' '.join(cmd)}")
    if args.dry_run:
        return 0
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"SERL E2E runtime weights: {output_path}")
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
    controller_python = getattr(args, "controller_python", None) or getattr(args, "python", None)
    controller_python_override = bool(controller_python)
    if controller_python:
        controller_python = str(Path(controller_python).expanduser().resolve())
        env["WEBOTS_CONTROLLER_PYTHON"] = controller_python
    else:
        controller_python = sys.executable
    python_bin_dir = Path(controller_python).resolve().parent
    env["PATH"] = f"{python_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    site_paths = [] if controller_python_override else [path for path in site.getsitepackages() if Path(path).exists()]
    if site_paths:
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(site_paths + ([existing_pythonpath] if existing_pythonpath else []))
    env["PYTHONUNBUFFERED"] = "1"
    env["RL_DEEPBOTS_PHASE"] = str(args.phase)
    env["RL_DEEPBOTS_MAX_STEPS"] = str(args.max_episode_steps)
    env["RL_DEEPBOTS_SEED"] = str(args.seed)
    if trace_path is not None:
        env["RL_DEEPBOTS_TRACE_PATH"] = str(trace_path)
        env["RL_TRACE_PATH"] = str(trace_path)
    if getattr(args, "weights", None):
        env["RL_POLICY_WEIGHTS_PATH"] = str(Path(args.weights).expanduser().resolve())
    if getattr(args, "actor_source", None):
        env["RL_DEEPBOTS_ACTOR_SOURCE"] = args.actor_source
    if getattr(args, "warm_start", None):
        env["RL_DEEPBOTS_WARM_START"] = args.warm_start
    if getattr(args, "warm_start_weights", None):
        env["RL_DEEPBOTS_WARM_START_WEIGHTS"] = str(Path(args.warm_start_weights).expanduser().resolve())
    if getattr(args, "sac_log_std_init", None) is not None:
        env["RL_DEEPBOTS_SAC_LOG_STD_INIT"] = str(args.sac_log_std_init)
    if getattr(args, "demo_traces", None):
        demo_paths = [str(Path(path).expanduser().resolve()) for path in args.demo_traces]
        env["RL_DEEPBOTS_DEMO_TRACES"] = os.pathsep.join(demo_paths)
    if getattr(args, "demo_limit", None) is not None:
        env["RL_DEEPBOTS_DEMO_LIMIT"] = str(args.demo_limit)
    if getattr(args, "ent_coef", None) is not None:
        env["RL_DEEPBOTS_ENT_COEF"] = str(args.ent_coef)
    if getattr(args, "gradient_steps", None) is not None:
        env["RL_DEEPBOTS_GRADIENT_STEPS"] = str(args.gradient_steps)
    if getattr(args, "terminate_out_of_play", False):
        env["RL_DEEPBOTS_TERMINATE_OUT_OF_PLAY"] = "1"
    return env


def cmd_deepbots_record(args):
    if args.policy == "sac" and not args.model:
        raise SystemExit("--policy sac requires --model pointing to an SB3 SAC .zip file.")

    run_name = _normalize_name(args.name or _timestamp())
    trace_path = _deepbots_trace_path(args, run_name=run_name)
    env = _base_deepbots_env(args, trace_path=trace_path)
    if args.policy == "sac":
        mode = "sac_eval"
    elif args.policy == "bootstrap":
        mode = "actor"
        env["RL_DEEPBOTS_ACTOR_SOURCE"] = "bootstrap"
    else:
        mode = args.policy
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
            "RL_DEEPBOTS_ACTOR_SOURCE",
            "RL_DEEPBOTS_MODEL_PATH",
        ),
    )
    return 0


def cmd_deepbots_teleop(args):
    run_name = _normalize_name(args.name or _timestamp())
    trace_path = _deepbots_trace_path(args, suffix="teleop", run_name=run_name)
    env = _base_deepbots_env(args, trace_path=trace_path)
    env["RL_DEEPBOTS_MODE"] = "teleop"

    print("Deepbots mode: teleop")
    print(f"Trace file: {trace_path}")
    print("Controls: W/Up forward, S/Down reverse, A/Left, D/Right, Space stop, R reset, Q quit.")

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
        ),
    )
    return 0


def cmd_deepbots_eval(args):
    eval_dir = _ensure_dir(args.eval_dir)
    run_name = _normalize_name(args.name or _timestamp())
    output_path = Path(args.output).expanduser().resolve() if args.output else eval_dir / f"{run_name}_summary.json"
    trace_path = _deepbots_trace_path(args, suffix="eval", run_name=run_name) if args.trace else None

    if args.policy == "sac" and not args.model:
        raise SystemExit("--policy sac requires --model pointing to an SB3 SAC .zip file.")

    env = _base_deepbots_env(args, trace_path=trace_path)
    env["RL_DEEPBOTS_MODE"] = "eval"
    env["RL_DEEPBOTS_EVAL_POLICY"] = args.policy
    env["RL_DEEPBOTS_EVAL_EPISODES"] = str(args.episodes)
    env["RL_DEEPBOTS_EVAL_OUTPUT"] = str(output_path)
    if args.policy == "bootstrap":
        env["RL_DEEPBOTS_EVAL_POLICY"] = "bootstrap"
        env["RL_DEEPBOTS_ACTOR_SOURCE"] = "bootstrap"
    if args.model:
        env["RL_DEEPBOTS_MODEL_PATH"] = str(Path(args.model).expanduser().resolve())

    print("Deepbots mode: eval")
    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.episodes}")
    print(f"Eval summary: {output_path}")
    if trace_path is not None:
        print(f"Eval trace: {trace_path}")

    if args.launch_webots:
        rc = _launch_webots(args, env, args.world)
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            print(
                "Eval result: "
                f"success_rate={summary.get('success_rate', 0.0):.3f} "
                f"successes={summary.get('successes', 0)}/{summary.get('episodes_completed', 0)} "
                f"wrong={summary.get('wrong_goals', 0)} "
                f"timeouts={summary.get('timeouts', 0)} "
                f"mean_steps={summary.get('mean_episode_steps', 0.0):.1f}"
            )
        return rc

    _print_env_exports(
        env,
        (
            "RL_DEEPBOTS_MODE",
            "RL_DEEPBOTS_EVAL_POLICY",
            "RL_DEEPBOTS_EVAL_EPISODES",
            "RL_DEEPBOTS_EVAL_OUTPUT",
            "RL_DEEPBOTS_TRACE_PATH",
            "RL_TRACE_PATH",
            "RL_DEEPBOTS_PHASE",
            "RL_DEEPBOTS_MAX_STEPS",
            "RL_DEEPBOTS_SEED",
            "RL_POLICY_WEIGHTS_PATH",
            "RL_DEEPBOTS_ACTOR_SOURCE",
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
    learning_starts = args.learning_starts
    if learning_starts is None:
        has_warm_start = args.warm_start != "none" or args.warm_start_weights is not None
        learning_starts = 0 if has_warm_start else 1000
    env["RL_DEEPBOTS_LEARNING_STARTS"] = str(learning_starts)
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
    if "RL_DEEPBOTS_WARM_START" in env:
        print(f"SAC warm start: {env['RL_DEEPBOTS_WARM_START']}")
    if "RL_DEEPBOTS_WARM_START_WEIGHTS" in env:
        print(f"SAC warm start weights: {env['RL_DEEPBOTS_WARM_START_WEIGHTS']}")
    print(f"SAC learning starts: {learning_starts}")

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
            "RL_DEEPBOTS_WARM_START",
            "RL_DEEPBOTS_WARM_START_WEIGHTS",
            "RL_DEEPBOTS_SAC_LOG_STD_INIT",
            "RL_DEEPBOTS_DEMO_TRACES",
            "RL_DEEPBOTS_DEMO_LIMIT",
            "RL_DEEPBOTS_ENT_COEF",
            "RL_DEEPBOTS_GRADIENT_STEPS",
            "RL_DEEPBOTS_TERMINATE_OUT_OF_PLAY",
            "RL_DEEPBOTS_LEARNING_STARTS",
            "RL_DEEPBOTS_BATCH_SIZE",
            "RL_DEEPBOTS_LR",
            "RL_DEEPBOTS_GAMMA",
            "RL_DEEPBOTS_TENSORBOARD_DIR",
        ),
    )
    return 0


def cmd_deepbots_serl_train(args):
    run_name = _normalize_name(args.name or _timestamp())
    trace_path = _deepbots_trace_path(args, suffix="serl_train", run_name=run_name)
    output_path = Path(args.output).expanduser().resolve() if args.output else RUNTIME_WEIGHTS_PATH
    if not args.demo_traces:
        args.demo_traces = [args.trace_dir / "*teleop*.jsonl"]

    env = _base_deepbots_env(args, trace_path=trace_path)
    env["RL_DEEPBOTS_MODE"] = "serl_train"
    env["RL_DEEPBOTS_TOTAL_STEPS"] = str(args.timesteps)
    env["RL_DEEPBOTS_EXPORT_PATH"] = str(output_path)
    env["SERL_ROOT"] = str(Path(args.serl_root).expanduser())
    env["RL_DEEPBOTS_BATCH_SIZE"] = str(args.batch_size)
    env["RL_DEEPBOTS_UTD_RATIO"] = str(args.utd_ratio)
    env["RL_DEEPBOTS_LR"] = str(args.lr)
    env["RL_DEEPBOTS_GAMMA"] = str(args.gamma)
    env["RL_DEEPBOTS_BUFFER_SIZE"] = str(args.capacity)
    env["RL_DEEPBOTS_CRITIC_ENSEMBLE_SIZE"] = str(args.critic_ensemble_size)
    env["RL_DEEPBOTS_CRITIC_SUBSAMPLE_SIZE"] = str(args.critic_subsample_size)
    env["RL_DEEPBOTS_SERL_PRETRAIN_STEPS"] = str(args.pretrain_steps)
    env["RL_DEEPBOTS_UPDATE_EVERY"] = str(args.update_every)
    env["RL_DEEPBOTS_UPDATES_PER_STEP"] = str(args.updates_per_step)
    env["RL_DEEPBOTS_RANDOM_STEPS"] = str(args.random_steps)
    env["RL_DEEPBOTS_LOG_PERIOD"] = str(args.log_period)
    if args.learning_starts is not None:
        env["RL_DEEPBOTS_LEARNING_STARTS"] = str(args.learning_starts)
    if args.success_only:
        env["RL_DEEPBOTS_DEMO_SUCCESS_ONLY"] = "1"
    if args.allow_empty_demos:
        env["RL_DEEPBOTS_ALLOW_EMPTY_DEMOS"] = "1"

    print("Deepbots mode: serl_train")
    print(f"Transition trace: {trace_path}")
    print(f"Runtime export: {output_path}")
    if "WEBOTS_CONTROLLER_PYTHON" in env:
        print(f"Webots controller Python: {env['WEBOTS_CONTROLLER_PYTHON']}")
    if "RL_DEEPBOTS_DEMO_TRACES" in env:
        print(f"Teleop demo traces: {env['RL_DEEPBOTS_DEMO_TRACES']}")
    print(f"SERL root: {env['SERL_ROOT']}")

    if args.launch_webots:
        return _launch_webots(args, env, args.world)

    _print_env_exports(
        env,
        (
            "WEBOTS_CONTROLLER_PYTHON",
            "SERL_ROOT",
            "RL_DEEPBOTS_MODE",
            "RL_DEEPBOTS_TRACE_PATH",
            "RL_TRACE_PATH",
            "RL_DEEPBOTS_PHASE",
            "RL_DEEPBOTS_MAX_STEPS",
            "RL_DEEPBOTS_SEED",
            "RL_DEEPBOTS_TOTAL_STEPS",
            "RL_DEEPBOTS_EXPORT_PATH",
            "RL_DEEPBOTS_DEMO_TRACES",
            "RL_DEEPBOTS_DEMO_SUCCESS_ONLY",
            "RL_DEEPBOTS_ALLOW_EMPTY_DEMOS",
            "RL_DEEPBOTS_BATCH_SIZE",
            "RL_DEEPBOTS_UTD_RATIO",
            "RL_DEEPBOTS_LR",
            "RL_DEEPBOTS_GAMMA",
            "RL_DEEPBOTS_BUFFER_SIZE",
            "RL_DEEPBOTS_CRITIC_ENSEMBLE_SIZE",
            "RL_DEEPBOTS_CRITIC_SUBSAMPLE_SIZE",
            "RL_DEEPBOTS_SERL_PRETRAIN_STEPS",
            "RL_DEEPBOTS_LEARNING_STARTS",
            "RL_DEEPBOTS_RANDOM_STEPS",
            "RL_DEEPBOTS_UPDATE_EVERY",
            "RL_DEEPBOTS_UPDATES_PER_STEP",
            "RL_DEEPBOTS_LOG_PERIOD",
            "RL_DEEPBOTS_TERMINATE_OUT_OF_PLAY",
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


def cmd_filter_success(args):
    trace_dir = _ensure_dir(args.trace_dir)
    trace_files = _collect_trace_files(args.traces, trace_dir)
    if not trace_files:
        raise SystemExit("No trace files found.")

    run_name = _normalize_name(args.name or _timestamp())
    output_path = Path(args.output).expanduser().resolve() if args.output else trace_dir / f"{run_name}_success.jsonl"

    episodes = {}
    order = []
    total_rows = 0
    for path in trace_files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total_rows += 1
                key = _trace_episode_key(path, row)
                if key not in episodes:
                    episodes[key] = {"rows": [], "success": False}
                    order.append(key)
                episodes[key]["rows"].append(row)
                if _row_scored_correct_goal(row):
                    episodes[key]["success"] = True

    selected = [episodes[key] for key in order if episodes[key]["success"]]
    if not selected:
        raise SystemExit("No successful correct-goal episodes found in the provided traces.")

    _ensure_dir(output_path.parent)
    written_rows = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for episode in selected:
            for row in episode["rows"]:
                handle.write(json.dumps(row) + "\n")
                written_rows += 1

    print(f"Input traces: {len(trace_files)}")
    print(f"Input rows: {total_rows}")
    print(f"Successful episodes: {len(selected)} / {len(episodes)}")
    print(f"Output rows: {written_rows}")
    print(f"Success trace: {output_path}")
    return 0


def summarize_trace_files(trace_files):
    summaries = []
    for path in trace_files:
        episodes = {}
        total_rows = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_rows += 1
            key = _trace_episode_key(path, row)
            info = row.get("info") or {}
            episode = episodes.setdefault(
                key,
                {"rows": 0, "success": False, "wrong": False, "max_step": 0},
            )
            episode["rows"] += 1
            episode["success"] = episode["success"] or _row_scored_correct_goal(row)
            episode["wrong"] = episode["wrong"] or bool(info.get("wrong_goal"))
            episode["max_step"] = max(
                episode["max_step"],
                int(row.get("episode_step", info.get("episode_step", 0)) or 0),
            )

        if not episodes:
            summaries.append(
                {
                    "path": str(path),
                    "rows": 0,
                    "episodes": 0,
                    "successes": 0,
                    "wrong_goals": 0,
                    "success_rate": 0.0,
                    "mean_episode_steps": 0.0,
                    "max_episode_steps": 0,
                }
            )
            continue

        successes = sum(episode["success"] for episode in episodes.values())
        wrong_goals = sum(episode["wrong"] for episode in episodes.values())
        lengths = [
            episode["max_step"] or episode["rows"]
            for episode in episodes.values()
        ]
        summaries.append(
            {
                "path": str(path),
                "rows": total_rows,
                "episodes": len(episodes),
                "successes": successes,
                "wrong_goals": wrong_goals,
                "success_rate": successes / len(episodes),
                "mean_episode_steps": float(np.mean(lengths)),
                "max_episode_steps": int(max(lengths)),
            }
        )
    return summaries


def cmd_trace_summary(args):
    trace_dir = _ensure_dir(args.trace_dir)
    trace_files = _collect_trace_files(args.traces, trace_dir)
    if not trace_files:
        raise SystemExit("No trace files found.")

    summaries = summarize_trace_files(trace_files)
    for summary in summaries:
        print(
            f"{Path(summary['path']).name}: "
            f"rows={summary['rows']} "
            f"episodes={summary['episodes']} "
            f"success={summary['successes']} "
            f"wrong={summary['wrong_goals']} "
            f"success_rate={summary['success_rate']:.3f} "
            f"mean_steps={summary['mean_episode_steps']:.1f} "
            f"max_steps={summary['max_episode_steps']}"
        )
    return 0


def _quantile(values, percentile):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    index = int(round((len(values) - 1) * percentile))
    index = max(0, min(len(values) - 1, index))
    return values[index]


def _metric_summary(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values),
    }


def _angle_error(a, b):
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _pose_error_from_row(row, pose_key):
    actual = row.get("actual_pose")
    pose = row.get(pose_key)
    if actual is None or pose is None:
        return None, None
    return (
        math.hypot(float(pose[0]) - float(actual[0]), float(pose[1]) - float(actual[1])),
        _angle_error(pose[2], actual[2]),
    )


def _pose_jump_from_rows(prev_row, row, pose_key):
    if prev_row is None:
        return None, None
    prev_pose = prev_row.get(pose_key)
    pose = row.get(pose_key)
    if prev_pose is None or pose is None:
        return None, None
    return (
        math.hypot(float(pose[0]) - float(prev_pose[0]), float(pose[1]) - float(prev_pose[1])),
        _angle_error(pose[2], prev_pose[2]),
    )


def _localization_from_row(row, prev_row=None):
    localization = dict(row.get("localization") or {})
    fused = dict(localization.get("fused") or {})
    localizer = dict(localization.get("localizer") or {})
    odom = dict(localization.get("odom") or {})

    fused_pos, fused_heading = _pose_error_from_row(row, "fused_pose" if row.get("fused_pose") is not None else "estimated_pose")
    localizer_pos, localizer_heading = _pose_error_from_row(row, "localizer_pose")
    odom_pos, odom_heading = _pose_error_from_row(row, "odom_pose")
    fused_jump, fused_heading_jump = _pose_jump_from_rows(prev_row, row, "fused_pose" if row.get("fused_pose") is not None else "estimated_pose")

    fused.setdefault("pos_error", fused_pos)
    fused.setdefault("heading_error", fused_heading)
    fused.setdefault("heading_error_deg", None if fused_heading is None else math.degrees(fused_heading))
    fused.setdefault("pos_jump", fused_jump)
    fused.setdefault("heading_jump", fused_heading_jump)
    fused.setdefault("heading_jump_deg", None if fused_heading_jump is None else math.degrees(fused_heading_jump))
    localizer.setdefault("pos_error", localizer_pos)
    localizer.setdefault("heading_error", localizer_heading)
    localizer.setdefault("heading_error_deg", None if localizer_heading is None else math.degrees(localizer_heading))
    odom.setdefault("pos_error", odom_pos)
    odom.setdefault("heading_error", odom_heading)
    odom.setdefault("heading_error_deg", None if odom_heading is None else math.degrees(odom_heading))

    if "landmark_bin" not in localization:
        goal_count = int(localization.get("goal_count", row.get("goal_count", 0)) or 0)
        corner_count = int(localization.get("corner_count", row.get("corner_count", 0)) or 0)
        cross_count = int(localization.get("cross_count", row.get("cross_count", 0)) or 0)
        center_count = int(localization.get("center_count", row.get("center_count", 0)) or 0)
        structure_count = corner_count + cross_count + center_count
        total_count = goal_count + structure_count
        if total_count == 0:
            landmark_bin = "0_landmarks"
        elif structure_count == 0 and goal_count == 1:
            landmark_bin = "1_goal_only"
        elif structure_count == 0:
            landmark_bin = "goal_only_multi"
        elif structure_count == 1 and goal_count == 0:
            landmark_bin = "1_structural"
        elif structure_count >= 1 and goal_count >= 1:
            landmark_bin = "goal_structural"
        else:
            landmark_bin = "2plus_mixed"
        localization.update(
            {
                "goal_count": goal_count,
                "corner_count": corner_count,
                "cross_count": cross_count,
                "center_count": center_count,
                "structure_count": structure_count,
                "total_count": total_count,
                "landmark_bin": landmark_bin,
            }
        )

    localization["fused"] = fused
    localization["localizer"] = localizer
    localization["odom"] = odom
    localization.setdefault("pose_correction_trust", row.get("pose_correction_trust"))
    localization.setdefault("localizer_confidence", row.get("localizer_confidence"))
    return localization


def _summarize_localization_rows(rows):
    by_bin = {}
    high_info = {
        "fused_pos_error": [],
        "fused_heading_error_deg": [],
        "fused_pos_jump": [],
        "localizer_pos_error": [],
        "odom_pos_error": [],
        "pose_correction_trust": [],
        "localizer_confidence": [],
    }
    overall = {
        "fused_pos_error": [],
        "fused_heading_error_deg": [],
        "fused_pos_jump": [],
        "localizer_pos_error": [],
        "odom_pos_error": [],
        "pose_correction_trust": [],
        "localizer_confidence": [],
    }
    calibration_violations = []
    large_jump_rows = []
    first_large_error = None
    prev_row = None
    prev_trace_path = None

    for row in rows:
        trace_path = row.get("_trace_path")
        if trace_path != prev_trace_path:
            prev_row = None
            prev_trace_path = trace_path
        localization = _localization_from_row(row, prev_row)
        prev_row = row
        bin_name = localization.get("landmark_bin", "unknown")
        bucket = by_bin.setdefault(
            bin_name,
            {
                "fused_pos_error": [],
                "fused_heading_error_deg": [],
                "fused_pos_jump": [],
                "localizer_pos_error": [],
                "odom_pos_error": [],
                "pose_correction_trust": [],
                "localizer_confidence": [],
            },
        )
        fused = localization.get("fused", {})
        localizer = localization.get("localizer", {})
        odom = localization.get("odom", {})
        values = {
            "fused_pos_error": fused.get("pos_error"),
            "fused_heading_error_deg": fused.get("heading_error_deg"),
            "fused_pos_jump": fused.get("pos_jump"),
            "localizer_pos_error": localizer.get("pos_error"),
            "odom_pos_error": odom.get("pos_error"),
            "pose_correction_trust": localization.get("pose_correction_trust"),
            "localizer_confidence": localization.get("localizer_confidence"),
        }
        for key, value in values.items():
            if value is not None:
                overall[key].append(value)
                bucket[key].append(value)
                if bin_name in HIGH_INFO_LANDMARK_BINS:
                    high_info[key].append(value)

        fused_error = values["fused_pos_error"]
        trust = values["pose_correction_trust"]
        jump = values["fused_pos_jump"]
        if first_large_error is None and fused_error is not None and fused_error > 0.5:
            first_large_error = {
                "trace_path": row.get("_trace_path"),
                "step": row.get("step"),
                "fused_pos_error": fused_error,
                "landmark_bin": bin_name,
                "state": row.get("state"),
                "rl_submode": row.get("rl_submode"),
            }
        if fused_error is not None and trust is not None and fused_error > 0.5 and trust >= 0.075:
            calibration_violations.append(
                {
                    "trace_path": row.get("_trace_path"),
                    "step": row.get("step"),
                    "fused_pos_error": fused_error,
                    "trust": trust,
                    "landmark_bin": bin_name,
                    "state": row.get("state"),
                }
            )
        if jump is not None and jump > 0.375:
            large_jump_rows.append(
                {
                    "trace_path": row.get("_trace_path"),
                    "step": row.get("step"),
                    "fused_pos_jump": jump,
                    "fused_pos_error": fused_error,
                    "landmark_bin": bin_name,
                    "state": row.get("state"),
                }
            )

    def summarize_bucket(bucket):
        return {
            key: _metric_summary(values)
            for key, values in bucket.items()
        }

    summary = {
        "rows": len(rows),
        "overall": summarize_bucket(overall),
        "high_info": summarize_bucket(high_info),
        "high_info_bins": sorted(HIGH_INFO_LANDMARK_BINS),
        "by_landmark_bin": {
            bin_name: summarize_bucket(bucket)
            for bin_name, bucket in sorted(by_bin.items())
        },
        "failure_windows": {
            "first_fused_error_gt_0_5m": first_large_error,
            "top_fused_errors": [],
            "top_fused_jumps": sorted(large_jump_rows, key=lambda item: item["fused_pos_jump"], reverse=True)[:10],
            "calibration_violations": calibration_violations[:20],
            "calibration_violation_count": len(calibration_violations),
        },
    }

    ranked_errors = []
    prev_row = None
    prev_trace_path = None
    for row in rows:
        trace_path = row.get("_trace_path")
        if trace_path != prev_trace_path:
            prev_row = None
            prev_trace_path = trace_path
        localization = _localization_from_row(row, prev_row)
        prev_row = row
        fused = localization.get("fused", {})
        fused_error = fused.get("pos_error")
        if fused_error is not None:
            ranked_errors.append(
                {
                    "trace_path": row.get("_trace_path"),
                    "step": row.get("step"),
                    "fused_pos_error": fused_error,
                    "fused_pos_jump": fused.get("pos_jump"),
                    "landmark_bin": localization.get("landmark_bin"),
                    "trust": localization.get("pose_correction_trust"),
                    "confidence": localization.get("localizer_confidence"),
                    "state": row.get("state"),
                    "rl_submode": row.get("rl_submode"),
                }
            )
    summary["failure_windows"]["top_fused_errors"] = sorted(
        ranked_errors, key=lambda item: item["fused_pos_error"], reverse=True
    )[:10]

    high_info_error_rows = [
        item
        for item in ranked_errors
        if item.get("landmark_bin") in HIGH_INFO_LANDMARK_BINS
    ]
    high_info_large_jumps = [
        item
        for item in high_info_error_rows
        if (item.get("fused_pos_jump") or 0.0) > 0.225
    ]
    summary["failure_windows"]["top_high_info_fused_errors"] = sorted(
        high_info_error_rows, key=lambda item: item["fused_pos_error"], reverse=True
    )[:10]
    summary["failure_windows"]["high_info_large_jump_count"] = len(high_info_large_jumps)
    summary["failure_windows"]["high_info_large_jumps"] = sorted(
        high_info_large_jumps,
        key=lambda item: item["fused_pos_jump"] or 0.0,
        reverse=True,
    )[:10]

    gates = {
        "fused_pos_p50_lt_0_10": (summary["overall"]["fused_pos_error"]["p50"] or float("inf")) < 0.10,
        "fused_pos_p95_lt_0_275": (summary["overall"]["fused_pos_error"]["p95"] or float("inf")) < 0.275,
        "heading_p50_lt_4deg": (summary["overall"]["fused_heading_error_deg"]["p50"] or float("inf")) < 4.0,
        "heading_p95_lt_14deg": (summary["overall"]["fused_heading_error_deg"]["p95"] or float("inf")) < 14.0,
        "jump_p99_lt_0_125": (summary["overall"]["fused_pos_jump"]["p99"] or float("inf")) < 0.125,
        "no_jump_gt_0_375": len(large_jump_rows) == 0,
        "trust_low_when_error_gt_0_5m": len(calibration_violations) == 0,
        "high_info_fused_pos_p50_lt_0_04": (summary["high_info"]["fused_pos_error"]["p50"] or float("inf")) < 0.04,
        "high_info_fused_pos_p95_lt_0_15": (summary["high_info"]["fused_pos_error"]["p95"] or float("inf")) < 0.15,
        "high_info_fused_pos_p99_lt_0_21": (summary["high_info"]["fused_pos_error"]["p99"] or float("inf")) < 0.21,
        "high_info_heading_p95_lt_3deg": (summary["high_info"]["fused_heading_error_deg"]["p95"] or float("inf")) < 3.0,
        "high_info_heading_p99_lt_6deg": (summary["high_info"]["fused_heading_error_deg"]["p99"] or float("inf")) < 6.0,
        "high_info_jump_p99_lt_0_05": (summary["high_info"]["fused_pos_jump"]["p99"] or float("inf")) < 0.05,
        "high_info_no_jump_gt_0_225": len(high_info_large_jumps) == 0,
    }
    summary["acceptance_gates"] = gates
    summary["passed"] = all(gates.values())
    return summary


def cmd_localization_summary(args):
    trace_dir = _ensure_dir(args.trace_dir)
    trace_files = _collect_trace_files(args.traces, trace_dir)
    if not trace_files:
        raise SystemExit("No trace files found.")

    rows = []
    for path in trace_files:
        prev_row = None
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_trace_path"] = str(path)
            row["_localization_for_summary"] = _localization_from_row(row, prev_row)
            prev_row = row
            rows.append(row)

    summary = _summarize_localization_rows(rows)
    summary["trace_files"] = [str(path) for path in trace_files]
    summary["run_name"] = _normalize_name(args.name or _timestamp())

    metrics_dir = _ensure_dir(args.metrics_dir)
    output_path = Path(args.output).expanduser().resolve() if args.output else metrics_dir / f"{summary['run_name']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    aggregate_path = metrics_dir / "aggregate.json"
    aggregate = []
    if aggregate_path.exists():
        try:
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            aggregate = []
    aggregate_entry = {
        "run_name": summary["run_name"],
        "output": str(output_path),
        "passed": summary["passed"],
        "rows": summary["rows"],
        "fused_pos_p50": summary["overall"]["fused_pos_error"]["p50"],
        "fused_pos_p95": summary["overall"]["fused_pos_error"]["p95"],
        "heading_p95_deg": summary["overall"]["fused_heading_error_deg"]["p95"],
        "jump_p99": summary["overall"]["fused_pos_jump"]["p99"],
        "high_info_fused_pos_p95": summary["high_info"]["fused_pos_error"]["p95"],
        "high_info_fused_pos_p99": summary["high_info"]["fused_pos_error"]["p99"],
        "high_info_heading_p95_deg": summary["high_info"]["fused_heading_error_deg"]["p95"],
        "high_info_heading_p99_deg": summary["high_info"]["fused_heading_error_deg"]["p99"],
        "high_info_jump_p99": summary["high_info"]["fused_pos_jump"]["p99"],
        "high_info_large_jumps": summary["failure_windows"]["high_info_large_jump_count"],
        "calibration_violations": summary["failure_windows"]["calibration_violation_count"],
    }
    aggregate.append(aggregate_entry)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(
        "Localization summary: "
        f"passed={summary['passed']} "
        f"rows={summary['rows']} "
        f"fused_p50={aggregate_entry['fused_pos_p50']} "
        f"fused_p95={aggregate_entry['fused_pos_p95']} "
        f"heading_p95={aggregate_entry['heading_p95_deg']} "
        f"jump_p99={aggregate_entry['jump_p99']} "
        f"high_info_p99={aggregate_entry['high_info_fused_pos_p99']} "
        f"high_info_heading_p99={aggregate_entry['high_info_heading_p99_deg']} "
        f"high_info_jump_p99={aggregate_entry['high_info_jump_p99']} "
        f"calib_violations={aggregate_entry['calibration_violations']}"
    )
    print(f"Metrics: {output_path}")
    print(f"Aggregate: {aggregate_path}")
    return 0 if summary["passed"] or not args.fail_on_gate else 1


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
    recent_evals = sorted(EVAL_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:5] if EVAL_DIR.exists() else []

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

    print("\nRecent evals:")
    if recent_evals:
        for path in recent_evals:
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

    submission_eval = subparsers.add_parser(
        "submission-eval",
        help="Evaluate the submission controller stack on soccer_solo.wbt with randomized ball starts.",
    )
    submission_eval.add_argument("--name", help="Run name for traces and summary files.")
    submission_eval.add_argument("--episodes", type=int, default=20)
    submission_eval.add_argument("--seed", type=int, default=7)
    submission_eval.add_argument("--x-min", type=float, default=-1.25)
    submission_eval.add_argument("--x-max", type=float, default=1.50)
    submission_eval.add_argument("--y-min", type=float, default=-1.50)
    submission_eval.add_argument("--y-max", type=float, default=1.50)
    submission_eval.add_argument("--fixed-ball-x", type=float, help="Use one fixed ball x coordinate for every episode.")
    submission_eval.add_argument("--fixed-ball-y", type=float, help="Use one fixed ball y coordinate for every episode.")
    submission_eval.add_argument("--min-robot-dist", type=float, default=0.35, help="Reject ball starts too close to the robot's fixed start pose.")
    submission_eval.add_argument("--max-steps", type=int, default=18000)
    submission_eval.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    submission_eval.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    submission_eval.add_argument("--output", type=Path, help="Summary JSON path.")
    submission_eval.add_argument("--weights", type=Path, help="Optional runtime .npz actor weights file to use.")
    submission_eval.add_argument("--residual-blend", type=float, help="Optional RL_RESIDUAL_BLEND override.")
    submission_eval.add_argument("--visualizer", action="store_true", help="Keep the live visualizer enabled during evaluation.")
    submission_eval.add_argument("--world", default=str(DEFAULT_WORLD), help="Webots world to launch.")
    submission_eval.add_argument("--webots-bin", help="Path to the Webots executable.")
    submission_eval.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="fast", help="Webots run mode.")
    submission_eval.add_argument("--batch", action="store_true", default=True, help="Launch Webots in batch mode.")
    submission_eval.add_argument("--no-batch", dest="batch", action="store_false", help="Show the Webots GUI during evaluation.")
    submission_eval.add_argument("--dry-run", action="store_true", help="Print launch commands without running them.")
    submission_eval.set_defaults(func=cmd_submission_eval)

    deepbots_record = subparsers.add_parser("deepbots-record", help="Run the deepbots world with live policy actions and transition logging.")
    deepbots_record.add_argument("name", nargs="?", help="Trace run name.")
    deepbots_record.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    deepbots_record.add_argument("--policy", choices=("actor", "bootstrap", "expert", "random", "sac"), default="actor", help="Policy source for live actions.")
    deepbots_record.add_argument("--weights", type=Path, help="Runtime .npz actor weights for --policy actor.")
    deepbots_record.add_argument("--model", type=Path, help="SB3 SAC .zip model for --policy sac.")
    deepbots_record.add_argument("--phase", type=int, default=2, help="Curriculum phase used by the deepbots reset sampler.")
    deepbots_record.add_argument("--max-episode-steps", type=int, default=DEEPBOTS_DEFAULT_MAX_EPISODE_STEPS)
    deepbots_record.add_argument("--terminate-out-of-play", action="store_true", help="End an episode when the ball reaches the arena buffer/wall.")
    deepbots_record.add_argument("--seed", type=int, default=7)
    deepbots_record.add_argument("--world", default=str(DEEPBOTS_WORLD), help="Deepbots Webots world to launch.")
    deepbots_record.add_argument("--webots-bin", help="Path to the Webots executable.")
    deepbots_record.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="realtime", help="Webots run mode.")
    deepbots_record.add_argument("--batch", action="store_true", help="Launch Webots in batch mode.")
    deepbots_record.add_argument("--no-launch-webots", dest="launch_webots", action="store_false", help="Only print environment exports.")
    deepbots_record.add_argument("--dry-run", action="store_true", help="Print the launch command without running it.")
    deepbots_record.set_defaults(launch_webots=True)
    deepbots_record.set_defaults(func=cmd_deepbots_record)

    deepbots_teleop = subparsers.add_parser("deepbots-teleop", help="Collect E2E demos with keyboard teleoperation.")
    deepbots_teleop.add_argument("name", nargs="?", help="Trace run name.")
    deepbots_teleop.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    deepbots_teleop.add_argument("--phase", type=int, default=3, help="Curriculum phase used by the deepbots reset sampler.")
    deepbots_teleop.add_argument("--max-episode-steps", type=int, default=DEEPBOTS_DEFAULT_MAX_EPISODE_STEPS)
    deepbots_teleop.add_argument("--terminate-out-of-play", action="store_true", help="End an episode when the ball reaches the arena buffer/wall.")
    deepbots_teleop.add_argument("--seed", type=int, default=7)
    deepbots_teleop.add_argument("--python", help="Python interpreter for the Webots controller; should have deepbots/SERL dependencies.")
    deepbots_teleop.add_argument("--world", default=str(DEEPBOTS_WORLD), help="Deepbots Webots world to launch.")
    deepbots_teleop.add_argument("--webots-bin", help="Path to the Webots executable.")
    deepbots_teleop.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="realtime", help="Webots run mode.")
    deepbots_teleop.add_argument("--batch", action="store_true", default=False, help="Launch Webots in batch mode.")
    deepbots_teleop.add_argument("--no-launch-webots", dest="launch_webots", action="store_false", help="Only print environment exports.")
    deepbots_teleop.add_argument("--dry-run", action="store_true", help="Print the launch command without running it.")
    deepbots_teleop.set_defaults(launch_webots=True)
    deepbots_teleop.set_defaults(func=cmd_deepbots_teleop)

    deepbots_eval = subparsers.add_parser("deepbots-eval", help="Evaluate a policy for a finite number of deepbots episodes.")
    deepbots_eval.add_argument("--name", help="Run name for the eval summary.")
    deepbots_eval.add_argument("--policy", choices=("actor", "bootstrap", "expert", "sac"), default="actor")
    deepbots_eval.add_argument("--weights", type=Path, help="Runtime .npz actor weights for --policy actor.")
    deepbots_eval.add_argument("--model", type=Path, help="SB3 SAC .zip model for --policy sac.")
    deepbots_eval.add_argument("--episodes", type=int, default=50)
    deepbots_eval.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    deepbots_eval.add_argument("--output", type=Path, help="Eval summary JSON path.")
    deepbots_eval.add_argument("--trace", action="store_true", help="Also write per-step transition trace for eval episodes.")
    deepbots_eval.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    deepbots_eval.add_argument("--phase", type=int, default=2)
    deepbots_eval.add_argument("--max-episode-steps", type=int, default=DEEPBOTS_DEFAULT_MAX_EPISODE_STEPS)
    deepbots_eval.add_argument("--terminate-out-of-play", action="store_true", help="End an episode when the ball reaches the arena buffer/wall.")
    deepbots_eval.add_argument("--seed", type=int, default=7)
    deepbots_eval.add_argument("--world", default=str(DEEPBOTS_WORLD), help="Deepbots Webots world to launch.")
    deepbots_eval.add_argument("--webots-bin", help="Path to the Webots executable.")
    deepbots_eval.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="fast")
    deepbots_eval.add_argument("--batch", action="store_true", default=True, help="Launch Webots in batch mode.")
    deepbots_eval.add_argument("--no-batch", dest="batch", action="store_false", help="Show the Webots GUI during evaluation.")
    deepbots_eval.add_argument("--no-launch-webots", dest="launch_webots", action="store_false", help="Only print environment exports.")
    deepbots_eval.add_argument("--dry-run", action="store_true", help="Print the launch command without running it.")
    deepbots_eval.set_defaults(launch_webots=True)
    deepbots_eval.set_defaults(func=cmd_deepbots_eval)

    deepbots_train = subparsers.add_parser("deepbots-train", help="Train SAC inside Webots through the deepbots controller.")
    deepbots_train.add_argument("--name", help="Run name for the SAC model and trace.")
    deepbots_train.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    deepbots_train.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    deepbots_train.add_argument("--output", type=Path, help="SB3 SAC .zip output path.")
    deepbots_train.add_argument("--activate", action="store_true", help="Export the trained SAC deterministic actor to the runtime .npz weights.")
    deepbots_train.add_argument("--runtime-output", type=Path, help="Runtime .npz output path for --activate.")
    deepbots_train.add_argument("--timesteps", type=int, default=50000)
    deepbots_train.add_argument("--learning-starts", type=int, help="Defaults to 0 when SAC warm-start is enabled, otherwise 1000.")
    deepbots_train.add_argument("--batch-size", type=int, default=256)
    deepbots_train.add_argument("--lr", type=float, default=3e-4)
    deepbots_train.add_argument("--gamma", type=float, default=0.995)
    deepbots_train.add_argument("--warm-start", choices=("bootstrap", "runtime", "none"), default="bootstrap", help="Initialize SAC actor from the embedded/runtime actor before online learning.")
    deepbots_train.add_argument("--warm-start-weights", type=Path, help="Specific runtime .npz actor weights for SAC warm-start.")
    deepbots_train.add_argument("--sac-log-std-init", type=float, default=-1.2, help="Initial SAC log standard deviation when warm-starting the actor.")
    deepbots_train.add_argument("--demo-traces", nargs="*", type=Path, help="Expert/demo JSONL traces to prefill the SAC replay buffer.")
    deepbots_train.add_argument("--demo-limit", type=int, default=0, help="Maximum demo transitions to prefill; 0 means all.")
    deepbots_train.add_argument("--ent-coef", default=None, help="SAC entropy coefficient, e.g. auto, 0.02, or 0.005.")
    deepbots_train.add_argument("--gradient-steps", type=int, default=1, help="Gradient updates per environment step.")
    deepbots_train.add_argument("--phase", type=int, default=2)
    deepbots_train.add_argument("--max-episode-steps", type=int, default=DEEPBOTS_DEFAULT_MAX_EPISODE_STEPS)
    deepbots_train.add_argument("--terminate-out-of-play", action="store_true", help="End an episode when the ball reaches the arena buffer/wall.")
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

    deepbots_serl_train = subparsers.add_parser("deepbots-serl-train", help="Train SERL SAC online inside Webots with separate teleop demo and online replay buffers.")
    deepbots_serl_train.add_argument("demo_traces", nargs="*", type=Path, help="Teleop JSONL traces, directories, or globs. Default: tmp/rl_traces/*teleop*.jsonl")
    deepbots_serl_train.add_argument("--name", help="Run name for the SERL online training trace.")
    deepbots_serl_train.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    deepbots_serl_train.add_argument("--output", type=Path, help="Runtime .npz output path. Defaults to e2e_rl_policy_weights.npz.")
    deepbots_serl_train.add_argument("--python", help="Python interpreter for the Webots controller; must have SERL/JAX dependencies.")
    deepbots_serl_train.add_argument("--serl-root", type=Path, default=Path(os.environ.get("SERL_ROOT", "~/serl")).expanduser())
    deepbots_serl_train.add_argument("--timesteps", type=int, default=50000, help="Online Webots environment steps.")
    deepbots_serl_train.add_argument("--pretrain-steps", type=int, default=500, help="SERL updates on the separate demo buffer before online rollout.")
    deepbots_serl_train.add_argument("--learning-starts", type=int, help="First online step at which updates are allowed.")
    deepbots_serl_train.add_argument("--random-steps", type=int, default=0, help="Initial random online exploration steps.")
    deepbots_serl_train.add_argument("--batch-size", type=int, default=256)
    deepbots_serl_train.add_argument("--utd-ratio", type=int, default=8)
    deepbots_serl_train.add_argument("--updates-per-step", type=int, default=1)
    deepbots_serl_train.add_argument("--update-every", type=int, default=1)
    deepbots_serl_train.add_argument("--lr", type=float, default=3e-4)
    deepbots_serl_train.add_argument("--gamma", type=float, default=0.995)
    deepbots_serl_train.add_argument("--capacity", type=int, default=200000)
    deepbots_serl_train.add_argument("--critic-ensemble-size", type=int, default=10)
    deepbots_serl_train.add_argument("--critic-subsample-size", type=int, default=2)
    deepbots_serl_train.add_argument("--success-only", action="store_true", help="Prefill replay only from successful teleop episodes.")
    deepbots_serl_train.add_argument("--allow-empty-demos", action="store_true", help="Allow online SERL from scratch without teleop replay.")
    deepbots_serl_train.add_argument("--log-period", type=int, default=100)
    deepbots_serl_train.add_argument("--phase", type=int, default=3)
    deepbots_serl_train.add_argument("--max-episode-steps", type=int, default=DEEPBOTS_DEFAULT_MAX_EPISODE_STEPS)
    deepbots_serl_train.add_argument("--terminate-out-of-play", action="store_true", help="End an episode when the ball reaches the arena buffer/wall.")
    deepbots_serl_train.add_argument("--seed", type=int, default=7)
    deepbots_serl_train.add_argument("--world", default=str(DEEPBOTS_WORLD), help="Deepbots Webots world to launch.")
    deepbots_serl_train.add_argument("--webots-bin", help="Path to the Webots executable.")
    deepbots_serl_train.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="fast", help="Webots run mode.")
    deepbots_serl_train.add_argument("--batch", action="store_true", default=True, help="Launch Webots in batch/no-rendering mode.")
    deepbots_serl_train.add_argument("--no-batch", dest="batch", action="store_false", help="Show the Webots GUI during online SERL training.")
    deepbots_serl_train.add_argument("--no-launch-webots", dest="launch_webots", action="store_false", help="Only print environment exports.")
    deepbots_serl_train.add_argument("--dry-run", action="store_true", help="Print the launch command without running it.")
    deepbots_serl_train.set_defaults(launch_webots=True)
    deepbots_serl_train.set_defaults(func=cmd_deepbots_serl_train)

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
        sub.add_argument("--weighting", choices=("advantage", "uniform"), default="advantage")
        sub.add_argument("--device", default="cpu")
        sub.set_defaults(func=func)

    serl_train = subparsers.add_parser("serl-train", help="Offline demo-only SERL SAC update/export. For online RL use deepbots-serl-train.")
    serl_train.add_argument("traces", nargs="*", help="Teleop trace files, directories, or globs. Default: tmp/rl_traces/*teleop*.jsonl")
    serl_train.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    serl_train.add_argument("--serl-root", type=Path, default=Path(os.environ.get("SERL_ROOT", "~/serl")).expanduser())
    serl_train.add_argument("--python", default=os.environ.get("SERL_PYTHON"), help="Python interpreter with SERL/JAX dependencies. Defaults to the current interpreter.")
    serl_train.add_argument("--output", type=Path, help="Runtime .npz output path. Defaults to e2e_rl_policy_weights.npz.")
    serl_train.add_argument("--steps", type=int, default=8000)
    serl_train.add_argument("--batch-size", type=int, default=256)
    serl_train.add_argument("--utd-ratio", type=int, default=8)
    serl_train.add_argument("--lr", type=float, default=3e-4)
    serl_train.add_argument("--gamma", type=float, default=0.995)
    serl_train.add_argument("--seed", type=int, default=7)
    serl_train.add_argument("--capacity", type=int, default=200000)
    serl_train.add_argument("--critic-ensemble-size", type=int, default=10)
    serl_train.add_argument("--critic-subsample-size", type=int, default=2)
    serl_train.add_argument("--success-only", action="store_true")
    serl_train.add_argument("--log-period", type=int, default=100)
    serl_train.add_argument("--dry-run", action="store_true")
    serl_train.set_defaults(func=cmd_serl_train)

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

    filter_success = subparsers.add_parser("filter-success", help="Keep only trace episodes that scored on the correct goal.")
    filter_success.add_argument("traces", nargs="*", help="Trace files, directories, or glob patterns. Default: tmp/rl_traces/*.jsonl")
    filter_success.add_argument("--name", help="Output run name.")
    filter_success.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    filter_success.add_argument("--output", type=Path, help="Filtered JSONL output path.")
    filter_success.set_defaults(func=cmd_filter_success)

    trace_summary = subparsers.add_parser("trace-summary", help="Summarize trace success rates and episode lengths.")
    trace_summary.add_argument("traces", nargs="*", help="Trace files, directories, or glob patterns. Default: tmp/rl_traces/*.jsonl")
    trace_summary.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    trace_summary.set_defaults(func=cmd_trace_summary)

    localization_summary = subparsers.add_parser(
        "localization-summary",
        help="Compute localization robustness metrics from transition traces.",
    )
    localization_summary.add_argument("traces", nargs="*", help="Trace files, directories, or glob patterns. Default: tmp/rl_traces/*.jsonl")
    localization_summary.add_argument("--name", help="Run name for metrics output.")
    localization_summary.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    localization_summary.add_argument("--metrics-dir", type=Path, default=LOCALIZATION_METRICS_DIR)
    localization_summary.add_argument("--output", type=Path, help="Metrics JSON output path.")
    localization_summary.add_argument("--fail-on-gate", action="store_true", help="Return non-zero if localization acceptance gates fail.")
    localization_summary.set_defaults(func=cmd_localization_summary)

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
