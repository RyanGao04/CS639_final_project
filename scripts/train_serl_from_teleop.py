#!/usr/bin/env python3
"""Train/export the E2E policy from teleop demos with SERL."""

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
RUNTIME_WEIGHTS = ROOT / "final_project" / "controllers" / "robot_one_controller" / "e2e_rl_policy_weights.npz"


def _timestamp_name(prefix):
    return prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _print_command(cmd):
    print(" ".join(shlex.quote(str(part)) for part in cmd))


def _run(cmd, env=None, dry_run=False):
    _print_command(cmd)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


def _add_train_args(parser):
    parser.add_argument("traces", nargs="*", help="Teleop trace files, directories, or globs.")
    parser.add_argument("--workflow-python", default=sys.executable, help="Python interpreter used for rl_workflow.py.")
    parser.add_argument("--serl-python", default=os.environ.get("SERL_PYTHON"), help="Python interpreter used by the Webots controller; must have SERL/JAX installed.")
    parser.add_argument("--serl-root", type=Path, default=Path(os.environ.get("SERL_ROOT", "~/serl")).expanduser())
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--output", type=Path, default=RUNTIME_WEIGHTS)
    parser.add_argument("--timesteps", "--steps", dest="timesteps", type=int, default=50000, help="Online Webots environment steps.")
    parser.add_argument("--pretrain-steps", type=int, default=500, help="SERL updates on the separate teleop demo buffer before online rollout.")
    parser.add_argument("--learning-starts", type=int, help="First online step at which SERL updates are allowed.")
    parser.add_argument("--random-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd-ratio", type=int, default=8)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capacity", type=int, default=200000)
    parser.add_argument("--critic-ensemble-size", type=int, default=10)
    parser.add_argument("--critic-subsample-size", type=int, default=2)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--allow-empty-demos", action="store_true")
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--phase", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--terminate-out-of-play", action="store_true")
    parser.add_argument("--offline-pretrain-only", action="store_true", help="Run the old demo-only SERL update without Webots online interaction.")


def _add_render_args(parser):
    parser.add_argument("--render", "--render-training", dest="render_training", action="store_true", help="Show the Webots GUI during online SERL training.")
    parser.add_argument("--render-before", action="store_true", help="Show a Webots GUI rollout before training as a baseline.")
    parser.add_argument("--render-after", action="store_true", help="Show a Webots GUI rollout after training.")
    parser.add_argument("--render-episodes", type=int, default=3)
    parser.add_argument("--render-phase", type=int, default=3)
    parser.add_argument("--render-max-episode-steps", type=int, default=6000)
    parser.add_argument("--render-seed", type=int, default=17)
    parser.add_argument("--render-name", default=None)
    parser.add_argument("--webots-bin", help="Path to Webots executable for rendered rollout.")
    parser.add_argument("--webots-mode", choices=("pause", "realtime", "fast"), default="realtime")
    parser.add_argument("--dry-run", action="store_true")


def _offline_train_cmd(args):
    traces = args.traces or [str(args.trace_dir / "*teleop*.jsonl")]
    cmd = [
        args.workflow_python,
        str(WORKFLOW),
        "serl-train",
        *traces,
        "--trace-dir",
        str(args.trace_dir),
        "--serl-root",
        str(args.serl_root),
        "--output",
        str(args.output),
        "--steps",
        str(args.timesteps),
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
    if args.serl_python:
        cmd.extend(["--python", args.serl_python])
    if args.success_only:
        cmd.append("--success-only")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _online_train_cmd(args):
    traces = args.traces or [str(args.trace_dir / "*teleop*.jsonl")]
    cmd = [
        args.workflow_python,
        str(WORKFLOW),
        "deepbots-serl-train",
        *traces,
        "--trace-dir",
        str(args.trace_dir),
        "--serl-root",
        str(args.serl_root),
        "--output",
        str(args.output),
        "--timesteps",
        str(args.timesteps),
        "--pretrain-steps",
        str(args.pretrain_steps),
        "--random-steps",
        str(args.random_steps),
        "--batch-size",
        str(args.batch_size),
        "--utd-ratio",
        str(args.utd_ratio),
        "--updates-per-step",
        str(args.updates_per_step),
        "--update-every",
        str(args.update_every),
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
        "--phase",
        str(args.phase),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--webots-mode",
        args.webots_mode if args.render_training else "fast",
    ]
    if args.serl_python:
        cmd.extend(["--python", args.serl_python])
    if args.learning_starts is not None:
        cmd.extend(["--learning-starts", str(args.learning_starts)])
    if args.success_only:
        cmd.append("--success-only")
    if args.allow_empty_demos:
        cmd.append("--allow-empty-demos")
    if args.terminate_out_of_play:
        cmd.append("--terminate-out-of-play")
    if args.webots_bin:
        cmd.extend(["--webots-bin", args.webots_bin])
    if args.render_training:
        cmd.append("--no-batch")
    else:
        cmd.append("--batch")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _render_cmd(args, label):
    name = args.render_name or _timestamp_name(label)
    cmd = [
        args.workflow_python,
        str(WORKFLOW),
        "deepbots-eval",
        "--policy",
        "actor",
        "--weights",
        str(args.output),
        "--episodes",
        str(args.render_episodes),
        "--name",
        name,
        "--trace",
        "--phase",
        str(args.render_phase),
        "--max-episode-steps",
        str(args.render_max_episode_steps),
        "--seed",
        str(args.render_seed),
        "--no-batch",
        "--webots-mode",
        args.webots_mode,
    ]
    if args.webots_bin:
        cmd.extend(["--webots-bin", args.webots_bin])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _add_train_args(parser)
    _add_render_args(parser)
    args = parser.parse_args()

    env = os.environ.copy()
    env["RL_POLICY_MODE"] = "e2e"

    if args.render_before:
        print("Rendered Webots baseline before SERL training:")
        rc = _run(_render_cmd(args, "serl_before"), env=env, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if args.offline_pretrain_only:
        print("SERL demo-only offline update from teleop demos:")
        rc = _run(_offline_train_cmd(args), env=env, dry_run=args.dry_run)
    else:
        print("SERL online RL training in Webots with separate teleop demo + online replay buffers:")
        rc = _run(_online_train_cmd(args), env=env, dry_run=args.dry_run)
    if rc != 0:
        return rc

    if args.render_after:
        print("Rendered Webots rollout after SERL training:")
        return _run(_render_cmd(args, "serl_after"), env=env, dry_run=args.dry_run)

    print(f"Runtime actor output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
