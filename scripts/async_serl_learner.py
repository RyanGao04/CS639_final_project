#!/usr/bin/env python3
"""Central SERL learner for async Webots actor workers.

Workers run Webots with ``RL_DEEPBOTS_MODE=async_actor`` and push JSON
transitions to this process over ZeroMQ. This learner owns the SERL agent and
replay buffers, performs GPU updates continuously, and periodically exports
runtime actor weights that workers can reload.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "tmp" / "rl_traces"
CHECKPOINT_DIR = ROOT / "tmp" / "rl_checkpoints"
DEFAULT_OUTPUT = CHECKPOINT_DIR / "async_serl_latest_weights.npz"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from serl_e2e_train import (  # noqa: E402
    _actor_arrays_from_params,
    _collect_trace_files,
    _create_agent,
    _import_runtime_controller,
    _import_serl,
    _insert_transitions,
    _load_transitions,
    _tree_to_jax,
)


def _flatten_scalar_info(info, prefix=""):
    scalars = {}
    if not isinstance(info, dict):
        return scalars
    for key, value in info.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            scalars.update(_flatten_scalar_info(value, name))
            continue
        try:
            array = np.asarray(value)
        except Exception:
            continue
        if array.shape != ():
            continue
        try:
            scalars[name] = float(array)
        except (TypeError, ValueError):
            continue
    return scalars


def _transition_from_message(message, feature_dim):
    obs = np.asarray(message.get("observations"), dtype=np.float32)
    next_obs = np.asarray(message.get("next_observations"), dtype=np.float32)
    action = np.asarray(message.get("actions"), dtype=np.float32)
    if obs.shape != (feature_dim,) or next_obs.shape != (feature_dim,) or action.shape[0] < 2:
        raise ValueError(
            f"bad transition shapes obs={obs.shape} next={next_obs.shape} action={action.shape}"
        )
    done = bool(message.get("dones", False))
    return {
        "observations": obs,
        "actions": action[:2],
        "next_observations": next_obs,
        "rewards": np.float32(float(message.get("rewards", 0.0))),
        "masks": np.float32(float(message.get("masks", 0.0 if done else 1.0))),
        "dones": done,
    }


def _concat_batches(jax, jnp, first_batch, second_batch):
    tree_map = getattr(jax, "tree_map", None) or jax.tree_util.tree_map
    return tree_map(lambda first, second: jnp.concatenate((first, second), axis=0), first_batch, second_batch)


def _sample_mixed_batch(
    replay_buffer,
    demo_buffer,
    jax,
    jnp,
    online_batch_size,
    demo_batch_size,
    utd_ratio,
):
    online_batch = _tree_to_jax(jax, jnp, replay_buffer.sample(batch_size=online_batch_size))
    if demo_buffer is None or demo_batch_size <= 0:
        batch = online_batch
    else:
        demo_batch = _tree_to_jax(jax, jnp, demo_buffer.sample(batch_size=demo_batch_size))
        batch = _concat_batches(jax, jnp, online_batch, demo_batch)
    return batch


def _mixed_update(
    agent,
    replay_buffer,
    demo_buffer,
    jax,
    jnp,
    online_batch_size,
    demo_batch_size,
    utd_ratio,
):
    batch = _sample_mixed_batch(
        replay_buffer,
        demo_buffer,
        jax,
        jnp,
        online_batch_size,
        demo_batch_size,
        utd_ratio,
    )
    return agent.update_high_utd(batch, utd_ratio=utd_ratio)


def _export_agent(agent, unfreeze, feature_dim, output_path):
    arrays = _actor_arrays_from_params(unfreeze(agent.state.params), feature_dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    np.savez(tmp_path, **arrays)
    produced = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    produced.replace(output_path)
    return output_path


def _drain_messages(socket, replay_buffer, feature_dim, max_messages):
    received = 0
    bad = 0
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("async_serl_learner.py requires pyzmq.") from exc

    while received < max_messages:
        try:
            message = socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        try:
            replay_buffer.insert(_transition_from_message(message, feature_dim))
            received += 1
        except Exception as exc:
            bad += 1
            if bad <= 5:
                print(f"[learner] dropped malformed transition: {exc}")
    return received, bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5557")
    parser.add_argument("--demo-traces", nargs="*", default=[str(TRACE_DIR / "*teleop*.jsonl")])
    parser.add_argument("--serl-root", type=Path, default=Path(os.environ.get("SERL_ROOT", "~/serl")).expanduser())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--updates", type=int, default=50000)
    parser.add_argument("--seconds", type=float, default=0.0, help="Optional wall-clock limit. 0 means no limit.")
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--min-online", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--utd-ratio", type=int, default=16)
    parser.add_argument("--demo-fraction", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capacity", type=int, default=500000)
    parser.add_argument("--critic-ensemble-size", type=int, default=10)
    parser.add_argument("--critic-subsample-size", type=int, default=2)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument("--export-period", type=int, default=500)
    parser.add_argument("--drain-batch", type=int, default=4096)
    parser.add_argument(
        "--updates-per-drain",
        type=int,
        default=1,
        help="Number of learner updates to run after each socket drain once replay has enough online data.",
    )
    parser.add_argument(
        "--reuse-batch-updates",
        type=int,
        default=1,
        help="Reuse one sampled replay/demo batch for this many consecutive learner updates.",
    )
    args = parser.parse_args()

    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("pyzmq is required for async SERL learning.") from exc

    feature_dim = _import_runtime_controller()
    gym, nn, jax, jnp, unfreeze, SACAgent, ReplayBuffer = _import_serl(args.serl_root)
    trace_files = _collect_trace_files(args.demo_traces)
    demo_transitions = _load_transitions(trace_files, feature_dim, success_only=args.success_only) if trace_files else []

    total_batch_size = max(16, args.batch_size) * max(1, args.utd_ratio)
    demo_batch_size = 0
    if demo_transitions:
        demo_batch_size = int(round(total_batch_size * min(max(args.demo_fraction, 0.0), 0.95)))
    online_batch_size = max(1, total_batch_size - demo_batch_size)

    obs_space = gym.spaces.Box(
        low=-np.ones(feature_dim, dtype=np.float32),
        high=np.ones(feature_dim, dtype=np.float32),
        shape=(feature_dim,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        low=-np.ones(2, dtype=np.float32),
        high=np.ones(2, dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )
    replay_buffer = ReplayBuffer(
        obs_space,
        action_space,
        capacity=max(args.capacity, args.min_online + total_batch_size),
    )
    demo_buffer = None
    if demo_transitions:
        demo_buffer = ReplayBuffer(
            obs_space,
            action_space,
            capacity=max(len(demo_transitions), demo_batch_size, total_batch_size),
        )
        _insert_transitions(demo_buffer, demo_transitions)

    agent = _create_agent(
        SACAgent,
        nn,
        jax,
        feature_dim,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        critic_ensemble_size=args.critic_ensemble_size,
        critic_subsample_size=args.critic_subsample_size,
    )

    print(
        "[learner] setup "
        f"bind={args.bind} demos={len(demo_transitions)} files={len(trace_files)} "
        f"online_batch={online_batch_size} demo_batch={demo_batch_size} "
        f"utd={args.utd_ratio} output={args.output}"
    )

    for pretrain_step in range(1, max(0, args.pretrain_steps) + 1):
        if demo_buffer is None:
            break
        batch = _tree_to_jax(jax, jnp, demo_buffer.sample(batch_size=total_batch_size))
        agent, info = agent.update_high_utd(batch, utd_ratio=args.utd_ratio)
        if pretrain_step == 1 or pretrain_step % args.log_period == 0:
            print(f"[learner] pretrain={pretrain_step} info={json.dumps(_flatten_scalar_info(info), sort_keys=True)}")

    output_path = args.output.expanduser().resolve()
    _export_agent(agent, unfreeze, feature_dim, output_path)
    print(f"[learner] exported initial policy {output_path}")

    context = zmq.Context.instance()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, max(1000, args.drain_batch * 4))
    socket.bind(args.bind)

    stop_requested = False

    def _handle_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    update_count = 0
    received_total = 0
    bad_total = 0
    last_info = {}
    while update_count < args.updates and not stop_requested:
        if deadline is not None and time.monotonic() >= deadline:
            break

        received, bad = _drain_messages(socket, replay_buffer, feature_dim, args.drain_batch)
        received_total += received
        bad_total += bad

        if len(replay_buffer) < args.min_online or len(replay_buffer) < online_batch_size:
            if received == 0:
                time.sleep(0.01)
            if update_count == 0 and received_total and received_total % args.log_period == 0:
                print(f"[learner] waiting online_replay={len(replay_buffer)}/{args.min_online}")
            continue

        cached_batch = None
        reuse_left = 0
        for _ in range(max(1, args.updates_per_drain)):
            if update_count >= args.updates or stop_requested:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break

            if args.reuse_batch_updates > 1:
                if cached_batch is None or reuse_left <= 0:
                    cached_batch = _sample_mixed_batch(
                        replay_buffer,
                        demo_buffer,
                        jax,
                        jnp,
                        online_batch_size,
                        demo_batch_size,
                        args.utd_ratio,
                    )
                    reuse_left = args.reuse_batch_updates
                agent, last_info = agent.update_high_utd(cached_batch, utd_ratio=args.utd_ratio)
                reuse_left -= 1
            else:
                agent, last_info = _mixed_update(
                    agent,
                    replay_buffer,
                    demo_buffer,
                    jax,
                    jnp,
                    online_batch_size,
                    demo_batch_size,
                    args.utd_ratio,
                )
            update_count += 1

            if update_count == 1 or update_count % args.log_period == 0:
                print(
                    "[learner] "
                    f"updates={update_count}/{args.updates} received={received_total} "
                    f"online_replay={len(replay_buffer)} bad={bad_total} "
                    f"info={json.dumps(_flatten_scalar_info(last_info), sort_keys=True)}"
                )
            if update_count % args.export_period == 0:
                _export_agent(agent, unfreeze, feature_dim, output_path)
                print(f"[learner] exported policy {output_path}")

    _export_agent(agent, unfreeze, feature_dim, output_path)
    print(
        f"[learner] done updates={update_count} received={received_total} "
        f"online_replay={len(replay_buffer)} output={output_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
