#!/usr/bin/env python3
"""Train the E2E actor from teleop demos with SERL's SAC/RLPD-style agent."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_DIR = ROOT / "final_project" / "controllers" / "robot_one_controller"
DEFAULT_SERL_ROOT = Path(os.environ.get("SERL_ROOT", "~/serl")).expanduser()
DEFAULT_RUNTIME_OUTPUT = CONTROLLER_DIR / "e2e_rl_policy_weights.npz"


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


def _import_runtime_controller():
    sys.path.insert(0, str(CONTROLLER_DIR))
    from starter_controller import E2E_INPUT_DIM

    return E2E_INPUT_DIM


def _import_serl(serl_root):
    serl_launcher = Path(serl_root).expanduser() / "serl_launcher"
    if not serl_launcher.exists():
        raise SystemExit(f"SERL launcher not found: {serl_launcher}")
    sys.path.insert(0, str(serl_launcher))

    try:
        import gym
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
        from flax.core import unfreeze
        from serl_launcher.agents.continuous.sac import SACAgent
        from serl_launcher.data.replay_buffer import ReplayBuffer
    except ImportError as exc:
        raise SystemExit(
            "SERL/JAX dependencies are missing. Install the SERL environment first, "
            "then run this command with that Python interpreter."
        ) from exc

    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map

    return gym, nn, jax, jnp, unfreeze, SACAgent, ReplayBuffer


def _collect_trace_files(patterns):
    files = []
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if any(ch in str(path) for ch in "*?[]"):
            files.extend(sorted(path.parent.glob(path.name)))
        elif path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            files.append(path)
    deduped = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _load_transitions(trace_files, feature_dim, success_only=False):
    transitions = []
    episodes = {}
    order = []

    for path in trace_files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                obs = row.get("features")
                next_obs = row.get("next_features")
                action = row.get("action")
                if obs is None or next_obs is None or action is None:
                    continue
                if len(obs) != feature_dim or len(next_obs) != feature_dim or len(action) < 2:
                    continue
                info = row.get("info") or {}
                done = bool(row.get("done", False))
                reward = float(row.get("reward", 0.0))
                transition = {
                    "observations": np.asarray(obs, dtype=np.float32),
                    "actions": np.asarray(action[:2], dtype=np.float32),
                    "next_observations": np.asarray(next_obs, dtype=np.float32),
                    "rewards": np.float32(reward),
                    "masks": np.float32(0.0 if done else 1.0),
                    "dones": bool(done),
                    "_success": bool(info.get("correct_goal")),
                }
                key = (str(path), row.get("episode", info.get("episode_index", 0)))
                if key not in episodes:
                    episodes[key] = []
                    order.append(key)
                episodes[key].append(transition)

    if success_only:
        for key in order:
            episode = episodes[key]
            if any(item["_success"] for item in episode):
                transitions.extend(episode)
    else:
        for key in order:
            transitions.extend(episodes[key])

    for transition in transitions:
        transition.pop("_success", None)
    return transitions


def _tree_to_jax(jax, jnp, batch):
    tree_map = getattr(jax, "tree_map", None)
    if tree_map is None:
        tree_map = jax.tree_util.tree_map
    return tree_map(lambda value: jnp.asarray(value), batch)


def _create_agent(SACAgent, nn, jax, feature_dim, seed, lr, gamma, critic_ensemble_size, critic_subsample_size):
    return SACAgent.create_states(
        jax.random.PRNGKey(seed),
        np.zeros((feature_dim,), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        actor_optimizer_kwargs={
            "learning_rate": lr,
            "warmup_steps": 0,
        },
        critic_optimizer_kwargs={
            "learning_rate": lr,
            "warmup_steps": 0,
        },
        temperature_optimizer_kwargs={
            "learning_rate": lr,
        },
        policy_network_kwargs={
            "hidden_dims": [64, 64],
            "activations": nn.tanh,
            "use_layer_norm": False,
        },
        critic_network_kwargs={
            "hidden_dims": [256, 256],
            "activations": nn.tanh,
            "use_layer_norm": True,
        },
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5.0,
        },
        temperature_init=1e-2,
        discount=gamma,
        backup_entropy=False,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
    )


def _insert_transitions(replay_buffer, transitions):
    for item in transitions:
        replay_buffer.insert(item)


def _actor_arrays_from_params(params, feature_dim):
    actor = params["actor"]
    mlp = None
    for value in actor.values():
        if not isinstance(value, dict):
            continue
        dense0 = value.get("Dense_0")
        dense1 = value.get("Dense_1")
        if dense0 is None or dense1 is None:
            continue
        kernel0 = np.asarray(dense0.get("kernel"))
        kernel1 = np.asarray(dense1.get("kernel"))
        if kernel0.shape == (feature_dim, 64) and kernel1.shape == (64, 64):
            mlp = value
            break
    if mlp is None:
        raise RuntimeError(f"Could not find 64x64 actor MLP in params keys: {sorted(actor.keys())}")

    mean_dense = actor.get("Dense_0")
    if mean_dense is None or np.asarray(mean_dense.get("kernel")).shape != (64, 2):
        candidates = [
            value
            for key, value in sorted(actor.items())
            if isinstance(value, dict)
            and key.startswith("Dense_")
            and np.asarray(value.get("kernel")).shape == (64, 2)
        ]
        if not candidates:
            raise RuntimeError("Could not find actor mean output layer.")
        mean_dense = candidates[0]

    return {
        "w1": np.asarray(mlp["Dense_0"]["kernel"], dtype=np.float32).T,
        "b1": np.asarray(mlp["Dense_0"]["bias"], dtype=np.float32),
        "w2": np.asarray(mlp["Dense_1"]["kernel"], dtype=np.float32).T,
        "b2": np.asarray(mlp["Dense_1"]["bias"], dtype=np.float32),
        "w3": np.asarray(mean_dense["kernel"], dtype=np.float32).T,
        "b3": np.asarray(mean_dense["bias"], dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="Teleop JSONL traces, directories, or globs.")
    parser.add_argument("--serl-root", type=Path, default=DEFAULT_SERL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUNTIME_OUTPUT)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd-ratio", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capacity", type=int, default=200000)
    parser.add_argument("--critic-ensemble-size", type=int, default=10)
    parser.add_argument("--critic-subsample-size", type=int, default=2)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--log-period", type=int, default=100)
    args = parser.parse_args()

    feature_dim = _import_runtime_controller()
    gym, nn, jax, jnp, unfreeze, SACAgent, ReplayBuffer = _import_serl(args.serl_root)

    trace_files = _collect_trace_files(args.traces)
    if not trace_files:
        raise SystemExit("No trace files found.")
    transitions = _load_transitions(trace_files, feature_dim, success_only=args.success_only)
    if not transitions:
        raise SystemExit("No usable E2E transitions found. Collect teleop demos first.")

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
        capacity=max(args.capacity, len(transitions)),
    )
    _insert_transitions(replay_buffer, transitions)
    print(f"Loaded {len(transitions)} transitions from {len(trace_files)} trace file(s).")

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

    train_batch_size = args.batch_size * args.utd_ratio
    for step in range(1, args.steps + 1):
        batch = replay_buffer.sample(batch_size=train_batch_size)
        batch = _tree_to_jax(jax, jnp, batch)
        agent, info = agent.update_high_utd(batch, utd_ratio=args.utd_ratio)
        if step == 1 or step % args.log_period == 0:
            scalar_info = _flatten_scalar_info(info)
            print(f"step {step}: {json.dumps(scalar_info, sort_keys=True)}")

    params = unfreeze(agent.state.params)
    arrays = _actor_arrays_from_params(params, feature_dim)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    print(f"Exported E2E runtime actor weights to {output_path}")


if __name__ == "__main__":
    main()
