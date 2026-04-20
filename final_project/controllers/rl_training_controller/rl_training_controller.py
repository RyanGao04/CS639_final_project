#!/usr/bin/env python3
"""Deepbots live RL harness for the CS639 solo soccer task.

This controller is development scaffolding. The submission-compatible controller
remains final_project/controllers/robot_one_controller/starter_controller.py.
"""

import atexit
import json
import math
import os
from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEEPBOTS_ROOT = ROOT / "deepbots"
ROBOT_CONTROLLER_DIR = ROOT / "final_project" / "controllers" / "robot_one_controller"

for candidate in (DEEPBOTS_ROOT, ROBOT_CONTROLLER_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)


def _install_gym_compat():
    try:
        import gym  # type: ignore

        return gym
    except ImportError:
        pass

    try:
        import gymnasium as gymnasium  # type: ignore

        sys.modules.setdefault("gym", gymnasium)
        return gymnasium
    except ImportError:
        pass

    class _MiniEnv:
        pass

    class _MiniBox:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.shape = tuple(shape or self.low.shape)
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    mini_gym = types.SimpleNamespace(
        Env=_MiniEnv,
        spaces=types.SimpleNamespace(Box=_MiniBox),
    )
    sys.modules["gym"] = mini_gym
    return mini_gym


gym = _install_gym_compat()

try:
    from controller import Supervisor
    from deepbots.supervisor import RobotSupervisorEnv
except Exception as exc:  # pragma: no cover - only importable inside Webots.
    raise RuntimeError(
        "rl_training_controller.py must run as a Webots controller, and the local "
        "deepbots/ folder must exist at the project root."
    ) from exc

from starter_controller import (  # noqa: E402
    ARENA_X_HALF,
    ARENA_Y_HALF,
    FIELD_X_HALF,
    FIELD_Y_HALF,
    GOAL_HALF_WIDTH,
    MAX_WHEEL_SPEED,
    OBSERVATION_FOV,
    RIGHT_GOAL,
    RL_FORWARD_SCALE,
    RL_TURN_SCALE,
    EmbeddedActorPolicy,
    clamp,
    clip_to_field,
    distance,
    polar_to_world,
    relative_polar,
    wrap_to_pi,
)


FEATURE_DIM = 16
ACTION_DIM = 2
BALL_Z = 0.07
RUNTIME_WEIGHTS_FILENAME = "rl_policy_weights.npz"
RUNTIME_WEIGHTS_PATH = ROBOT_CONTROLLER_DIR / RUNTIME_WEIGHTS_FILENAME


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _normalize_xy(dx, dy):
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return 1.0, 0.0
    return dx / norm, dy / norm


class JsonlTransitionLogger:
    def __init__(self):
        self.handle = None
        trace_path = (
            os.environ.get("RL_DEEPBOTS_TRACE_PATH")
            or os.environ.get("RL_TRACE_PATH")
        )
        if not trace_path:
            return

        path = Path(trace_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(path, "a", encoding="utf-8", buffering=1)
        atexit.register(self.close)

    def log(self, payload):
        if self.handle is None:
            return
        self.handle.write(json.dumps(payload) + "\n")

    def close(self):
        if self.handle is None:
            return
        try:
            self.handle.close()
        finally:
            self.handle = None


class SoccerSoloDeepbotsEnv(RobotSupervisorEnv):
    """Robot-supervisor deepbots environment for visible-ball soccer play."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.robot_node = self.getFromDef("ROBOT_ONE")
        self.ball_node = self.getFromDef("BALL")
        if self.robot_node is None or self.ball_node is None:
            raise RuntimeError("World must define ROBOT_ONE and BALL nodes.")

        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.compass = None
        try:
            self.compass = self.getDevice("compass")
            self.compass.enable(self.timestep)
        except Exception:
            self.compass = None

        self.rng = np.random.default_rng(_env_int("RL_DEEPBOTS_SEED", 7))
        self.phase = max(1, min(3, _env_int("RL_DEEPBOTS_PHASE", 2)))
        self.max_episode_steps = max(100, _env_int("RL_DEEPBOTS_MAX_STEPS", 1800))
        self.control_noise_pct = max(0.0, _env_float("RL_DEEPBOTS_CONTROL_NOISE_PCT", 0.05))
        self.ball_range_noise_pct = max(0.0, _env_float("RL_DEEPBOTS_BALL_RANGE_NOISE_PCT", 0.02))
        self.ball_angle_noise = max(0.0, _env_float("RL_DEEPBOTS_BALL_ANGLE_NOISE", 0.015))
        self.ball_dropout = clamp(_env_float("RL_DEEPBOTS_BALL_DROPOUT", 0.08 if self.phase >= 3 else 0.0), 0.0, 0.9)

        self.observation_space = gym.spaces.Box(
            low=-np.ones(FEATURE_DIM, dtype=np.float32),
            high=np.ones(FEATURE_DIM, dtype=np.float32),
            shape=(FEATURE_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(ACTION_DIM, dtype=np.float32),
            high=np.ones(ACTION_DIM, dtype=np.float32),
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        self.logger = JsonlTransitionLogger()
        self.episode = 0
        self.episode_step = 0
        self.global_step = 0
        self.prev_forward_cmd = 0.0
        self.prev_turn_cmd = 0.0
        self.ball_age = 0
        self.ball_estimate = None
        self.last_seen_ball_rel = None
        self.localizer_confidence = 1.0
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_control = {"left_motor": 0.0, "right_motor": 0.0}
        self.last_observation = np.zeros(FEATURE_DIM, dtype=np.float32)
        self._prev_metrics = self._metrics()

    def _robot_pose(self):
        translation = self.robot_node.getField("translation").getSFVec3f()
        if self.compass is not None:
            values = self.compass.getValues()
            theta = math.atan2(values[0], values[1])
        else:
            rotation = self.robot_node.getField("rotation").getSFRotation()
            theta = rotation[3] if rotation[2] >= 0.0 else -rotation[3]
        return float(translation[0]), float(translation[1]), wrap_to_pi(float(theta))

    def _ball_position(self):
        translation = self.ball_node.getField("translation").getSFVec3f()
        return float(translation[0]), float(translation[1])

    def _wall_margin(self, pose):
        return min(ARENA_X_HALF - abs(pose[0]), ARENA_Y_HALF - abs(pose[1]))

    def _staging_point(self, ball_position):
        goal_dx = RIGHT_GOAL[0] - ball_position[0]
        goal_dy = RIGHT_GOAL[1] - ball_position[1]
        unit_x, unit_y = _normalize_xy(goal_dx, goal_dy)
        staging = (
            ball_position[0] - 0.28 * unit_x,
            ball_position[1] - 0.28 * unit_y,
        )
        return clip_to_field(staging, padding=0.20)

    def _is_visible(self, pose, point):
        _, angle = relative_polar(pose, point)
        return abs(angle) <= 0.5 * OBSERVATION_FOV

    def _observe_ball(self, pose, ball_position):
        visible = self._is_visible(pose, ball_position)
        if visible and self.ball_dropout > 0.0 and self.rng.random() < self.ball_dropout:
            visible = False

        if visible:
            dist, angle = relative_polar(pose, ball_position)
            noisy_dist = max(
                0.02,
                dist + float(self.rng.normal(0.0, self.ball_range_noise_pct * max(dist, 0.5))),
            )
            noisy_angle = wrap_to_pi(angle + float(self.rng.normal(0.0, self.ball_angle_noise)))
            ball_rel = (noisy_dist, noisy_angle)
            self.ball_estimate = polar_to_world(pose, ball_rel)
            self.last_seen_ball_rel = ball_rel
            self.ball_age = 0
            return 1.0, ball_rel, self.ball_estimate

        self.ball_age += 1
        if self.ball_estimate is not None:
            return 0.0, relative_polar(pose, self.ball_estimate), self.ball_estimate
        if self.last_seen_ball_rel is not None:
            return 0.0, self.last_seen_ball_rel, None
        return 0.0, (2.0, 0.0), None

    def _metrics(self):
        pose = self._robot_pose()
        ball = self._ball_position()
        return {
            "pose": pose,
            "ball": ball,
            "robot_to_ball": distance(pose[:2], ball),
            "ball_to_goal": distance(ball, RIGHT_GOAL),
            "wall_margin": self._wall_margin(pose),
            "correct_goal": self._is_correct_goal(ball),
            "wrong_goal": self._is_wrong_goal(ball),
        }

    def _is_correct_goal(self, ball):
        return ball[0] > FIELD_X_HALF + 0.02 and abs(ball[1]) <= GOAL_HALF_WIDTH

    def _is_wrong_goal(self, ball):
        return ball[0] < -FIELD_X_HALF - 0.02 and abs(ball[1]) <= GOAL_HALF_WIDTH

    def _is_out_of_play(self, ball):
        return (
            abs(ball[0]) > ARENA_X_HALF - 0.04
            or abs(ball[1]) > ARENA_Y_HALF - 0.04
        )

    def _sample_initial_state(self):
        for _ in range(200):
            ball_x = float(self.rng.uniform(-1.3, 1.5))
            ball_y = float(self.rng.uniform(-1.5, 1.5))
            goal_dx = RIGHT_GOAL[0] - ball_x
            goal_dy = RIGHT_GOAL[1] - ball_y
            goal_unit_x, goal_unit_y = _normalize_xy(goal_dx, goal_dy)
            perp_x, perp_y = -goal_unit_y, goal_unit_x

            if self.phase == 1:
                dist_to_ball = float(self.rng.uniform(0.4, 1.0))
                side_offset = float(self.rng.normal(0.0, 0.15))
                robot_x = ball_x - dist_to_ball * goal_unit_x + side_offset * perp_x
                robot_y = ball_y - dist_to_ball * goal_unit_y + side_offset * perp_y
                heading = math.atan2(ball_y - robot_y, ball_x - robot_x)
                heading += float(self.rng.normal(0.0, 0.25))
            else:
                dist_to_ball = float(self.rng.uniform(0.4, 1.8))
                bearing = float(self.rng.uniform(-math.pi, math.pi))
                robot_x = ball_x + dist_to_ball * math.cos(bearing)
                robot_y = ball_y + dist_to_ball * math.sin(bearing)
                heading_to_ball = math.atan2(ball_y - robot_y, ball_x - robot_x)
                heading = heading_to_ball + float(self.rng.uniform(-0.45 * OBSERVATION_FOV, 0.45 * OBSERVATION_FOV))

            robot_x = clamp(robot_x, -FIELD_X_HALF + 0.35, FIELD_X_HALF - 0.35)
            robot_y = clamp(robot_y, -FIELD_Y_HALF + 0.35, FIELD_Y_HALF - 0.35)
            pose = (robot_x, robot_y, wrap_to_pi(heading))
            if self._is_visible(pose, (ball_x, ball_y)):
                return pose, (ball_x, ball_y)

        return (-1.0, 0.0, 0.0), (0.0, 0.0)

    def _place_node(self, node, x, y, z, theta=None):
        node.getField("translation").setSFVec3f([float(x), float(y), float(z)])
        if theta is not None:
            node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, wrap_to_pi(float(theta))])
        node.resetPhysics()

    def get_default_observation(self):
        return self.last_observation.copy()

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.episode += 1
        self.episode_step = 0
        self.prev_forward_cmd = 0.0
        self.prev_turn_cmd = 0.0
        self.ball_age = 0
        self.ball_estimate = None
        self.last_seen_ball_rel = None
        self.localizer_confidence = 1.0
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_control = {"left_motor": 0.0, "right_motor": 0.0}
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        pose, ball = self._sample_initial_state()
        self._place_node(self.robot_node, pose[0], pose[1], 0.0, theta=pose[2])
        self._place_node(self.ball_node, ball[0], ball[1], BALL_Z)
        self.simulationResetPhysics()
        super(Supervisor, self).step(self.timestep)

        self._prev_metrics = self._metrics()
        self.last_observation = self.get_observations()
        return self.last_observation.copy()

    def get_observations(self):
        pose = self._robot_pose()
        ball = self._ball_position()
        ball_visible, ball_rel, ball_estimate = self._observe_ball(pose, ball)

        self.localizer_confidence = clamp(
            1.0 - 0.025 * self.ball_age + float(self.rng.normal(0.0, 0.01 if self.phase >= 3 else 0.0)),
            0.0,
            1.0,
        )

        goal_rel = relative_polar(pose, RIGHT_GOAL)
        if ball_estimate is not None:
            staging_point = self._staging_point(ball_estimate)
            staging_rel = relative_polar(pose, staging_point)
            ball_goal_alignment = math.cos(wrap_to_pi(goal_rel[1] - ball_rel[1]))
        else:
            staging_rel = (2.0, 0.0)
            ball_goal_alignment = -1.0

        wall_margin_norm = clamp(self._wall_margin(pose) / 1.5, 0.0, 1.0)
        features = np.array(
            [
                ball_visible,
                clamp(ball_rel[0] / 2.0, 0.0, 1.0),
                math.sin(ball_rel[1]),
                math.cos(ball_rel[1]),
                clamp(goal_rel[0] / 9.0, 0.0, 1.0),
                math.sin(goal_rel[1]),
                math.cos(goal_rel[1]),
                clamp(staging_rel[0] / 2.0, 0.0, 1.0),
                math.sin(staging_rel[1]),
                math.cos(staging_rel[1]),
                ball_goal_alignment,
                self.localizer_confidence,
                clamp(self.ball_age / 30.0, 0.0, 1.0),
                clamp(self.prev_forward_cmd, -1.0, 1.0),
                clamp(self.prev_turn_cmd, -1.0, 1.0),
                wall_margin_norm,
            ],
            dtype=np.float32,
        )
        self.last_observation = features
        return features.copy()

    def apply_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < ACTION_DIM:
            action = np.pad(action, (0, ACTION_DIM - action.shape[0]))

        a_forward = clamp(float(action[0]), -1.0, 1.0)
        a_turn = clamp(float(action[1]), -1.0, 1.0)
        forward = RL_FORWARD_SCALE * max(0.0, 0.5 * (a_forward + 1.0))
        turn = RL_TURN_SCALE * a_turn
        self.prev_forward_cmd = clamp(forward / RL_FORWARD_SCALE, -1.0, 1.0)
        self.prev_turn_cmd = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)

        left = clamp(forward - turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        right = clamp(forward + turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        if self.control_noise_pct > 0.0:
            left += float(self.rng.normal(0.0, self.control_noise_pct * abs(left)))
            right += float(self.rng.normal(0.0, self.control_noise_pct * abs(right)))
        left = clamp(left, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        right = clamp(right, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)
        self.last_action = np.array([a_forward, a_turn], dtype=np.float32)
        self.last_control = {"left_motor": float(left), "right_motor": float(right)}

    def step(self, action):
        prev_obs = self.last_observation.copy()
        self._prev_metrics = self._metrics()

        self.apply_action(action)
        if super(Supervisor, self).step(self.timestep) == -1:
            self.close()
            sys.exit(0)

        self.global_step += 1
        self.episode_step += 1
        obs = self.get_observations()
        reward = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()
        self._log_transition(prev_obs, obs, reward, done, info)
        return obs, reward, done, info

    def get_reward(self, action):
        del action
        prev = self._prev_metrics
        curr = self._metrics()
        reward = -0.01

        robot_progress = prev["robot_to_ball"] - curr["robot_to_ball"]
        ball_progress = prev["ball_to_goal"] - curr["ball_to_goal"]
        reward += 1.2 * robot_progress
        reward += 2.0 * ball_progress

        features = self.last_observation
        ball_angle = math.atan2(float(features[2]), float(features[3]))
        goal_angle = math.atan2(float(features[5]), float(features[6]))
        staging_error = 2.0 * float(features[7])
        reward += 0.5 * math.exp(-4.0 * abs(ball_angle))
        reward += 0.35 * math.exp(-2.5 * staging_error)
        reward += 0.25 * math.exp(-2.0 * abs(goal_angle))

        if features[0] < 0.5:
            reward -= 0.2
        if curr["wall_margin"] < 0.30 and ball_progress < 0.003:
            reward -= 0.3
        if abs(float(self.last_action[1])) > 0.75 and ball_progress < 0.003:
            reward -= 0.4

        if curr["correct_goal"]:
            reward += 12.0
        elif curr["wrong_goal"] or (self._is_out_of_play(curr["ball"]) and not curr["correct_goal"]):
            reward -= 8.0

        return float(reward)

    def is_done(self):
        metrics = self._metrics()
        if metrics["correct_goal"] or metrics["wrong_goal"]:
            return True
        if self._is_out_of_play(metrics["ball"]):
            return True
        return self.episode_step >= self.max_episode_steps

    def get_info(self):
        metrics = self._metrics()
        pose = metrics["pose"]
        ball = metrics["ball"]
        return {
            # SB3 Monitor reserves info["episode"] for its own episode summary dict.
            "episode_index": self.episode,
            "episode_step": self.episode_step,
            "global_step": self.global_step,
            "robot_pose": [float(pose[0]), float(pose[1]), float(pose[2])],
            "ball_position": [float(ball[0]), float(ball[1])],
            "correct_goal": bool(metrics["correct_goal"]),
            "wrong_goal": bool(metrics["wrong_goal"]),
            "ball_to_goal": float(metrics["ball_to_goal"]),
            "robot_to_ball": float(metrics["robot_to_ball"]),
        }

    def _log_transition(self, prev_obs, obs, reward, done, info):
        self.logger.log(
            {
                "episode": self.episode,
                "step": self.global_step,
                "episode_step": self.episode_step,
                "state": "RL_BALL_PLAY",
                "features": [float(v) for v in prev_obs],
                "action": [float(v) for v in self.last_action],
                "reward": float(reward),
                "next_features": [float(v) for v in obs],
                "done": bool(done),
                "estimated_pose": info["robot_pose"],
                "actual_pose": info["robot_pose"],
                "estimated_ball": info["ball_position"],
                "actual_ball": info["ball_position"],
                "control": self.last_control,
                "info": info,
            }
        )

    def close(self):
        try:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        except Exception:
            pass
        self.logger.close()


class GymnasiumAdapter(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, deepbots_env):
        self.deepbots_env = deepbots_env
        self.action_space = deepbots_env.action_space
        self.observation_space = deepbots_env.observation_space

    def reset(self, *, seed=None, options=None):
        obs = self.deepbots_env.reset(seed=seed, options=options)
        return obs, self.deepbots_env.get_info()

    def step(self, action):
        obs, reward, done, info = self.deepbots_env.step(action)
        return obs, reward, done, False, info

    def close(self):
        self.deepbots_env.close()


def _actor_arrays_from_embedded_policy():
    policy = EmbeddedActorPolicy()
    return {
        "w1": np.asarray(policy.w1, dtype=np.float32),
        "b1": np.asarray(policy.b1, dtype=np.float32),
        "w2": np.asarray(policy.w2, dtype=np.float32),
        "b2": np.asarray(policy.b2, dtype=np.float32),
        "w3": np.asarray(policy.w3, dtype=np.float32),
        "b3": np.asarray(policy.b3, dtype=np.float32),
    }, policy.source


def _load_actor_arrays_npz(path):
    expected = {
        "w1": (64, FEATURE_DIM),
        "b1": (64,),
        "w2": (64, 64),
        "b2": (64,),
        "w3": (ACTION_DIM, 64),
        "b3": (ACTION_DIM,),
    }
    path = Path(path).expanduser()
    with np.load(path) as data:
        arrays = {}
        for key, shape in expected.items():
            if key not in data:
                raise RuntimeError(f"{path} is missing actor array {key}")
            array = np.asarray(data[key], dtype=np.float32)
            if array.shape != shape:
                raise RuntimeError(f"{path}:{key} has shape {array.shape}, expected {shape}")
            arrays[key] = array
    return arrays, str(path)


def _copy_linear(layer, weight, bias):
    import torch

    if tuple(layer.weight.shape) != tuple(weight.shape):
        raise RuntimeError(f"Linear weight shape {tuple(layer.weight.shape)} does not match {weight.shape}")
    if tuple(layer.bias.shape) != tuple(bias.shape):
        raise RuntimeError(f"Linear bias shape {tuple(layer.bias.shape)} does not match {bias.shape}")

    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weight, dtype=layer.weight.dtype, device=layer.weight.device))
        layer.bias.copy_(torch.as_tensor(bias, dtype=layer.bias.dtype, device=layer.bias.device))


def warm_start_sac_actor(model, arrays, log_std_init):
    import torch
    import torch.nn as nn

    actor = model.policy.actor
    linear_layers = [module for module in actor.latent_pi.modules() if isinstance(module, nn.Linear)]
    if len(linear_layers) != 2:
        raise RuntimeError("Expected SAC latent_pi to contain exactly two Linear layers.")

    _copy_linear(linear_layers[0], arrays["w1"], arrays["b1"])
    _copy_linear(linear_layers[1], arrays["w2"], arrays["b2"])
    _copy_linear(actor.mu, arrays["w3"], arrays["b3"])

    if hasattr(actor, "log_std") and isinstance(actor.log_std, nn.Linear):
        with torch.no_grad():
            actor.log_std.weight.zero_()
            actor.log_std.bias.fill_(float(log_std_init))


def resolve_warm_start_arrays(mode, weights_path):
    mode = (mode or "none").strip().lower()
    if weights_path:
        return _load_actor_arrays_npz(weights_path)
    if mode == "none":
        return None, None
    if mode == "bootstrap":
        return _actor_arrays_from_embedded_policy()
    if mode == "runtime":
        if not RUNTIME_WEIGHTS_PATH.exists():
            raise RuntimeError(f"Runtime actor weights not found: {RUNTIME_WEIGHTS_PATH}")
        return _load_actor_arrays_npz(RUNTIME_WEIGHTS_PATH)
    raise RuntimeError(f"Unsupported RL_DEEPBOTS_WARM_START={mode!r}")


def export_sac_actor(model, output_path):
    import torch.nn as nn

    actor = model.policy.actor
    latent_pi = getattr(actor, "latent_pi", None)
    mu_layer = getattr(actor, "mu", None)
    if latent_pi is None or mu_layer is None:
        raise RuntimeError("Unsupported SAC actor layout; expected latent_pi and mu modules.")

    linear_layers = [module for module in latent_pi.modules() if isinstance(module, nn.Linear)]
    if len(linear_layers) != 2 or not isinstance(mu_layer, nn.Linear):
        raise RuntimeError("SAC actor must use exactly two hidden Linear layers plus mu output.")

    arrays = {
        "w1": linear_layers[0].weight.detach().cpu().numpy().astype(np.float32),
        "b1": linear_layers[0].bias.detach().cpu().numpy().astype(np.float32),
        "w2": linear_layers[1].weight.detach().cpu().numpy().astype(np.float32),
        "b2": linear_layers[1].bias.detach().cpu().numpy().astype(np.float32),
        "w3": mu_layer.weight.detach().cpu().numpy().astype(np.float32),
        "b3": mu_layer.bias.detach().cpu().numpy().astype(np.float32),
    }
    expected = {
        "w1": (64, FEATURE_DIM),
        "b1": (64,),
        "w2": (64, 64),
        "b2": (64,),
        "w3": (ACTION_DIM, 64),
        "b3": (ACTION_DIM,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise RuntimeError(f"{key} has shape {arrays[key].shape}, expected {shape}")

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"Exported runtime actor weights to {path}")


def run_actor_loop(env):
    actor = EmbeddedActorPolicy()
    obs = env.reset()
    while True:
        action = actor(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()


def run_random_loop(env):
    obs = env.reset()
    del obs
    while True:
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()


def run_sac_eval(env):
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for RL_DEEPBOTS_MODE=sac_eval. "
            "Run: python scripts/rl_workflow.py install-rl-deps"
        ) from exc

    model_path = os.environ.get("RL_DEEPBOTS_MODEL_PATH")
    if not model_path:
        raise RuntimeError("Set RL_DEEPBOTS_MODEL_PATH for sac_eval mode.")

    model = SAC.load(model_path)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()


def run_sac_training(env):
    try:
        import torch
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError(
            "SAC training requires torch, gymnasium, and stable-baselines3. "
            "Run: python scripts/rl_workflow.py install-rl-deps"
        ) from exc

    model_path = Path(
        os.environ.get(
            "RL_DEEPBOTS_MODEL_PATH",
            str(ROOT / "tmp" / "rl_checkpoints" / "deepbots_sac.zip"),
        )
    ).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = os.environ.get("RL_DEEPBOTS_TENSORBOARD_DIR")
    total_timesteps = max(1, _env_int("RL_DEEPBOTS_TOTAL_STEPS", 50000))
    learning_starts = max(0, _env_int("RL_DEEPBOTS_LEARNING_STARTS", 1000))
    batch_size = max(16, _env_int("RL_DEEPBOTS_BATCH_SIZE", 256))

    wrapped_env = GymnasiumAdapter(env)
    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": [64, 64],
    }
    ent_coef_raw = os.environ.get("RL_DEEPBOTS_ENT_COEF", "auto")
    ent_coef = ent_coef_raw if ent_coef_raw.startswith("auto") else float(ent_coef_raw)
    model = SAC(
        "MlpPolicy",
        wrapped_env,
        learning_rate=_env_float("RL_DEEPBOTS_LR", 3e-4),
        buffer_size=max(total_timesteps, _env_int("RL_DEEPBOTS_BUFFER_SIZE", 100000)),
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=_env_float("RL_DEEPBOTS_TAU", 0.02),
        gamma=_env_float("RL_DEEPBOTS_GAMMA", 0.995),
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_dir,
    )

    warm_start_mode = os.environ.get("RL_DEEPBOTS_WARM_START", "none")
    warm_start_weights = os.environ.get("RL_DEEPBOTS_WARM_START_WEIGHTS")
    log_std_init = _env_float("RL_DEEPBOTS_SAC_LOG_STD_INIT", -1.2)
    warm_arrays, warm_source = resolve_warm_start_arrays(warm_start_mode, warm_start_weights)
    if warm_arrays is not None:
        warm_start_sac_actor(model, warm_arrays, log_std_init=log_std_init)
        print(f"Warm-started SAC actor from {warm_source} with log_std_init={log_std_init}")

    model.learn(total_timesteps=total_timesteps, log_interval=10)
    model.save(str(model_path))
    print(f"Saved SAC model to {model_path}")

    export_path = os.environ.get("RL_DEEPBOTS_EXPORT_PATH")
    if export_path:
        export_sac_actor(model, export_path)

    env.close()
    env.simulationQuit(0)


def main():
    env = SoccerSoloDeepbotsEnv()
    mode = os.environ.get("RL_DEEPBOTS_MODE", "actor").strip().lower()
    try:
        if mode == "actor":
            run_actor_loop(env)
        elif mode == "random":
            run_random_loop(env)
        elif mode == "sac_train":
            run_sac_training(env)
        elif mode == "sac_eval":
            run_sac_eval(env)
        else:
            raise RuntimeError(f"Unsupported RL_DEEPBOTS_MODE={mode!r}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
