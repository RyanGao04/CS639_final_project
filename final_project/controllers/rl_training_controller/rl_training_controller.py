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
SCRIPTS_DIR = ROOT / "scripts"
ROBOT_CONTROLLER_DIR = ROOT / "final_project" / "controllers" / "robot_one_controller"

for candidate in (DEEPBOTS_ROOT, ROBOT_CONTROLLER_DIR, SCRIPTS_DIR):
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
    from controller import Keyboard, Supervisor
    from deepbots.supervisor import RobotSupervisorEnv
except Exception as exc:  # pragma: no cover - only importable inside Webots.
    raise RuntimeError(
        "rl_training_controller.py must run as a Webots controller, and the local "
        "deepbots/ folder must exist at the project root."
    ) from exc

from starter_controller import (  # noqa: E402
    ARENA_X_HALF,
    ARENA_Y_HALF,
    E2E_FEATURE_NAMES,
    E2E_REVERSE_SCALE,
    EndToEndActorPolicy,
    FIELD_X_HALF,
    FIELD_Y_HALF,
    GOAL_HALF_WIDTH,
    MAX_WHEEL_SPEED,
    OBSERVATION_FOV,
    RIGHT_GOAL,
    RL_FORWARD_SCALE,
    RL_TURN_SCALE,
    StudentController,
    clamp,
    clip_to_field,
    distance,
    polar_to_world,
    relative_polar,
    wrap_to_pi,
)


FEATURE_DIM = len(E2E_FEATURE_NAMES)
ACTION_DIM = 2
BALL_Z = 0.07
RUNTIME_WEIGHTS_FILENAME = EndToEndActorPolicy.RUNTIME_WEIGHTS_FILENAME
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


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        self.opponent_node = self.getFromDef("ROBOT_TWO")
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
        self.object_range_noise_pct = max(0.0, _env_float("RL_DEEPBOTS_OBJECT_RANGE_NOISE_PCT", 0.0))
        self.object_angle_noise = max(0.0, _env_float("RL_DEEPBOTS_OBJECT_ANGLE_NOISE", 0.0))
        self.ball_range_noise_pct = max(0.0, _env_float("RL_DEEPBOTS_BALL_RANGE_NOISE_PCT", self.object_range_noise_pct))
        self.ball_angle_noise = max(0.0, _env_float("RL_DEEPBOTS_BALL_ANGLE_NOISE", self.object_angle_noise))
        self.odometry_noise = max(0.0, _env_float("RL_DEEPBOTS_ODOMETRY_NOISE", 0.01))
        self.ball_dropout = clamp(_env_float("RL_DEEPBOTS_BALL_DROPOUT", 0.08 if self.phase >= 3 else 0.0), 0.0, 0.9)
        self.tournament_mirror = os.environ.get("RL_DEEPBOTS_TOURNAMENT_MIRROR", "1") != "0"
        reset_sampler = os.environ.get("RL_DEEPBOTS_RESET_SAMPLER", "").strip().lower().replace("-", "_")
        if not reset_sampler:
            reset_sampler = "full_field" if self.phase >= 3 else "central"
        if reset_sampler in ("legacy", "curriculum", "phase3"):
            reset_sampler = "central"
        if reset_sampler not in ("central", "full_field"):
            print(f"Unknown RL_DEEPBOTS_RESET_SAMPLER={reset_sampler!r}; using full_field.")
            reset_sampler = "full_field"
        self.reset_sampler = reset_sampler
        self.reset_ball_margin = clamp(
            _env_float("RL_DEEPBOTS_RESET_BALL_MARGIN", 0.12),
            0.02,
            min(FIELD_X_HALF, FIELD_Y_HALF) - 0.05,
        )
        self.reset_edge_band = clamp(
            _env_float("RL_DEEPBOTS_RESET_EDGE_BAND", 0.75),
            0.05,
            min(FIELD_X_HALF, FIELD_Y_HALF),
        )
        self.reset_edge_fraction = clamp(_env_float("RL_DEEPBOTS_RESET_EDGE_FRACTION", 0.30), 0.0, 1.0)
        self.reset_corner_fraction = clamp(_env_float("RL_DEEPBOTS_RESET_CORNER_FRACTION", 0.20), 0.0, 1.0)
        self.reset_behind_fraction = clamp(_env_float("RL_DEEPBOTS_RESET_BEHIND_FRACTION", 0.25), 0.0, 1.0)
        self.reset_min_robot_dist = clamp(_env_float("RL_DEEPBOTS_RESET_MIN_ROBOT_DIST", 0.35), 0.0, 2.0)
        self.attack_sign = 1.0
        self.prev_position = None
        self.prev_rotation = None
        os.environ.setdefault("WEBOTS_LIVE_VISUALIZER", "0")
        self.belief_controller = StudentController()

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

    def _attack_goal(self):
        return self.attack_sign * RIGHT_GOAL[0], RIGHT_GOAL[1]

    def _staging_point(self, ball_position):
        attack_goal = self._attack_goal()
        goal_dx = attack_goal[0] - ball_position[0]
        goal_dy = attack_goal[1] - ball_position[1]
        unit_x, unit_y = _normalize_xy(goal_dx, goal_dy)
        staging = (
            ball_position[0] - 0.28 * unit_x,
            ball_position[1] - 0.28 * unit_y,
        )
        return clip_to_field(staging, padding=0.20)

    def _is_visible(self, pose, point):
        _, angle = relative_polar(pose, point)
        return abs(angle) <= 0.5 * OBSERVATION_FOV

    def _get_polar_obs(self, pose, point, range_noise_pct=0.0, angle_noise=0.0, dropout=0.0):
        visible = self._is_visible(pose, point)
        if visible and dropout > 0.0 and self.rng.random() < dropout:
            visible = False

        if not visible:
            return None

        dist, angle = relative_polar(pose, point)
        noisy_dist = max(
            0.02,
            dist + float(self.rng.normal(0.0, range_noise_pct * max(dist, 0.5))),
        )
        noisy_angle = wrap_to_pi(angle + float(self.rng.normal(0.0, angle_noise)))
        return noisy_dist, noisy_angle

    def _provide_odometry(self):
        pose = self._robot_pose()
        rotation = pose[2]
        position = pose[:2]
        if self.prev_position is None:
            self.prev_position = position
        if self.prev_rotation is None:
            self.prev_rotation = rotation

        delta_forward = math.hypot(position[0] - self.prev_position[0], position[1] - self.prev_position[1])
        delta_rotation = rotation - self.prev_rotation
        if abs(delta_rotation) > math.pi:
            if rotation < 0:
                delta_rotation = (2.0 * math.pi + rotation) - self.prev_rotation
            else:
                delta_rotation = rotation - (2.0 * math.pi + self.prev_rotation)

        self.prev_position = position
        self.prev_rotation = rotation
        odometry = np.array([delta_forward, delta_rotation], dtype=np.float32)
        if self.odometry_noise > 0.0:
            odometry += self.rng.normal(0.0, self.odometry_noise, size=2).astype(np.float32)
        return odometry

    def _build_student_sensors(self):
        pose = self._robot_pose()
        ball = self._ball_position()
        ball_obs = self._get_polar_obs(
            pose,
            ball,
            range_noise_pct=self.ball_range_noise_pct,
            angle_noise=self.ball_angle_noise,
            dropout=self.ball_dropout,
        )
        goals = []
        for goal in ((4.5, 0.0), (-4.5, 0.0)):
            obs = self._get_polar_obs(
                pose,
                goal,
                range_noise_pct=self.object_range_noise_pct,
                angle_noise=self.object_angle_noise,
            )
            if obs is not None:
                goals.append(obs)

        crosses = []
        for cross in ((3.25, 0.0), (-3.25, 0.0)):
            obs = self._get_polar_obs(
                pose,
                cross,
                range_noise_pct=self.object_range_noise_pct,
                angle_noise=self.object_angle_noise,
            )
            if obs is not None:
                crosses.append(obs)

        corners = []
        for corner in ((-4.5, 3.0), (-4.5, -3.0), (4.5, 3.0), (4.5, -3.0)):
            obs = self._get_polar_obs(
                pose,
                corner,
                range_noise_pct=self.object_range_noise_pct,
                angle_noise=self.object_angle_noise,
            )
            if obs is not None:
                corners.append(obs)

        center = self._get_polar_obs(
            pose,
            (0.0, 0.0),
            range_noise_pct=self.object_range_noise_pct,
            angle_noise=self.object_angle_noise,
        )

        opponent = None
        if self.opponent_node is not None:
            opponent_position = self.opponent_node.getField("translation").getSFVec3f()[:2]
            opponent = self._get_polar_obs(
                pose,
                opponent_position,
                range_noise_pct=self.object_range_noise_pct,
                angle_noise=self.object_angle_noise,
            )

        return {
            "ball": ball_obs,
            "goal": goals,
            "center_circle": center,
            "penalty_cross": crosses,
            "corners": corners,
            "opponent": opponent,
            "odometry": self._provide_odometry(),
            "debug_truth": {
                "robot_pose": [float(pose[0]), float(pose[1]), float(pose[2])],
                "ball_position": [float(ball[0]), float(ball[1])],
            },
        }

    def _metrics(self):
        pose = self._robot_pose()
        ball = self._ball_position()
        return {
            "pose": pose,
            "ball": ball,
            "robot_to_ball": distance(pose[:2], ball),
            "ball_to_goal": distance(ball, self._attack_goal()),
            "wall_margin": self._wall_margin(pose),
            "correct_goal": self._is_correct_goal(ball),
            "wrong_goal": self._is_wrong_goal(ball),
        }

    def _is_correct_goal(self, ball):
        return self.attack_sign * ball[0] > FIELD_X_HALF + 0.02 and abs(ball[1]) <= GOAL_HALF_WIDTH

    def _is_wrong_goal(self, ball):
        return self.attack_sign * ball[0] < -FIELD_X_HALF - 0.02 and abs(ball[1]) <= GOAL_HALF_WIDTH

    def _is_out_of_play(self, ball):
        return (
            abs(ball[0]) > ARENA_X_HALF - 0.04
            or abs(ball[1]) > ARENA_Y_HALF - 0.04
        )

    def _phase3_pose_and_ball(self, internal_ball):
        if self.attack_sign > 0.0:
            return (-1.0, 0.0, 0.0), internal_ball
        return (1.0, 0.0, math.pi), (-internal_ball[0], -internal_ball[1])

    def _is_valid_phase3_internal_ball(self, internal_ball):
        return math.hypot(internal_ball[0] + 1.0, internal_ball[1]) >= self.reset_min_robot_dist

    def _sample_phase3_central_ball(self):
        return (
            float(self.rng.uniform(-1.25, 1.50)),
            float(self.rng.uniform(-1.50, 1.50)),
        )

    def _sample_near_field_edge(self, half_extent):
        sign = -1.0 if self.rng.random() < 0.5 else 1.0
        outer = max(self.reset_ball_margin, half_extent - self.reset_ball_margin)
        inner = max(self.reset_ball_margin, half_extent - self.reset_edge_band)
        if inner >= outer:
            value = outer
        else:
            value = float(self.rng.uniform(inner, outer))
        return sign * value

    def _sample_phase3_full_field_ball(self):
        min_x = -FIELD_X_HALF + self.reset_ball_margin
        max_x = FIELD_X_HALF - self.reset_ball_margin
        min_y = -FIELD_Y_HALF + self.reset_ball_margin
        max_y = FIELD_Y_HALF - self.reset_ball_margin

        corner_fraction = self.reset_corner_fraction
        edge_fraction = self.reset_edge_fraction
        behind_fraction = self.reset_behind_fraction
        special_total = corner_fraction + edge_fraction + behind_fraction
        if special_total > 1.0:
            corner_fraction /= special_total
            edge_fraction /= special_total
            behind_fraction /= special_total

        draw = self.rng.random()
        if draw < corner_fraction:
            return (
                self._sample_near_field_edge(FIELD_X_HALF),
                self._sample_near_field_edge(FIELD_Y_HALF),
            )
        draw -= corner_fraction

        if draw < edge_fraction:
            if self.rng.random() < 0.5:
                return (
                    self._sample_near_field_edge(FIELD_X_HALF),
                    float(self.rng.uniform(min_y, max_y)),
                )
            return (
                float(self.rng.uniform(min_x, max_x)),
                self._sample_near_field_edge(FIELD_Y_HALF),
            )
        draw -= edge_fraction

        if draw < behind_fraction:
            behind_max_x = min(-1.02, max_x)
            if behind_max_x > min_x:
                return (
                    float(self.rng.uniform(min_x, behind_max_x)),
                    float(self.rng.uniform(min_y, max_y)),
                )

        return (
            float(self.rng.uniform(min_x, max_x)),
            float(self.rng.uniform(min_y, max_y)),
        )

    def _sample_initial_state(self):
        self.attack_sign = 1.0
        if self.phase >= 3:
            self.attack_sign = -1.0 if self.tournament_mirror and self.rng.random() < 0.5 else 1.0
            sampler = self._sample_phase3_full_field_ball if self.reset_sampler == "full_field" else self._sample_phase3_central_ball
            for _ in range(400):
                internal_ball = sampler()
                if not self._is_valid_phase3_internal_ball(internal_ball):
                    continue
                return self._phase3_pose_and_ball(internal_ball)
            return self._phase3_pose_and_ball((0.0, 0.0))

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
        self.prev_position = None
        self.prev_rotation = None
        self.belief_controller.close()
        self.belief_controller = StudentController()
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        pose, ball = self._sample_initial_state()
        self._place_node(self.robot_node, pose[0], pose[1], 0.0, theta=pose[2])
        self._place_node(self.ball_node, ball[0], ball[1], BALL_Z)
        self.simulationResetPhysics()
        super(Supervisor, self).step(self.timestep)
        current_pose = self._robot_pose()
        self.prev_position = current_pose[:2]
        self.prev_rotation = current_pose[2]

        self._prev_metrics = self._metrics()
        self.last_observation = self.get_observations()
        return self.last_observation.copy()

    def get_observations(self):
        sensors = self._build_student_sensors()
        features, _, _ = self.belief_controller.observe_e2e_features(sensors)
        self.last_observation = features
        return features.copy()

    def apply_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < ACTION_DIM:
            action = np.pad(action, (0, ACTION_DIM - action.shape[0]))

        a_forward = clamp(float(action[0]), -1.0, 1.0)
        a_turn = clamp(float(action[1]), -1.0, 1.0)
        forward = RL_FORWARD_SCALE * a_forward if a_forward >= 0.0 else E2E_REVERSE_SCALE * a_forward
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
            "attack_sign": float(self.attack_sign),
        }

    def _log_transition(self, prev_obs, obs, reward, done, info):
        self.logger.log(
            {
                "episode": self.episode,
                "step": self.global_step,
                "episode_step": self.episode_step,
                "state": f"E2E_{os.environ.get('RL_DEEPBOTS_MODE', 'train').upper()}",
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
    policy = EndToEndActorPolicy()
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
    actor = EndToEndActorPolicy()
    obs = env.reset()
    while True:
        action = actor(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()


def _policy_weight_mtime(path):
    if path is None:
        return None
    try:
        return Path(path).expanduser().stat().st_mtime_ns
    except OSError:
        return None


def _async_policy_path():
    raw_path = (
        os.environ.get("RL_ASYNC_POLICY_WEIGHTS_PATH")
        or os.environ.get("E2E_POLICY_WEIGHTS_PATH")
        or os.environ.get("RL_POLICY_WEIGHTS_PATH")
    )
    if raw_path:
        return Path(raw_path).expanduser()
    if RUNTIME_WEIGHTS_PATH.exists():
        return RUNTIME_WEIGHTS_PATH
    return None


def run_async_actor_worker(env):
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("RL_DEEPBOTS_MODE=async_actor requires pyzmq.") from exc

    endpoint = os.environ.get("RL_ASYNC_TRANSITION_PUSH", "tcp://127.0.0.1:5557")
    worker_id = os.environ.get("RL_ASYNC_WORKER_ID", str(os.getpid()))
    reload_every = max(1, _env_int("RL_ASYNC_RELOAD_EVERY", 250))
    log_period = max(1, _env_int("RL_ASYNC_LOG_PERIOD", 1000))
    max_steps = max(0, _env_int("RL_ASYNC_MAX_STEPS", 0))
    send_hwm = max(100, _env_int("RL_ASYNC_SEND_HWM", 10000))

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, send_hwm)
    socket.connect(endpoint)

    policy_path = _async_policy_path()
    if policy_path is not None:
        os.environ["E2E_POLICY_WEIGHTS_PATH"] = str(policy_path)
    actor = EndToEndActorPolicy()
    actor_mtime = _policy_weight_mtime(policy_path)

    obs = env.reset()
    step_count = 0
    episode_count = 0
    episode_return = 0.0
    dropped = 0
    print(
        "async actor worker: "
        f"id={worker_id} endpoint={endpoint} policy={policy_path} "
        f"reload_every={reload_every} max_steps={max_steps or 'unbounded'}"
    )

    while True:
        if policy_path is not None and step_count % reload_every == 0:
            current_mtime = _policy_weight_mtime(policy_path)
            if current_mtime is not None and current_mtime != actor_mtime:
                actor = EndToEndActorPolicy()
                actor_mtime = current_mtime
                print(f"async actor worker={worker_id} reloaded policy {policy_path}")

        action = np.asarray(actor(obs), dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        next_obs, reward, done, info = env.step(action)
        step_count += 1
        episode_return += float(reward)

        payload = {
            "worker_id": worker_id,
            "step": step_count,
            "episode_index": episode_count,
            "observations": [float(value) for value in obs],
            "actions": [float(value) for value in action],
            "next_observations": [float(value) for value in next_obs],
            "rewards": float(reward),
            "masks": 0.0 if done else 1.0,
            "dones": bool(done),
            "info": info,
        }
        try:
            socket.send_json(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            dropped += 1

        if done:
            episode_count += 1
            if info.get("correct_goal"):
                outcome = "correct_goal"
            elif info.get("wrong_goal"):
                outcome = "wrong_goal"
            elif int(info.get("episode_step", 0)) >= env.max_episode_steps:
                outcome = "timeout"
            else:
                outcome = "out_of_play"
            print(
                f"async actor worker={worker_id} episode={episode_count} "
                f"outcome={outcome} steps={info.get('episode_step')} "
                f"return={episode_return:.3f} dropped={dropped}"
            )
            obs = env.reset()
            episode_return = 0.0
        else:
            obs = next_obs

        if step_count == 1 or step_count % log_period == 0:
            print(
                f"async actor worker={worker_id} step={step_count} "
                f"episodes={episode_count} dropped={dropped}"
            )

        if max_steps and step_count >= max_steps:
            print(f"async actor worker={worker_id} reached max_steps={max_steps}")
            env.close()
            env.simulationQuit(0)
            return


def run_random_loop(env):
    obs = env.reset()
    del obs
    while True:
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()


def run_teleop_loop(env):
    keyboard = Keyboard()
    keyboard.enable(env.timestep)
    obs = env.reset()
    print(
        "Teleop demo collection: W/Up forward, S/Down reverse, "
        "A/Left turn left, D/Right turn right, Space stop, R reset, Q quit."
    )

    while True:
        linear = 0.0
        angular = 0.0
        reset_requested = False
        quit_requested = False

        key = keyboard.getKey()
        while key != -1:
            if key in (Keyboard.UP, ord("W"), ord("w")):
                linear += 0.85
            elif key in (Keyboard.DOWN, ord("S"), ord("s")):
                linear -= 0.60
            elif key in (Keyboard.LEFT, ord("A"), ord("a")):
                angular += 0.80
            elif key in (Keyboard.RIGHT, ord("D"), ord("d")):
                angular -= 0.80
            elif key in (ord(" "),):
                linear = 0.0
                angular = 0.0
            elif key in (ord("R"), ord("r")):
                reset_requested = True
            elif key in (ord("Q"), ord("q")):
                quit_requested = True
            key = keyboard.getKey()

        if quit_requested:
            env.close()
            env.simulationQuit(0)
            return

        if reset_requested:
            obs = env.reset()
            continue

        action = np.array(
            [clamp(linear, -1.0, 1.0), clamp(angular, -1.0, 1.0)],
            dtype=np.float32,
        )
        obs, reward, done, info = env.step(action)
        del obs, reward
        if done:
            if info.get("correct_goal"):
                outcome = "correct_goal"
            elif info.get("wrong_goal"):
                outcome = "wrong_goal"
            elif int(info.get("episode_step", 0)) >= env.max_episode_steps:
                outcome = "timeout"
            else:
                outcome = "out_of_play"
            print(f"teleop episode done: outcome={outcome} steps={info.get('episode_step')}")
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


def run_eval_loop(env):
    policy_name = os.environ.get("RL_DEEPBOTS_EVAL_POLICY", "actor").strip().lower()
    episodes = max(1, _env_int("RL_DEEPBOTS_EVAL_EPISODES", 20))
    output_path = os.environ.get("RL_DEEPBOTS_EVAL_OUTPUT")

    if policy_name == "sac":
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise RuntimeError(
                "stable-baselines3 is required for RL_DEEPBOTS_EVAL_POLICY=sac. "
                "Run: python scripts/rl_workflow.py install-rl-deps"
            ) from exc
        model_path = os.environ.get("RL_DEEPBOTS_MODEL_PATH")
        if not model_path:
            raise RuntimeError("Set RL_DEEPBOTS_MODEL_PATH for SAC evaluation.")
        model = SAC.load(model_path)

        def act(observation):
            action, _ = model.predict(observation, deterministic=True)
            return action

    elif policy_name in ("actor", "bootstrap"):
        actor = EndToEndActorPolicy()

        def act(observation):
            if policy_name == "bootstrap":
                return np.zeros(ACTION_DIM, dtype=np.float32)
            return actor(observation)

    else:
        raise RuntimeError(f"Unsupported RL_DEEPBOTS_EVAL_POLICY={policy_name!r}")

    results = []
    for episode_index in range(1, episodes + 1):
        obs = env.reset()
        done = False
        info = env.get_info()
        total_reward = 0.0
        while not done:
            obs, reward, done, info = env.step(act(obs))
            total_reward += float(reward)

        if info.get("correct_goal"):
            outcome = "correct_goal"
        elif info.get("wrong_goal"):
            outcome = "wrong_goal"
        elif int(info.get("episode_step", 0)) >= env.max_episode_steps:
            outcome = "timeout"
        else:
            outcome = "out_of_play"

        results.append(
            {
                "episode": episode_index,
                "outcome": outcome,
                "episode_step": int(info.get("episode_step", 0)),
                "return": total_reward,
                "correct_goal": bool(info.get("correct_goal")),
                "wrong_goal": bool(info.get("wrong_goal")),
                "attack_sign": float(info.get("attack_sign", 1.0)),
                "ball_position": info.get("ball_position"),
                "robot_pose": info.get("robot_pose"),
            }
        )
        print(
            f"eval episode={episode_index}/{episodes} "
            f"outcome={outcome} steps={results[-1]['episode_step']} "
            f"return={total_reward:.3f}"
        )

    successes = sum(result["outcome"] == "correct_goal" for result in results)
    wrong_goals = sum(result["outcome"] == "wrong_goal" for result in results)
    timeouts = sum(result["outcome"] == "timeout" for result in results)
    summary = {
        "policy": policy_name,
        "episodes_requested": episodes,
        "episodes_completed": len(results),
        "successes": successes,
        "wrong_goals": wrong_goals,
        "timeouts": timeouts,
        "success_rate": successes / len(results) if results else 0.0,
        "mean_episode_steps": float(np.mean([result["episode_step"] for result in results])) if results else 0.0,
        "results": results,
    }
    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote eval summary to {path}")

    env.close()
    env.simulationQuit(0)


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


def _serl_demo_patterns():
    raw = os.environ.get("RL_DEEPBOTS_DEMO_TRACES", "")
    return [item for item in raw.split(os.pathsep) if item]


def _serl_transition(obs, action, next_obs, reward, done):
    return {
        "observations": np.asarray(obs, dtype=np.float32),
        "actions": np.asarray(action, dtype=np.float32),
        "next_observations": np.asarray(next_obs, dtype=np.float32),
        "rewards": np.float32(reward),
        "masks": np.float32(0.0 if done else 1.0),
        "dones": bool(done),
    }


def _save_serl_actor(agent, unfreeze, actor_arrays_from_params, output_path):
    params = unfreeze(agent.state.params)
    arrays = actor_arrays_from_params(params, FEATURE_DIM)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"Exported SERL E2E runtime actor weights to {path}")


def _serl_update(agent, replay_buffer, jax, jnp, tree_to_jax, total_batch_size, utd_ratio):
    batch = replay_buffer.sample(batch_size=total_batch_size)
    batch = tree_to_jax(jax, jnp, batch)
    return agent.update_high_utd(batch, utd_ratio=utd_ratio)


def _concat_serl_batches(jax, jnp, first_batch, second_batch):
    tree_map = getattr(jax, "tree_map", None)
    if tree_map is None:
        tree_map = jax.tree_util.tree_map
    return tree_map(lambda first, second: jnp.concatenate((first, second), axis=0), first_batch, second_batch)


def _serl_mixed_update(
    agent,
    replay_buffer,
    demo_buffer,
    jax,
    jnp,
    tree_to_jax,
    online_batch_size,
    demo_batch_size,
    utd_ratio,
):
    online_batch = tree_to_jax(jax, jnp, replay_buffer.sample(batch_size=online_batch_size))
    if demo_buffer is None or demo_batch_size <= 0:
        batch = online_batch
    else:
        demo_batch = tree_to_jax(jax, jnp, demo_buffer.sample(batch_size=demo_batch_size))
        batch = _concat_serl_batches(jax, jnp, online_batch, demo_batch)
    return agent.update_high_utd(batch, utd_ratio=utd_ratio)


def run_serl_training(env):
    try:
        from serl_e2e_train import (
            _actor_arrays_from_params,
            _collect_trace_files,
            _create_agent,
            _import_serl,
            _insert_transitions,
            _load_transitions,
            _tree_to_jax,
        )
    except ImportError as exc:
        raise RuntimeError(
            "SERL training support is missing. Run through a Python environment "
            "that can import scripts/serl_e2e_train.py and SERL/JAX."
        ) from exc

    serl_root = Path(os.environ.get("SERL_ROOT", "~/serl")).expanduser()
    gym_serl, nn, jax, jnp, serl_unfreeze, SACAgent, ReplayBuffer = _import_serl(serl_root)
    unfreeze = serl_unfreeze

    demo_patterns = _serl_demo_patterns()
    demo_files = _collect_trace_files(demo_patterns) if demo_patterns else []
    success_only = _env_bool("RL_DEEPBOTS_DEMO_SUCCESS_ONLY", False)
    demo_transitions = _load_transitions(demo_files, FEATURE_DIM, success_only=success_only) if demo_files else []
    if not demo_transitions and not _env_bool("RL_DEEPBOTS_ALLOW_EMPTY_DEMOS", False):
        raise RuntimeError(
            "SERL online training needs teleop demos for sample efficiency. "
            "Set RL_DEEPBOTS_DEMO_TRACES or RL_DEEPBOTS_ALLOW_EMPTY_DEMOS=1."
        )

    total_steps = max(1, _env_int("RL_DEEPBOTS_TOTAL_STEPS", 50000))
    batch_size = max(16, _env_int("RL_DEEPBOTS_BATCH_SIZE", 256))
    utd_ratio = max(1, _env_int("RL_DEEPBOTS_UTD_RATIO", 8))
    total_update_batch_size = batch_size * utd_ratio
    online_batch_size = total_update_batch_size
    demo_batch_size = 0
    if demo_transitions:
        online_batch_size = max(1, total_update_batch_size // 2)
        demo_batch_size = total_update_batch_size - online_batch_size

    replay_capacity = max(
        _env_int("RL_DEEPBOTS_BUFFER_SIZE", 200000),
        total_steps + total_update_batch_size,
    )
    demo_capacity = max(
        len(demo_transitions),
        demo_batch_size,
        total_update_batch_size,
    )
    pretrain_steps = max(0, _env_int("RL_DEEPBOTS_SERL_PRETRAIN_STEPS", 500 if demo_transitions else 0))
    learning_starts = max(0, _env_int("RL_DEEPBOTS_LEARNING_STARTS", online_batch_size if demo_transitions else total_update_batch_size))
    random_steps = max(0, _env_int("RL_DEEPBOTS_RANDOM_STEPS", 0 if demo_transitions else 1000))
    update_every = max(1, _env_int("RL_DEEPBOTS_UPDATE_EVERY", 1))
    updates_per_step = max(1, _env_int("RL_DEEPBOTS_UPDATES_PER_STEP", 1))
    log_period = max(1, _env_int("RL_DEEPBOTS_LOG_PERIOD", 100))
    export_path = os.environ.get("RL_DEEPBOTS_EXPORT_PATH", str(RUNTIME_WEIGHTS_PATH))

    obs_space = gym_serl.spaces.Box(
        low=-np.ones(FEATURE_DIM, dtype=np.float32),
        high=np.ones(FEATURE_DIM, dtype=np.float32),
        shape=(FEATURE_DIM,),
        dtype=np.float32,
    )
    action_space = gym_serl.spaces.Box(
        low=-np.ones(ACTION_DIM, dtype=np.float32),
        high=np.ones(ACTION_DIM, dtype=np.float32),
        shape=(ACTION_DIM,),
        dtype=np.float32,
    )
    replay_buffer = ReplayBuffer(obs_space, action_space, capacity=replay_capacity)
    demo_buffer = None
    if demo_transitions:
        demo_buffer = ReplayBuffer(obs_space, action_space, capacity=demo_capacity)
        _insert_transitions(demo_buffer, demo_transitions)
    print(
        "SERL online training: "
        f"demos={len(demo_transitions)} files={len(demo_files)} "
        f"steps={total_steps} batch={batch_size} utd={utd_ratio} "
        f"online_batch={online_batch_size} demo_batch={demo_batch_size} "
        f"replay_capacity={replay_capacity} demo_capacity={demo_capacity}"
    )

    agent = _create_agent(
        SACAgent,
        nn,
        jax,
        FEATURE_DIM,
        seed=_env_int("RL_DEEPBOTS_SEED", 7),
        lr=_env_float("RL_DEEPBOTS_LR", 3e-4),
        gamma=_env_float("RL_DEEPBOTS_GAMMA", 0.995),
        critic_ensemble_size=max(1, _env_int("RL_DEEPBOTS_CRITIC_ENSEMBLE_SIZE", 10)),
        critic_subsample_size=max(1, _env_int("RL_DEEPBOTS_CRITIC_SUBSAMPLE_SIZE", 2)),
    )

    for update_step in range(1, pretrain_steps + 1):
        if demo_buffer is None:
            break
        agent, info = _serl_update(agent, demo_buffer, jax, jnp, _tree_to_jax, total_update_batch_size, utd_ratio)
        if update_step == 1 or update_step % log_period == 0:
            scalar_info = _flatten_scalar_info(info)
            print(f"serl pretrain update={update_step}: {json.dumps(scalar_info, sort_keys=True)}")

    rng = jax.random.PRNGKey(_env_int("RL_DEEPBOTS_SEED", 7) + 1009)
    obs = env.reset()
    episode_return = 0.0
    episode_count = 0
    update_count = pretrain_steps
    last_info = {}

    for step in range(1, total_steps + 1):
        if step <= random_steps:
            action = env.action_space.sample()
        else:
            rng, action_key = jax.random.split(rng)
            action = np.asarray(
                agent.sample_actions(jnp.asarray(obs[None, :]), seed=action_key)[0],
                dtype=np.float32,
            )
            action = np.clip(action, -1.0, 1.0)

        next_obs, reward, done, info = env.step(action)
        replay_buffer.insert(_serl_transition(obs, action, next_obs, reward, done))
        episode_return += float(reward)
        last_info = info

        if step >= learning_starts and len(replay_buffer) >= online_batch_size and step % update_every == 0:
            for _ in range(updates_per_step):
                agent, train_info = _serl_mixed_update(
                    agent,
                    replay_buffer,
                    demo_buffer,
                    jax,
                    jnp,
                    _tree_to_jax,
                    online_batch_size,
                    demo_batch_size,
                    utd_ratio,
                )
                update_count += 1
        else:
            train_info = {}

        if done:
            episode_count += 1
            if info.get("correct_goal"):
                outcome = "correct_goal"
            elif info.get("wrong_goal"):
                outcome = "wrong_goal"
            elif int(info.get("episode_step", 0)) >= env.max_episode_steps:
                outcome = "timeout"
            else:
                outcome = "out_of_play"
            print(
                f"serl episode={episode_count} outcome={outcome} "
                f"steps={info.get('episode_step')} return={episode_return:.3f} "
                f"online_replay={len(replay_buffer)} "
                f"demo_replay={0 if demo_buffer is None else len(demo_buffer)} "
                f"updates={update_count}"
            )
            obs = env.reset()
            episode_return = 0.0
        else:
            obs = next_obs

        if step == 1 or step % log_period == 0:
            scalar_info = _flatten_scalar_info(train_info)
            print(
                f"serl step={step}/{total_steps} online_replay={len(replay_buffer)} "
                f"demo_replay={0 if demo_buffer is None else len(demo_buffer)} "
                f"updates={update_count} last_reward={float(reward):.3f} "
                f"ball_to_goal={float(last_info.get('ball_to_goal', 0.0)):.3f} "
                f"info={json.dumps(scalar_info, sort_keys=True)}"
            )

    _save_serl_actor(agent, unfreeze, _actor_arrays_from_params, export_path)
    env.close()
    env.simulationQuit(0)


def main():
    env = SoccerSoloDeepbotsEnv()
    mode = os.environ.get("RL_DEEPBOTS_MODE", "actor").strip().lower()
    try:
        if mode == "actor":
            run_actor_loop(env)
        elif mode == "async_actor":
            run_async_actor_worker(env)
        elif mode == "random":
            run_random_loop(env)
        elif mode == "teleop":
            run_teleop_loop(env)
        elif mode == "eval":
            run_eval_loop(env)
        elif mode == "sac_train":
            run_sac_training(env)
        elif mode == "serl_train":
            run_serl_training(env)
        elif mode == "sac_eval":
            run_sac_eval(env)
        else:
            raise RuntimeError(f"Unsupported RL_DEEPBOTS_MODE={mode!r}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
