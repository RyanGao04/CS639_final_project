"""student_controller controller."""

import atexit
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import numpy as np


FIELD_X_HALF = 4.5
FIELD_Y_HALF = 3.0
RIGHT_GOAL = (4.5, 0.0)
LEFT_GOAL = (-4.5, 0.0)
GOAL_DEPTH = 0.6
GOAL_HALF_WIDTH = 0.8
ARENA_X_HALF = 5.1
ARENA_Y_HALF = 3.6
CORNERS = [(-4.5, 3.0), (-4.5, -3.0), (4.5, 3.0), (4.5, -3.0)]
PENALTY_CROSSES = [(3.25, 0.0), (-3.25, 0.0)]
CENTER_CIRCLE = (0.0, 0.0)
MAX_WHEEL_SPEED = 6.0
RL_FORWARD_SCALE = 5.6
RL_TURN_SCALE = 3.2
VISUALIZER_DEFAULT_ENABLED = True
OBSERVATION_FOV = math.pi / 2.0


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def relative_polar(pose, point):
    dx = point[0] - pose[0]
    dy = point[1] - pose[1]
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    return dist, wrap_to_pi(heading - pose[2])


def point_in_fov(pose, point, fov=OBSERVATION_FOV):
    _, rel_angle = relative_polar(pose, point)
    return abs(rel_angle) <= 0.5 * fov


def polar_to_world(pose, obs):
    dist, rel_angle = obs
    world_angle = pose[2] + rel_angle
    return (
        pose[0] + dist * math.cos(world_angle),
        pose[1] + dist * math.sin(world_angle),
    )


def clip_to_field(point, padding=0.05):
    return (
        clamp(point[0], -FIELD_X_HALF + padding, FIELD_X_HALF - padding),
        clamp(point[1], -FIELD_Y_HALF + padding, FIELD_Y_HALF - padding),
    )


def clip_to_arena(point, padding=0.05):
    return (
        clamp(point[0], -ARENA_X_HALF + padding, ARENA_X_HALF - padding),
        clamp(point[1], -ARENA_Y_HALF + padding, ARENA_Y_HALF - padding),
    )


def clip_ball_position(point, padding=0.05):
    return clip_to_arena(point, padding=padding)


@dataclass
class PoseHypothesis:
    x: float
    y: float
    theta: float
    validity: float
    weight: float

    def copy(self):
        return PoseHypothesis(
            self.x,
            self.y,
            self.theta,
            self.validity,
            self.weight,
        )


class MultiHypothesisLocalizer:
    def __init__(self):
        self.num_hypotheses = 35
        self.base_validity_weight = 0.1
        self.rng = np.random.default_rng(7)
        self.landmark_specs = {
            "goal": {
                "points": [RIGHT_GOAL, LEFT_GOAL],
                "sigma_r": 0.20,
                "sigma_a": 0.15,
                "gate": 0.85,
            },
            "center_circle": {
                "points": [CENTER_CIRCLE],
                "sigma_r": 0.18,
                "sigma_a": 0.12,
                "gate": 0.50,
            },
            "penalty_cross": {
                "points": PENALTY_CROSSES,
                "sigma_r": 0.18,
                "sigma_a": 0.12,
                "gate": 0.60,
            },
            "corners": {
                "points": CORNERS,
                "sigma_r": 0.25,
                "sigma_a": 0.14,
                "gate": 0.90,
            },
        }
        self.pose = (-1.0, 0.0, 0.0)
        self.confidence = 0.5
        self.hypotheses = self._initial_hypotheses()

    def _initial_hypotheses(self):
        hypotheses = []
        headings_per_x = 5
        x_columns = self.num_hypotheses // headings_per_x
        x_mid = x_columns // 2
        theta_mid = headings_per_x // 2
        dx_step = 0.10
        dtheta_step = math.radians(5.0)
        for i in range(self.num_hypotheses):
            x_bucket = i // headings_per_x
            theta_bucket = i % headings_per_x
            dx = (x_bucket - x_mid) * dx_step
            dtheta = (theta_bucket - theta_mid) * dtheta_step
            dy = float(self.rng.normal(0.0, 0.06))
            hypotheses.append(
                PoseHypothesis(
                    x=-1.0 + dx,
                    y=dy,
                    theta=dtheta,
                    validity=0.5,
                    weight=1.0 / self.num_hypotheses,
                )
            )
        return hypotheses

    def predict(self, odometry):
        if odometry is None:
            return

        forward = float(odometry[0])
        rotation = float(odometry[1])
        trans_noise = 0.01 + 0.12 * abs(forward)
        lateral_noise = 0.004 + 0.05 * abs(forward)
        rot_noise = math.radians(1.5) + 0.20 * abs(rotation) + 0.04 * abs(forward)

        for hypothesis in self.hypotheses:
            noisy_forward = forward + float(self.rng.normal(0.0, trans_noise))
            noisy_lateral = float(self.rng.normal(0.0, lateral_noise))
            noisy_rotation = rotation + float(self.rng.normal(0.0, rot_noise))
            mid_heading = hypothesis.theta + 0.5 * noisy_rotation
            hypothesis.x += noisy_forward * math.cos(mid_heading) - noisy_lateral * math.sin(mid_heading)
            hypothesis.y += noisy_forward * math.sin(mid_heading) + noisy_lateral * math.cos(mid_heading)
            hypothesis.theta = wrap_to_pi(hypothesis.theta + noisy_rotation)
            hypothesis.x = clamp(hypothesis.x, -FIELD_X_HALF - 0.5, FIELD_X_HALF + 0.5)
            hypothesis.y = clamp(hypothesis.y, -FIELD_Y_HALF - 0.5, FIELD_Y_HALF + 0.5)

    def update(self, sensors):
        measurements, total_count = self._collect_measurements(sensors)
        if total_count == 0:
            for hypothesis in self.hypotheses:
                hypothesis.validity *= 0.995
                hypothesis.weight = 1.0 / self.num_hypotheses
            self.pose = self._estimate_pose()
            self.confidence = max(h.validity for h in self.hypotheses)
            return self.pose

        for hypothesis in self.hypotheses:
            self._update_hypothesis(hypothesis, measurements, total_count)

        self._normalize_weights()
        self.pose = self._estimate_pose()
        self.confidence = max(h.validity for h in self.hypotheses)
        self._resample()
        return self.pose

    def _collect_measurements(self, sensors):
        measurements = {}
        for key in ("goal", "penalty_cross", "corners"):
            values = sensors.get(key, [])
            if values:
                measurements[key] = [tuple(v) for v in values]
        center_obs = sensors.get("center_circle")
        if center_obs is not None:
            measurements["center_circle"] = [tuple(center_obs)]
        total_count = sum(len(values) for values in measurements.values())
        return measurements, total_count

    def _associate_landmark(self, hypothesis, obs, points, gate):
        obs_world = polar_to_world((hypothesis.x, hypothesis.y, hypothesis.theta), obs)
        best_point = None
        best_dist_sq = None
        for point in points:
            dist_sq = (obs_world[0] - point[0]) ** 2 + (obs_world[1] - point[1]) ** 2
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_point = point
        if best_dist_sq is None or best_dist_sq > gate * gate:
            return None
        return best_point

    def _nudge_towards_landmark(self, hypothesis, obs, landmark):
        world_bearing = math.atan2(landmark[1] - hypothesis.y, landmark[0] - hypothesis.x)
        target_theta = wrap_to_pi(world_bearing - obs[1])
        theta_error = wrap_to_pi(target_theta - hypothesis.theta)
        hypothesis.theta = wrap_to_pi(hypothesis.theta + 0.18 * theta_error)

        inferred_x = landmark[0] - obs[0] * math.cos(hypothesis.theta + obs[1])
        inferred_y = landmark[1] - obs[0] * math.sin(hypothesis.theta + obs[1])
        hypothesis.x += 0.22 * (inferred_x - hypothesis.x)
        hypothesis.y += 0.22 * (inferred_y - hypothesis.y)

    def _update_hypothesis(self, hypothesis, measurements, total_count):
        matched = 0
        error_sum = 0.0
        pose = (hypothesis.x, hypothesis.y, hypothesis.theta)

        for name, obs_list in measurements.items():
            spec = self.landmark_specs[name]
            for obs in obs_list:
                landmark = self._associate_landmark(hypothesis, obs, spec["points"], spec["gate"])
                if landmark is None:
                    error_sum += 6.0
                    continue

                matched += 1
                predicted_dist, predicted_angle = relative_polar(pose, landmark)
                dist_error = obs[0] - predicted_dist
                angle_error = wrap_to_pi(obs[1] - predicted_angle)
                error_sum += (dist_error / spec["sigma_r"]) ** 2
                error_sum += (angle_error / spec["sigma_a"]) ** 2
                self._nudge_towards_landmark(hypothesis, obs, landmark)
                pose = (hypothesis.x, hypothesis.y, hypothesis.theta)

        current_validity = matched / float(total_count)
        hypothesis.validity = 0.92 * hypothesis.validity + 0.08 * current_validity
        hypothesis.weight = (
            self.base_validity_weight
            + (1.0 - self.base_validity_weight) * hypothesis.validity
        ) * math.exp(-0.5 * min(error_sum, 60.0))

    def _normalize_weights(self):
        total_weight = sum(h.weight for h in self.hypotheses)
        if total_weight <= 1e-9:
            uniform = 1.0 / self.num_hypotheses
            for hypothesis in self.hypotheses:
                hypothesis.weight = uniform
            return

        for hypothesis in self.hypotheses:
            hypothesis.weight /= total_weight

    def _estimate_pose(self):
        weight_sum = sum(h.weight for h in self.hypotheses)
        if weight_sum <= 1e-9:
            return self.pose

        x = sum(h.weight * h.x for h in self.hypotheses) / weight_sum
        y = sum(h.weight * h.y for h in self.hypotheses) / weight_sum
        cos_sum = sum(h.weight * math.cos(h.theta) for h in self.hypotheses)
        sin_sum = sum(h.weight * math.sin(h.theta) for h in self.hypotheses)
        theta = math.atan2(sin_sum, cos_sum)
        return x, y, theta

    def _resample(self):
        weights = np.array([h.weight for h in self.hypotheses], dtype=float)
        if weights.sum() <= 1e-9:
            return
        weights /= weights.sum()

        best_index = int(np.argmax(weights))
        best = self.hypotheses[best_index].copy()

        cumulative = np.cumsum(weights)
        step = 1.0 / self.num_hypotheses
        start = float(self.rng.uniform(0.0, step))
        new_hypotheses = [best]
        idx = 0

        for sample_idx in range(1, self.num_hypotheses):
            threshold = start + sample_idx * step
            while idx < self.num_hypotheses - 1 and threshold > cumulative[idx]:
                idx += 1
            source = self.hypotheses[idx]
            new_hypotheses.append(
                PoseHypothesis(
                    x=source.x + float(self.rng.normal(0.0, 0.03)),
                    y=source.y + float(self.rng.normal(0.0, 0.03)),
                    theta=wrap_to_pi(source.theta + float(self.rng.normal(0.0, math.radians(1.5)))),
                    validity=source.validity,
                    weight=1.0 / self.num_hypotheses,
                )
            )

        best.weight = 1.0 / self.num_hypotheses
        self.hypotheses = new_hypotheses

    def debug_landmark_estimates(self, sensors, pose=None):
        if pose is None:
            pose = self.pose

        measurements, _ = self._collect_measurements(sensors)
        projected = {key: [] for key in self.landmark_specs}
        matched = {key: [] for key in self.landmark_specs}
        all_landmarks = {}
        visible_landmarks = {}
        hypothesis = PoseHypothesis(
            x=pose[0],
            y=pose[1],
            theta=pose[2],
            validity=1.0,
            weight=1.0,
        )

        for name, spec in self.landmark_specs.items():
            points = [tuple(point) for point in spec["points"]]
            all_landmarks[name] = points
            visible_landmarks[name] = [
                point for point in points if point_in_fov(pose, point)
            ]

        for name, obs_list in measurements.items():
            spec = self.landmark_specs[name]
            for obs in obs_list:
                point = polar_to_world(pose, obs)
                projected[name].append(point)
                match = self._associate_landmark(hypothesis, obs, spec["points"], spec["gate"])
                if match is not None:
                    matched[name].append(match)

        return {
            "all": all_landmarks,
            "visible": visible_landmarks,
            "projected": projected,
            "matched": matched,
        }


class BallTracker:
    def __init__(self):
        self.position = None
        self.age = 10**9
        self.confidence = 0.0

    def update(self, pose, ball_obs):
        if ball_obs is not None:
            observed_position = clip_ball_position(
                polar_to_world(pose, ball_obs),
                padding=0.10,
            )
            if self.position is None or self.age > 20:
                self.position = observed_position
            else:
                alpha = 0.35
                self.position = (
                    (1.0 - alpha) * self.position[0] + alpha * observed_position[0],
                    (1.0 - alpha) * self.position[1] + alpha * observed_position[1],
                )
            self.age = 0
            self.confidence = min(1.0, 0.75 * self.confidence + 0.40)
        else:
            self.age += 1
            self.confidence *= 0.97

    def is_fresh(self, max_age):
        return self.position is not None and self.age <= max_age and self.confidence > 0.05

    def relative_ball(self, pose):
        if self.position is None:
            return None
        return relative_polar(pose, self.position)


class LiveVisualizerClient:
    def __init__(self):
        env_toggle = os.environ.get("WEBOTS_LIVE_VISUALIZER")
        self.enabled = VISUALIZER_DEFAULT_ENABLED if env_toggle is None else env_toggle != "0"
        self.process = None

        if not self.enabled:
            return

        script_path = Path(__file__).with_name("localization_visualizer.py")
        if not script_path.exists():
            self.enabled = False
            return

        try:
            self.process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            atexit.register(self.close)
        except Exception:
            self.enabled = False
            self.process = None

    def publish(self, snapshot):
        if not self.enabled or self.process is None or self.process.stdin is None:
            return

        if self.process.poll() is not None:
            self.close()
            return

        try:
            self.process.stdin.write(json.dumps(snapshot) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.close()

    def close(self):
        if self.process is None:
            return

        try:
            if self.process.stdin is not None and not self.process.stdin.closed:
                self.process.stdin.close()
        except OSError:
            pass

        try:
            if self.process.poll() is None:
                self.process.terminate()
        except OSError:
            pass

        self.process = None
        self.enabled = False


class TransitionLogger:
    def __init__(self):
        self.handle = None
        trace_path = os.environ.get("RL_TRACE_PATH")
        if not trace_path:
            return
        try:
            self.handle = open(trace_path, "a", encoding="utf-8", buffering=1)
            atexit.register(self.close)
        except OSError:
            self.handle = None

    def log(self, payload):
        if self.handle is None:
            return
        try:
            self.handle.write(json.dumps(payload) + "\n")
        except OSError:
            self.close()

    def close(self):
        if self.handle is None:
            return
        try:
            self.handle.close()
        except OSError:
            pass
        self.handle = None


class EmbeddedActorPolicy:
    INPUT_DIM = 16
    HIDDEN_DIM = 64
    RUNTIME_WEIGHTS_FILENAME = "rl_policy_weights.npz"
    FEATURE_NAMES = (
        "ball_visible",
        "ball_dist_norm",
        "sin_ball_angle",
        "cos_ball_angle",
        "goal_dist_norm",
        "sin_goal_angle",
        "cos_goal_angle",
        "staging_dist_norm",
        "sin_staging_angle",
        "cos_staging_angle",
        "ball_goal_alignment_norm",
        "localizer_confidence",
        "ball_age_norm",
        "prev_forward_cmd",
        "prev_turn_cmd",
        "wall_margin_norm",
    )

    def __init__(self):
        self.source = "embedded_bootstrap"
        runtime_weights = self._load_runtime_actor()
        if runtime_weights is not None:
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = runtime_weights
        else:
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = self._build_bootstrap_actor()

    def _runtime_weights_path(self):
        env_path = os.environ.get("RL_POLICY_WEIGHTS_PATH")
        if env_path:
            candidate = Path(env_path).expanduser()
            if candidate.exists():
                return candidate
            print(f"RL policy weights path not found, falling back to embedded weights: {candidate}")
            return None

        candidate = Path(__file__).with_name(self.RUNTIME_WEIGHTS_FILENAME)
        if candidate.exists():
            return candidate
        return None

    def _load_runtime_actor(self):
        candidate = self._runtime_weights_path()
        if candidate is None:
            return None

        expected_shapes = {
            "w1": (self.HIDDEN_DIM, self.INPUT_DIM),
            "b1": (self.HIDDEN_DIM,),
            "w2": (self.HIDDEN_DIM, self.HIDDEN_DIM),
            "b2": (self.HIDDEN_DIM,),
            "w3": (2, self.HIDDEN_DIM),
            "b3": (2,),
        }

        try:
            with np.load(candidate) as data:
                arrays = []
                for key, shape in expected_shapes.items():
                    if key not in data:
                        raise KeyError(f"missing key {key}")
                    array = np.asarray(data[key], dtype=np.float32)
                    if array.shape != shape:
                        raise ValueError(f"{key} has shape {array.shape}, expected {shape}")
                    arrays.append(array)
        except Exception as exc:
            print(f"Failed to load RL policy weights from {candidate}: {exc}")
            return None

        self.source = str(candidate)
        print(f"Loaded RL policy weights from {candidate}")
        return tuple(arrays)

    def _build_bootstrap_actor(self):
        w1 = np.zeros((self.HIDDEN_DIM, self.INPUT_DIM), dtype=np.float32)
        b1 = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        w2 = np.zeros((self.HIDDEN_DIM, self.HIDDEN_DIM), dtype=np.float32)
        b2 = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        w3 = np.zeros((2, self.HIDDEN_DIM), dtype=np.float32)
        b3 = np.array([-0.55, 0.0], dtype=np.float32)

        for i in range(self.INPUT_DIM):
            w1[i, i] = 1.35
            w1[16 + i, i] = -1.35
            w2[i, i] = 1.0
            w2[16 + i, 16 + i] = 1.0

        # Hand-shaped combination units. These are a bootstrap actor that can later
        # be replaced by exported trained weights without changing controller code.
        w1[32, 0] = 1.0
        w1[32, 1] = 1.0
        w1[32, 11] = 0.7

        w1[33, 7] = 1.0
        w1[33, 9] = 0.9
        w1[33, 3] = 0.4

        w1[34, 2] = 0.8
        w1[34, 8] = 1.2
        w1[34, 5] = 0.4

        w1[35, 10] = 1.0
        w1[35, 6] = 0.5
        w1[35, 15] = 0.4

        w1[36, 12] = -1.0
        w1[36, 15] = 0.8
        w1[36, 11] = 0.4

        w1[37, 13] = 0.8
        w1[37, 14] = -0.4
        w1[37, 1] = 0.4

        w1[38, 5] = 0.9
        w1[38, 2] = -0.5

        for i in range(32, 39):
            w2[i, i] = 1.0

        # Forward action.
        w3[0, 0] = 0.70   # ball_visible
        w3[0, 1] = 0.85   # ball_dist_norm
        w3[0, 3] = 0.75   # cos_ball_angle
        w3[0, 4] = 0.20   # goal_dist_norm
        w3[0, 6] = 0.20   # cos_goal_angle
        w3[0, 7] = 0.55   # staging_dist_norm
        w3[0, 9] = 0.40   # cos_staging_angle
        w3[0, 10] = 0.55  # ball_goal_alignment_norm
        w3[0, 11] = 0.45  # localizer_confidence
        w3[0, 12] = -0.55 # ball_age_norm
        w3[0, 13] = 0.20  # prev_forward_cmd
        w3[0, 15] = 0.45  # wall_margin_norm
        w3[0, 32] = 0.55
        w3[0, 33] = 0.50
        w3[0, 35] = 0.30
        w3[0, 36] = 0.35
        w3[0, 37] = 0.18

        # Turn action.
        w3[1, 2] = 0.65   # sin_ball_angle
        w3[1, 5] = 0.45   # sin_goal_angle
        w3[1, 8] = 1.10   # sin_staging_angle
        w3[1, 14] = 0.18  # prev_turn_cmd
        w3[1, 34] = 0.95
        w3[1, 38] = 0.55
        w3[1, 36] = -0.10

        return w1, b1, w2, b2, w3, b3

    def __call__(self, features):
        h1 = np.tanh(self.w1 @ features + self.b1)
        h2 = np.tanh(self.w2 @ h1 + self.b2)
        return np.tanh(self.w3 @ h2 + self.b3)


class StudentController:
    def __init__(self):
        self.localizer = MultiHypothesisLocalizer()
        self.ball_tracker = BallTracker()
        self.visualizer = LiveVisualizerClient()
        self.transition_logger = TransitionLogger()
        self.actor = EmbeddedActorPolicy()

        self.state = "SEARCH_BALL"
        self.state_age = 0
        self.step_count = 0
        self.search_direction = 1.0
        self.attack_goal = RIGHT_GOAL

        self.prev_forward_cmd = 0.0
        self.prev_turn_cmd = 0.0
        self.low_confidence_steps = 0
        self.no_progress_steps = 0
        self.prev_ball_goal_dist = None
        self.last_seen_ball_rel = None
        self.last_rl_features = np.zeros(self.actor.INPUT_DIM, dtype=np.float32)
        self.last_rl_action = np.zeros(2, dtype=np.float32)
        self.last_rl_context = {}

    def _set_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            self.state_age = 0
            if new_state != "RL_BALL_PLAY":
                self.no_progress_steps = 0
                self.prev_ball_goal_dist = None

    def _wheel_command(self, forward, turn):
        self.prev_forward_cmd = clamp(forward / RL_FORWARD_SCALE, -1.0, 1.0)
        self.prev_turn_cmd = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
        left = clamp(forward - turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        right = clamp(forward + turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        return {"left_motor": left, "right_motor": right}

    def _search_control(self):
        if self.step_count % 80 == 0:
            self.search_direction *= -1.0
        forward = 0.3 if self.localizer.confidence > 0.25 else 0.0
        turn = 2.3 * self.search_direction
        return self._wheel_command(forward, turn)

    def _wall_margin(self, pose):
        return min(ARENA_X_HALF - abs(pose[0]), ARENA_Y_HALF - abs(pose[1]))

    def _staging_point(self, ball_position):
        goal_dx = self.attack_goal[0] - ball_position[0]
        goal_dy = self.attack_goal[1] - ball_position[1]
        norm = math.hypot(goal_dx, goal_dy)
        if norm < 1e-6:
            norm = 1.0
        offset = 0.28
        staging = (
            ball_position[0] - offset * goal_dx / norm,
            ball_position[1] - offset * goal_dy / norm,
        )
        return clip_to_field(staging, padding=0.20)

    def _ball_observation(self, pose, sensors):
        ball_obs = sensors.get("ball")
        if ball_obs is not None:
            self.last_seen_ball_rel = tuple(ball_obs)
            return tuple(ball_obs), self.ball_tracker.position
        if self.ball_tracker.is_fresh(max_age=35):
            rel_ball = self.ball_tracker.relative_ball(pose)
            if rel_ball is not None:
                return rel_ball, self.ball_tracker.position
        return None, None

    def extract_rl_features(self, pose, sensors):
        ball_visible = 1.0 if sensors.get("ball") is not None else 0.0
        ball_rel, ball_position = self._ball_observation(pose, sensors)

        if ball_rel is None:
            if self.last_seen_ball_rel is not None:
                ball_rel = self.last_seen_ball_rel
            else:
                ball_rel = (2.0, 0.0)

        if ball_position is None and self.ball_tracker.position is not None:
            ball_position = self.ball_tracker.position

        goal_rel = relative_polar(pose, self.attack_goal)
        if ball_position is not None:
            staging_point = self._staging_point(ball_position)
            staging_rel = relative_polar(pose, staging_point)
            ball_goal_alignment = math.cos(wrap_to_pi(goal_rel[1] - ball_rel[1]))
        else:
            staging_point = None
            staging_rel = (2.0, 0.0)
            ball_goal_alignment = -1.0

        wall_margin = self._wall_margin(pose)
        wall_margin_norm = clamp(wall_margin / 1.5, 0.0, 1.0)

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
                clamp(self.localizer.confidence, 0.0, 1.0),
                clamp(self.ball_tracker.age / 30.0, 0.0, 1.0),
                clamp(self.prev_forward_cmd, -1.0, 1.0),
                clamp(self.prev_turn_cmd, -1.0, 1.0),
                wall_margin_norm,
            ],
            dtype=np.float32,
        )

        context = {
            "ball_rel": ball_rel,
            "ball_position": ball_position,
            "goal_rel": goal_rel,
            "staging_point": staging_point,
            "staging_rel": staging_rel,
            "wall_margin_norm": wall_margin_norm,
        }
        return features, context

    def rl_policy(self, features):
        return self.actor(features)

    def should_enter_rl(self, sensors):
        return sensors.get("ball") is not None

    def _update_rl_progress(self, pose, context):
        if self.localizer.confidence < 0.20:
            self.low_confidence_steps += 1
        else:
            self.low_confidence_steps = 0

        ball_position = context.get("ball_position")
        if ball_position is None:
            self.no_progress_steps = 0
            self.prev_ball_goal_dist = None
            return

        ball_goal_dist = distance(ball_position, self.attack_goal)
        if self.prev_ball_goal_dist is None:
            self.prev_ball_goal_dist = ball_goal_dist
            self.no_progress_steps = 0
            return

        progress = self.prev_ball_goal_dist - ball_goal_dist
        wall_margin = self._wall_margin(pose)
        if wall_margin < 0.35 and progress < 0.003:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        self.prev_ball_goal_dist = ball_goal_dist

    def should_exit_rl(self, pose, sensors, context):
        del sensors  # kept for required interface symmetry
        self._update_rl_progress(pose, context)
        if self.ball_tracker.age > 8:
            return True
        if self.low_confidence_steps >= 10:
            return True
        if self.no_progress_steps >= 20:
            return True
        return False

    def recover_control(self, pose, context):
        del context
        ball_rel = None
        if self.ball_tracker.is_fresh(max_age=45):
            ball_rel = self.ball_tracker.relative_ball(pose)
        elif self.last_seen_ball_rel is not None:
            ball_rel = self.last_seen_ball_rel

        if ball_rel is None:
            return self._search_control()

        turn = clamp(2.8 * ball_rel[1], -2.8, 2.8)
        forward = 0.9 if abs(ball_rel[1]) < 0.35 else 0.0
        if self.localizer.confidence < 0.15:
            forward = 0.0
        return self._wheel_command(forward, turn)

    def _action_to_control(self, action):
        a_forward = clamp(float(action[0]), -1.0, 1.0)
        a_turn = clamp(float(action[1]), -1.0, 1.0)
        forward = RL_FORWARD_SCALE * max(0.0, 0.5 * (a_forward + 1.0))
        turn = RL_TURN_SCALE * a_turn
        return self._wheel_command(forward, turn)

    def _log_transition(self, sensors, pose, control):
        if self.transition_logger.handle is None:
            return
        truth = sensors.get("debug_truth", {})
        self.transition_logger.log(
            {
                "step": self.step_count,
                "state": self.state,
                "estimated_pose": [float(v) for v in pose],
                "actual_pose": truth.get("robot_pose"),
                "estimated_ball": None if self.ball_tracker.position is None else [float(v) for v in self.ball_tracker.position],
                "actual_ball": truth.get("ball_position"),
                "features": [float(v) for v in self.last_rl_features],
                "action": [float(v) for v in self.last_rl_action],
                "control": {
                    "left_motor": float(control["left_motor"]),
                    "right_motor": float(control["right_motor"]),
                },
            }
        )

    def _publish_visualizer(self, sensors, pose):
        if self.visualizer is None or not self.visualizer.enabled:
            return
        if self.step_count % 2 != 0:
            return

        landmark_debug = self.localizer.debug_landmark_estimates(sensors, pose)
        truth = sensors.get("debug_truth", {})
        actual_pose = truth.get("robot_pose")
        actual_ball = truth.get("ball_position")
        estimated_ball = self.ball_tracker.position

        snapshot = {
            "state": self.state,
            "confidence": float(self.localizer.confidence),
            "estimated_pose": [float(pose[0]), float(pose[1]), float(pose[2])],
            "actual_pose": None if actual_pose is None else [float(v) for v in actual_pose],
            "estimated_ball": None if estimated_ball is None else [float(v) for v in estimated_ball],
            "actual_ball": None if actual_ball is None else [float(v) for v in actual_ball],
            "hypotheses": [[float(h.x), float(h.y), float(h.theta)] for h in self.localizer.hypotheses],
            "landmark_points": {
                "all": {
                    key: [[float(pt[0]), float(pt[1])] for pt in values]
                    for key, values in landmark_debug["all"].items()
                },
                "visible": {
                    key: [[float(pt[0]), float(pt[1])] for pt in values]
                    for key, values in landmark_debug["visible"].items()
                },
                "projected": {
                    key: [[float(pt[0]), float(pt[1])] for pt in values]
                    for key, values in landmark_debug["projected"].items()
                },
                "matched": {
                    key: [[float(pt[0]), float(pt[1])] for pt in values]
                    for key, values in landmark_debug["matched"].items()
                },
            },
            "rl_features": [float(v) for v in self.last_rl_features],
            "rl_action": [float(v) for v in self.last_rl_action],
        }
        self.visualizer.publish(snapshot)

    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict: dict, contains control for "left_motor" and "right_motor"
        """
        self.step_count += 1
        self.state_age += 1

        self.localizer.predict(sensors.get("odometry"))
        pose = self.localizer.update(sensors)
        self.ball_tracker.update(pose, sensors.get("ball"))

        features, context = self.extract_rl_features(pose, sensors)
        self.last_rl_features = features
        self.last_rl_context = context

        if self.state == "SEARCH_BALL" and self.should_enter_rl(sensors):
            self._set_state("RL_BALL_PLAY")
        elif self.state == "RECOVER":
            if self.should_enter_rl(sensors):
                self._set_state("RL_BALL_PLAY")
            elif self.state_age >= 30:
                self._set_state("SEARCH_BALL")

        if self.state == "RL_BALL_PLAY":
            if self.should_exit_rl(pose, sensors, context):
                self._set_state("RECOVER")
                self.last_rl_action = np.zeros(2, dtype=np.float32)
                control = self.recover_control(pose, context)
            else:
                action = self.rl_policy(features)
                self.last_rl_action = np.asarray(action, dtype=np.float32)
                control = self._action_to_control(action)
        elif self.state == "RECOVER":
            self.last_rl_action = np.zeros(2, dtype=np.float32)
            control = self.recover_control(pose, context)
        else:
            self.last_rl_action = np.zeros(2, dtype=np.float32)
            control = self._search_control()

        self._log_transition(sensors, pose, control)
        self._publish_visualizer(sensors, pose)
        return control
