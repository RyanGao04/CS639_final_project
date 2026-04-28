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


def limit_vector(dx, dy, max_norm):
    norm = math.hypot(dx, dy)
    if norm <= max_norm or norm < 1e-9:
        return dx, dy
    scale = max_norm / norm
    return dx * scale, dy * scale


def relative_polar(pose, point):
    dx = point[0] - pose[0]
    dy = point[1] - pose[1]
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    return dist, wrap_to_pi(heading - pose[2])


def point_in_fov(pose, point, fov=OBSERVATION_FOV):
    _, rel_angle = relative_polar(pose, point)
    return abs(rel_angle) <= 0.5 * fov


def pose_error(estimated_pose, actual_pose):
    if estimated_pose is None or actual_pose is None:
        return None, None
    pos_error = math.hypot(
        float(estimated_pose[0]) - float(actual_pose[0]),
        float(estimated_pose[1]) - float(actual_pose[1]),
    )
    heading_error = abs(wrap_to_pi(float(estimated_pose[2]) - float(actual_pose[2])))
    return pos_error, heading_error


def pose_jump(current_pose, previous_pose):
    if current_pose is None or previous_pose is None:
        return 0.0, 0.0
    pos_jump = math.hypot(
        float(current_pose[0]) - float(previous_pose[0]),
        float(current_pose[1]) - float(previous_pose[1]),
    )
    heading_jump = abs(wrap_to_pi(float(current_pose[2]) - float(previous_pose[2])))
    return pos_jump, heading_jump


def landmark_bin(stats):
    total_count = int(stats.get("total_count", 0))
    goal_count = int(stats.get("goal_count", 0))
    structure_count = int(stats.get("structure_count", 0))
    if total_count == 0:
        return "0_landmarks"
    if structure_count == 0:
        if goal_count == 1:
            return "1_goal_only"
        return "goal_only_multi"
    if structure_count == 1 and goal_count == 0:
        return "1_structural"
    if structure_count >= 1 and goal_count >= 1:
        return "goal_structural"
    return "2plus_mixed"


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
        self.num_hypotheses = 75
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
        self.diagnostics = self._empty_diagnostics()
        self.direct_pose_diagnostics = None
        self.hypotheses = self._initial_hypotheses()

    def _empty_diagnostics(self):
        return {
            "goal_count": 0,
            "corner_count": 0,
            "cross_count": 0,
            "center_count": 0,
            "structure_count": 0,
            "total_count": 0,
            "landmark_bin": "0_landmarks",
            "matched_count": 0,
            "association_count": 0,
            "mean_residual": None,
            "max_residual": None,
            "residual_quality": 0.0,
            "observability": 0.0,
            "hypothesis_spread_xy": 0.0,
            "heading_spread": 0.0,
            "effective_sample_size": float(self.num_hypotheses),
            "effective_sample_size_norm": 1.0,
            "resampled": False,
            "strong_update_allowed": False,
            "best_validity": 0.0,
        }

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
            hypothesis.x = clamp(hypothesis.x, -ARENA_X_HALF - 0.05, ARENA_X_HALF + 0.05)
            hypothesis.y = clamp(hypothesis.y, -ARENA_Y_HALF - 0.05, ARENA_Y_HALF + 0.05)

    def update(self, sensors):
        measurements, total_count, stats = self._collect_measurements(sensors)
        if total_count == 0:
            self.direct_pose_diagnostics = None
            for hypothesis in self.hypotheses:
                hypothesis.validity *= 0.995
            self._normalize_weights()
            self.pose = self._estimate_pose()
            self.confidence = max(0.0, 0.995 * self.confidence)
            self.diagnostics = self._build_diagnostics(stats, [], resampled=False)
            return self.pose

        update_results = []
        for hypothesis in self.hypotheses:
            update_results.append(self._update_hypothesis(hypothesis, measurements, total_count, stats))

        self._normalize_weights()
        self.pose = self._estimate_pose()
        self.confidence = self._estimate_confidence(stats, update_results)
        self.direct_pose_diagnostics = self._direct_pose_from_measurements(measurements, self.pose)
        if self._direct_pose_reliable(stats):
            direct_pose = self.direct_pose_diagnostics["pose"]
            self._anchor_hypotheses(
                direct_pose,
                position_sigma=0.030 if stats["total_count"] >= 3 else 0.045,
                heading_sigma=math.radians(1.0 if stats["total_count"] >= 3 else 1.8),
            )
            self.pose = direct_pose
            self.confidence = max(self.confidence, 0.72 if stats["total_count"] >= 3 else 0.48)
        should_resample = self._should_resample(stats)
        if should_resample:
            self._resample()
        self.diagnostics = self._build_diagnostics(stats, update_results, resampled=should_resample)
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
        stats = {
            "goal_count": len(measurements.get("goal", [])),
            "corner_count": len(measurements.get("corners", [])),
            "cross_count": len(measurements.get("penalty_cross", [])),
            "center_count": len(measurements.get("center_circle", [])),
        }
        stats["structure_count"] = (
            stats["corner_count"] + stats["cross_count"] + stats["center_count"]
        )
        stats["total_count"] = total_count
        stats["landmark_bin"] = landmark_bin(stats)
        return measurements, total_count, stats

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

    def _observation_residual(self, pose, obs, landmark, spec):
        predicted_dist, predicted_angle = relative_polar(pose, landmark)
        dist_error = obs[0] - predicted_dist
        angle_error = wrap_to_pi(obs[1] - predicted_angle)
        normalized = (dist_error / spec["sigma_r"]) ** 2
        normalized += (angle_error / spec["sigma_a"]) ** 2
        return normalized, dist_error, angle_error

    def _association_candidates(self, hypothesis, name, obs):
        spec = self.landmark_specs[name]
        pose = (hypothesis.x, hypothesis.y, hypothesis.theta)
        candidates = []
        for point in spec["points"]:
            projected = polar_to_world(pose, obs)
            projected_error = distance(projected, point)
            residual, dist_error, angle_error = self._observation_residual(pose, obs, point, spec)
            if projected_error <= spec["gate"]:
                candidates.append(
                    {
                        "name": name,
                        "obs": obs,
                        "landmark": point,
                        "residual": residual,
                        "projected_error": projected_error,
                        "dist_error": dist_error,
                        "angle_error": angle_error,
                    }
                )
        candidates.sort(key=lambda item: item["residual"])
        return candidates

    def _best_consistent_assignment(self, hypothesis, measurements):
        observations = []
        for name, obs_list in measurements.items():
            for obs in obs_list:
                observations.append((name, obs))

        if not observations:
            return [], 0.0, 0.0

        candidate_lists = []
        for name, obs in observations:
            candidates = self._association_candidates(hypothesis, name, obs)
            unmatched_penalty = 10.0 if name == "goal" else 12.0
            candidates = candidates[:4] + [
                {
                    "name": name,
                    "obs": obs,
                    "landmark": None,
                    "residual": unmatched_penalty,
                    "projected_error": None,
                    "dist_error": None,
                    "angle_error": None,
                }
            ]
            candidate_lists.append(candidates)

        best_assignment = []
        best_score = float("inf")

        def search(index, used, assignment, score):
            nonlocal best_assignment, best_score
            if score >= best_score:
                return
            if index >= len(candidate_lists):
                best_assignment = list(assignment)
                best_score = score
                return
            for candidate in candidate_lists[index]:
                landmark = candidate["landmark"]
                used_key = None
                duplicate_penalty = 0.0
                if landmark is not None:
                    used_key = (candidate["name"], landmark)
                    if used_key in used:
                        duplicate_penalty = 8.0
                if used_key is not None and duplicate_penalty == 0.0:
                    used.add(used_key)
                assignment.append(candidate)
                search(index + 1, used, assignment, score + candidate["residual"] + duplicate_penalty)
                assignment.pop()
                if used_key is not None and duplicate_penalty == 0.0:
                    used.remove(used_key)

        search(0, set(), [], 0.0)

        ambiguity_penalty = 0.0
        for candidates in candidate_lists:
            real = [candidate for candidate in candidates if candidate["landmark"] is not None]
            if len(real) >= 2:
                gap = real[1]["residual"] - real[0]["residual"]
                if gap < 2.0:
                    ambiguity_penalty += 2.0 - gap

        return best_assignment, best_score + ambiguity_penalty, ambiguity_penalty

    def _observability_score(self, stats):
        if stats["total_count"] == 0:
            return 0.0
        if stats["structure_count"] == 0:
            return 0.08 if stats["goal_count"] <= 1 else 0.18
        score = (
            1.20 * stats["center_count"]
            + 1.00 * stats["cross_count"]
            + 0.75 * min(stats["corner_count"], 2)
            + 0.45 * min(stats["goal_count"], 2)
        )
        return clamp(score / 2.75, 0.18, 1.0)

    def _update_gains(self, stats):
        if stats["structure_count"] == 0:
            return 0.04, 0.0
        if stats["structure_count"] == 1 and stats["goal_count"] == 0:
            return 0.14, 0.040
        if stats["structure_count"] == 1:
            return 0.14, 0.070
        return 0.18, 0.16

    def _direct_pose_from_measurements(self, measurements, prior_pose):
        observations = []
        for name, obs_list in measurements.items():
            for obs in obs_list:
                dist, angle = obs
                observations.append(
                    {
                        "name": name,
                        "obs": obs,
                        "robot_point": (
                            float(dist) * math.cos(float(angle)),
                            float(dist) * math.sin(float(angle)),
                        ),
                        "candidates": [tuple(point) for point in self.landmark_specs[name]["points"]],
                    }
                )
        if len(observations) < 2:
            return None

        best = None
        assignments = []
        assignment_count = 0
        max_assignments = 3000

        def estimate_pose(robot_points, world_points):
            n = len(robot_points)
            px = sum(point[0] for point in robot_points) / n
            py = sum(point[1] for point in robot_points) / n
            qx = sum(point[0] for point in world_points) / n
            qy = sum(point[1] for point in world_points) / n
            cross = 0.0
            dot = 0.0
            for p, q in zip(robot_points, world_points):
                ux = p[0] - px
                uy = p[1] - py
                vx = q[0] - qx
                vy = q[1] - qy
                cross += ux * vy - uy * vx
                dot += ux * vx + uy * vy
            if abs(cross) + abs(dot) < 1e-9:
                return None
            theta = math.atan2(cross, dot)
            c = math.cos(theta)
            s = math.sin(theta)
            tx = qx - (c * px - s * py)
            ty = qy - (s * px + c * py)
            residuals = []
            for p, q in zip(robot_points, world_points):
                wx = tx + c * p[0] - s * p[1]
                wy = ty + s * p[0] + c * p[1]
                residuals.append(math.hypot(wx - q[0], wy - q[1]))
            return (
                (tx, ty, wrap_to_pi(theta)),
                sum(residuals) / len(residuals),
                max(residuals),
            )

        def consider():
            nonlocal best
            robot_points = [item["robot_point"] for item, _ in assignments]
            world_points = [point for _, point in assignments]
            result = estimate_pose(robot_points, world_points)
            if result is None:
                return
            pose, mean_residual, max_residual = result
            if abs(pose[0]) > ARENA_X_HALF + 0.08 or abs(pose[1]) > ARENA_Y_HALF + 0.08:
                return
            prior_pos = distance(pose[:2], prior_pose[:2])
            prior_heading = abs(wrap_to_pi(pose[2] - prior_pose[2]))
            if prior_pos > 1.80 or prior_heading > math.radians(95.0):
                return
            score = (
                mean_residual
                + 0.10 * max_residual
                + 0.018 * min(prior_pos, 2.0) ** 2
                + 0.018 * min(prior_heading, math.pi) ** 2
            )
            candidate = {
                "pose": pose,
                "mean_residual": mean_residual,
                "max_residual": max_residual,
                "prior_pos": prior_pos,
                "prior_heading": prior_heading,
                "score": score,
                "count": len(assignments),
                "kind": "multi",
            }
            if best is None or score < best["score"]:
                best = candidate

        def search(index, used):
            nonlocal assignment_count
            if assignment_count >= max_assignments:
                return
            if index >= len(observations):
                assignment_count += 1
                consider()
                return
            item = observations[index]
            for point in item["candidates"]:
                key = (item["name"], point)
                if key in used:
                    continue
                used.add(key)
                assignments.append((item, point))
                search(index + 1, used)
                assignments.pop()
                used.remove(key)

        search(0, set())
        return best

    def _direct_pose_reliable(self, stats):
        candidate = self.direct_pose_diagnostics
        if candidate is None:
            return False
        if stats["landmark_bin"] in ("0_landmarks", "1_goal_only", "goal_only_multi", "1_structural"):
            return False
        if candidate["prior_pos"] > 1.15 or candidate["prior_heading"] > math.radians(65.0):
            return False
        if stats["total_count"] >= 3:
            return candidate["mean_residual"] < 0.10 and candidate["max_residual"] < 0.24
        return candidate["mean_residual"] < 0.035 and candidate["max_residual"] < 0.08

    def _anchor_hypotheses(self, pose, position_sigma=0.035, heading_sigma=0.020):
        for index, hypothesis in enumerate(self.hypotheses):
            if index == 0:
                hypothesis.x = pose[0]
                hypothesis.y = pose[1]
                hypothesis.theta = pose[2]
            else:
                hypothesis.x = pose[0] + float(self.rng.normal(0.0, position_sigma))
                hypothesis.y = pose[1] + float(self.rng.normal(0.0, position_sigma))
                hypothesis.theta = wrap_to_pi(pose[2] + float(self.rng.normal(0.0, heading_sigma)))
            hypothesis.validity = max(hypothesis.validity, 0.82)
            hypothesis.weight = 1.0 / self.num_hypotheses

    def _nudge_towards_landmark(self, hypothesis, obs, landmark, theta_gain, position_gain):
        world_bearing = math.atan2(landmark[1] - hypothesis.y, landmark[0] - hypothesis.x)
        target_theta = wrap_to_pi(world_bearing - obs[1])
        theta_error = wrap_to_pi(target_theta - hypothesis.theta)
        hypothesis.theta = wrap_to_pi(hypothesis.theta + theta_gain * theta_error)

        inferred_x = landmark[0] - obs[0] * math.cos(hypothesis.theta + obs[1])
        inferred_y = landmark[1] - obs[0] * math.sin(hypothesis.theta + obs[1])
        hypothesis.x += position_gain * (inferred_x - hypothesis.x)
        hypothesis.y += position_gain * (inferred_y - hypothesis.y)

    def _update_hypothesis(self, hypothesis, measurements, total_count, stats):
        assignment, score, ambiguity_penalty = self._best_consistent_assignment(hypothesis, measurements)
        matched_items = [item for item in assignment if item["landmark"] is not None]
        matched = len(matched_items)
        error_sum = sum(float(item["residual"]) for item in assignment)
        theta_gain, position_gain = self._update_gains(stats)

        for item in matched_items:
            self._nudge_towards_landmark(
                hypothesis,
                item["obs"],
                item["landmark"],
                theta_gain=theta_gain,
                position_gain=position_gain,
            )

        current_validity = matched / float(max(total_count, 1))
        hypothesis.validity = 0.92 * hypothesis.validity + 0.08 * current_validity
        mean_error = score / float(max(total_count, 1))
        hypothesis.weight = (
            self.base_validity_weight
            + (1.0 - self.base_validity_weight) * hypothesis.validity
        ) * math.exp(-0.5 * min(mean_error, 30.0))

        return {
            "matched": matched,
            "association_count": len(matched_items),
            "error_sum": error_sum,
            "score": score,
            "mean_residual": mean_error,
            "max_residual": max((float(item["residual"]) for item in assignment), default=0.0),
            "ambiguity_penalty": ambiguity_penalty,
        }

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

    def _hypothesis_stats(self):
        weights = np.array([h.weight for h in self.hypotheses], dtype=float)
        if weights.sum() <= 1e-9:
            weights = np.ones(self.num_hypotheses, dtype=float) / self.num_hypotheses
        else:
            weights = weights / weights.sum()

        xs = np.array([h.x for h in self.hypotheses], dtype=float)
        ys = np.array([h.y for h in self.hypotheses], dtype=float)
        thetas = np.array([h.theta for h in self.hypotheses], dtype=float)
        mean_x = float(np.sum(weights * xs))
        mean_y = float(np.sum(weights * ys))
        spread_xy = math.sqrt(
            float(np.sum(weights * ((xs - mean_x) ** 2 + (ys - mean_y) ** 2)))
        )
        cos_sum = float(np.sum(weights * np.cos(thetas)))
        sin_sum = float(np.sum(weights * np.sin(thetas)))
        resultant = min(1.0, math.hypot(cos_sum, sin_sum))
        heading_spread = math.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1e-6))))
        ess = float(1.0 / max(float(np.sum(weights * weights)), 1e-9))
        return {
            "spread_xy": spread_xy,
            "heading_spread": heading_spread,
            "effective_sample_size": ess,
            "effective_sample_size_norm": ess / float(self.num_hypotheses),
        }

    def _estimate_confidence(self, stats, update_results):
        hypothesis_stats = self._hypothesis_stats()
        observability = self._observability_score(stats)
        if not update_results:
            return clamp(0.995 * self.confidence, 0.0, 1.0)
        best_validity = max(h.validity for h in self.hypotheses)
        weighted_mean_residual = self._weighted_result_value(update_results, "mean_residual", default=10.0)
        residual_quality = math.exp(-0.5 * min(weighted_mean_residual, 18.0))
        compactness = (
            1.0 - clamp(hypothesis_stats["spread_xy"] / 0.90, 0.0, 1.0)
        ) * (
            1.0 - clamp(hypothesis_stats["heading_spread"] / math.radians(55.0), 0.0, 1.0)
        )
        ess_quality = clamp(hypothesis_stats["effective_sample_size_norm"], 0.0, 1.0)
        confidence = (
            0.12 * best_validity
            + 0.68 * observability * residual_quality * compactness
            + 0.20 * observability * ess_quality
        )
        if stats["landmark_bin"] in ("0_landmarks", "1_goal_only"):
            confidence = min(confidence, 0.14 if stats["landmark_bin"] == "1_goal_only" else 0.04)
        elif stats["landmark_bin"] == "goal_only_multi":
            confidence = min(confidence, 0.24)
        return clamp(confidence, 0.0, 1.0)

    def _weighted_result_value(self, update_results, key, default=0.0):
        if not update_results:
            return default
        weights = [h.weight for h in self.hypotheses]
        weight_sum = sum(weights)
        if weight_sum <= 1e-9:
            return float(np.mean([result.get(key, default) for result in update_results]))
        return float(
            sum(
                weight * float(result.get(key, default))
                for weight, result in zip(weights, update_results)
            )
            / weight_sum
        )

    def _should_resample(self, stats):
        if stats["landmark_bin"] in ("0_landmarks", "1_goal_only"):
            return False
        hypothesis_stats = self._hypothesis_stats()
        if stats["structure_count"] <= 1 and stats["goal_count"] == 0:
            return hypothesis_stats["effective_sample_size_norm"] < 0.35
        if stats["structure_count"] == 0:
            return False
        return hypothesis_stats["effective_sample_size_norm"] < 0.65

    def _build_diagnostics(self, stats, update_results, resampled):
        hypothesis_stats = self._hypothesis_stats()
        matched_count = self._weighted_result_value(update_results, "matched", default=0.0)
        association_count = self._weighted_result_value(update_results, "association_count", default=0.0)
        mean_residual = None
        max_residual = None
        residual_quality = 0.0
        if update_results:
            mean_residual = self._weighted_result_value(update_results, "mean_residual", default=10.0)
            max_residual = max(result.get("max_residual", 0.0) for result in update_results)
            residual_quality = math.exp(-0.5 * min(mean_residual, 18.0))
        diagnostics = {
            **stats,
            "matched_count": float(matched_count),
            "association_count": float(association_count),
            "mean_residual": None if mean_residual is None else float(mean_residual),
            "max_residual": None if max_residual is None else float(max_residual),
            "residual_quality": float(residual_quality),
            "observability": float(self._observability_score(stats)),
            "hypothesis_spread_xy": float(hypothesis_stats["spread_xy"]),
            "heading_spread": float(hypothesis_stats["heading_spread"]),
            "effective_sample_size": float(hypothesis_stats["effective_sample_size"]),
            "effective_sample_size_norm": float(hypothesis_stats["effective_sample_size_norm"]),
            "resampled": bool(resampled),
            "strong_update_allowed": stats["landmark_bin"] not in ("0_landmarks", "1_goal_only"),
            "best_validity": float(max((h.validity for h in self.hypotheses), default=0.0)),
            "direct_pose_used": bool(self._direct_pose_reliable(stats)),
            "direct_pose_residual": None if self.direct_pose_diagnostics is None else float(self.direct_pose_diagnostics["mean_residual"]),
            "direct_pose_prior_pos": None if self.direct_pose_diagnostics is None else float(self.direct_pose_diagnostics["prior_pos"]),
        }
        return diagnostics

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

        measurements, _, _ = self._collect_measurements(sensors)
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

    def update(self, pose, ball_obs, pose_trust=1.0):
        if ball_obs is not None:
            observed_position = clip_ball_position(
                polar_to_world(pose, ball_obs),
                padding=0.10,
            )
            if self.position is None or self.age > 120:
                self.position = observed_position
            else:
                pose_trust = clamp(float(pose_trust), 0.0, 1.0)
                jump = distance(self.position, observed_position)
                near_contact = ball_obs[0] < 0.58
                if self.age <= 8:
                    max_jump = (0.22 + 0.55 * pose_trust) if near_contact else (0.03 + 0.12 * pose_trust)
                    if jump > max_jump:
                        dx = observed_position[0] - self.position[0]
                        dy = observed_position[1] - self.position[1]
                        step_dx, step_dy = limit_vector(dx, dy, max_jump)
                        observed_position = (
                            self.position[0] + step_dx,
                            self.position[1] + step_dy,
                        )
                        jump = max_jump

                alpha = 0.32 + 0.20 * pose_trust
                if not near_contact and pose_trust < 0.75:
                    alpha = min(alpha, 0.05)
                elif not near_contact:
                    alpha = min(alpha, 0.18)
                if jump > 0.12:
                    alpha = max(alpha, 0.55 + 0.15 * pose_trust)
                if pose_trust > 0.55 and (observed_position[0] > 2.5 or ball_obs[0] < 0.55):
                    alpha = max(alpha, 0.74)
                if pose_trust > 0.78 and (observed_position[0] > 3.5 or ball_obs[0] < 0.35):
                    alpha = max(alpha, 0.90)
                if jump > 0.55 and pose_trust < 0.35:
                    alpha = min(alpha, 0.28)
                if not near_contact and pose_trust < 0.75:
                    alpha = min(alpha, 0.05)
                elif not near_contact:
                    alpha = min(alpha, 0.18)
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

        process = self.process
        self.process = None
        self.enabled = False

        try:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                process.stdin.close()
        except OSError:
            pass

        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=1.0)
            except OSError:
                pass
        except OSError:
            pass


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
    INPUT_DIM = 19
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
        "mode_orbit",
        "mode_align",
        "mode_push",
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
        self.opening_active = os.environ.get("OPENING_VISIBLE_OVERRIDE", "1") != "0"
        self.opening_ball_position = None
        self.opening_ball_age = 10**9
        self.opening_seen_steps = 0
        self.opening_push_steps = 0
        self.opening_phase = "acquire"
        self.opening_control_pose = None
        self.odom_pose = (-1.0, 0.0, 0.0)
        self.fused_pose = (-1.0, 0.0, 0.0)
        self.pose_correction_trust = 0.0
        self.last_measurement_stats = {
            "goal_count": 0,
            "corner_count": 0,
            "cross_count": 0,
            "center_count": 0,
            "structure_count": 0,
            "total_count": 0,
        }

        self.prev_forward_cmd = 0.0
        self.prev_turn_cmd = 0.0
        self.low_confidence_steps = 0
        self.no_progress_steps = 0
        self.prev_ball_goal_dist = None
        self.prev_side_progress_metric = None
        self.side_stall_steps = 0
        self.last_seen_ball_rel = None
        self.rl_submode = "orbit"
        self.rl_push_steps = 0
        self.last_rl_features = np.zeros(self.actor.INPUT_DIM, dtype=np.float32)
        self.last_rl_action = np.zeros(2, dtype=np.float32)
        self.last_rl_context = {}
        self.prev_diagnostic_fused_pose = None
        self.prev_diagnostic_localizer_pose = None
        self.prev_diagnostic_odom_pose = None
        self.last_localization_diagnostics = {}
        self.no_landmark_steps = 0
        self.no_structure_steps = 0
        self.low_observability_steps = 0
        self.localization_recovery_steps = 0

    def close(self):
        self.visualizer.close()
        self.transition_logger.close()

    def _set_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            self.state_age = 0
            if new_state != "RL_BALL_PLAY":
                self.no_progress_steps = 0
                self.prev_ball_goal_dist = None
                self.prev_side_progress_metric = None
                self.side_stall_steps = 0
                self.rl_submode = "orbit"
                self.rl_push_steps = 0

    def _wheel_command(self, forward, turn):
        self.prev_forward_cmd = clamp(forward / RL_FORWARD_SCALE, -1.0, 1.0)
        self.prev_turn_cmd = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
        left = clamp(forward - turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        right = clamp(forward + turn, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        return {"left_motor": left, "right_motor": right}

    def _update_opening_ball(self, ball_obs, pose):
        if ball_obs is None:
            self.opening_ball_age += 1
            return

        self.last_seen_ball_rel = tuple(ball_obs)
        raw_observed = polar_to_world(pose, ball_obs)
        observed = clip_to_field(raw_observed, padding=0.12)
        if self.opening_ball_position is None:
            self.opening_ball_position = observed
            self.opening_phase = "stage"
        elif self.opening_phase in ("stage", "align"):
            old_x, old_y = self.opening_ball_position
            observed_y = observed[1]
            if (
                abs(old_y) > 0.55
                and old_y * observed_y < 0.0
                and self.localizer.confidence < 0.55
            ):
                observed_y = old_y
            jump = distance((old_x, old_y), observed)
            if jump > 0.45 and self.pose_correction_trust >= 0.35:
                alpha = 0.72
            elif jump > 0.45:
                alpha = 0.42
            else:
                alpha = 0.30
            if ball_obs[0] > 0.65 and self.pose_correction_trust < 0.25:
                alpha = min(alpha, 0.12)
            self.opening_ball_position = (
                (1.0 - alpha) * old_x + alpha * observed[0],
                (1.0 - alpha) * old_y + alpha * observed_y,
            )
            self.opening_ball_age = 0
            self.opening_seen_steps += 1
            return
        elif self.opening_phase == "push":
            old_x, old_y = self.opening_ball_position
            observed_y = observed[1]
            if (
                abs(old_y) > 0.55
                and old_y * observed_y < 0.0
                and self.localizer.confidence < 0.55
            ):
                observed_y = old_y
            new_x = max(old_x, 0.55 * old_x + 0.45 * observed[0])
            new_y = 0.94 * old_y + 0.06 * observed_y
            self.opening_ball_position = (
                new_x,
                new_y,
            )
            self.opening_ball_age = 0
            self.opening_seen_steps += 1
            return
        else:
            old_x, old_y = self.opening_ball_position
            jump = distance((old_x, old_y), observed)
            max_jump = 0.05 if ball_obs[0] > 0.55 else 0.18
            if jump > max_jump:
                step_dx, step_dy = limit_vector(observed[0] - old_x, observed[1] - old_y, max_jump)
                observed = (old_x + step_dx, old_y + step_dy)

            alpha = 0.34 if ball_obs[0] > 0.55 else 0.52
            self.opening_ball_position = (
                (1.0 - alpha) * old_x + alpha * observed[0],
                (1.0 - alpha) * old_y + alpha * observed[1],
            )

        self.opening_ball_age = 0
        self.opening_seen_steps += 1

    def _opening_drive_to_point(self, pose, target, max_forward=2.4, slow_radius=0.18):
        target_dist, target_angle = relative_polar(pose, target)
        turn = clamp(2.9 * target_angle, -RL_TURN_SCALE, RL_TURN_SCALE)
        abs_angle = abs(target_angle)
        if target_dist < slow_radius:
            forward = 0.35
        elif abs_angle < 0.40:
            forward = max_forward
        elif abs_angle < 0.85:
            forward = min(max_forward, 1.05)
        elif abs_angle < 1.35:
            forward = 0.30
        else:
            forward = 0.0
        return forward, turn, target_dist, target_angle

    def _opening_heading_error(self, pose, target_heading=0.0):
        del pose
        heading = self.localizer.pose[2]
        return wrap_to_pi(target_heading - heading)

    def _opening_pose_for_control(self, base_pose):
        target_pose = tuple(base_pose)
        if self.opening_active and self.localizer.confidence > 0.55:
            target_pose = tuple(self.localizer.pose)

        previous = self.opening_control_pose
        if previous is None:
            previous = tuple(base_pose)
        dx = target_pose[0] - previous[0]
        dy = target_pose[1] - previous[1]
        corr_dx, corr_dy = limit_vector(dx, dy, 0.18)
        smoothed = (
            previous[0] + corr_dx,
            previous[1] + corr_dy,
            wrap_to_pi(target_pose[2]),
        )
        self.opening_control_pose = smoothed
        return smoothed

    def _opening_local_push_control(self, pose, ball, ball_dist, ball_angle):
        heading_error = self._opening_heading_error(pose)
        lane_ball_x, lane_ball_y = ball
        if abs(lane_ball_y) > 1.10:
            lane_ball_y = math.copysign(1.10, lane_ball_y)

        if lane_ball_x > 3.15:
            target_heading = math.atan2(
                clamp(-0.75 * lane_ball_y, -1.0, 1.0),
                max(0.35, 4.85 - lane_ball_x),
            )
            heading_error = self._opening_heading_error(pose, target_heading=target_heading)
        turn = clamp(
            1.35 * ball_angle + 1.45 * heading_error,
            -RL_TURN_SCALE,
            RL_TURN_SCALE,
        )
        if abs(ball_angle) > 1.05:
            forward = 0.0
        elif abs(heading_error) < 1.10:
            forward = 5.0 if abs(ball_angle) < 0.72 else 2.2
        else:
            forward = 0.8
        return self._wheel_command(forward, turn)

    def _opening_heading_prior(self):
        if (
            not self.opening_active
            or self.opening_phase != "push"
            or self.opening_ball_position is None
        ):
            return None
        lane_ball_x, lane_ball_y = self.opening_ball_position
        if lane_ball_x < 3.20:
            return None
        if abs(lane_ball_y) > 1.10:
            lane_ball_y = math.copysign(1.10, lane_ball_y)
        return math.atan2(
            clamp(-0.45 * lane_ball_y, -0.70, 0.70),
            max(0.65, 4.85 - lane_ball_x),
        )

    def _opening_visible_ball_control(self, sensors, pose):
        if not self.opening_active:
            return None

        if (
            self.no_landmark_steps >= 360
            and self.localizer.confidence < 0.12
        ):
            return self._localization_scan_control(turn_scale=2.45)
        if (
            self.low_observability_steps >= 900
            and self.localizer.confidence < 0.12
            and self.pose_correction_trust < 0.03
        ):
            return self._localization_scan_control(turn_scale=2.35)

        ball_obs = sensors.get("ball")
        if ball_obs is None and self.opening_ball_position is None:
            return None

        self._update_opening_ball(ball_obs, pose)
        ball = self.opening_ball_position
        if ball is None:
            return None
        ball_rel = tuple(ball_obs) if ball_obs is not None else relative_polar(pose, ball)
        ball_dist, ball_angle = ball_rel

        if self.opening_phase == "push" and ball_obs is not None and ball_dist < 0.62:
            self.opening_push_steps += 1
            return self._opening_local_push_control(pose, ball, ball_dist, ball_angle)

        # This local opening mode is only for the initial get-behind problem.
        if (
            self.opening_ball_age > 5000
            or abs(ball[1]) > 2.20
        ):
            self.opening_active = False
            return None

        lane_target_y = clamp(-0.55 * ball[1], -0.42, 0.42)
        target_dx = 4.85 - ball[0]
        target_dy = lane_target_y - ball[1]
        target_norm = max(1e-6, math.hypot(target_dx, target_dy))
        target_unit_x = target_dx / target_norm
        target_unit_y = target_dy / target_norm
        lane_heading = math.atan2(target_unit_y, target_unit_x)
        lane_heading_error = wrap_to_pi(lane_heading - pose[2])
        staging_offset = 0.72
        staging = clip_to_field(
            (
                ball[0] - staging_offset * target_unit_x,
                ball[1] - staging_offset * target_unit_y,
            ),
            padding=0.18,
        )
        staging_dist, staging_angle = relative_polar(pose, staging)

        if self.opening_phase == "stage":
            if staging_dist < 0.18:
                self.opening_phase = "align"
            else:
                forward, turn, _, _ = self._opening_drive_to_point(
                    pose,
                    staging,
                    max_forward=2.5,
                    slow_radius=0.18,
                )
                if ball_obs is not None and ball_dist < 0.55:
                    forward = min(forward, 0.25)
                return self._wheel_command(forward, turn)

        if self.opening_phase == "align":
            if abs(lane_heading_error) < 0.20:
                self.opening_phase = "push"
            else:
                turn = clamp(2.8 * lane_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                return self._wheel_command(0.0, turn)

        if self.opening_phase == "push":
            self.opening_push_steps += 1
            if ball_obs is None:
                frozen_dist, frozen_angle = relative_polar(pose, ball)
                if frozen_dist > 0.45 or abs(frozen_angle) > 0.75:
                    self.opening_phase = "stage"
                    forward, turn, _, _ = self._opening_drive_to_point(
                        pose,
                        staging,
                        max_forward=2.0,
                        slow_radius=0.18,
                    )
                    return self._wheel_command(forward, turn)
            center_bias = clamp(-1.05 * ball[1], -0.95, 0.95)
            if self.opening_push_steps > 420:
                ball_gain = 0.45
                lane_gain = 2.85
                center_gain = 0.28
            else:
                ball_gain = 1.25
                lane_gain = 1.55
                center_gain = 0.12
            visible_turn = 0.0 if ball_obs is None else ball_gain * ball_angle
            turn = clamp(
                visible_turn + lane_gain * lane_heading_error + center_gain * center_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            if ball_obs is not None and abs(ball_angle) > 1.05:
                forward = 0.0
            elif abs(lane_heading_error) < 0.75:
                forward = 4.8
            else:
                forward = 1.4
            return self._wheel_command(forward, turn)

        robot_from_ball_x = pose[0] - ball[0]
        robot_from_ball_y = pose[1] - ball[1]
        behind_depth = robot_from_ball_x * (-target_unit_x) + robot_from_ball_y * (-target_unit_y)
        lateral_error = abs(robot_from_ball_x * (-target_unit_y) + robot_from_ball_y * target_unit_x)
        lane_heading = math.atan2(lane_target_y - ball[1], 4.85 - ball[0])
        lane_heading_error = wrap_to_pi(lane_heading - pose[2])
        centered_behind = behind_depth > 0.24 and lateral_error < 0.30
        close_behind = behind_depth > 0.10 and lateral_error < 0.40
        push_ready = (
            ball_dist < 0.72
            and centered_behind
            and abs(ball_angle) < 0.82
            and abs(lane_heading_error) < 0.95
        )

        if push_ready:
            self.opening_push_steps += 1
            center_bias = clamp(-1.10 * ball[1], -1.0, 1.0)
            turn = clamp(
                1.15 * ball_angle + 2.35 * lane_heading_error + 0.65 * center_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            forward = 4.7 if abs(ball_angle) < 0.55 and abs(lane_heading_error) < 0.70 else 2.1
            return self._wheel_command(forward, turn)

        self.opening_push_steps = 0
        if close_behind and ball_dist < 0.78:
            turn = clamp(
                1.15 * ball_angle + 2.25 * lane_heading_error,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            forward = 0.65 if abs(ball_angle) < 0.95 and abs(lane_heading_error) < 1.15 else 0.0
            return self._wheel_command(forward, turn)

        staging_offset = 0.74 if ball_dist > 0.52 or lateral_error > 0.42 else 0.58
        staging = clip_to_field(
            (
                ball[0] - staging_offset * target_unit_x,
                ball[1] - staging_offset * target_unit_y,
            ),
            padding=0.18,
        )
        forward, turn, staging_dist, staging_angle = self._opening_drive_to_point(
            pose,
            staging,
            max_forward=2.7,
            slow_radius=0.16,
        )

        if ball_obs is not None and ball_dist < 0.48 and not close_behind:
            forward = 0.0
            turn = clamp(2.2 * staging_angle, -RL_TURN_SCALE, RL_TURN_SCALE)
        elif ball_obs is not None and ball_dist < 0.70 and not centered_behind:
            forward = min(forward, 0.45 if abs(staging_angle) < 0.95 else 0.0)
        elif staging_dist < 0.20 and abs(lane_heading_error) > 0.65:
            forward = 0.0
            turn = clamp(2.7 * lane_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)

        return self._wheel_command(forward, turn)

    def _advance_pose(self, pose, odometry):
        if odometry is None:
            return pose
        forward = float(odometry[0])
        rotation = float(odometry[1])
        x, y, theta = pose
        mid_theta = theta + 0.5 * rotation
        x += forward * math.cos(mid_theta)
        y += forward * math.sin(mid_theta)
        theta = wrap_to_pi(theta + rotation)
        return x, y, theta

    def _filtered_fused_odometry(self, odometry):
        if odometry is None:
            return None
        forward = float(odometry[0])
        raw_rotation = float(odometry[1])
        commanded_turn = self.prev_turn_cmd * RL_TURN_SCALE
        if self.last_measurement_stats.get("total_count", 0) <= 0:
            rotation = 0.2915 * raw_rotation + 0.00450 * commanded_turn
        else:
            rotation = 0.0160 * raw_rotation + 0.00727 * commanded_turn
        if abs(commanded_turn) < 0.18 and abs(raw_rotation) < 0.025:
            rotation = 0.12 * raw_rotation
        return forward, rotation

    def _integrate_odometry_pose(self, odometry):
        if odometry is None:
            return self.odom_pose
        self.odom_pose = self._advance_pose(self.odom_pose, odometry)
        self.fused_pose = self._advance_pose(self.fused_pose, self._filtered_fused_odometry(odometry))
        return self.odom_pose

    def _observation_stats(self, sensors):
        goal_count = len(sensors.get("goal", []))
        corner_count = len(sensors.get("corners", []))
        cross_count = len(sensors.get("penalty_cross", []))
        center_count = 1 if sensors.get("center_circle") is not None else 0
        structure_count = corner_count + cross_count + center_count
        stats = {
            "goal_count": goal_count,
            "corner_count": corner_count,
            "cross_count": cross_count,
            "center_count": center_count,
            "structure_count": structure_count,
            "total_count": goal_count + structure_count,
        }
        stats["landmark_bin"] = landmark_bin(stats)
        return stats

    def _ball_finish_zone(self, ball_position):
        if ball_position is None:
            return False
        return distance(ball_position, self.attack_goal) < 1.8 or ball_position[0] > 3.25

    def _update_localization_visibility_counters(self):
        stats = self.last_measurement_stats
        if stats.get("total_count", 0) <= 0:
            self.no_landmark_steps += 1
        else:
            self.no_landmark_steps = 0

        if stats.get("structure_count", 0) <= 0:
            self.no_structure_steps += 1
        else:
            self.no_structure_steps = 0

        if stats.get("landmark_bin") in ("goal_structural", "2plus_mixed"):
            self.low_observability_steps = 0
        else:
            self.low_observability_steps += 1

    def _needs_localization_recovery(self, sensors):
        if self.state == "RL_BALL_PLAY":
            if (
                self.no_landmark_steps >= 360
                and self.localizer.confidence < 0.12
            ):
                return True
            if (
                self.low_observability_steps >= 900
                and self.localizer.confidence < 0.10
                and self.pose_correction_trust < 0.03
            ):
                return True
            if sensors.get("ball") is not None and self.ball_tracker.age <= 4:
                return False
            if (
                self.no_landmark_steps >= 240
                and self.localizer.confidence < 0.10
                and (self.ball_tracker.position is None or self.ball_tracker.age > 30)
            ):
                return True
            if self.no_landmark_steps >= 900 and self.localizer.confidence < 0.16:
                return True
            return False
        if os.environ.get("LOCALIZATION_RECOVERY_OVERRIDE", "0") != "1":
            return (
                (
                    self.no_landmark_steps >= 360
                    and self.localizer.confidence < 0.12
                )
                or (
                    self.low_observability_steps >= 900
                    and self.localizer.confidence < 0.12
                    and self.pose_correction_trust < 0.03
                )
            )
        if sensors.get("ball") is not None and self.ball_tracker.age <= 4:
            return False
        if self.opening_active and self.opening_ball_age <= 90:
            return False
        if self.no_landmark_steps >= 120:
            return True
        if self.no_structure_steps >= 220 and self.pose_correction_trust < 0.08:
            return True
        fused_heading_error = (
            self.last_localization_diagnostics.get("fused", {}).get("heading_error_deg")
            if self.last_localization_diagnostics
            else None
        )
        if fused_heading_error is not None and fused_heading_error > 45.0 and self.no_structure_steps >= 80:
            return True
        return False

    def _low_info_stale_ball(self, max_no_landmark=120, max_ball_age=10):
        stats = self.last_measurement_stats
        return (
            self.no_landmark_steps >= max_no_landmark
            and stats.get("total_count", 0) == 0
            and self.localizer.confidence < 0.10
            and (self.ball_tracker.position is None or self.ball_tracker.age > max_ball_age)
        )

    def _localization_scan_control(self, turn_scale=2.35):
        self.localization_recovery_steps += 1
        direction = 1.0 if (self.localization_recovery_steps // 140) % 2 == 0 else -1.0
        return self._wheel_command(0.0, turn_scale * direction)

    def _localization_recovery_control(self):
        return self._localization_scan_control(turn_scale=2.45)

    def _localizer_trust(self, sensors):
        stats = self._observation_stats(sensors)
        self.last_measurement_stats = stats
        if stats["total_count"] == 0:
            return 0.0, stats

        diagnostics = getattr(self.localizer, "diagnostics", {})
        structure_score = float(diagnostics.get("observability", 0.0))
        residual_quality = float(diagnostics.get("residual_quality", 0.0))
        spread_xy = float(diagnostics.get("hypothesis_spread_xy", 1.0))
        heading_spread = float(diagnostics.get("heading_spread", math.pi))
        compactness = (
            1.0 - clamp(spread_xy / 0.90, 0.0, 1.0)
        ) * (
            1.0 - clamp(heading_spread / math.radians(55.0), 0.0, 1.0)
        )
        trust = clamp(self.localizer.confidence, 0.0, 1.0) * structure_score
        trust *= 0.35 + 0.65 * residual_quality
        trust *= 0.40 + 0.60 * compactness

        if stats["landmark_bin"] == "1_goal_only":
            trust = min(trust * 0.08, 0.08)
        elif stats["landmark_bin"] == "0_landmarks":
            trust = 0.0
        elif stats["structure_count"] == 0:
            trust = min(trust * 0.22, 0.16)
        elif stats["landmark_bin"] == "1_structural" and residual_quality > 0.45:
            trust = max(trust, 0.055)
        elif stats["structure_count"] >= 1 and residual_quality > 0.65 and self.localizer.confidence > 0.25:
            trust = max(trust, 0.075 * structure_score)

        if self._ball_finish_zone(self.ball_tracker.position):
            trust *= 0.55
        if self.state == "RL_BALL_PLAY":
            if stats["structure_count"] < 2:
                trust *= 0.45
            if stats["center_count"] == 0 and stats["cross_count"] == 0:
                trust *= 0.35
            else:
                trust *= 0.65
            if (
                stats["landmark_bin"] == "1_structural"
                and residual_quality > 0.70
                and self.localizer.confidence > 0.14
            ):
                trust = max(trust, 0.055)
        if self.state == "SEARCH_BALL":
            trust = min(1.0, trust * 1.25)
        return trust, stats

    def _fuse_pose_estimate(self, localizer_pose, sensors):
        odom_pose = self.odom_pose
        base_pose = self.fused_pose
        if localizer_pose is None:
            self.pose_correction_trust = 0.0
            self.fused_pose = base_pose
            return self.fused_pose

        trust, stats = self._localizer_trust(sensors)
        self.pose_correction_trust = trust
        base_outside_arena = (
            abs(base_pose[0]) > ARENA_X_HALF + 0.20
            or abs(base_pose[1]) > ARENA_Y_HALF + 0.20
        )
        localizer_plausible = (
            abs(localizer_pose[0]) <= ARENA_X_HALF + 0.15
            and abs(localizer_pose[1]) <= ARENA_Y_HALF + 0.15
        )
        dx = float(localizer_pose[0] - base_pose[0])
        dy = float(localizer_pose[1] - base_pose[1])
        theta_error = wrap_to_pi(localizer_pose[2] - base_pose[2])
        pos_error = math.hypot(dx, dy)
        finish_zone = self._ball_finish_zone(self.ball_tracker.position)
        side_estimate = self._is_side_channel(self.ball_tracker.position)
        diagnostics = getattr(self.localizer, "diagnostics", {})
        strong_update_allowed = bool(diagnostics.get("strong_update_allowed", False))
        residual_quality = float(diagnostics.get("residual_quality", 0.0))
        low_info = stats["landmark_bin"] in ("0_landmarks", "1_goal_only", "goal_only_multi")
        weak_info = low_info or stats["landmark_bin"] == "1_structural"
        high_info = stats["landmark_bin"] in ("goal_structural", "2plus_mixed")
        high_info_recovery = (
            high_info
            and localizer_plausible
            and strong_update_allowed
            and residual_quality >= 0.55
            and self.localizer.confidence >= 0.30
        )
        if (
            (
                base_outside_arena
                or (
                    self.state == "RL_BALL_PLAY"
                    and pos_error > 0.75
                    and (finish_zone or side_estimate or abs(odom_pose[1]) > FIELD_Y_HALF - 0.35)
                )
            )
            and localizer_plausible
            and strong_update_allowed
            and residual_quality >= 0.18
            and self.localizer.confidence >= 0.20
        ):
            strong_pos_cap = 0.18 if stats["landmark_bin"] in ("goal_structural", "2plus_mixed") else 0.28
            corr_dx, corr_dy = limit_vector(dx, dy, strong_pos_cap)
            corr_theta = clamp(theta_error, -math.radians(32.0), math.radians(32.0))
            self.fused_pose = (
                base_pose[0] + corr_dx,
                base_pose[1] + corr_dy,
                wrap_to_pi(base_pose[2] + corr_theta),
            )
            recovery_trust = max(trust, 0.22)
            if pos_error > 0.45:
                recovery_trust = min(recovery_trust, 0.07)
            self.pose_correction_trust = recovery_trust
            return self.fused_pose

        if weak_info:
            corr_dx, corr_dy = limit_vector(dx, dy, 0.28)
            self.fused_pose = (
                base_pose[0] + corr_dx,
                base_pose[1] + corr_dy,
                wrap_to_pi(localizer_pose[2]),
            )
            if stats["landmark_bin"] == "1_structural":
                self.pose_correction_trust = min(self.pose_correction_trust, 0.055)
            else:
                self.pose_correction_trust = 0.0
            return self.fused_pose

        if trust <= 0.02:
            if high_info_recovery:
                corr_dx, corr_dy = limit_vector(dx, dy, 0.045)
                corr_theta = clamp(theta_error, -math.radians(38.0), math.radians(38.0))
                self.fused_pose = (
                    base_pose[0] + corr_dx,
                    base_pose[1] + corr_dy,
                    wrap_to_pi(base_pose[2] + corr_theta),
                )
                self.pose_correction_trust = min(trust, 0.07)
                return self.fused_pose
            if stats["landmark_bin"] == "0_landmarks":
                heading_prior = self._opening_heading_prior()
                if heading_prior is not None:
                    prior_error = wrap_to_pi(heading_prior - base_pose[2])
                    prior_cap = math.radians(4.0)
                    self.fused_pose = (
                        base_pose[0],
                        base_pose[1],
                        wrap_to_pi(base_pose[2] + clamp(prior_error, -prior_cap, prior_cap)),
                    )
                    return self.fused_pose
            self.fused_pose = base_pose
            return self.fused_pose

        if low_info and (pos_error > 0.16 or abs(theta_error) > math.radians(10.0)):
            self.pose_correction_trust = min(self.pose_correction_trust, 0.08)
            self.fused_pose = base_pose
            return self.fused_pose

        if self.state == "RL_BALL_PLAY" and pos_error > (0.30 if strong_update_allowed else 0.18):
            if not strong_update_allowed or residual_quality < 0.10:
                self.fused_pose = base_pose
                return self.fused_pose

        if not strong_update_allowed and (pos_error > 0.20 or abs(theta_error) > math.radians(12.0)):
            self.fused_pose = base_pose
            return self.fused_pose

        pos_cap = 0.015 + 0.45 * trust
        theta_cap = math.radians(1.5) + math.radians(34.0) * trust

        if stats["structure_count"] >= 2 and self.localizer.confidence > 0.45 and not finish_zone:
            pos_cap += 0.16
            theta_cap += math.radians(14.0)
        elif stats["structure_count"] >= 1 and self.localizer.confidence > 0.35 and not finish_zone:
            pos_cap += 0.08
            theta_cap += math.radians(7.0)

        if self.state == "SEARCH_BALL" and stats["structure_count"] >= 1 and self.localizer.confidence > 0.35:
            pos_cap += 0.08
            theta_cap += math.radians(8.0)

        if (
            self.state != "RL_BALL_PLAY"
            and stats["center_count"] > 0
            and stats["structure_count"] >= 2
            and self.localizer.confidence > 0.65
        ):
            pos_cap = max(pos_cap, 0.36)
            theta_cap = max(theta_cap, math.radians(28.0))

        if stats["landmark_bin"] == "1_goal_only":
            pos_cap = min(pos_cap, 0.010)
            theta_cap = min(theta_cap, math.radians(1.2))
        elif stats["landmark_bin"] == "1_structural":
            pos_cap = min(pos_cap, 0.075)
            theta_cap = min(theta_cap, math.radians(12.0))
        elif high_info:
            pos_cap = min(pos_cap, 0.045 if high_info_recovery else 0.035)
            theta_cap = min(
                max(theta_cap, math.radians(38.0)) if high_info_recovery else theta_cap,
                math.radians(38.0 if high_info_recovery else 4.5),
            )

        pos_cap = min(pos_cap, 0.48)
        theta_cap = min(theta_cap, math.radians(34.0))

        corr_dx, corr_dy = limit_vector(dx, dy, pos_cap)
        corr_theta = clamp(theta_error, -theta_cap, theta_cap)
        self.fused_pose = (
            base_pose[0] + corr_dx,
            base_pose[1] + corr_dy,
            wrap_to_pi(base_pose[2] + corr_theta),
        )
        if stats["landmark_bin"] in ("goal_structural", "2plus_mixed") and pos_error > 0.20:
            self.pose_correction_trust = min(self.pose_correction_trust, 0.07)
        return self.fused_pose

    def _search_control(self):
        if self.ball_tracker.is_fresh(max_age=60):
            ball_position = self.ball_tracker.position
            pose = self.fused_pose
            ball_rel = self.ball_tracker.relative_ball(pose)
            if ball_rel is not None:
                return self._action_to_control(
                    self._geometric_rl_prior(
                        {
                            "pose": pose,
                            "ball_position": ball_position,
                            "ball_rel": ball_rel,
                        }
                    )
                )

        if self.last_seen_ball_rel is not None and (
            self.state_age < 18
            or (self.last_seen_ball_rel[0] < 1.0 and self.state_age < 140)
        ):
            turn = clamp(2.8 * self.last_seen_ball_rel[1], -2.8, 2.8)
            forward = 0.0 if self.last_seen_ball_rel[0] < 1.0 else 0.7 if abs(self.last_seen_ball_rel[1]) < 0.40 else 0.0
            return self._wheel_command(forward, turn)

        sweep_phase = self.state_age // 120
        if sweep_phase == 0:
            self.search_direction = 1.0
            forward = 0.0
            turn = 2.8
        elif sweep_phase == 1:
            self.search_direction = -1.0
            forward = 0.0
            turn = -2.8
        else:
            self.search_direction = -1.0 if sweep_phase % 2 == 0 else 1.0
            forward = 0.8 if self.localizer.confidence > 0.25 else 0.0
            turn = 1.9 * self.search_direction
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

    def _is_side_channel(self, ball_position):
        return (
            ball_position is not None
            and (
                abs(ball_position[1]) > 2.05
                or (ball_position[0] > 0.50 and abs(ball_position[1]) > 1.35)
                or (ball_position[0] > 1.45 and abs(ball_position[1]) > 0.85)
            )
        )

    def _side_channel_resolved(self, ball_position):
        if ball_position is None:
            return True
        if abs(ball_position[1]) < 0.62:
            return True
        return ball_position[0] > 4.15 and abs(ball_position[1]) <= GOAL_HALF_WIDTH - 0.08

    def _update_rl_submode(self, pose, ball_position, ball_rel):
        if ball_position is None:
            self.rl_submode = "orbit"
            self.rl_push_steps = 0
            return

        goal_dx = self.attack_goal[0] - ball_position[0]
        goal_dy = self.attack_goal[1] - ball_position[1]
        norm = max(1e-6, math.hypot(goal_dx, goal_dy))
        goal_unit_x = goal_dx / norm
        goal_unit_y = goal_dy / norm
        goal_heading = math.atan2(goal_unit_y, goal_unit_x)
        goal_heading_error = wrap_to_pi(goal_heading - pose[2])
        field_heading_error = wrap_to_pi(0.0 - pose[2])
        ball_to_goal = norm
        robot_from_ball_x = pose[0] - ball_position[0]
        robot_from_ball_y = pose[1] - ball_position[1]
        robot_ball_radius = math.hypot(robot_from_ball_x, robot_from_ball_y)
        behind_score = (
            robot_from_ball_x * (-goal_unit_x)
            + robot_from_ball_y * (-goal_unit_y)
        )
        lateral_error = abs(
            robot_from_ball_x * (-goal_unit_y)
            + robot_from_ball_y * goal_unit_x
        )
        near_goal_side = ball_position[0] > 3.0 and abs(ball_position[1]) > 0.58
        finish_zone = ball_to_goal < 1.8 or ball_position[0] > 3.25
        side_channel = self._is_side_channel(ball_position)
        well_behind = (
            behind_score > (0.30 if near_goal_side else 0.42)
            and lateral_error < (0.42 if near_goal_side else 0.34)
            and 0.42 <= robot_ball_radius <= (1.10 if near_goal_side else 0.95)
        )
        close_push_ready = (
            ball_rel[0] < 0.42
            and behind_score > 0.05
            and lateral_error < (0.36 if near_goal_side else 0.32)
            and abs(goal_heading_error) < (1.15 if near_goal_side else 1.05)
        )

        if self.rl_submode == "recenter":
            self.rl_push_steps = 0
            if not self._side_channel_resolved(ball_position):
                return
            if well_behind:
                self.rl_submode = "align"
            else:
                self.rl_submode = "orbit"

        if side_channel and not self._side_channel_resolved(ball_position):
            self.rl_submode = "recenter"
            self.rl_push_steps = 0
            return

        if close_push_ready:
            self.rl_submode = "push"
            return

        if finish_zone and ball_rel[0] < 0.95:
            if near_goal_side and not (behind_score > 0.20 and lateral_error < 0.30):
                self.rl_submode = "recenter"
                self.rl_push_steps = 0
                return
            self.rl_submode = "push"
            return

        if self.rl_submode == "push":
            self.rl_push_steps += 1
            if ball_rel[0] > 0.95 or abs(ball_rel[1]) > (1.05 if near_goal_side else 1.35):
                self.rl_submode = "orbit"
                self.rl_push_steps = 0
        elif self.rl_submode == "align":
            if not well_behind:
                self.rl_submode = "orbit"
            elif (
                abs(goal_heading_error) < (0.62 if near_goal_side else 0.72)
                and abs(ball_rel[1]) < (0.90 if near_goal_side else 0.95)
                and ball_rel[0] < 0.90
            ):
                self.rl_submode = "push"
                self.rl_push_steps = 0
        elif well_behind:
            self.rl_submode = "align"

    def _rl_submode_features(self):
        if self.rl_submode in ("align", "recenter"):
            return (0.0, 1.0, 0.0)
        if self.rl_submode == "push":
            return (0.0, 0.0, 1.0)
        return (1.0, 0.0, 0.0)

    def _ball_observation(self, pose, sensors):
        ball_obs = sensors.get("ball")
        if ball_obs is not None:
            self.last_seen_ball_rel = tuple(ball_obs)
            return tuple(ball_obs), self.ball_tracker.position
        if self.ball_tracker.is_fresh(max_age=120):
            rel_ball = self.ball_tracker.relative_ball(pose)
            if rel_ball is not None:
                return rel_ball, self.ball_tracker.position
        return None, None

    def _has_reliable_ball_estimate(self, max_age=35):
        if self.ball_tracker.position is None:
            return False
        if self.ball_tracker.age <= max_age:
            return True
        if self.ball_tracker.confidence >= 0.25 and self.ball_tracker.age <= max(max_age, 80):
            return True
        return False

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
        self._update_rl_submode(pose, ball_position, ball_rel)
        mode_orbit, mode_align, mode_push = self._rl_submode_features()

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
                mode_orbit,
                mode_align,
                mode_push,
            ],
            dtype=np.float32,
        )

        context = {
            "ball_visible": sensors.get("ball") is not None,
            "ball_rel": ball_rel,
            "ball_position": ball_position,
            "goal_rel": goal_rel,
            "staging_point": staging_point,
            "staging_rel": staging_rel,
            "wall_margin_norm": wall_margin_norm,
            "rl_submode": self.rl_submode,
            "pose": pose,
        }
        return features, context

    def _geometric_rl_prior(self, context):
        pose = context.get("pose", self.localizer.pose)
        ball_visible = bool(context.get("ball_visible", False))
        ball = context.get("ball_position")
        ball_rel = context.get("ball_rel")
        if ball is None or ball_rel is None:
            return np.array([-1.0, 0.0], dtype=np.float32)

        if not ball_visible and ball_rel[0] < 0.85:
            if ball[0] > 4.25:
                turn = clamp(1.2 * ball_rel[1], -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 2.2 if abs(ball_rel[1]) < 0.45 else 1.2 if abs(ball_rel[1]) < 1.10 else 0.35
            elif self.rl_submode == "push" and ball_rel[0] < 0.65:
                field_heading_error = wrap_to_pi(0.0 - pose[2])
                centerline_bias = clamp(-1.10 * ball[1], -1.0, 1.0)
                turn = clamp(
                    0.95 * ball_rel[1] + 1.45 * field_heading_error + 0.75 * centerline_bias,
                    -RL_TURN_SCALE,
                    RL_TURN_SCALE,
                )
                forward = 2.4 if abs(field_heading_error) < 1.25 else 1.1
            else:
                turn = clamp(2.8 * ball_rel[1], -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 0.25 if abs(ball_rel[1]) < 0.25 else 0.0
            a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
            a_turn = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
            return np.array([a_forward, a_turn], dtype=np.float32)

        goal_dx = self.attack_goal[0] - ball[0]
        goal_dy = self.attack_goal[1] - ball[1]
        goal_norm = max(1e-6, math.hypot(goal_dx, goal_dy))
        goal_unit_x = goal_dx / goal_norm
        goal_unit_y = goal_dy / goal_norm
        goal_heading = math.atan2(goal_unit_y, goal_unit_x)
        goal_heading_error = wrap_to_pi(goal_heading - pose[2])
        field_heading_error = wrap_to_pi(0.0 - pose[2])
        robot_from_ball_x = pose[0] - ball[0]
        robot_from_ball_y = pose[1] - ball[1]
        robot_ball_radius = max(0.05, math.hypot(robot_from_ball_x, robot_from_ball_y))
        robot_ball_angle = math.atan2(robot_from_ball_y, robot_from_ball_x)
        desired_behind_angle = math.atan2(-goal_unit_y, -goal_unit_x)
        orbit_error = wrap_to_pi(desired_behind_angle - robot_ball_angle)
        behind_score = robot_from_ball_x * (-goal_unit_x) + robot_from_ball_y * (-goal_unit_y)
        lateral_error = abs(robot_from_ball_x * (-goal_unit_y) + robot_from_ball_y * goal_unit_x)
        near_goal_side = ball[0] > 3.0 and abs(ball[1]) > 0.58
        finish_zone = goal_norm < 1.8 or ball[0] > 3.25

        if ball_visible and ball_rel[0] < 0.42 and not (ball[0] > 4.25 and abs(ball[1]) > 0.95):
            ball_angle = ball_rel[1]
            lane_target_y = 0.0 if abs(ball[1]) > 0.55 else 0.35 * ball[1]
            lane_heading = math.atan2(lane_target_y - ball[1], 4.85 - ball[0])
            lane_heading_error = wrap_to_pi(lane_heading - pose[2])
            centerline_bias = clamp(-1.15 * ball[1], -1.15, 1.15)
            close_behind_score = robot_from_ball_x * (-goal_unit_x) + robot_from_ball_y * (-goal_unit_y)
            close_lateral_error = abs(robot_from_ball_x * (-goal_unit_y) + robot_from_ball_y * goal_unit_x)
            if ball_rel[0] < 0.30 and not (close_behind_score > 0.10 and close_lateral_error < 0.46):
                if (
                    self.rl_submode == "recenter"
                    and ball[0] > 2.60
                    and abs(ball[1]) > 1.45
                ):
                    return self._recenter_action(pose, ball, ball_rel, ball_visible=True)
                turn = clamp(
                    0.85 * ball_angle + 1.55 * lane_heading_error + 0.95 * centerline_bias,
                    -RL_TURN_SCALE,
                    RL_TURN_SCALE,
                )
                if (
                    self.rl_submode == "recenter"
                    and ball_rel[0] < 0.14
                    and abs(ball_angle) < 0.55
                    and abs(field_heading_error) < 1.45
                ):
                    if abs(ball[1]) > 0.25:
                        side_sign = 1.0 if ball[1] > 0.0 else -1.0
                        inward_turn = -side_sign * max(0.25, min(abs(turn), 1.15))
                        turn = clamp(0.45 * turn + 0.55 * inward_turn, -RL_TURN_SCALE, RL_TURN_SCALE)
                    forward = 1.05 if abs(ball_angle) < 0.35 else 0.55
                    a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
                    return np.array([a_forward, clamp(turn / RL_TURN_SCALE, -1.0, 1.0)], dtype=np.float32)
                if (
                    self.rl_submode == "recenter"
                    and ball_rel[0] < 0.28
                    and (abs(ball[1]) > 1.05 or self._wall_margin(pose) < 0.24)
                    and abs(field_heading_error) < 1.45
                ):
                    side_sign = 1.0 if ball[1] > 0.0 else -1.0
                    inward_turn = -side_sign * max(0.45, min(abs(turn), RL_TURN_SCALE))
                    turn = clamp(0.35 * turn + 0.65 * inward_turn, -RL_TURN_SCALE, RL_TURN_SCALE)
                    forward = 0.70 if abs(ball_angle) < 1.05 else 0.35
                    a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
                    return np.array([a_forward, clamp(turn / RL_TURN_SCALE, -1.0, 1.0)], dtype=np.float32)
                return np.array([-1.0, clamp(turn / RL_TURN_SCALE, -1.0, 1.0)], dtype=np.float32)
            turn = clamp(
                0.75 * ball_angle + 2.45 * lane_heading_error + 0.85 * centerline_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            if abs(ball[1]) > 0.95:
                forward = 1.6 if abs(lane_heading_error) < 0.85 and abs(ball_angle) < 0.90 else 0.55
            elif abs(lane_heading_error) < 0.75 and abs(ball_angle) < 0.70:
                forward = 3.0
            elif abs(lane_heading_error) < 1.25 and abs(ball_angle) < 0.95:
                forward = 1.35
            else:
                forward = 0.35 if abs(ball_angle) < 1.20 else 0.0
            a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
            a_turn = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
            return np.array([a_forward, a_turn], dtype=np.float32)

        recent_ball = ball_visible or self.ball_tracker.age <= 4
        if recent_ball and ball[0] < 3.05 and abs(ball[1]) < 1.55:
            target_y = ball[1] if abs(ball[1]) <= GOAL_HALF_WIDTH + 0.20 else 0.0
            target_dx = 4.80 - ball[0]
            target_dy = target_y - ball[1]
            target_norm = max(1e-6, math.hypot(target_dx, target_dy))
            target_unit_x = target_dx / target_norm
            target_unit_y = target_dy / target_norm
            desired_heading = math.atan2(target_unit_y, target_unit_x)
            desired_heading_error = wrap_to_pi(desired_heading - pose[2])
            target_behind_score = (
                robot_from_ball_x * (-target_unit_x)
                + robot_from_ball_y * (-target_unit_y)
            )
            target_lateral_error = abs(
                robot_from_ball_x * (-target_unit_y)
                + robot_from_ball_y * target_unit_x
            )
            push_ready = (
                ball_rel[0] < 0.72
                and target_behind_score > 0.12
                and target_lateral_error < 0.34
                and abs(desired_heading_error) < 1.15
            )
            behind_but_misaligned = (
                ball_rel[0] < 0.78
                and target_behind_score > 0.08
                and target_lateral_error < 0.46
            )

            if push_ready:
                ball_angle = ball_rel[1]
                turn = clamp(
                    1.05 * ball_angle + 2.15 * desired_heading_error,
                    -RL_TURN_SCALE,
                    RL_TURN_SCALE,
                )
                if abs(ball[1]) > 0.72 and abs(desired_heading_error) > 0.42:
                    forward = 0.75
                elif abs(desired_heading_error) < 0.48:
                    forward = 4.8 if abs(ball_angle) < 0.80 else 2.5
                elif abs(desired_heading_error) < 1.10:
                    forward = 1.25
                else:
                    forward = 0.0
                if ball[0] < 1.8 and ball_rel[0] < 0.30:
                    forward = min(forward, 2.0 if abs(desired_heading_error) < 0.35 else 0.75)
            elif behind_but_misaligned:
                ball_angle = ball_rel[1]
                turn = clamp(
                    1.15 * ball_angle + 2.65 * desired_heading_error,
                    -RL_TURN_SCALE,
                    RL_TURN_SCALE,
                )
                if abs(desired_heading_error) < 1.10 and abs(ball_angle) < 0.90:
                    forward = 1.1
                elif abs(desired_heading_error) < 1.65 and abs(ball_angle) < 1.10:
                    forward = 0.35
                else:
                    forward = 0.0
            else:
                staging = clip_to_field(
                    (
                        ball[0] - 0.78 * target_unit_x,
                        ball[1] - 0.78 * target_unit_y,
                    ),
                    padding=0.18,
                )
                staging_dist, staging_angle = relative_polar(pose, staging)
                turn = clamp(2.65 * staging_angle, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = (
                    2.4
                    if abs(staging_angle) < 0.45
                    else 0.9
                    if abs(staging_angle) < 0.90
                    else 0.25
                    if abs(staging_angle) < 1.35
                    else 0.0
                )
                if staging_dist < 0.18:
                    forward = min(forward, 0.55)
                if ball_visible and ball_rel[0] < 0.70 and abs(ball_rel[1]) < 1.15:
                    forward = min(forward, 0.35 if abs(staging_angle) < 0.75 else 0.0)

            a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
            a_turn = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
            return np.array([a_forward, a_turn], dtype=np.float32)

        if ball_visible and ball_rel[0] < 0.38 and not (ball[0] > 4.25 and abs(ball[1]) > 0.95):
            ball_angle = ball_rel[1]
            centerline_bias = clamp(-1.05 * ball[1], -1.0, 1.0)
            turn = clamp(
                0.95 * ball_angle + 1.75 * field_heading_error + 0.70 * centerline_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            forward = 3.5 if abs(field_heading_error) < 1.15 else 1.4
            if abs(ball_angle) > 0.95:
                forward = min(forward, 1.2)
            a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
            a_turn = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
            return np.array([a_forward, a_turn], dtype=np.float32)

        if (
            ball_visible
            and ball[0] < 1.45
            and ball_rel[0] < 1.05
            and abs(ball_rel[1]) < 1.15
        ):
            behind_score = robot_from_ball_x * (-goal_unit_x) + robot_from_ball_y * (-goal_unit_y)
            lateral_error = abs(robot_from_ball_x * (-goal_unit_y) + robot_from_ball_y * goal_unit_x)
            outer_side = ball[1] * (pose[1] - ball[1]) > 0.0
            close_outer_push = (
                ball_rel[0] < 0.36
                and outer_side
                and behind_score > 0.07
                and lateral_error < 0.28
                and abs(goal_heading_error) < 0.95
            )
            ready_to_push = (
                close_outer_push
                or (
                    behind_score > 0.24
                    and lateral_error < 0.36
                    and abs(goal_heading_error) < 1.10
                )
            )
            if ready_to_push:
                ball_angle = ball_rel[1]
                centerline_bias = clamp(-1.05 * ball[1], -1.0, 1.0)
                turn = clamp(
                    1.15 * ball_angle
                    + 0.75 * goal_heading_error
                    + 0.95 * field_heading_error
                    + 0.70 * centerline_bias,
                    -RL_TURN_SCALE,
                    RL_TURN_SCALE,
                )
                forward = 2.1 if close_outer_push else 2.4 if abs(ball_angle) < 0.65 else 1.1
            elif abs(ball[1]) > 0.42:
                if ball_rel[0] < 0.50:
                    lateral_offset = -0.36 if ball[1] > 0.0 else 0.36
                    staging = clip_to_field(
                        (
                            ball[0] - 0.56 * goal_unit_x,
                            ball[1] - 0.30 * goal_unit_y + lateral_offset,
                        ),
                        padding=0.20,
                    )
                    _, target_angle = relative_polar(pose, staging)
                    turn = clamp(2.45 * target_angle, -RL_TURN_SCALE, RL_TURN_SCALE)
                    forward = 0.75 if abs(target_angle) < 0.70 else 0.25 if abs(target_angle) < 1.35 else 0.0
                else:
                    ball_angle = ball_rel[1]
                    centerline_bias = clamp(-1.25 * ball[1], -1.15, 1.15)
                    turn = clamp(1.75 * ball_angle + centerline_bias, -RL_TURN_SCALE, RL_TURN_SCALE)
                    forward = 2.5 if abs(ball_angle) < 0.85 else 1.3 if abs(ball_angle) < 1.15 else 0.7
            else:
                lateral_offset = 0.0
                if abs(ball[1]) > 0.45:
                    lateral_offset = -0.36 if ball[1] > 0.0 else 0.36
                staging = clip_to_field(
                    (
                        ball[0] - 0.56 * goal_unit_x,
                        ball[1] - 0.30 * goal_unit_y + lateral_offset,
                    ),
                    padding=0.20,
                )
                target_dist, target_angle = relative_polar(pose, staging)
                turn = clamp(1.85 * target_angle, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = (
                    2.4
                    if abs(target_angle) < 0.45
                    else 1.8
                    if abs(target_angle) < 1.20
                    else 1.15
                    if abs(target_angle) < 1.85
                    else 0.45
                    if abs(target_angle) < 2.35
                    else 0.0
                )
                if target_dist < 0.16:
                    forward = 0.35
            a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
            a_turn = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
            return np.array([a_forward, a_turn], dtype=np.float32)

        if self.rl_submode == "recenter" or (
            self._is_side_channel(ball) and not self._side_channel_resolved(ball)
        ):
            return self._recenter_action(pose, ball, ball_rel, ball_visible=ball_visible)

        if finish_zone:
            return self._goal_finish_action(pose, ball, ball_rel, ball_visible=ball_visible)

        if self.rl_submode == "orbit":
            if robot_ball_radius > 1.05:
                staging = clip_to_field((ball[0] - 0.60 * goal_unit_x, ball[1] - 0.60 * goal_unit_y), padding=0.25)
                target_dist, target_angle = relative_polar(pose, staging)
                turn = clamp(2.7 * target_angle, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 3.6 if abs(target_angle) < 0.45 else 1.0 if abs(target_angle) < 1.0 else 0.0
                if target_dist < 0.20:
                    forward = 0.8
            else:
                radial_x = robot_from_ball_x / robot_ball_radius
                radial_y = robot_from_ball_y / robot_ball_radius
                tangent_x = -radial_y
                tangent_y = radial_x
                orbit_sign = 1.0 if orbit_error >= 0.0 else -1.0
                target_radius = 0.76
                radial_gain = clamp(target_radius - robot_ball_radius, -0.8, 0.8)
                tangent_gain = clamp(abs(orbit_error) / math.pi, 0.25, 1.0)
                vx = 1.15 * tangent_gain * orbit_sign * tangent_x + 1.25 * radial_gain * radial_x
                vy = 1.15 * tangent_gain * orbit_sign * tangent_y + 1.25 * radial_gain * radial_y
                if abs(orbit_error) < 0.45:
                    staging = (ball[0] - 0.70 * goal_unit_x, ball[1] - 0.70 * goal_unit_y)
                    vx += 0.8 * (staging[0] - pose[0])
                    vy += 0.8 * (staging[1] - pose[1])
                target = clip_to_field((pose[0] + vx, pose[1] + vy), padding=0.25)
                target_dist, target_angle = relative_polar(pose, target)
                forward = 3.2 if abs(target_angle) < 0.35 else 1.0 if abs(target_angle) < 1.0 else 0.0
                if target_dist < 0.18:
                    forward = 0.6
                turn = clamp(3.0 * target_angle, -RL_TURN_SCALE, RL_TURN_SCALE)
        elif self.rl_submode == "align":
            staging = clip_to_field((ball[0] - 0.66 * goal_unit_x, ball[1] - 0.66 * goal_unit_y), padding=0.25)
            staging_dist, staging_angle = relative_polar(pose, staging)
            if staging_dist > 0.18:
                turn = clamp(3.0 * staging_angle, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 2.8 if abs(staging_angle) < 0.38 else 0.8 if abs(staging_angle) < 1.0 else 0.0
            elif abs(goal_heading_error) < (0.72 if near_goal_side else 0.82):
                turn = clamp(1.4 * ball_rel[1] + 0.9 * goal_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 2.2 if ball_rel[0] > 0.36 else 1.0
            else:
                turn = clamp(3.0 * goal_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 0.0
            if ball_rel[0] < 0.42 and abs(ball_rel[1]) < 0.85 and abs(goal_heading_error) < 0.65:
                turn = clamp(1.8 * ball_rel[1] + 0.8 * goal_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 2.4
        else:
            ball_angle = ball_rel[1]
            ball_to_goal = math.hypot(goal_dx, goal_dy)
            finish_strip = ball[0] > 4.1 and abs(ball[1]) <= GOAL_HALF_WIDTH + 0.12
            if finish_strip:
                target_y = clamp(0.20 * ball[1], -0.16, 0.16)
                finish_heading = math.atan2(target_y - ball[1], 4.9 - ball[0])
                finish_heading_error = wrap_to_pi(finish_heading - pose[2])
                turn = clamp(1.1 * ball_angle + 2.4 * finish_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 5.3 if abs(finish_heading_error) < 1.2 else 2.8
            elif ball_rel[0] > 0.55:
                turn = clamp(2.1 * ball_angle + 0.9 * goal_heading_error + 0.75 * field_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 3.4 if abs(ball_angle) < 0.65 else 1.0
            elif near_goal_side:
                turn = clamp(0.8 * ball_angle + 2.0 * goal_heading_error + 0.75 * field_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 4.4 if abs(goal_heading_error) < 0.65 else 2.9
            else:
                turn = clamp(1.7 * ball_angle + 0.9 * goal_heading_error + 1.15 * field_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 5.2 if abs(ball_angle) < 0.70 and abs(field_heading_error) < 0.90 else 3.0
            if abs(goal_heading_error) > 1.45 and self.rl_push_steps > 20 and ball_to_goal > 1.4:
                forward = min(forward, 0.8)

        a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
        a_turn = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
        return np.array([a_forward, a_turn], dtype=np.float32)

    def _recenter_action(self, pose, ball, ball_rel, ball_visible=True):
        side_sign = 1.0 if ball[1] > 0.0 else -1.0
        target_dx = 4.80 - ball[0]
        target_dy = -ball[1]
        target_norm = max(1e-6, math.hypot(target_dx, target_dy))
        target_unit_x = target_dx / target_norm
        target_unit_y = target_dy / target_norm
        desired_heading = math.atan2(target_unit_y, target_unit_x)
        desired_heading_error = wrap_to_pi(desired_heading - pose[2])
        staging = clip_to_field(
            (ball[0] - 0.58 * target_unit_x, ball[1] - 0.58 * target_unit_y),
            padding=0.15,
        )
        staging_dist, staging_angle = relative_polar(pose, staging)
        centerline_bias = clamp(-1.45 * ball[1], -1.2, 1.2)

        robot_from_ball_x = pose[0] - ball[0]
        robot_from_ball_y = pose[1] - ball[1]
        behind_score = robot_from_ball_x * (-target_unit_x) + robot_from_ball_y * (-target_unit_y)
        lateral_error = abs(robot_from_ball_x * (-target_unit_y) + robot_from_ball_y * target_unit_x)
        robot_behind = behind_score > 0.24 and lateral_error < 0.44

        if ball_visible and ball_rel[0] < 0.95 and robot_behind:
            turn = clamp(
                1.10 * ball_rel[1] + 1.75 * desired_heading_error + 0.35 * centerline_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            forward = 2.7 if abs(desired_heading_error) < 0.80 and abs(ball_rel[1]) < 0.80 else 1.0
        elif ball_visible and ball_rel[0] < 0.45:
            turn = clamp(2.4 * staging_angle + 0.45 * centerline_bias, -RL_TURN_SCALE, RL_TURN_SCALE)
            forward = 0.65 if abs(staging_angle) < 0.75 else 0.25 if abs(staging_angle) < 1.35 else 0.0
        else:
            turn = clamp(2.9 * staging_angle + 0.55 * centerline_bias, -RL_TURN_SCALE, RL_TURN_SCALE)
            forward = 2.2 if abs(staging_angle) < 0.50 else 0.9 if abs(staging_angle) < 1.15 else 0.0
            if staging_dist < 0.18:
                forward = 0.45

        a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
        a_turn = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
        return np.array([a_forward, a_turn], dtype=np.float32)

    def _goal_finish_action(self, pose, ball, ball_rel, ball_visible=True):
        target_y = clamp(0.22 * ball[1], -0.12, 0.12)
        target = (4.95, target_y)
        target_dx = target[0] - ball[0]
        target_dy = target[1] - ball[1]
        target_norm = max(1e-6, math.hypot(target_dx, target_dy))
        goal_heading = math.atan2(target_dy, target_dx)
        goal_heading_error = wrap_to_pi(goal_heading - pose[2])
        staging = clip_to_field(
            (
                ball[0] - 0.32 * target_dx / target_norm,
                ball[1] - 0.32 * target_dy / target_norm,
            ),
            padding=0.10,
        )
        staging_dist, staging_angle = relative_polar(pose, staging)
        centerline_bias = clamp(-1.35 * ball[1], -1.1, 1.1)
        near_goal_side = ball[0] > 3.10 and abs(ball[1]) > 0.55
        finish_corridor = ball[0] > 4.15 and abs(ball[1]) <= GOAL_HALF_WIDTH - 0.08
        ahead_margin = 0.18 if finish_corridor else 0.12
        robot_ahead_of_ball = pose[0] > ball[0] + ahead_margin
        if finish_corridor and ball_visible and ball_rel[0] > 0.16 and abs(ball_rel[1]) < 0.45:
            robot_ahead_of_ball = False

        if near_goal_side and abs(ball[1]) > GOAL_HALF_WIDTH - 0.02:
            if ball_visible and ball_rel[0] < 0.95:
                turn = clamp(
                    1.2 * ball_rel[1] + 2.0 * centerline_bias,
                    -RL_TURN_SCALE,
                    RL_TURN_SCALE,
                )
                forward = 3.6 if abs(ball_rel[1]) < 0.75 else 1.4
            else:
                inward_scale = clamp((abs(ball[1]) - 0.55) / 0.45, 0.25, 1.0)
                center_target = clip_to_field(
                    (
                        ball[0] - 0.22,
                        ball[1] - 0.32 * inward_scale * math.copysign(1.0, ball[1]),
                    ),
                    padding=0.10,
                )
                center_dist, center_angle = relative_polar(pose, center_target)
                turn = clamp(
                    2.5 * center_angle + 1.2 * centerline_bias,
                    -RL_TURN_SCALE,
                    RL_TURN_SCALE,
                )
                forward = 0.9 if center_dist > 0.15 else 0.4
        elif finish_corridor and ball_visible and ball_rel[0] > 0.16 and abs(ball_rel[1]) < 0.60:
            turn = clamp(
                1.6 * ball_rel[1] + 1.4 * centerline_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            forward = 5.4 if abs(ball_rel[1]) < 0.35 else 4.2
        elif robot_ahead_of_ball and ball_rel[0] < 0.85:
            if near_goal_side or finish_corridor:
                inward_scale = clamp(abs(ball[1]) / max(GOAL_HALF_WIDTH, 1e-6), 0.35, 1.0)
                inward_offset = -0.24 * inward_scale * math.copysign(1.0, ball[1])
                reposition = clip_to_field(
                    (ball[0] - 0.24, ball[1] + inward_offset),
                    padding=0.10,
                )
                reposition_dist, reposition_angle = relative_polar(pose, reposition)
                turn = clamp(
                    2.6 * reposition_angle + 0.15 * goal_heading_error + 1.00 * centerline_bias,
                    -RL_TURN_SCALE,
                    RL_TURN_SCALE,
                )
                if ball_visible:
                    forward = 2.0 if abs(reposition_angle) < 0.55 else 1.1 if abs(reposition_angle) < 1.25 else 0.65
                else:
                    forward = 1.0 if reposition_dist > 0.18 or abs(reposition_angle) < 1.25 else 0.45
            else:
                turn = clamp(
                    2.8 * staging_angle + 0.55 * goal_heading_error + 0.45 * centerline_bias,
                    -RL_TURN_SCALE,
                    RL_TURN_SCALE,
                )
                if ball_visible:
                    forward = 1.8 if abs(staging_angle) < 0.40 else 0.9 if abs(staging_angle) < 1.00 else 0.45
                else:
                    forward = 0.6 if abs(staging_angle) < 1.20 else 0.3
        elif finish_corridor:
            turn = clamp(
                1.3 * ball_rel[1] + 1.6 * centerline_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            forward = 5.4 if abs(ball_rel[1]) < 0.75 and ball_rel[0] < 0.75 else 3.6
        elif ball[0] > 4.35 and abs(ball[1]) <= GOAL_HALF_WIDTH - 0.05:
            turn = clamp(
                0.9 * ball_rel[1] + 1.6 * goal_heading_error + 0.7 * centerline_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            forward = 5.4 if abs(goal_heading_error) < 1.25 else 3.1
        elif staging_dist > 0.18 and ball_rel[0] > 0.28:
            turn = clamp(
                2.6 * staging_angle + 0.5 * goal_heading_error + 0.35 * centerline_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            forward = 2.6 if abs(staging_angle) < 0.45 else 0.7 if abs(staging_angle) < 1.0 else 0.0
        else:
            turn = clamp(
                1.0 * ball_rel[1] + 2.5 * goal_heading_error + 0.65 * centerline_bias,
                -RL_TURN_SCALE,
                RL_TURN_SCALE,
            )
            forward = 5.0 if abs(goal_heading_error) < 1.2 else 2.8

        a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
        a_turn = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
        return np.array([a_forward, a_turn], dtype=np.float32)

    def rl_policy(self, features):
        prior = self._geometric_rl_prior(self.last_rl_context)
        actor_action = self.actor(features)
        policy_mode = os.environ.get("RL_POLICY_MODE", "").lower()
        if policy_mode == "actor":
            return np.clip(actor_action, -1.0, 1.0)
        if policy_mode == "servo":
            return self._local_servo_prior(self.last_rl_context)
        blend = float(os.environ.get("RL_RESIDUAL_BLEND", "0.0"))
        if self.actor.source == "embedded_bootstrap":
            blend = 0.0
        blend = clamp(blend, 0.0, 0.25)
        return np.clip((1.0 - blend) * prior + blend * actor_action, -1.0, 1.0)

    def _local_servo_prior(self, context):
        pose = context.get("pose", self.fused_pose)
        ball_rel = context.get("ball_rel")
        if ball_rel is None:
            return np.array([-1.0, 0.0], dtype=np.float32)

        ball_visible = bool(context.get("ball_visible", False))
        ball_dist, ball_angle = ball_rel
        field_heading_error = wrap_to_pi(0.0 - pose[2])

        if ball_visible:
            if ball_dist > 0.58:
                turn = clamp(2.35 * ball_angle + 0.45 * field_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 3.4 if abs(ball_angle) < 0.55 else 1.2 if abs(ball_angle) < 1.05 else 0.0
            else:
                turn = clamp(1.15 * ball_angle + 1.65 * field_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
                forward = 5.1 if abs(field_heading_error) < 0.85 and abs(ball_angle) < 0.75 else 2.1
        elif ball_dist < 0.80:
            turn = clamp(1.35 * ball_angle + 1.35 * field_heading_error, -RL_TURN_SCALE, RL_TURN_SCALE)
            forward = 2.2 if abs(field_heading_error) < 1.20 else 0.8
        else:
            turn = clamp(2.4 * ball_angle, -RL_TURN_SCALE, RL_TURN_SCALE)
            forward = 0.0

        a_forward = 2.0 * clamp(forward / RL_FORWARD_SCALE, 0.0, 1.0) - 1.0
        a_turn = clamp(turn / RL_TURN_SCALE, -1.0, 1.0)
        return np.array([a_forward, a_turn], dtype=np.float32)


    def should_enter_rl(self, sensors):
        return sensors.get("ball") is not None or self._has_reliable_ball_estimate(max_age=35)

    def _update_rl_progress(self, pose, context):
        if self.localizer.confidence < 0.20:
            self.low_confidence_steps += 1
        else:
            self.low_confidence_steps = 0

        ball_position = context.get("ball_position")
        if ball_position is None:
            self.no_progress_steps = 0
            self.prev_ball_goal_dist = None
            self.prev_side_progress_metric = None
            self.side_stall_steps = 0
            return

        ball_goal_dist = distance(ball_position, self.attack_goal)
        if self.prev_ball_goal_dist is None:
            self.prev_ball_goal_dist = ball_goal_dist
            self.no_progress_steps = 0
            self.prev_side_progress_metric = None
            self.side_stall_steps = 0
            return

        progress = self.prev_ball_goal_dist - ball_goal_dist
        wall_margin = self._wall_margin(pose)
        finish_zone = ball_goal_dist < 1.8 or ball_position[0] > 3.25
        if finish_zone:
            if progress < -0.002:
                self.no_progress_steps += 1
            else:
                self.no_progress_steps = 0
        elif wall_margin < 0.35 and progress < 0.003:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        self.prev_ball_goal_dist = ball_goal_dist

        side_region = 2.8 <= ball_position[0] <= 3.7 and abs(ball_position[1]) > 0.75
        if side_region:
            side_metric = ball_position[0] - 0.45 * abs(ball_position[1])
            if self.prev_side_progress_metric is not None:
                side_progress = side_metric - self.prev_side_progress_metric
                if side_progress < 0.002:
                    self.side_stall_steps += 1
                else:
                    self.side_stall_steps = 0
            self.prev_side_progress_metric = side_metric
            if self.side_stall_steps >= 180:
                self.rl_submode = "recenter"
                self.rl_push_steps = 0
        else:
            self.prev_side_progress_metric = None
            self.side_stall_steps = 0

    def should_exit_rl(self, pose, sensors, context):
        del sensors  # kept for required interface symmetry
        self._update_rl_progress(pose, context)
        ball_visible = bool(context.get("ball_visible", False))
        ball_position = context.get("ball_position")
        finish_zone = False
        if ball_position is not None:
            finish_zone = distance(ball_position, self.attack_goal) < 1.8 or ball_position[0] > 3.25
        reliable_estimate = self._has_reliable_ball_estimate(max_age=180 if finish_zone else 140)

        if ball_visible:
            return False
        if not finish_zone and self.ball_tracker.age > 14:
            return True
        if reliable_estimate:
            if (
                self.low_confidence_steps >= (70 if finish_zone else 45)
                and self.localizer.confidence < 0.08
                and self.ball_tracker.age > (160 if finish_zone else 120)
            ):
                return True
            return False

        if self.ball_tracker.age > (180 if finish_zone else 140):
            return True
        if self.low_confidence_steps >= (55 if finish_zone else 28):
            return True
        if self.no_progress_steps >= (240 if finish_zone else 90):
            return True
        return False

    def recover_control(self, pose, context):
        ball_position = context.get("ball_position")
        ball_rel = context.get("ball_rel")
        if ball_position is None and self.ball_tracker.is_fresh(max_age=45):
            ball_position = self.ball_tracker.position
            ball_rel = self.ball_tracker.relative_ball(pose)
        elif ball_rel is None and self.last_seen_ball_rel is not None:
            ball_rel = self.last_seen_ball_rel

        if ball_position is not None and ball_rel is not None:
            if not context.get("ball_visible", False) and ball_rel[0] < 0.85:
                if ball_position[0] > 4.25:
                    turn = clamp(1.2 * ball_rel[1], -2.8, 2.8)
                    forward = 2.0 if abs(ball_rel[1]) < 0.50 else 1.0 if abs(ball_rel[1]) < 1.10 else 0.25
                    return self._wheel_command(forward, turn)
                turn = clamp(2.8 * ball_rel[1], -2.8, 2.8)
                return self._wheel_command(0.0, turn)
            action = self._geometric_rl_prior(
                {
                    "pose": pose,
                    "ball_position": ball_position,
                    "ball_rel": ball_rel,
                }
            )
            return self._action_to_control(action)

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

    def _pose_diagnostics(self, pose, actual_pose, previous_pose):
        pos_error, heading_error = pose_error(pose, actual_pose)
        pos_jump, heading_jump = pose_jump(pose, previous_pose)
        return {
            "pos_error": None if pos_error is None else float(pos_error),
            "heading_error": None if heading_error is None else float(heading_error),
            "heading_error_deg": None if heading_error is None else float(math.degrees(heading_error)),
            "pos_jump": float(pos_jump),
            "heading_jump": float(heading_jump),
            "heading_jump_deg": float(math.degrees(heading_jump)),
        }

    def _build_localization_diagnostics(self, sensors, fused_pose):
        truth = sensors.get("debug_truth", {})
        actual_pose = truth.get("robot_pose")
        actual_ball = truth.get("ball_position")
        localizer_diag = dict(getattr(self.localizer, "diagnostics", {}))
        stats = self.last_measurement_stats or self._observation_stats(sensors)
        for key, value in stats.items():
            localizer_diag.setdefault(key, value)
        localizer_diag.setdefault("landmark_bin", landmark_bin(stats))

        localizer_pose = self.localizer.pose
        odom_pose = self.odom_pose
        ball_error = None
        if actual_ball is not None and self.ball_tracker.position is not None:
            ball_error = distance(self.ball_tracker.position, actual_ball)

        diagnostics = {
            **localizer_diag,
            "pose_correction_trust": float(self.pose_correction_trust),
            "localizer_confidence": float(self.localizer.confidence),
            "no_landmark_steps": int(self.no_landmark_steps),
            "no_structure_steps": int(self.no_structure_steps),
            "low_observability_steps": int(self.low_observability_steps),
            "localization_recovery_steps": int(self.localization_recovery_steps),
            "fused": self._pose_diagnostics(fused_pose, actual_pose, self.prev_diagnostic_fused_pose),
            "localizer": self._pose_diagnostics(localizer_pose, actual_pose, self.prev_diagnostic_localizer_pose),
            "odom": self._pose_diagnostics(odom_pose, actual_pose, self.prev_diagnostic_odom_pose),
            "ball_error": None if ball_error is None else float(ball_error),
        }
        return diagnostics

    def _remember_diagnostic_poses(self, fused_pose):
        self.prev_diagnostic_fused_pose = tuple(fused_pose)
        self.prev_diagnostic_localizer_pose = tuple(self.localizer.pose)
        self.prev_diagnostic_odom_pose = tuple(self.odom_pose)

    def _log_transition(self, sensors, pose, control):
        localization = self._build_localization_diagnostics(sensors, pose)
        self.last_localization_diagnostics = localization
        truth = sensors.get("debug_truth", {})
        payload = {
            "step": self.step_count,
            "state": self.state,
            "rl_submode": self.rl_submode,
            "opening_phase": self.opening_phase,
            "opening_active": bool(self.opening_active),
            "opening_ball": None if self.opening_ball_position is None else [float(v) for v in self.opening_ball_position],
            "opening_ball_age": int(self.opening_ball_age),
            "estimated_pose": [float(v) for v in pose],
            "fused_pose": [float(v) for v in pose],
            "odom_pose": [float(v) for v in self.odom_pose],
            "localizer_pose": [float(v) for v in self.localizer.pose],
            "localizer_confidence": float(self.localizer.confidence),
            "pose_correction_trust": float(self.pose_correction_trust),
            "actual_pose": truth.get("robot_pose"),
            "estimated_ball": None if self.ball_tracker.position is None else [float(v) for v in self.ball_tracker.position],
            "actual_ball": truth.get("ball_position"),
            "localization": localization,
            "features": [float(v) for v in self.last_rl_features],
            "action": [float(v) for v in self.last_rl_action],
            "control": {
                "left_motor": float(control["left_motor"]),
                "right_motor": float(control["right_motor"]),
            },
        }
        if self.transition_logger.handle is not None:
            self.transition_logger.log(payload)
        self._remember_diagnostic_poses(pose)

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
            "fused_pose": [float(pose[0]), float(pose[1]), float(pose[2])],
            "localizer_pose": [float(v) for v in self.localizer.pose],
            "odom_pose": [float(v) for v in self.odom_pose],
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
            "localization": self.last_localization_diagnostics,
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

        odometry = sensors.get("odometry")
        self._integrate_odometry_pose(odometry)
        self.localizer.predict(odometry)
        localizer_pose = self.localizer.update(sensors)
        pose = self._fuse_pose_estimate(localizer_pose, sensors)
        self._update_localization_visibility_counters()
        opening_pose = self._opening_pose_for_control(pose)
        opening_control = self._opening_visible_ball_control(sensors, opening_pose)
        if opening_control is not None:
            self.ball_tracker.update(opening_pose, None)
            self.last_rl_action = np.zeros(2, dtype=np.float32)
            self._log_transition(sensors, opening_pose, opening_control)
            self._publish_visualizer(sensors, opening_pose)
            return opening_control
        self.opening_control_pose = tuple(pose)

        ball_pose_trust = clamp(0.08 + 0.92 * self.pose_correction_trust, 0.0, 1.0)
        self.ball_tracker.update(pose, sensors.get("ball"), pose_trust=ball_pose_trust)

        features, context = self.extract_rl_features(pose, sensors)
        self.last_rl_features = features
        self.last_rl_context = context

        if self.state == "SEARCH_BALL" and self.should_enter_rl(sensors):
            self._set_state("RL_BALL_PLAY")
        elif self.state == "RECOVER":
            recover_timeout = 240 if self._has_reliable_ball_estimate(max_age=140) else 45
            if self.should_enter_rl(sensors):
                self._set_state("RL_BALL_PLAY")
            elif self.state_age >= recover_timeout:
                self._set_state("SEARCH_BALL")

        if self._needs_localization_recovery(sensors):
            self.last_rl_action = np.zeros(2, dtype=np.float32)
            control = self._localization_recovery_control()
            self._log_transition(sensors, pose, control)
            self._publish_visualizer(sensors, pose)
            return control
        self.localization_recovery_steps = 0

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
