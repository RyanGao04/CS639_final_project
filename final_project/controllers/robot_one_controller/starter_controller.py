"""student_controller controller."""

import atexit
import json
import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import subprocess
import sys

import numpy as np


FIELD_X_HALF = 4.5
FIELD_Y_HALF = 3.0
RIGHT_GOAL = (4.5, 0.0)
LEFT_GOAL = (-4.5, 0.0)
GOAL_HALF_WIDTH = 0.8
ARENA_X_HALF = 5.1
ARENA_Y_HALF = 3.6
CORNERS = [(-4.5, 3.0), (-4.5, -3.0), (4.5, 3.0), (4.5, -3.0)]
PENALTY_CROSSES = [(3.25, 0.0), (-3.25, 0.0)]
CENTER_CIRCLE = (0.0, 0.0)
ODOM_TURN_COMMAND_SCALE = 3.2
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
        trace_path = os.environ.get("CONTROL_TRACE_PATH")
        if not trace_path:
            return
        try:
            self.handle = open(trace_path, "a", encoding="utf-8", buffering=1)
            atexit.register(self.close)
        except OSError:
            self.handle = None

    def log(self, payload):
        if self.handle is None:
            if os.environ.get("CONTROL_TRACE_VALIDATE", "1") != "0":
                try:
                    json.dumps(payload)
                except (TypeError, ValueError):
                    pass
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


class BallControlState(Enum):
    NAVIGATING_TO_BEHIND_BALL = auto()
    LINING_UP = auto()
    LINED_UP = auto()
    QUICK_LINEUP = auto()
    DRIBBLING = auto()
    SCORE = auto()
    BRING_INTO_FIELD = auto()


class BallControlFSM:
    """Finite-state ball control policy driven by the controller's pose and ball estimates."""

    MAX_SPEED = 6.67

    def __init__(self):
        self.goals = [[4.5, 0.0], [-4.5, 0.0]]
        self.posts = [[4.55, 0.8], [4.55, -0.8], [-4.55, 0.8], [-4.55, -0.8]]
        self.state = BallControlState.NAVIGATING_TO_BEHIND_BALL
        self.counter = 0
        self.counter_without_ball = 0
        self.pose = (-1.0, 0.0, 0.0)
        self.global_pos_ball = [0.0, 0.0]
        self.has_ball_estimate = False
        self.ball_angle = 0.0
        self.ball_distance = -1.0
        self.ball_distance_from_goal = 0.0
        self.distance_to_behind_ball = 0.0
        self.x_behind_ball = -10.0
        self.y_behind_ball = -10.0
        self.ball_angle_behind = 0.0
        self.target = [0.0, 0.0]
        self.turn = 0.0
        self.path_to_behind_ball = []
        self.path_from_behind_ball_to_goal = []
        self.full_path = []
        self.started_dribbling = False
        self.has_lined_up = False
        self.is_in_line = False

    def step(self, pose, ball_position, ball_rel, ball_visible):
        self.counter += 1
        self.pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        self._update_ball_data(ball_position, ball_rel, ball_visible)

        if not self.has_ball_estimate:
            left_speed, right_speed = self._search_wheel_speeds()
            self.turn = 0.0
        elif ((not self.full_path and not ball_visible) or self.counter_without_ball > 100):
            left_speed, right_speed = self._search_wheel_speeds()
            self.turn = 0.0
        elif ball_visible and not self.full_path:
            left_speed, right_speed = [-1.0, 1.0]
            self.turn = 0.0
            (
                self.path_to_behind_ball,
                self.path_from_behind_ball_to_goal,
                self.full_path,
            ) = self.generate_path()
        else:
            left_speed, right_speed = self.update()

        control = {
            "left_motor": (left_speed - self.turn) * self.MAX_SPEED,
            "right_motor": (right_speed + self.turn) * self.MAX_SPEED,
        }
        return control, self.debug_payload()

    def _search_wheel_speeds(self):
        if self.ball_is_to_right_of_robot():
            return [1.0, -1.0]
        return [-1.0, 1.0]

    def _update_ball_data(self, ball_position, ball_rel, ball_visible):
        if ball_visible:
            self.counter_without_ball = 0
        else:
            self.counter_without_ball += 1

        if ball_position is not None:
            self.has_ball_estimate = True
            self.global_pos_ball = [float(ball_position[0]), float(ball_position[1])]
            if ball_visible:
                self.global_pos_ball[1] -= 0.02

        if not self.has_ball_estimate:
            return

        if ball_rel is None:
            self.ball_distance, self.ball_angle = relative_polar(self.pose, self.global_pos_ball)
        else:
            self.ball_distance = float(ball_rel[0])
            self.ball_angle = wrap_to_pi(float(ball_rel[1]))

        goal = self.goals[0]
        self.ball_distance_from_goal = math.sqrt(
            (goal[0] - self.global_pos_ball[0]) ** 2
            + (goal[1] - self.global_pos_ball[1]) ** 2
        )
        target_goal_x = goal[0] + 0.5 if self.ball_distance_from_goal <= 0.5 else goal[0]
        global_angle_ball_to_goal = wrap_to_pi(
            math.atan2(goal[1] - self.global_pos_ball[1], target_goal_x - self.global_pos_ball[0])
        )
        behind_offset = 0.3 if ball_visible else 0.25
        self.x_behind_ball = self.global_pos_ball[0] - behind_offset * math.cos(global_angle_ball_to_goal)
        self.y_behind_ball = self.global_pos_ball[1] - behind_offset * math.sin(global_angle_ball_to_goal)
        self.ball_angle_behind = wrap_to_pi(
            math.atan2(self.y_behind_ball - self.pose[1], self.x_behind_ball - self.pose[0])
        )
        self.distance_to_behind_ball = math.sqrt(
            (self.x_behind_ball - self.pose[0]) ** 2
            + (self.y_behind_ball - self.pose[1]) ** 2
        )

    def update(self):
        (
            self.path_to_behind_ball,
            self.path_from_behind_ball_to_goal,
            self.full_path,
        ) = self.generate_path()
        self.path_to_behind_ball = self.densify_path(
            self.path_to_behind_ball,
            num_points=int(self.ball_distance * 20) + 5,
        )
        self.path_from_behind_ball_to_goal = self.densify_path(
            self.path_from_behind_ball_to_goal,
            num_points=int(self.ball_distance_from_goal * 20) + 5,
        )
        self.path_to_behind_ball = self.smooth_path(self.path_to_behind_ball)
        self.path_from_behind_ball_to_goal = self.smooth_path(self.path_from_behind_ball_to_goal)
        self.full_path = self.path_to_behind_ball + self.path_from_behind_ball_to_goal
        self._maybe_transition()

        left_speed = right_speed = 1.0
        if self.state is BallControlState.DRIBBLING:
            self.target = self.get_path_target(self.path_from_behind_ball_to_goal)
            if self.close_to_goal():
                self.turn, speed = self.do_control_dribbling([self.goals[0][0] + 0.5, self.goals[0][1]])
            else:
                self.turn, speed = self.do_control_dribbling([self.goals[0][0], self.goals[0][1]])
            left_speed = right_speed = speed
            self.has_lined_up = False
        elif self.state is BallControlState.LINED_UP:
            self.target = self.get_path_target(self.path_from_behind_ball_to_goal)
            self.turn = self.do_control(self.target)
        elif self.state is BallControlState.LINING_UP:
            self.have_lined_up_behind_ball()
            self.turn, speed = self.do_control_lined_up()
            left_speed = right_speed = speed
        elif self.state is BallControlState.QUICK_LINEUP:
            self.turn, speed = self.do_control_quick_lineup()
            self.reached_target_behind_ball()
            left_speed = right_speed = speed
        elif self.state is BallControlState.SCORE:
            self.target = self.get_path_target(self.path_from_behind_ball_to_goal)
            self.turn, speed = self.do_control_dribbling([self.goals[0][0] + 0.5, self.goals[0][1]])
            left_speed = right_speed = speed
        else:
            self.reached_target_behind_ball()
            if not self.path_to_behind_ball:
                self.turn = self.do_control([self.x_behind_ball, self.y_behind_ball])
            else:
                self.target = self.get_path_target(self.path_to_behind_ball)
                self.turn = self.do_control(self.target)
        return left_speed, right_speed

    def _maybe_transition(self):
        current = self.state
        next_state = current
        if current is BallControlState.NAVIGATING_TO_BEHIND_BALL:
            if self.behind_ball() and self.is_in_line:
                next_state = BallControlState.LINING_UP
            elif not self.ball_is_in_field():
                next_state = BallControlState.BRING_INTO_FIELD
        elif current is BallControlState.LINING_UP:
            if not self.behind_ball():
                next_state = BallControlState.NAVIGATING_TO_BEHIND_BALL
            elif self.has_lined_up:
                next_state = BallControlState.LINED_UP
            elif not self.ball_is_in_field():
                next_state = BallControlState.BRING_INTO_FIELD
        elif current is BallControlState.LINED_UP:
            if not self.behind_ball():
                next_state = BallControlState.NAVIGATING_TO_BEHIND_BALL
            elif self.dribbling() and self.is_in_line:
                next_state = BallControlState.DRIBBLING
            elif not self.ball_is_in_field():
                next_state = BallControlState.BRING_INTO_FIELD
        elif current is BallControlState.QUICK_LINEUP:
            if self.behind_ball() and self.is_in_line:
                next_state = BallControlState.LINING_UP
            elif not self.behind_ball():
                next_state = BallControlState.NAVIGATING_TO_BEHIND_BALL
            elif not self.ball_is_in_field():
                next_state = BallControlState.BRING_INTO_FIELD
        elif current is BallControlState.DRIBBLING:
            if not self.ball_is_close_enough_to_dribble():
                next_state = BallControlState.QUICK_LINEUP
            elif not self.behind_ball():
                next_state = BallControlState.NAVIGATING_TO_BEHIND_BALL
            elif self.close_to_goal() and self.is_in_line:
                next_state = BallControlState.SCORE
            elif not self.ball_is_in_field():
                next_state = BallControlState.BRING_INTO_FIELD
        elif current is BallControlState.SCORE:
            if not self.ball_is_close_enough_to_dribble():
                next_state = BallControlState.QUICK_LINEUP
            elif not self.behind_ball():
                next_state = BallControlState.NAVIGATING_TO_BEHIND_BALL
            elif not self.ball_is_in_field():
                next_state = BallControlState.BRING_INTO_FIELD
        elif current is BallControlState.BRING_INTO_FIELD:
            if self.ball_is_in_field():
                next_state = BallControlState.NAVIGATING_TO_BEHIND_BALL
        self._transition_to(next_state)

    def _transition_to(self, next_state):
        if next_state is self.state:
            return
        exit_hook = {
            BallControlState.DRIBBLING: self._on_exit_dribbling,
            BallControlState.BRING_INTO_FIELD: self._on_exit_bring_to_field,
            BallControlState.NAVIGATING_TO_BEHIND_BALL: self._on_exit_navigating,
        }.get(self.state)
        if exit_hook:
            exit_hook()
        self.state = next_state
        enter_hook = {
            BallControlState.NAVIGATING_TO_BEHIND_BALL: self._on_enter_navigating,
            BallControlState.QUICK_LINEUP: self._on_enter_quick_lineup,
            BallControlState.DRIBBLING: self._on_enter_dribbling,
            BallControlState.BRING_INTO_FIELD: self._on_enter_bring_to_field,
        }.get(self.state)
        if enter_hook:
            enter_hook()

    def _on_enter_navigating(self):
        self.started_dribbling = False
        self.has_lined_up = False
        self.is_in_line = False

    def _on_enter_quick_lineup(self):
        self.has_lined_up = False
        self.is_in_line = False

    def _on_enter_dribbling(self):
        self.started_dribbling = True

    def _on_enter_bring_to_field(self):
        self.goals[0][0] = 0.0
        self.goals[0][1] = 0.0

    def _on_exit_dribbling(self):
        pass

    def _on_exit_bring_to_field(self):
        self.goals[0][0] = 4.5
        self.goals[0][1] = 0.0

    def _on_exit_navigating(self):
        self.is_in_line = True

    def densify_path(self, path, num_points=50):
        path = np.array(path, dtype=float)
        if len(path) < 2:
            return path.tolist()
        total_dist = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        if total_dist == 0:
            return [path[0].tolist()]
        if num_points < 2:
            num_points = 2
        spacing = total_dist / (num_points - 1)
        new_path = [path[0].tolist()]
        accumulated = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            segment = p2 - p1
            seg_len = np.linalg.norm(segment)
            if seg_len == 0:
                continue
            direction = segment / seg_len
            while accumulated + seg_len >= spacing:
                remaining = spacing - accumulated
                new_point = p1 + direction * remaining
                new_path.append(new_point.tolist())
                p1 = new_point
                seg_len -= remaining
                accumulated = 0.0
            accumulated += seg_len
        if not np.allclose(new_path[-1], path[-1]):
            new_path.append(path[-1].tolist())
        return new_path

    def move_points(self, points, obstacles, influence_radius=0.3, strength=0.12):
        new_points = []
        for x, y in points:
            total_dx = 0.0
            total_dy = 0.0
            for ox, oy in obstacles:
                dx = x - ox
                dy = y - oy
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 1e-6:
                    continue
                if dist < influence_radius:
                    repulse = min(strength * (1.0 / dist - 1.0 / influence_radius), strength * 3)
                    total_dx += repulse * dx / dist
                    total_dy += repulse * dy / dist
            new_points.append([x + total_dx, y + total_dy])
        return new_points

    def smooth_path(self, path, alpha=0.2):
        smoothed = [list(point) for point in path]
        for i in range(1, len(path) - 1):
            smoothed[i][0] = (1 - alpha) * path[i][0] + alpha * (path[i - 1][0] + path[i + 1][0]) / 2
            smoothed[i][1] = (1 - alpha) * path[i][1] + alpha * (path[i - 1][1] + path[i + 1][1]) / 2
        return smoothed

    def generate_path(self):
        if self.distance_to_behind_ball > 0.1 and not self.dribbling():
            path_to_behind_ball = [
                [self.pose[0], self.pose[1]],
                [self.x_behind_ball, self.y_behind_ball],
            ]
            path_to_behind_ball = self.densify_path(path_to_behind_ball, num_points=int(self.ball_distance * 20))
            obstacles = [[float(self.global_pos_ball[0]), float(self.global_pos_ball[1])]]
            obstacles.extend([[float(pt[0]), float(pt[1])] for pt in self.posts])
            path_to_behind_ball = self.move_points(path_to_behind_ball, obstacles)
        else:
            path_to_behind_ball = [[self.pose[0], self.pose[1]]]

        if self.ball_distance_from_goal > 1:
            path_from_behind_ball_to_goal = [
                [float(self.global_pos_ball[0]), float(self.global_pos_ball[1])],
                [float(self.goals[0][0]), float(self.goals[0][1])],
                [float(self.goals[0][0] + 0.5), float(self.goals[0][1])],
            ]
        else:
            path_from_behind_ball_to_goal = [
                [float(self.global_pos_ball[0]), float(self.global_pos_ball[1])],
                [float(self.goals[0][0] + 0.5), float(self.goals[0][1])],
            ]
        path_from_behind_ball_to_goal = self.densify_path(
            path_from_behind_ball_to_goal,
            num_points=int((self.ball_distance + self.ball_distance_from_goal) * 20 + 5),
        )
        obstacles = [[float(pt[0]), float(pt[1])] for pt in self.posts]
        path_from_behind_ball_to_goal = self.move_points(path_from_behind_ball_to_goal, obstacles)
        return path_to_behind_ball, path_from_behind_ball_to_goal, path_to_behind_ball + path_from_behind_ball_to_goal[1:]

    def dribbling(self):
        if self.ball_distance < 0.15:
            return self.ball_is_close_enough_to_dribble()
        return False

    def ball_is_close_enough_to_dribble(self):
        angle_to_ball = wrap_to_pi(
            math.atan2(-self.pose[1] + self.global_pos_ball[1], -self.pose[0] + self.global_pos_ball[0])
        )
        if self.ball_distance_from_goal > 0.5:
            angle_between_post_and_goal = wrap_to_pi(
                math.atan2(self.posts[0][1] - self.global_pos_ball[1], self.posts[0][0] - self.global_pos_ball[0])
                - math.atan2(self.goals[0][1] - self.global_pos_ball[1], self.goals[0][0] - self.global_pos_ball[0])
            )
        else:
            angle_between_post_and_goal = wrap_to_pi(
                math.atan2(self.posts[0][1] - self.global_pos_ball[1], self.posts[0][0] - self.global_pos_ball[0])
                - math.atan2(self.goals[0][1] + 0.5 - self.global_pos_ball[1], self.goals[0][0] - self.global_pos_ball[0])
            )
        return abs(angle_to_ball) < (
            abs(angle_between_post_and_goal) + math.radians(10 * self.ball_distance_from_goal)
        )

    def do_control(self, target):
        angle_to_target = math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])
        angle_diff = wrap_to_pi(angle_to_target - self.pose[2])
        return max(min(angle_diff, 1), -1)

    def do_control_dribbling(self, target):
        angle_to_target = math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])
        angle_diff = wrap_to_pi(angle_to_target - self.pose[2] + self.ball_angle / 5)
        turn = max(min(angle_diff, 1), -1)
        if abs(self.ball_angle) > math.radians(20):
            turn *= 3
        return turn, 1.0

    def get_path_target(self, path):
        if len(path) < 2:
            return self.global_pos_ball
        min_dist = float("inf")
        closest_idx = 0
        for i, point in enumerate(path):
            dist = math.sqrt((point[0] - self.pose[0]) ** 2 + (point[1] - self.pose[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return path[min(closest_idx + 3, len(path) - 1)]

    def behind_ball(self):
        return self.pose[0] < self.global_pos_ball[0]

    def have_lined_up_behind_ball(self):
        if self.ball_distance_from_goal > 0.5:
            angle_ball_to_goal = math.atan2(
                self.goals[0][1] - self.global_pos_ball[1],
                self.goals[0][0] - self.global_pos_ball[0],
            )
        else:
            angle_ball_to_goal = math.atan2(
                self.goals[0][1] - self.global_pos_ball[1],
                self.goals[0][0] + 0.5 - self.global_pos_ball[0],
            )
        angle_diff = wrap_to_pi(angle_ball_to_goal - self.pose[2])
        self.has_lined_up = abs(angle_diff) < math.radians(2)
        return abs(angle_diff) < math.radians(3)

    def reached_target_behind_ball(self, tolerance=0.02):
        rx, ry = self.pose[0], self.pose[1]
        bx, by = self.global_pos_ball[0], self.global_pos_ball[1]
        if self.ball_distance_from_goal > 0.5:
            gx, gy = self.goals[0][0], self.goals[0][1]
        else:
            gx, gy = self.goals[0][0] + 0.5, self.goals[0][1]
        line_vec = np.array([gx - bx, gy - by])
        robot_vec = np.array([rx - bx, ry - by])
        line_length = np.linalg.norm(line_vec)
        if line_length < 1e-6:
            return False
        perpendicular_dist = abs(line_vec[0] * robot_vec[1] - line_vec[1] * robot_vec[0]) / line_length
        projection = np.dot(robot_vec, line_vec) / line_length
        self.is_in_line = perpendicular_dist < tolerance and projection < 0
        return self.is_in_line

    def do_control_lined_up(self):
        if self.ball_distance_from_goal > 0.5:
            angle_ball_to_goal = math.atan2(
                self.goals[0][1] - self.global_pos_ball[1],
                self.goals[0][0] - self.global_pos_ball[0],
            )
        else:
            angle_ball_to_goal = math.atan2(
                self.goals[0][1] - self.global_pos_ball[1],
                self.goals[0][0] + 0.5 - self.global_pos_ball[0],
            )
        angle_diff = wrap_to_pi(angle_ball_to_goal - self.pose[2])
        return max(min(angle_diff * 15, 1), -1), 0.0

    def do_control_quick_lineup(self):
        dx = self.x_behind_ball - self.pose[0]
        dy = self.y_behind_ball - self.pose[1]
        angle_to_behind = math.atan2(dy, dx)
        backward_heading = wrap_to_pi(self.pose[2] + math.pi)
        angle_diff = wrap_to_pi(angle_to_behind - backward_heading)
        return max(min(angle_diff * 15, 1), -1), -1.0

    def close_to_goal(self):
        return self.ball_distance_from_goal < 0.5

    def ball_is_in_field(self):
        return -4.4 < self.global_pos_ball[0] < 4.4 and -2.9 < self.global_pos_ball[1] < 2.9

    def ball_is_to_right_of_robot(self):
        return self.ball_angle < 0

    def debug_payload(self):
        return {
            "mode": "ball_control_fsm",
            "state": self.state.name,
            "target": [float(self.target[0]), float(self.target[1])],
            "staging": [float(self.x_behind_ball), float(self.y_behind_ball)],
            "goal_target": [float(self.goals[0][0]), float(self.goals[0][1])],
            "path": [[float(point[0]), float(point[1])] for point in self.full_path],
            "ball_distance": float(self.ball_distance),
            "ball_angle": float(self.ball_angle),
            "counter_without_ball": int(self.counter_without_ball),
        }


class StudentController:
    def __init__(self):
        seed_raw = os.environ.get("STARTER_NUMPY_SEED", "7")
        if seed_raw:
            np.random.seed(int(seed_raw))
        self.localizer = MultiHypothesisLocalizer()
        self.ball_tracker = BallTracker()
        self.visualizer = LiveVisualizerClient()
        self.transition_logger = TransitionLogger()
        self.ball_control = BallControlFSM()

        self.state = "BALL_CONTROL_INIT"
        self.state_age = 0
        self.step_count = 0
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
            "landmark_bin": "0_landmarks",
        }

        self.prev_forward_cmd = 0.0
        self.prev_turn_cmd = 0.0
        self.prev_diagnostic_fused_pose = None
        self.prev_diagnostic_localizer_pose = None
        self.prev_diagnostic_odom_pose = None
        self.last_localization_diagnostics = {}
        self.no_landmark_steps = 0
        self.no_structure_steps = 0
        self.low_observability_steps = 0
        self.last_control_plan = {}

    def close(self):
        self.visualizer.close()
        self.transition_logger.close()

    def _set_control_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            self.state_age = 0

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
        commanded_turn = self.prev_turn_cmd * ODOM_TURN_COMMAND_SCALE
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
        return distance(ball_position, RIGHT_GOAL) < 1.8 or ball_position[0] > 3.25

    def _is_side_channel(self, ball_position):
        if ball_position is None:
            return False
        return abs(ball_position[1]) > FIELD_Y_HALF - 0.55

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
                    pos_error > 0.75
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
            self.fused_pose = base_pose
            return self.fused_pose

        if low_info and (pos_error > 0.16 or abs(theta_error) > math.radians(10.0)):
            self.pose_correction_trust = min(self.pose_correction_trust, 0.08)
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

        return {
            **localizer_diag,
            "pose_correction_trust": float(self.pose_correction_trust),
            "localizer_confidence": float(self.localizer.confidence),
            "no_landmark_steps": int(self.no_landmark_steps),
            "no_structure_steps": int(self.no_structure_steps),
            "low_observability_steps": int(self.low_observability_steps),
            "fused": self._pose_diagnostics(fused_pose, actual_pose, self.prev_diagnostic_fused_pose),
            "localizer": self._pose_diagnostics(localizer_pose, actual_pose, self.prev_diagnostic_localizer_pose),
            "odom": self._pose_diagnostics(odom_pose, actual_pose, self.prev_diagnostic_odom_pose),
            "ball_error": None if ball_error is None else float(ball_error),
        }

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
            "control_mode": "ball_control_fsm",
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
            "motion_plan": self.last_control_plan,
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
            "motion_plan": self.last_control_plan,
            "localization": self.last_localization_diagnostics,
        }
        self.visualizer.publish(snapshot)

    def _ball_control_action(self, pose, sensors):
        ball_visible = sensors.get("ball") is not None
        if ball_visible:
            ball_rel = sensors.get("ball")
        elif self.ball_tracker.position is not None:
            ball_rel = self.ball_tracker.relative_ball(pose)
        else:
            ball_rel = None

        control, debug = self.ball_control.step(
            pose,
            self.ball_tracker.position,
            ball_rel,
            ball_visible,
        )
        self._set_control_state(f"BALL_CONTROL_{debug['state']}")
        self.last_control_plan = {
            "mode": debug["mode"],
            "control_state": debug["state"],
            "target": debug["target"],
            "staging": debug["staging"],
            "goal_target": debug["goal_target"],
            "path": debug["path"],
            "ball_distance": debug["ball_distance"],
            "ball_angle": debug["ball_angle"],
            "counter_without_ball": debug["counter_without_ball"],
        }
        return control

    def step(self, sensors):
        self.step_count += 1
        self.state_age += 1

        odometry = sensors.get("odometry")
        self._integrate_odometry_pose(odometry)
        self.localizer.predict(odometry)
        localizer_pose = self.localizer.update(sensors)
        pose = self._fuse_pose_estimate(localizer_pose, sensors)
        self._update_localization_visibility_counters()

        ball_pose_trust = clamp(0.08 + 0.92 * self.pose_correction_trust, 0.0, 1.0)
        self.ball_tracker.update(pose, sensors.get("ball"), pose_trust=ball_pose_trust)

        control = self._ball_control_action(pose, sensors)
        self._log_transition(sensors, pose, control)
        self._publish_visualizer(sensors, pose)
        return control
