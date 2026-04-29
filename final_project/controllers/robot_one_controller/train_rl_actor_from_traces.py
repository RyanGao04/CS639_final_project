"""Train an E2E belief-feature 64->64->2 tanh actor from RL trace JSONL files."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from starter_controller import E2E_INPUT_DIM


FIELD_X_HALF = 4.5
GOAL_HALF_WIDTH = 0.8
RIGHT_GOAL = np.array([4.5, 0.0], dtype=np.float32)
LEFT_GOAL = np.array([-4.5, 0.0], dtype=np.float32)
FEATURE_DIM = E2E_INPUT_DIM


def _as_np_xy(value):
    if value is None:
        return None
    return np.asarray(value[:2], dtype=np.float32)


def _goal_for_ball(ball_xy):
    if ball_xy is None:
        return RIGHT_GOAL
    return RIGHT_GOAL if ball_xy[0] >= 0.0 else LEFT_GOAL


def _angle_from_sin_cos(sin_value, cos_value):
    return math.atan2(float(sin_value), float(cos_value))


def _compute_step_reward(prev_row, row):
    prev_ball = _as_np_xy(prev_row.get("actual_ball"))
    if prev_ball is None:
        prev_ball = _as_np_xy(prev_row.get("estimated_ball"))

    ball = _as_np_xy(row.get("actual_ball"))
    if ball is None:
        ball = _as_np_xy(row.get("estimated_ball"))

    prev_pose = _as_np_xy(prev_row.get("actual_pose"))
    if prev_pose is None:
        prev_pose = _as_np_xy(prev_row.get("estimated_pose"))

    pose = _as_np_xy(row.get("actual_pose"))
    if pose is None:
        pose = _as_np_xy(row.get("estimated_pose"))

    features = np.asarray(row["features"], dtype=np.float32)
    action = np.asarray(row["action"], dtype=np.float32)

    reward = -0.01

    if prev_ball is not None and ball is not None and prev_pose is not None and pose is not None:
        goal = _goal_for_ball(ball)
        prev_robot_to_ball = float(np.linalg.norm(prev_pose - prev_ball))
        robot_to_ball = float(np.linalg.norm(pose - ball))
        prev_ball_to_goal = float(np.linalg.norm(prev_ball - goal))
        ball_to_goal = float(np.linalg.norm(ball - goal))

        reward += 1.2 * (prev_robot_to_ball - robot_to_ball)
        reward += 2.0 * (prev_ball_to_goal - ball_to_goal)

        if ball[0] > FIELD_X_HALF and abs(ball[1]) < GOAL_HALF_WIDTH:
            reward += 12.0
        elif ball[0] < -FIELD_X_HALF and abs(ball[1]) < GOAL_HALF_WIDTH:
            reward -= 8.0
    else:
        reward -= 0.05

    ball_visible = float(features[0])
    ball_angle = _angle_from_sin_cos(features[3], features[4])
    goal_angle = _angle_from_sin_cos(features[12], features[13])
    behind_score = float(features[19])
    lateral_error = abs(float(features[20]))
    wall_margin_norm = float(features[28])

    reward += 0.5 * math.exp(-4.0 * abs(ball_angle))
    reward += 0.25 * math.exp(-2.0 * abs(goal_angle))
    reward += 0.25 * max(0.0, behind_score)
    reward -= 0.15 * lateral_error

    if ball_visible < 0.5:
        reward -= 0.2
    if wall_margin_norm < 0.2:
        reward -= 0.3
    if abs(float(action[1])) > 0.75:
        reward -= 0.05

    return reward


def _load_trace_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("state") != "RL_BALL_PLAY":
                continue
            if "features" not in row or "action" not in row:
                continue
            if len(row["features"]) != FEATURE_DIM:
                continue
            rows.append(row)
    return rows


def _discounted_returns(rewards, gamma):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        returns[idx] = running
    return returns


def load_dataset(paths, gamma):
    features = []
    actions = []
    weights = []

    for path in paths:
        rows = _load_trace_file(path)
        if len(rows) < 2:
            continue

        rewards = [0.0]
        for idx in range(1, len(rows)):
            rewards.append(_compute_step_reward(rows[idx - 1], rows[idx]))
        returns = _discounted_returns(rewards, gamma)

        baseline = float(np.median(returns))
        advantages = np.maximum(returns - baseline, 0.0)
        sample_weights = 0.25 + advantages

        for row, sample_weight in zip(rows, sample_weights):
            features.append(np.asarray(row["features"], dtype=np.float32))
            actions.append(np.asarray(row["action"], dtype=np.float32))
            weights.append(float(sample_weight))

    if not features:
        raise ValueError("No usable RL_BALL_PLAY transitions found in provided traces.")

    features = np.stack(features)
    actions = np.stack(actions)
    weights = np.asarray(weights, dtype=np.float32)
    weights /= max(weights.mean(), 1e-6)
    weights = np.clip(weights, 0.1, 8.0)
    return features, actions, weights


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

    def export_arrays(self):
        layers = [module for module in self.net if isinstance(module, nn.Linear)]
        return {
            "w1": layers[0].weight.detach().cpu(),
            "b1": layers[0].bias.detach().cpu(),
            "w2": layers[1].weight.detach().cpu(),
            "b2": layers[1].bias.detach().cpu(),
            "w3": layers[2].weight.detach().cpu(),
            "b3": layers[2].bias.detach().cpu(),
        }


def train_actor(features, actions, weights, epochs, batch_size, lr, device):
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(actions),
        torch.from_numpy(weights),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    actor = Actor().to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

    for epoch in range(epochs):
        actor.train()
        total_loss = 0.0
        total_samples = 0

        for batch_features, batch_actions, batch_weights in loader:
            batch_features = batch_features.to(device)
            batch_actions = batch_actions.to(device)
            batch_weights = batch_weights.to(device)

            pred = actor(batch_features)
            mse = ((pred - batch_actions) ** 2).mean(dim=1)
            loss = (batch_weights * mse).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * batch_features.shape[0]
            total_samples += batch_features.shape[0]

        mean_loss = total_loss / max(total_samples, 1)
        print(f"epoch {epoch + 1:03d}: loss={mean_loss:.6f}")

    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path, help="Trace JSONL files produced via RL_TRACE_PATH.")
    parser.add_argument("--output", type=Path, default=Path("trained_rl_actor.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    features, actions, weights = load_dataset(args.trace, gamma=args.gamma)
    print(f"loaded {len(features)} transitions from {len(args.trace)} trace file(s)")

    actor = train_actor(
        features,
        actions,
        weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    checkpoint = actor.export_arrays()
    torch.save(checkpoint, args.output)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
