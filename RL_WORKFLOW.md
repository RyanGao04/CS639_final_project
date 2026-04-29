# RL Workflow

This document describes the local RL workflow for the CS639 soccer project.

The final Gradescope-compatible controller is still:

```text
final_project/controllers/robot_one_controller/starter_controller.py
```

The deepbots world, training controller, traces, checkpoints, and evaluation files are local development scaffolding. They are useful for iteration, but the assignment submission should not depend on Webots training harness files.

## Current Recommended Setup

Use the local virtual environment:

```bash
.venv/bin/python scripts/rl_workflow.py status
```

Important runtime files:

```text
final_project/controllers/robot_one_controller/starter_controller.py
final_project/controllers/robot_one_controller/e2e_rl_policy_weights.npz
tmp/rl_traces/
tmp/rl_checkpoints/
tmp/rl_eval/
```

`e2e_rl_policy_weights.npz` is optional at runtime. The default submission path still uses the existing heuristic controller unless `RL_POLICY_MODE=e2e` is set. If E2E mode is enabled and the weights file is missing or invalid, the E2E actor is a zero-action placeholder and should not be used for scoring runs.

## Tournament-Aware E2E RL Path

The new E2E policy path is:

```text
noisy relative sensors -> localizer / ball tracker / belief encoder -> one RL policy -> wheel action
```

Enable it explicitly:

```bash
RL_POLICY_MODE=e2e .venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy actor \
  --episodes 20 \
  --name e2e_probe \
  --trace
```

The E2E actor observation is a fixed 35-dimensional belief vector. It removes the old `orbit/align/push` mode one-hot features and adds opponent-aware tournament features. Training/evaluation harnesses now build observations by generating the same relative polar sensor dictionary used by the assignment wrapper, then calling `StudentController.observe_e2e_features()`.

Deepbots phase 3 samples the assignment-style start pose and, by default, mirrors half of episodes to represent the tournament robot starting on the opposite side. Disable mirrored tournament starts with:

```bash
RL_DEEPBOTS_TOURNAMENT_MIRROR=0 ...
```

## Recommended Validation Command

Run this before trusting a policy:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy actor \
  --episodes 50 \
  --name eval_runtime \
  --max-episode-steps 6000 \
  --trace
```

The current target is:

```text
correct-goal success rate > 95%
wrong goals = 0
```

There is not yet a validated SERL-trained E2E policy. Treat any newly exported
`e2e_rl_policy_weights.npz` as experimental until it passes the 50-episode
evaluation gate above.

## Localization Robustness Workflow

Run fixed-seed submission evaluations before trusting localization changes:

```bash
.venv/bin/python scripts/rl_workflow.py submission-eval \
  --name loc_final_seed20260425 \
  --episodes 50 \
  --seed 20260425 \
  --max-steps 18000

.venv/bin/python scripts/rl_workflow.py submission-eval \
  --name loc_final_seed314 \
  --episodes 20 \
  --seed 314 \
  --max-steps 18000

.venv/bin/python scripts/rl_workflow.py submission-eval \
  --name loc_final_lower_side \
  --episodes 5 \
  --fixed-ball-x 1.1983 \
  --fixed-ball-y -1.3367 \
  --max-steps 18000
```

After any traced run, compute the localization scorecard:

```bash
.venv/bin/python scripts/rl_workflow.py localization-summary \
  tmp/rl_traces/loc_final_seed20260425_submission_ep*.jsonl \
  --name loc_final_seed20260425
```

Metrics are written under `tmp/localization_metrics/`, and the running decision log is `LOCALIZATION_PROGRESS.md`.

Current localization gates:

- Overall fused position error: p50 `< 0.20m`, p95 `< 0.55m`.
- Overall fused heading error: p50 `< 8deg`, p95 `< 28deg`.
- Overall fused pose jump: p99 `< 0.25m/step`, no jump `> 0.75m`.
- Confidence calibration: if fused position error `> 1.0m`, `pose_correction_trust < 0.15`.
- High-information landmark windows only (`goal_structural`, `2plus_mixed`): position p50 `< 0.08m`, p95 `< 0.30m`, p99 `< 0.42m`; heading p95 `< 6deg`, p99 `< 12deg`; jump p99 `< 0.10m/step`; no high-info jump `> 0.45m`.

## Controller Design

Default submission behavior still uses the heuristic FSM:

```text
SEARCH_BALL -> RL_BALL_PLAY -> RECOVER
```

When `RL_POLICY_MODE=e2e` is enabled, the FSM action logic is bypassed:

```text
noisy sensors -> belief features -> E2E actor -> wheel command
```

The E2E actor observation is 35-dimensional:

```text
belief pose/ball/goal geometry + localization confidence + opponent features
```

Default deployed behavior remains heuristic-safe:

```text
RL_POLICY_MODE unset -> geometric/FSM controller
```

Enable the full-game E2E actor only after training a validated runtime weights file:

```bash
RL_POLICY_MODE=e2e .venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy actor \
  --episodes 50 \
  --name eval_e2e_runtime
```

If `RL_POLICY_MODE=e2e` is set without valid `e2e_rl_policy_weights.npz`, the actor returns zero actions.

## Install Dependencies

If the virtual environment is missing packages:

```bash
.venv/bin/python scripts/rl_workflow.py install-rl-deps
```

The workflow automatically launches Webots and injects the needed Python paths for deepbots/SB3 runs.

## Evaluation

Evaluate the runtime actor:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy actor \
  --episodes 50 \
  --name eval_runtime \
  --trace
```

Evaluate the embedded fallback actor:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy bootstrap \
  --episodes 50 \
  --name eval_bootstrap \
  --trace
```

Evaluate the ground-truth expert/prior used for data generation:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy expert \
  --episodes 50 \
  --name eval_expert \
  --trace
```

Evaluate an SB3 SAC model:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy sac \
  --model tmp/rl_checkpoints/sac_v1_sac.zip \
  --episodes 50 \
  --name eval_sac_v1 \
  --trace
```

## Generate Expert Data

The preferred demo source is now manual keyboard teleoperation, because the
current heuristic is not strong enough to be a reliable expert.

Collect keyboard demos through the deepbots harness:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-teleop manual_v1 \
  --phase 3 \
  --webots-mode realtime
```

Convenience wrapper:

```bash
.venv/bin/python scripts/collect_teleop_demos.py manual_v1
```

Controls:

```text
W/Up: forward
S/Down: reverse
A/Left: turn left
D/Right: turn right
Space: stop
R: reset episode
Q: quit
```

Teleop traces are written to `tmp/rl_traces/*_teleop.jsonl`. Each transition
uses the same 35-dimensional E2E belief vector as runtime evaluation, generated
through the noisy sensor -> localizer -> ball tracker path.

Summarize manual traces:

```bash
.venv/bin/python scripts/rl_workflow.py trace-summary \
  tmp/rl_traces/manual_v1_teleop.jsonl
```

## SERL E2E Training

Use `~/serl` as the training framework, run online RL inside Webots, and export
a runtime NumPy actor:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-serl-train \
  'tmp/rl_traces/*teleop*.jsonl' \
  --serl-root ~/serl \
  --python /path/to/serl/env/bin/python \
  --timesteps 50000 \
  --batch-size 256 \
  --utd-ratio 8
```

Convenience wrapper:

```bash
.venv/bin/python scripts/train_serl_from_teleop.py \
  'tmp/rl_traces/*teleop*.jsonl' \
  --serl-python /path/to/serl/env/bin/python
```

This wrapper launches online SERL training in Webots by default. Teleop demos
go into a separate demo buffer, online rollouts go into the online replay
buffer, and each SERL/RLPD update samples a mixed batch from both buffers.

The online reward is intentionally simple: small time penalty, robot-to-ball
progress, ball-to-correct-goal progress, and terminal correct/wrong goal
rewards. It does not use a learned reward classifier.

To render Webots during the online training process:

```bash
.venv/bin/python scripts/train_serl_from_teleop.py \
  'tmp/rl_traces/*teleop*.jsonl' \
  --serl-python /path/to/serl/env/bin/python \
  --render
```

To visually inspect the exported policy after training:

```bash
.venv/bin/python scripts/train_serl_from_teleop.py \
  'tmp/rl_traces/*teleop*.jsonl' \
  --serl-python /path/to/serl/env/bin/python \
  --render-after
```

The old demo-only SERL update is still available for quick pretraining checks,
but it is not the full online RL loop:

```bash
.venv/bin/python scripts/train_serl_from_teleop.py \
  'tmp/rl_traces/*teleop*.jsonl' \
  --serl-python /path/to/serl/env/bin/python \
  --offline-pretrain-only
```

This runs a SERL SAC/RLPD-style update with demo replay, high UTD updates, and
an ensemble critic, then writes:

```text
final_project/controllers/robot_one_controller/e2e_rl_policy_weights.npz
```

If the local `.venv` does not have JAX/SERL installed, pass the Python
interpreter from the SERL environment with `--python` or `SERL_PYTHON`. The
workflow wrapper prints the underlying command with:

```bash
.venv/bin/python scripts/rl_workflow.py serl-train --dry-run
```

After training, evaluate with E2E mode enabled:

```bash
RL_POLICY_MODE=e2e .venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy actor \
  --episodes 50 \
  --name eval_serl_e2e \
  --trace
```

## Legacy Expert Data

The following commands are still available for comparison, but should not be
treated as the primary demo source until the heuristic is stronger.

Record expert/prior trajectories:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-record \
  expert_v1 \
  --policy expert \
  --batch \
  --webots-mode fast
```

Filter only correct-goal successful episodes:

```bash
.venv/bin/python scripts/rl_workflow.py filter-success \
  tmp/rl_traces/expert_v1_deepbots.jsonl \
  --name expert_v1_success
```

Summarize traces:

```bash
.venv/bin/python scripts/rl_workflow.py trace-summary \
  tmp/rl_traces/expert_v1_deepbots.jsonl
```

## Legacy Behavior Cloning

Train a 35-dimensional E2E actor from traces and activate it:

```bash
.venv/bin/python scripts/rl_workflow.py train-activate \
  tmp/rl_traces/expert_v1_success_success.jsonl \
  --name bc_expert_v1 \
  --epochs 120 \
  --batch-size 512 \
  --lr 2e-4 \
  --weighting uniform
```

Use `--weighting uniform` when you want the actor to imitate all phases equally. Use `--weighting advantage` when you want to emphasize high-return parts of a trajectory, usually near successful pushes.

Activate an existing checkpoint:

```bash
.venv/bin/python scripts/rl_workflow.py activate \
  tmp/rl_checkpoints/bc_expert_v1.pt
```

Deactivate runtime weights and fall back to embedded weights:

```bash
.venv/bin/python scripts/rl_workflow.py deactivate
```

## Legacy SB3 SAC Training

SAC training runs inside Webots through:

```text
final_project/worlds/soccer_solo_deepbots.wbt
final_project/controllers/rl_training_controller/rl_training_controller.py
```

Basic SAC run:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-train \
  --name sac_v1 \
  --timesteps 50000 \
  --activate
```

Recommended SAC run with warm start and expert replay prefill:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-train \
  --name sac_from_expert \
  --warm-start runtime \
  --timesteps 80000 \
  --lr 1e-4 \
  --batch-size 512 \
  --sac-log-std-init -3.0 \
  --ent-coef 0.005 \
  --gradient-steps 2 \
  --demo-traces tmp/rl_traces/expert_v1_success_success.jsonl \
  --demo-limit 20000
```

Only add `--activate` after the SAC model has been evaluated and is better than the current runtime actor.

Export an SB3 SAC model to runtime `.npz`:

```bash
.venv/bin/python scripts/rl_workflow.py activate-sb3 \
  tmp/rl_checkpoints/sac_from_expert_sac.zip
```

## Webots Launch Options

Default training/evaluation uses fast batch mode:

```text
--batch --webots-mode fast
```

Show the Webots GUI:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy actor \
  --episodes 5 \
  --name eval_gui \
  --no-batch \
  --webots-mode realtime
```

Print the launch command without running Webots:

```bash
.venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy actor \
  --dry-run
```

## Legacy Trace Recording

The older `record` command launches the normal assignment world and records traces through `starter_controller.py`:

```bash
.venv/bin/python scripts/rl_workflow.py record run_001
```

Use this mostly for debugging the submission controller. For online RL training and automatic reset/evaluation, prefer `deepbots-record`, `deepbots-train`, and `deepbots-eval`.

## Troubleshooting

Check current workflow state:

```bash
.venv/bin/python scripts/rl_workflow.py status
```

If Webots cannot find the right Python packages, rerun commands through `.venv/bin/python`, not plain `python`.

If a training run writes a bad runtime actor, restore a known checkpoint:

```bash
.venv/bin/python scripts/rl_workflow.py activate \
  tmp/rl_checkpoints/bc_expert_v6_mode_uniform.pt
```

If you want the safest current behavior, leave E2E mode disabled:

```bash
unset RL_POLICY_MODE
```

Runtime/config files generated for Webots Python setup are ignored by git:

```text
final_project/controllers/rl_training_controller/runtime.ini
final_project/controllers/rl_training_controller/config.ini
```
