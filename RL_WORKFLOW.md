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
final_project/controllers/robot_one_controller/rl_policy_weights.npz
tmp/rl_traces/
tmp/rl_checkpoints/
tmp/rl_eval/
```

`rl_policy_weights.npz` is optional at runtime. If it is missing or invalid, `starter_controller.py` falls back to the embedded bootstrap actor and the geometric RL prior.

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

The latest validated run was:

```text
eval_actor_residual_blend0
successes = 48/50
success_rate = 0.96
wrong_goals = 0
```

Summary file:

```text
tmp/rl_eval/eval_actor_residual_blend0_summary.json
```

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

High-level FSM remains:

```text
SEARCH_BALL -> RL_BALL_PLAY -> RECOVER
```

Inside `RL_BALL_PLAY`, the controller uses an RL-style submode:

```text
orbit -> align -> push
```

The actor observation is now 19-dimensional:

```text
16 belief features + 3 submode one-hot features
```

Default deployed behavior is residual-safe:

```text
final_action = geometric_rl_prior
```

You can enable neural residual correction during experiments:

```bash
RL_RESIDUAL_BLEND=0.05 .venv/bin/python scripts/rl_workflow.py deepbots-eval \
  --policy actor \
  --episodes 50 \
  --name eval_residual_005
```

Keep `RL_RESIDUAL_BLEND=0.0` for the stable baseline unless a new residual actor has been validated.

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

## Behavior Cloning

Train a 19-dimensional actor from traces and activate it:

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

## SAC Training

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

If you want the safest current behavior, set:

```bash
export RL_RESIDUAL_BLEND=0.0
```

Runtime/config files generated for Webots Python setup are ignored by git:

```text
final_project/controllers/rl_training_controller/runtime.ini
final_project/controllers/rl_training_controller/config.ini
```
