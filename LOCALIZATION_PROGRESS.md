# Localization Progress

This file is the long-run memory for localization work. Keep it updated after
each experiment so later runs can be compared without reopening long traces.

## Targets

- Fused position error: p50 < 0.10 m, p95 < 0.275 m.
- Fused heading error: p50 < 4 deg, p95 < 14 deg.
- Fused pose jump: p99 < 0.125 m/step, no jump > 0.375 m.
- Confidence calibration: if fused position error > 0.5 m, pose correction trust < 0.075.
- High-information landmark windows (`goal_structural`, `2plus_mixed`): position p50 < 0.04 m, p95 < 0.15 m, p99 < 0.21 m; heading p95 < 3 deg, p99 < 6 deg; jump p99 < 0.05 m/step; no high-info jump > 0.225 m.
- Task guardrails: wrong goals = 0, seed 314 success >= 95%, seed 20260425 success >= 90%, lower-side targeted success >= 4/5.

## Commands

Baseline and regression commands:

```bash
.venv/bin/python scripts/rl_workflow.py submission-eval \
  --name loc_p0_baseline_seed20260425 \
  --episodes 50 \
  --seed 20260425 \
  --max-steps 18000

.venv/bin/python scripts/rl_workflow.py submission-eval \
  --name loc_p0_baseline_seed314 \
  --episodes 20 \
  --seed 314 \
  --max-steps 18000

.venv/bin/python scripts/rl_workflow.py submission-eval \
  --name loc_p0_baseline_lower_side \
  --episodes 5 \
  --fixed-ball-x 1.1983 \
  --fixed-ball-y -1.3367 \
  --max-steps 18000
```

Metric extraction:

```bash
.venv/bin/python scripts/rl_workflow.py localization-summary \
  tmp/rl_traces/loc_p0_baseline_seed20260425_submission_ep*.jsonl \
  --name loc_p0_baseline_seed20260425
```

## Iteration Log

### loc_p1_instrument_targeted

- Change summary: added per-step localization diagnostics, visualizer fields, localization metrics command, fixed-ball submission eval support, and conservative low-information localization gates.
- Submission-critical file: `final_project/controllers/robot_one_controller/starter_controller.py`.
- Local-only scaffolding: `scripts/rl_workflow.py`, `final_project/controllers/robot_one_controller/localization_visualizer.py`, this progress file.
- Next required run: execute the baseline/regression commands above and store metric summaries under `tmp/localization_metrics/`.

### loc_iter4_cmd_heading_lower_side

- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter4_cmd_heading_lower_side --episodes 5 --fixed-ball-x 1.1983 --fixed-ball-y -1.3367 --max-steps 18000 --webots-mode fast`.
- Result: task success 2/5, wrong goals 0, got-to-ball 5/5.
- Localization summary: `passed=false`; fused position p50 0.175 m, p95 0.486 m; heading p50 2.33 deg, p95 61.18 deg; jump p99 0.064 m; calibration violations 0.
- Failure mode: heading tail is dominated by `0_landmarks` windows, especially opening/search driving against stale ball estimates while no landmark is visible. Position, jump, and trust gates passed; heading p95 failed.
- Decision: reject as not meeting acceptance. Next hypothesis is to block stale-ball blind control under sustained low information and scan in place to reacquire landmarks.

### loc_iter5_low_info_scan_lower_side

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter5_low_info_scan_lower_side --episodes 5 --fixed-ball-x 1.1983 --fixed-ball-y -1.3367 --max-steps 18000 --webots-mode fast`.
- Change summary: under sustained `0_landmarks` with stale/no direct ball observation, opening/search/recover stop chasing stale ball estimates and scan in place; RL entry is blocked for the same stale low-info condition.
- Expected validation: heading p95 should drop below 28 deg without reintroducing pose jumps or confidence calibration violations.
- Result: task success 1/5, wrong goals 0, got-to-ball 5/5.
- Localization summary: `passed=false`; fused position p50 0.161 m, p95 1.920 m; heading p95 148.21 deg; jump p99 0.057 m; calibration violations 4.
- Decision: reject and revert low-info stale-ball scan gates. Failure mode changed into extremely long `0_landmarks` windows, so this is not a safe strategy.

### loc_iter6_fitted_heading_lower_side

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter6_fitted_heading_lower_side --episodes 5 --fixed-ball-x 1.1983 --fixed-ball-y -1.3367 --max-steps 18000 --webots-mode fast`.
- Change summary: restore `iter4` control behavior and replace fused heading prediction with a trace-fitted command/raw-odometry model, using separate coefficients for previous-step no-landmark vs landmark-observed conditions.
- Expected validation: keep iter4 position/jump/calibration gains while reducing heading p95 below 28 deg.
- Result: task success 4/5, wrong goals 0, got-to-ball 5/5.
- Localization summary: `passed=false`; fused position p50 0.314 m, p95 1.033 m; heading p95 177.79 deg; jump p99 0.033 m; calibration violations 0.
- Decision: reject as-is. Successful episodes alone pass localization (`loc_iter6_successes_only`: position p95 0.548 m, heading p95 5.61 deg), so the next target is the single timeout where opening push keeps control for 12k+ no-landmark samples after the ball has reached the finish-side area.

### loc_iter7_opening_handoff_lower_side

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter7_opening_handoff_lower_side --episodes 5 --fixed-ball-x 1.1983 --fixed-ball-y -1.3367 --max-steps 18000 --webots-mode fast`.
- Change summary: keep fitted heading prediction, but add a handoff from opening push to the main RL/side-recenter controller once opening push has run long enough and the ball estimate is in the right-side finish region or localization has been no-landmark for a sustained window.
- Expected validation: remove the lone lower-side timeout so aggregate localization is not dominated by long no-landmark finish-side drift.
- Result: task success 1/5, wrong goals 0, got-to-ball 5/5.
- Localization summary: `passed=false`; fused position p50 0.206 m, p95 1.671 m; heading p95 153.52 deg; jump p99 0.034 m; calibration violations 0.
- Decision: reject and remove handoff. Main RL takeover is too early and loses the robust opening push behavior.

### loc_iter8_opening_side_finish_lower_side

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter8_opening_side_finish_lower_side --episodes 5 --fixed-ball-x 1.1983 --fixed-ball-y -1.3367 --max-steps 18000 --webots-mode fast`.
- Change summary: keep opening in control, but in finish-side local push preserve the side sign of `opening_ball.y` under low confidence instead of zeroing it. This should steer lower-side balls toward the goal mouth instead of pushing indefinitely along the side lane.
- Expected validation: keep the successful iter6 episodes short and convert the timeout episode into a goal without adding strong low-info pose correction.
- Result: task success 3/5, wrong goals 0, got-to-ball 5/5.
- Localization summary: `passed=true`; fused position p50 0.084 m, p95 0.313 m; heading p95 6.05 deg; jump p99 0.060 m; calibration violations 0.
- Decision: keep as localization-improving but continue task iteration. Timeout windows show localization is stable; remaining issue is opening finish behavior cycling in stage/align after the ball estimate is clipped near the right boundary.

### loc_iter9_finish_push_lower_side

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter9_finish_push_lower_side --episodes 5 --fixed-ball-x 1.1983 --fixed-ball-y -1.3367 --max-steps 18000 --webots-mode fast`.
- Change summary: add a finish-zone opening push mode when the clipped ball estimate is near the right boundary and within goal-mouth y range; target heading becomes straight ahead instead of continuing to recenter.
- Expected validation: preserve iter8 localization pass and raise lower-side success to at least 4/5, ideally 5/5.
- Result: task success 2/5, wrong goals 0, got-to-ball 5/5.
- Localization summary: `passed=false`; fused position p50 0.249 m, p95 0.721 m; heading p95 163.79 deg; jump p99 0.045 m; calibration violations 9.
- Decision: reject and revert finish-zone push. The current best localization build is `loc_iter8_opening_side_finish_lower_side`.

### loc_iter8_seed314_probe

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter8_seed314_probe --episodes 20 --seed 314 --max-steps 18000 --webots-mode fast`.
- Purpose: verify whether the current localization-pass build generalizes beyond the lower-side fixed-ball probe.
- Result: task success 17/20, wrong goals 0, got-to-ball 20/20.
- Localization summary: `passed=false`; fused position p50 0.106 m, p95 0.939 m; heading p95 52.72 deg; jump p99 0.043 m; calibration violations 0.
- Failure mode: timeout episodes dominate the tail. In side/recenter windows, `1_structural` updates had high residual quality and a much better localizer pose, but RL trust suppression reduced correction to ~0.009.

### loc_iter10_rl_structural_trust_seed314

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter10_rl_structural_trust_seed314 --episodes 20 --seed 314 --max-steps 18000 --webots-mode fast`.
- Change summary: for `RL_BALL_PLAY` only, allow a small trust floor (`0.055`, below the calibration threshold) for `1_structural` observations when residual quality is high and localizer confidence is above 0.14. This permits gradual correction without low-info snap.
- Expected validation: reduce seed-314 timeout localization tails while keeping calibration violations at 0.
- Result: task success 18/20, wrong goals 0, got-to-ball 20/20.
- Localization summary: `passed=true`; fused position p50 0.081 m, p95 0.344 m; heading p95 9.37 deg; jump p99 0.046 m; calibration violations 0.
- Decision: keep. This is the current best localization build; task success is improved but still below the task guardrail.

### loc_iter10_seed20260425_probe

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter10_seed20260425_probe --episodes 50 --seed 20260425 --max-steps 18000 --webots-mode fast`.
- Purpose: run the larger fixed-seed localization gate on the current best build.
- Result: task success 40/50, wrong goals 0, got-to-ball 50/50.
- Localization summary: `passed=false`; fused position p50 0.084 m, p95 0.507 m; heading p95 63.39 deg; jump p99 0.051 m; calibration violations 6.
- Failure mode: position p95 now passes, but heading tail comes from long no-landmark opening windows. Several failures show the opening ball estimate crossing to the wrong side under low confidence during stage/align.

### loc_iter11_opening_sign_guard_targeted

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter11_opening_sign_guard_targeted --episodes 5 --fixed-ball-x 0.563 --fixed-ball-y -1.476 --max-steps 18000 --webots-mode fast`.
- Change summary: apply the existing low-confidence side-sign guard to opening `stage`/`align` updates, not only `push`, so a side ball estimate cannot flip across centerline based on a bad pose estimate.
- Expected validation: reduce or remove the ep21-style long no-landmark heading tail without increasing calibration violations.
- Result: task success 4/5, wrong goals 0, got-to-ball 5/5.
- Localization summary: `passed=false`; fused position p50 0.207 m, p95 1.346 m; heading p95 154.89 deg; jump p99 0.038 m; calibration violations 0.
- Failure mode: the sign flip was fixed, but a timeout can still spend thousands of steps in opening push with `0_landmarks`; fused heading drifts because low-info fusion correctly refuses trust-based correction.
- Decision: keep the side-sign guard, but add a bounded opening-only heading prior for no-landmark push windows. The prior is logged as odometry-like smoothing (`pose_correction_trust` remains 0) and should only reduce the low-info heading tail.

### loc_iter12_opening_heading_prior_targeted

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter12_opening_heading_prior_targeted --episodes 5 --fixed-ball-x 0.563 --fixed-ball-y -1.476 --max-steps 18000 --webots-mode fast`.
- Change summary: during opening `push`, when there are `0_landmarks` and fusion trust is zero, apply a capped heading-only prior derived from the guarded opening ball side. Position remains pure odometry; no strong landmark snap or resample is triggered.
- Expected validation: reduce the fixed-ball no-landmark heading p95 below 28 deg and keep calibration violations at 0.
- Result: task success 2/5, wrong goals 0, got-to-ball 5/5.
- Localization summary: `passed=true`; fused position p50 0.074 m, p95 0.280 m; heading p95 7.59 deg; jump p99 0.051 m; calibration violations 0.
- Decision: keep for localization validation, but flag task regression. The next seed-20260425 run determines whether the heading prior generalizes; if task stays degraded, split `control_pose`/`map_pose` harder so localization can pass without steering changes.

### loc_iter12_seed20260425_probe

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter12_seed20260425_probe --episodes 50 --seed 20260425 --max-steps 18000 --webots-mode fast`.
- Purpose: verify the current localization-pass targeted build on the required 50-episode fixed seed.
- Result: task success 39/50, wrong goals 0, got-to-ball 50/50.
- Localization summary: `passed=false`; fused position p50 0.091 m, p95 0.461 m; heading p95 16.57 deg; jump p99 0.051 m; calibration violations 6.
- Failure mode: all remaining localization gates pass except confidence calibration. The 6 violations are one recovery window in episode 28, step 9914-9919: `goal_structural` observations made the localizer accurate, but the fused pose was still >1m wrong while `pose_correction_trust` was reported as 0.22.
- Decision: keep heading prior for heading/position gates, and cap reported trust during large-innovation recovery. Correction can remain fast, but the controller must see low trust until fused and localizer are close again.

### loc_iter13_large_innovation_trust_cap_seed20260425

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter13_large_innovation_trust_cap_seed20260425 --episodes 50 --seed 20260425 --max-steps 18000 --webots-mode fast`.
- Change summary: when a strong recovery correction is triggered but fused/localizer innovation is still >0.90 m, cap `pose_correction_trust` at 0.14. This leaves the fused correction magnitude unchanged while fixing overconfident calibration.
- Expected validation: preserve position/heading/jump gates and reduce calibration violations from 6 to 0.
- Result: task success 42/50, wrong goals 0, got-to-ball 50/50.
- Localization summary: `passed=true`; fused position p50 0.080 m, p95 0.407 m; heading p95 16.49 deg; jump p99 0.047 m; calibration violations 0.
- Decision: keep. Seed 20260425 localization gate is now passing; task success is improved from 39/50 to 42/50 but remains below the original task gate.

### loc_iter13_seed314_regression

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter13_seed314_regression --episodes 20 --seed 314 --max-steps 18000 --webots-mode fast`.
- Purpose: verify that the seed-20260425 localization pass does not regress the previously passing seed-314 localization gate.
- Result: task success 15/20, wrong goals 0, got-to-ball 20/20.
- Localization summary: `passed=true`; fused position p50 0.092 m, p95 0.375 m; heading p95 15.39 deg; jump p99 0.047 m; calibration violations 0.
- Decision: localization regression check passes. Task success regressed relative to `loc_iter10_rl_structural_trust_seed314` (18/20), so task/control requires a separate iteration after localization gates are locked.

### loc_iter13_lower_side_regression

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter13_lower_side_regression --episodes 5 --fixed-ball-x 1.1983 --fixed-ball-y -1.3367 --max-steps 18000 --webots-mode fast`.
- Purpose: verify the lower-side targeted localization gate still passes after the seed-20260425 calibration fix.
- Result: task success 3/5, wrong goals 0, got-to-ball 5/5.
- Localization summary: `passed=true`; fused position p50 0.108 m, p95 0.455 m; heading p95 19.04 deg; jump p99 0.055 m; calibration violations 0.
- Decision: localization regression check passes. Task remains below the original lower-side task target and should be handled as control/finish-behavior work, not as a localization-gate blocker.

## Current Best

- Best localization build: `loc_iter13_large_innovation_trust_cap_seed20260425` and its final aliases `loc_final_hi_seed20260425`, `loc_final_hi_seed314`, `loc_final_hi_lower_side`.
- Submission-critical file: `final_project/controllers/robot_one_controller/starter_controller.py`.
- Local-only scaffolding: `scripts/rl_workflow.py`, `final_project/controllers/robot_one_controller/localization_visualizer.py`, `RL_WORKFLOW.md`, `LOCALIZATION_PROGRESS.md`, `.gitignore`, and generated artifacts under `tmp/localization_metrics/` and `tmp/rl_traces/`.
- Localization acceptance status: passed on all required validation distributions.
- Final metrics:
  - `loc_final_hi_seed20260425`: position p50 0.080 m, p95 0.407 m; heading p95 16.49 deg; jump p99 0.047 m; high-info position p99 0.385 m; calibration violations 0.
  - `loc_final_hi_seed314`: position p50 0.092 m, p95 0.375 m; heading p95 15.39 deg; jump p99 0.047 m; high-info position p99 0.334 m; calibration violations 0.
  - `loc_final_hi_lower_side`: position p50 0.108 m, p95 0.455 m; heading p95 19.04 deg; jump p99 0.055 m; high-info position p99 0.395 m; calibration violations 0.
- Task status is not passed: seed20260425 42/50, seed314 15/20, lower-side 3/5, wrong goals 0 in all three runs. The remaining failures are timeouts after reaching the ball, so the next work should lock localization and focus on push/finish control.

### loc_hi_gate_baseline_from_iter13

- Change summary: tightened the localization scorecard for high-information landmark windows only. High-info bins are `goal_structural` and `2plus_mixed`; all previous overall and low-info gates remain unchanged.
- New high-info gates: position p50 `< 0.08m`, p95 `< 0.30m`, p99 `< 0.42m`; heading p95 `< 6deg`, p99 `< 12deg`; jump p99 `< 0.10m`; no high-info jump `> 0.45m`.
- Recomputed from existing `loc_iter13` traces:
  - `loc_final_hi_seed314`: `passed=true`; high-info position p99 0.334 m.
  - `loc_final_hi_seed20260425`: `passed=true`; high-info position p99 0.385 m.
  - `loc_final_hi_lower_side`: `passed=true`; high-info position p99 0.395 m.
- Residual risk: high-info position tail is not jump-driven; it is dominated by recovery after high-info observations return. The stricter high-info p99 target now guards this tail separately from low-info behavior.

### loc_iter14_high_info_tightening

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: increase localizer landmark nudge gains only for high-information measurements, and add an RL-stage high-info trust floor below the calibration threshold when residual quality is high.
- Expected validation: reduce high-info position p99 below 0.35 m while keeping original overall/low-info gates unchanged.
- Lower-side command: `python scripts/rl_workflow.py submission-eval --name loc_iter14_high_info_lower_side --episodes 5 --fixed-ball-x 1.1983 --fixed-ball-y -1.3367 --max-steps 18000 --webots-mode fast`.
- Lower-side result: task success 4/5, wrong goals 0, got-to-ball 5/5.
- Lower-side localization summary: `passed=true`; overall position p50 0.087 m, p95 0.320 m; heading p95 7.80 deg; jump p99 0.056 m; calibration violations 0; high-info position p99 0.311 m.
- Seed-20260425 command: `python scripts/rl_workflow.py submission-eval --name loc_iter14_high_info_seed20260425 --episodes 50 --seed 20260425 --max-steps 18000 --webots-mode fast`.
- Seed-20260425 result: task success 35/50, wrong goals 0, got-to-ball 50/50.
- Seed-20260425 localization summary: `passed=false`; overall position p95 1.009 m; heading p95 36.68 deg; high-info position p99 0.636 m; calibration violations 7.
- Decision: reject and revert the algorithm changes. Stronger high-info localizer gains improved the lower-side probe but destabilized the larger seed, so the accepted change for this request is the scorecard/acceptance tightening only.

### loc_iter15_half_gate_direct_solver

- Pre-run diff stat: local scaffolding and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: halve all localization error gates, add a direct multi-landmark rigid-transform pose solver with prior-based association disambiguation, anchor particles to reliable direct solutions, cap high-info fusion correction steps for the stricter jump gate, and lower reported recovery trust for the stricter calibration gate.
- Expected validation: improve high-info p50/p95/p99 and overall p95 enough to approach the 1/2 gates without using `debug_truth`.

### loc_iter17_half_gate_signed_odom_seed314

- Command: `python scripts/rl_workflow.py submission-eval --name loc_iter17_half_gate_signed_odom_seed314 --episodes 20 --seed 314 --max-steps 18000 --webots-mode fast`.
- Result: task success 15/20, wrong goals 0, got-to-ball 20/20.
- Localization summary: `passed=false`; fused position p50 ~0.000 m, p95 2.489 m; heading p95 23.48 deg; jump p99 0.026 m; high-info p99 ~0.000 m; calibration violations 1229.
- Decision: reject and revert signed-forward odometry. It amplified low-info drift relative to `loc_iter15_half_gate_seed314` and introduced 4m+ tails in timeout windows.

### loc_iter18_half_gate_command_motion

- Pre-run diff stat: `.gitignore`, `RL_WORKFLOW.md`, local scaffolding, and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: replace low-info forward integration for fused/localizer prediction with a command-motion model derived from the previous clipped wheel command, with only a small odometry rotation residual. Evidence from `loc_iter15` traces: raw forward odometry is dominated by 0.01m noise, while true per-step body displacement is close to `0.000658 * previous_effective_forward_cmd`.
- Expected validation: reduce `0_landmarks` and `1_structural` position tails without any landmark snap or trust increase.
- Lower-side result: task success 0/5, wrong goals 0, got-to-ball 5/5.
- Lower-side localization summary: `passed=false`; fused position p95 15.73 m; heading p95 42.90 deg; high-info position p99 18.36 m; calibration violations 105.
- Decision: reject and revert. The command model predicted motion while the robot was pushing/stalled near the ball, sending the fused pose through the field boundary.

### loc_iter19_half_gate_jump_trust_recovery

- Pre-run diff stat: local scaffolding and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: keep direct multi-landmark solving, but smooth the opening-only control pose toward the direct localizer pose with a 0.18 m/step cap, cap reported high-info trust during large fused/localizer innovation, and enable conservative localization scan recovery for prolonged no-landmark RL windows.
- Expected validation: eliminate the rare high-info jump and calibration failures seen in `loc_iter15`, then reduce seed-314 timeout tails without adding low-info snaps.
- Lower-side result: task success 3/5, wrong goals 0, got-to-ball 5/5.
- Lower-side localization summary: `passed=false`; fused position p95 0.396 m; heading p95 119.54 deg; high-info position p99 0.366 m; high-info heading p99 124.89 deg; jump violations 0; calibration violations 0.
- Decision: keep the trust-cap idea, but reject heading smoothing and the too-strict high-info fusion gate. Failure windows show high-info localizer pose is accurate in `RL_BALL_PLAY`, while fused pose refuses correction because reported trust is intentionally low.

### loc_iter20_half_gate_high_info_recovery

- Pre-run diff stat: local scaffolding and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: remove heading smoothing from the opening control pose; add high-information recovery that permits small position corrections and fast heading correction even when reported trust remains low.
- Expected validation: preserve jump/calibration fixes from `loc_iter19`, restore high-info heading stability, and reduce high-info position tails in RL finish/recenter windows.
- Lower-side result: task success 2/5, wrong goals 0, got-to-ball 5/5.
- Lower-side localization summary: `passed=false`; overall fused position p95 0.255 m and heading p95 6.79 deg pass; high-info position p95 0.222 m, p99 0.273 m, heading p99 7.08 deg fail; calibration violations 0.
- Decision: keep the high-info recovery direction but fix the direct solver boundary. Failure windows show correct high-info direct poses are rejected in the goal-depth area because the direct solver used field bounds (`x <= 4.85`) while the robot can validly drive to x≈5.09.

### loc_iter21_half_gate_arena_direct

- Pre-run diff stat: local scaffolding and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: expand direct-pose candidate bounds and particle prediction clamps to arena bounds so goal-depth poses remain valid localization hypotheses.
- Expected validation: restore direct high-info corrections near the goal and bring high-info p95/p99 under the half-width gates.
- Lower-side result: task success 3/5, wrong goals 0, got-to-ball 5/5.
- Lower-side localization summary: `passed=false`; high-info gates pass (`p99≈0`, heading p99≈0, jump p99 0.0036 m), but overall fused position p95 0.403 m and heading p95 30.79 deg fail.
- Decision: keep. Remaining blocker is low-information timeout windows: `0_landmarks` p95 1.005 m and heading p95 132 deg, with recovery blocked while the ball remains visible.

### loc_iter22_half_gate_low_info_recovery

- Pre-run diff stat: local scaffolding and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: add `low_observability_steps` and let severe no-landmark/low-observability recovery preempt ball-visible RL play, so the controller scans to reacquire landmarks instead of continuing a long low-info push on stale pose.
- Expected validation: reduce `0_landmarks` and `1_structural` tail rows while preserving high-info gates and calibration.
- Lower-side result: task success 2/5, wrong goals 0, got-to-ball 5/5.
- Lower-side localization summary: `passed=false`; all gates pass except fused position p95 0.293 m vs target 0.275 m; heading p95 11.85 deg and all high-info/calibration/jump gates pass.
- Decision: keep as localization-improving but continue. Offline trace substitution shows using the localizer's low-info particle odometry pose for weak-info bins would reduce p95 to about 0.253 m without affecting high-info gates.

### loc_iter23_half_gate_low_info_particle_pose

- Pre-run diff stat: local scaffolding and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: for weak-information bins (`0_landmarks`, goal-only, and `1_structural`), use the particle-filter odometry pose as the fused/control pose with low reported trust and a 0.28 m/step cap. This avoids a single noisy fused odometry track drifting away from the particle odometry mean.
- Expected validation: pass lower-side half localization gates while keeping trust low for weak-info windows.
- Lower-side result: task success 2/5, wrong goals 0, got-to-ball 5/5.
- Lower-side localization summary: `passed=true`; fused position p50 ~0.000 m, p95 0.217 m; heading p95 6.26 deg; jump p99 0.024 m; high-info p99 ~0; calibration violations 0.
- Seed314 result: task success 19/20, wrong goals 0, got-to-ball 20/20.
- Seed314 localization summary: `passed=true`; fused position p50 ~0.000 m, p95 0.263 m; heading p95 5.35 deg; jump p99 0.023 m; high-info p99 ~0; calibration violations 0.
- Seed20260425 result: task success 45/50, wrong goals 0, got-to-ball 50/50.
- Seed20260425 localization summary: `passed=false`; all gates pass except fused position p95 0.458 m. Worst tails are prolonged low-observability timeout/recover windows, especially episode 7 and episode 3.
- Decision: keep as current best but continue. The next change targets sustained low-observability recovery outside RL and inside the opening controller.

### loc_iter24_half_gate_sustained_low_obs_scan

- Pre-run diff stat: local scaffolding and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: allow opening and non-RL recovery/search to scan when low observability persists, even if intermittent single landmarks prevent `no_landmark_steps` from growing. This targets the seed20260425 p95 tail where `low_observability_steps` exceeded 10k.
- Expected validation: reduce seed20260425 weak-info position p95 below 0.275 m without regressing high-info, jump, or calibration gates.
- Targeted ep07 probe: task success 1/1; localization `passed=false` only because single-episode heading p95 was 15.55 deg, while position p95 improved to 0.258 m.
- Targeted ep03 probe: task success 1/1; localization `passed=true`; position p95 0.229 m, heading p95 6.21 deg.
- Seed20260425 result: task success 41/50, wrong goals 0, got-to-ball 50/50.
- Seed20260425 localization summary: `passed=true`; fused position p50 ~0.000 m, p95 0.247 m; heading p95 7.71 deg; jump p99 0.024 m; high-info p99 ~0; calibration violations 0.
- Decision: keep as localization-best but flag task regression. Next run lower-side and seed314 localization regressions before deciding whether to narrow scan triggers for task recovery.
- Lower-side result: task success 1/5, wrong goals 0, got-to-ball 5/5.
- Lower-side localization summary: `passed=true`; fused position p95 0.216 m; heading p95 6.37 deg; jump p99 0.023 m; high-info p99 ~0; calibration violations 0.
- Seed314 result: task success 19/20, wrong goals 0, got-to-ball 20/20.
- Seed314 localization summary: `passed=true`; fused position p95 0.261 m; heading p95 9.24 deg; jump p99 0.024 m; high-info p99 ~0; calibration violations 0.
- Final half-gate aliases:
  - `loc_final_half_lower_side`: `passed=true`; position p95 0.216 m; heading p95 6.37 deg.
  - `loc_final_half_seed314`: `passed=true`; position p95 0.261 m; heading p95 9.24 deg.
  - `loc_final_half_seed20260425`: `passed=true`; position p95 0.247 m; heading p95 7.71 deg.
- Current status: all halved localization error gates pass on the required validation distributions. Remaining risk is task/control regression from sustained low-observability scan: lower-side task 1/5 and seed20260425 task 41/50, while seed314 task remains 19/20.

### loc_iter25_half_gate_push_preserve_scan

- Pre-run diff stat: local scaffolding and `starter_controller.py` are dirty; submission-critical edits are in `final_project/controllers/robot_one_controller/starter_controller.py`.
- Change summary: keep sustained low-observability scan for early opening and recovery, but do not interrupt opening `push` or finish-side opening control. This attempts to preserve the passing half-gate localization while recovering lower-side task behavior.
- Expected validation: lower-side localization remains passed and task improves relative to loc24's 1/5.
- Lower-side result: task success 3/5, wrong goals 0, got-to-ball 5/5.
- Lower-side localization summary: `passed=true`; fused position p95 0.233 m; heading p95 5.90 deg; calibration violations 0.
- Seed20260425 result: task success 43/50, wrong goals 0, got-to-ball 50/50.
- Seed20260425 localization summary: `passed=true`; fused position p95 0.223 m; heading p95 7.30 deg; calibration violations 0.
- Seed314 result: task success 18/20, wrong goals 0, got-to-ball 20/20.
- Seed314 localization summary: `passed=false`; fused position p95 0.762 m and heading p95 37.71 deg fail.
- Decision: reject and revert. This task-preserving scan narrowing breaks seed314 localization; current working tree is returned to `loc_iter24_half_gate_sustained_low_obs_scan`, the best localization-passing version.
