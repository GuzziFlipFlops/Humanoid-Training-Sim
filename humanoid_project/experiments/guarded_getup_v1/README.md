# Guarded get-up v1 — isolated recovery training

Prepared September 3, 2026; extended after explicit authorization to start recovery training. The original design record below is retained, with implementation updates here. No production task, robot asset, motor limit, or locomotion checkpoint is changed. Consult the latest `logs/guarded_getup_v1_*/state.json` for live status; a launched supervisor or completed PPO budget is not proof of successful recovery.

## What is implemented and what is not

- `reward_core.py`: standalone batched PyTorch recovery/hold reward and success tracker.
- `test_reward_core.py`: synthetic adversarial tests, including partial resets and control-step scaling. These do not prove physical feasibility or learning convergence.
- This document: old failure audit, proposed reward weights, reset/evaluation/Isaac integration contract.
- `candidate.py`, `telemetry.py`: isolated programmatic Isaac task, real filtered ground contacts, exact box penetration and sole support-hull checks. Instantiated directly by `worker.py`, not globally registered with Gym.
- `assets/recovery.usda`: self-collision-enabled overlay of the unchanged original asset. Thirty-four direct mechanical-neighbor/gimbal pairs are excluded; cross-leg and forearm/torso collisions remain active. Simplified boxes and solver tolerances remain approximations, not hardware validation.
- `assets/pose_bank.pt` / `pose_audit.json`: 1,280 physics-settled poses, 256 each for back, front, left side, right side, and standing rehearsal. Each group has 128 training and 128 held-out poses. Accepted poses settled for 2 seconds without external support. Side starts that rolled onto their back/front were rejected. Actual robot mass is 38 kg.
- `worker.py`: fresh recovery PPO policy (288 actor inputs with IMU/encoder/action history; 101 privileged critic inputs), fixed action scale .7, gamma .9985, 25-second episodes. It does not load the walking network.
- `evaluation.py`: independent 2+5-second hold verification from raw pre-reset telemetry, preserving first-trial failures across resets. `test_evaluation.py` plus core tests cover 24 synthetic cases.
- `chain.py`: 800 supine discovery + 800 four-pose discovery + 400 refinement iteration ceilings, evaluated every 200; early advancement only on per-pose gates. No-progress or failed final gates stop the chain for diagnosis. Small train/save/reload smoke and 4096/6144/8192 benchmarks precede main training, with 512 MiB free-memory reserve and memory-only fallback.
- `monitor.py`: current block, iterations/second, next evaluation ETA, stage cap ETA, and independently measured success. No automatic running-policy handoff is implemented.

### Integration findings (September 3)

Live launch record: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_getup_v1_20260903_0804\state.json`. Small 64-env/64-iteration training and saved-checkpoint reload completed; 4096/6144/8192 scaling tests passed. The first main 8192-env supine block began September 3 at 08:08 EDT and produced finite PPO updates. Future status must be read from the live state; this is a launch record, not a recovery-success claim. The accepted runner/config/USD hashes remained unchanged. Monitor with the environment Python and `experiments/guarded_getup_v1/monitor.py`.

- GPU filtered contact views did not accept the initial static-floor filter. Each environment now has a fixed kinematic floor body and one separately ground-filtered sensor per robot body. Measured ground load must match robot weight; an all-zero force signal fails the pose audit. Self-contact forces cannot count as foot support.
- Preflight initially classified the first random policy movement as a bad reset: a side pose's forearm/hip overlap moved from 9.2 mm to 16.0 mm after that action. The 15 mm severe-penetration threshold was **not loosened**. `ResetAwarePositionAction` holds audited joint positions for one 20 ms motor-controlled tick to refresh contacts before validating the start. Subsequent excessive penetration ends that attempt as a failure instead of aborting the process as a reset error.
- Isaac may exit with process code zero during application cleanup after a Python exception. Workers print exceptions before cleanup; the supervisor requires `result.json` and a real checkpoint, not exit code alone.
- Standing feasibility: holding saved **measured** joint angles as new PD targets caused further gravity deflection and falls. A separate calibration using the original nominal targets plus ankle IMU feedback passed the unchanged 2+5-second hold and full 25-second tail in 15/16 standing trials (`logs/getup_hold_20260903_0801/evaluation.json`). One initial trial failed; none of the 15 successful holds were subsequently lost. This controller is used only by `--mode hold` to test feasibility. It supplies no assistance to recovery PPO and is not a 93.75% get-up result.
- Pose/budget/gate settings are engineering choices for this simplified robot, not guarantees of convergence. Original pre-integration prerequisites and reward reasoning follow.

## Research used, and the boundary of the comparison

[HumanUP](https://arxiv.org/abs/2502.12152) treats get-up as a complex-contact, sparse-reward problem. Its discovery/refinement split and emphasis on accurate collision geometry support separating exploration from later motion cleanup. Unlike its initially relaxed discovery constraints, this proposal keeps this robot's approved physical limits. It is not a reproduction of HumanUP or a transfer of G1 weights.

[HoST](https://arxiv.org/abs/2502.08378) uses posture/terrain curricula and motion regularization, with a multi-critic architecture, to learn standing up. This motivates testing diverse poses and measuring violent/oscillatory motion, not merely maximizing height. Our current prototype is not HoST's implementation and has no multi-critic trainer yet.

Do not copy a walking cadence, fixed stride, fixed knee trajectory, equal instantaneous foot load, or running air-time reward into recovery. Rolling or briefly lowering the torso can be necessary; the reward must permit this.

## Why the previous implementation could look successful but fail

Inspected source: `source/humanoid_tasks/curriculum/skill_mdp.py` (`reset_recovery_pose`, `sustained_getup_reward`, `getup_success_termination`) and `env_cfg.py` (`RecoveryRewardsCfg`, `RecoveryRefineRewardsCfg`).

1. Reset used a fixed 0.22 m root height with four rotated poses and random joints. This does not establish collision-free, settled poses. Ten percent of starts were upright, which must not inflate a lying-to-standing success denominator.
2. Discovery also paid continuous height/upright rewards. Refinement removed those separate terms, but the sequence still paid repeated partial-height/partial-upright shaping. Partial rising could dominate without successful recovery.
3. Completion checked the root/pelvis, not the independently articulated torso. This repeats the pelvis-versus-torso blind spot found in locomotion.
4. Non-foot contact was penalized separately but not forbidden by the completion predicate. A robot leaning on its hands could get too much credit. During the rise, however, hand/knee support is legitimate and must NOT be globally forbidden.
5. Success ended the episode after a short dwell (currently 1.25 s discovery / 2.5 s refine), hiding what happened later. Drift/foot anchoring was absent; root height was not terrain-relative.
6. Isaac's `RewardManager.compute` multiplies every weighted term by `dt`. A one-time bonus of 14 returned as a rate at dt=.02 contributes only .28 before its outer weight. Event and continuous-rate units must be explicit.
7. Current self-collision is disabled in `humanoid_cfg.py` because neighboring box colliders overlap. Rewards cannot repair physically impossible limb paths. Implicit-drive torque remains estimated, and the ankle model remains an approximation.

## Proposed task

Keep the accepted walk/run network frozen and train recovery as a separate policy. Start with a single verified supine family, then add prone and left/right sides; retain earlier pose samples. Crouched/easy starts may help discovery but carry separate labels and never count as lying recoveries. Add perturbed limbs/heading/friction only after nominal recovery works. Do not add turning, jumping, or backflip rewards.

The completion sequence is:

`verified settled lying start → any feasible rise → stable two-foot stand for 2 s → stay in that stand for another 5 s`

No exact intermediate joint angles or mandatory kneeling sequence. No reset at the first upright frame, at 2 s, or at the first success event. Continue to the trial horizon, proposed 25 s, and invalidate a trial if the robot subsequently walks off, falls, or loses the final support conditions. Recovery must finish its 7 s hold before the horizon. The hold anchor may reset after an unsuccessful attempt, but never after a credited success.

Provisional final hold tolerances (calibrate against an actually stationary pose before training; these are engineering starting values, not values asserted by the cited papers):

| Signal | Required during hold |
|---|---|
| Pelvis height above local ground | 0.78–0.94 m, based on this model's 0.84 m reference |
| Pelvis AND actual torso tilt | Each <=12 degrees |
| Planar / vertical / angular speed | <=0.08 m/s / 0.08 m/s / 0.25 rad/s |
| Joint speed RMS | <=0.8 rad/s |
| Loaded feet | Each >=10% total weight; sum 75–125% total weight |
| Sole orientation / contact-point slip | <=15 degrees / <=0.05 m/s |
| Non-foot ground support | No force above 5 N |
| COM support margin | At least 10 mm inside the loaded-sole support polygon |
| Drift from fixed hold anchor | Pelvis <=50 mm, each sole <=20 mm, yaw <=10 degrees |
| Physics/contact validity | Fresh samples, no severe impact or penetration |

Do not infer support from a root coordinate or a foot body center alone. Ground-relative pelvis height is not total robot height. The force ratios allow asymmetric static loading; they do not force 50/50 load at each step.

## Reward proposal and units

| Term | Proposed contribution | Purpose |
|---|---|---|
| New best height/torso progress | <=1 total per episode | Sparse exploration help; not repeatable by bouncing |
| Full 2+5 s completion | +20 once | Pays for the requested recovery and settled stand |
| Subsequent loss of the stand | -20 once, invalidates success | Revokes credit if it walks/falls away later |
| Timeout/episode end without valid recovery | -3 once | Missing the task is not a success |
| Terminal motion | -0.10 × bounded normalized motion per second | Damps motion once upright, foot-supported standing is feasible |
| Physical joint limits / unreachable targets | Proposed -0.5 / -0.25 rates | Respect available motion; never enlarge limits to rescue learning |
| Ankle differential envelope | Proposed -1 rate | Use explicit `[LP, LR, RP, RR]` names, not implicit indexing |
| Action change / power | Discovery -0.001 / -2e-6; refine only after success | Weak exploration costs, then measured smoothing |

The first five rows are implemented in the independent core; the other regularizers are now in candidate.py. Their measured magnitudes still require learning-time calibration. Height/upright progress uses the increase of the episode's best combined score, not reward on every frame. It allows temporary downward/rolling motion, is bounded, and is NOT claimed to be policy-invariant potential shaping. Initialize the score from the settled start to avoid a reset windfall. There is no free standing/alive-rate reward to farm.

`step_reward` is the already integrated contribution. Return `rate=step_reward/dt` from an Isaac term with outer weight 1.0. Regularization rates multiply dt exactly once. Tests check event contributions at dt .01/.02/.04. PPO's discount and horizon still affect the relative value of distant success: inspect discounted returns and contact/impact behavior in the smoke test before scheduling a long run. Reward weights alone do not guarantee convergence or exclude every exploit.

## Before connecting to Isaac

1. Create a separate asset variant with usable collision geometry. Filter only necessary adjacent pairs; keep non-neighbor limbs/torso collision active. Audit all four lying families and large-motion joint sweeps. Do not globally enable overlapping boxes or modify the successful runner's asset underneath it.
2. Generate/audit a pose bank using the actual articulation and per-body collision bounds. Place each pose above its lowest collision point, settle without external support, reject penetration/NaNs/bad velocities, and retain valid pose IDs and orientation labels. A boolean `validated_pose=True` is not a substitute for that audit.
3. Fill every `Frame` field from measured physics before automatic reset. Resolve bodies/joints by explicit name. Use actual torso orientation, total mass (including any randomized mass), upward foot ground forces, sole contact-point velocity, and a signed 2D COM margin inside the convex hull of the two loaded sole footprints. Do not use a broad AABB as a support polygon or silently default missing fields to zero. Audit ground-versus-self force filtering; `net_forces_w` alone does not label every contact's other body.
4. Run the tracker once per control step; cache its output for rewards, terminations and logging. Isaac can ask terms more than once and computes terminations/rewards in a specific order. Provide actual impending timeout/failure before resetting. Preserve live post-success invalidation. Missing/stale contact data cannot satisfy a hold. Choose a documented short contact filter if solver noise warrants it; do not hide real one-foot hops.
5. Keep a new recovery observation/normalizer schema. Include pose group, best-progress/hold state and fixed-anchor displacement in privileged critic input; use appropriate proprioceptive history in the actor. Do not alter the accepted runner's 80-dimensional observation or its native .35 action scale. Pick and retain one recovery action scale through discovery/refinement; the old .70→.35 change is not an equivalent playback of the same weights.
6. Remove locomotion low-height/orientation/non-foot-contact terminations during rising. Retain numerical/hard-limit/severe-impact safeguards. Arms/knees may support the rise. No policy-generated episode reset. Continuous/peak thermal feasibility and coupled actuator limits remain prerequisites for hardware claims.
7. Evaluate the zero-speed controller handoff separately: after a valid hold, blend controls cautiously and measure a further stationary window. Do not presume a running policy is a reliable standing controller. Keep recovery holding until the handoff passes. No automatic claim of one unified neural network.

## Measured gates and proposed execution plan

Prepare collision/pose smoke at 16–64 environments, then a short train/save/reload smoke. Start discovery in bounded 200-iteration blocks with matched evaluation between them; this is a checkpoint interval, not a promise of success in 200 iterations. Decide the total budget from learning signal and true success, rather than repeating an ineffective 600/800-iteration run. Benchmark 8192/6144/4096 only after the new contact workload is stable; use the existing 512 MiB reserve unless the user changes it.

For each frozen evaluation, use disjoint pose-bank samples, at least 128 trials per pose family, two fixed seeds, and nominal versus modest perturbation batches. Report strict final success, post-rise failures, median/p95 rise time, tail drift, contacts/penetration, torque-speed saturation, RMS effort and action oscillation separately for every pose. Show sample counts and confidence intervals; exclude easy upright starts from the recovery denominator. Initial proposal: >=90% hard recovery in EACH required nominal pose family, >=85% in EACH perturbation family, no severe numerical/penetration failures. Thresholds must be frozen before candidate comparisons, not loosened to pass a weak policy.

The independent evaluator must recompute holds from raw telemetry, not just trust `_getup_success` or the new core's event. Preserve first-trial failures across resets. Archive exact config/source/asset/checkpoint hashes and only promote passing checkpoints. If no true recoveries appear, inspect achievable contacts/reward signal and change one hypothesis at a time; do not blindly increase steps or strengthen all gait penalties.

Invoke `chain.py` explicitly to launch the isolated training; importing the reward module does not start anything. Check the run's `state.json`, stdout/stderr, GPU CSV and preflight record before reporting it as running.

## Local tests

```powershell
Set-Location "E:\Aditya\robot_sim\humanoid_project\experiments\guarded_getup_v1"
& "E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe" -m unittest test_reward_core -v
```

These synthetic tests exercise logic, not simulator dynamics or policy quality. The validated runner's GUI uses `experiments/guarded_run_v4/play.py --speed 4 --real-time`; it does not use or train this recovery prototype.

### Checks completed September 3, 2026

- All **18 CPU contract tests passed**, including repeated reward farming, partial resets, stale contacts, invalid starts, torso-only cheating, support failure, sliding/shuffling, yaw wrapping, later walkoff credit revocation, and dt scaling.
- A synthetic **8,192-instance CUDA** batch completed the 350 control-step / 7 s standing hold with exactly one completion per instance and the expected 21 total reward for the ideal trace. This is a reward-tensor check, NOT an 8,192-robot Isaac simulation or memory benchmark.
- New Python files passed compilation. Updated `humanoid-robot-training` skill passed its validator.
- The separate v4 runner playback script loaded the exact final checkpoint at .35 scale and ran **200 headless inference steps / 4 s**, no falls. Command stayed 4.0 m/s; reported instantaneous speeds were 3.50, 3.98, 3.98 and 3.93 m/s at seconds 1–4. No PPO training or policy export occurred.
- Repository `Learning/` documentation remains excluded by `.git/info/exclude`. No public ignore rules, journal passages or production robot/task files were changed. The pre-existing modified `CAD/README.md` was left alone.
