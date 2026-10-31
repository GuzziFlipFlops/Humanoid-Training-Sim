# Original five-foot walker versus physical V29 R22

Checked October 2, 2026, at approximately 18:19 EDT (22:19 UTC). The four-hour `humanoid-training-four-hour-check` automation was deleted through the app, which confirmed deletion. Its local configuration is absent. Training itself was not stopped.

## Is the current training working?

The supervisor and its single simulation worker are running, but the locomotion objective has **not** succeeded. R22 is still learning standing; walking and running have never started in this run. At the live snapshot, 703 full PPO updates were checkpointed and the next worker was training at 8192 environments. The supervisor is PID 31408; worker wrapper PID 24348 owns actual Python worker PID 31208. The latest progress is fresh. Free VRAM was 1209 MiB, above the 300 MiB reserve; available host RAM was 4639 MiB.

The latest completed independent test, after 600 updates, used 128 robots for 20 seconds. All survived, but they drifted 0.487 m against a 0.25 m limit and averaged 0.150 m/s against a 0.12 m/s stationary limit. Pelvis tilt was 12.52 degrees and torso tilt 10.97 degrees. It is not stable stationary standing. The reported support metric also failed, but that metric has a coordinate-frame defect described below.

| Nominal standing test | Survival | Final drift | Mean planar speed | Pelvis tilt |
|---|---:|---:|---:|---:|
| 150 updates | 100% | 0.321 m | 0.052 m/s | 2.82 degrees |
| 300 updates | 43% | 0.923 m | 0.252 m/s | 6.25 degrees |
| 450 updates | 100% | 0.829 m | 0.215 m/s | 9.01 degrees |
| 600 updates | 100% | 0.487 m | 0.150 m/s | 12.52 degrees |

The earlier improvement in survival was real. Later tests do not support a claim of steady progress toward the standing gate. Pushed standing has not been reached. High survival is insufficient to qualify standing, much less walking.

## Which old policy actually walked?

The original approved walker is:

`humanoid_project/logs/rsl_rl/humanoid5ft_natural_walk/2026-08-31_17-06-10/model_2098.pt`

SHA-256: `dbddd7222f591d8d43d5e13d679a8ec4f9deb251dd864a15f78e9f88dcf6e174`.

I recomputed results from its saved 200 Hz physics arrays and episode records, rather than relying on reward or a journal claim. The native action scale is 0.35. Each condition contains eight walking robots with a 1.0 m/s command, observed for 24 seconds; velocity statistics exclude the first three seconds and use the first episode. The saved checkpoint hash matches and the checked telemetry is finite.

| Original walker condition | Forward speed | Mean absolute speed error | Falls |
|---|---:|---:|---:|
| Nominal | 0.9938 m/s | 0.0113 m/s | 0/8 |
| Randomized | 0.9946 m/s | 0.0117 m/s | 0/8 |

Raw knee motion and foot-force records show repeated bilateral stepping with approximately 0.473-second stride periods. This is actual walking at the tested command. It is asymmetric: nominal left/right load sharing is about 35/65%, with stance durations of 0.146/0.303 seconds. This evidence does not establish humanlike gait, unrestricted robustness, hardware readiness, recovery, or backflips.

A later guarded V4 checkpoint also retained walking:

`humanoid_project/logs/guarded_run_v4_20260903_0305/06_speed_4_0600_8192/candidate.pt`

Its saved independent evaluation has 24 trials per command for 20 seconds. At 1.0 m/s, nominal speed is 1.0016 m/s and pushed speed is 1.0013 m/s, both with zero falls. The saved V4 gate reports walking retention passed. That historical gate differs from the current stricter bilateral gait gate; its roughly 0.309 stance asymmetry would not establish compliance with the new 0.18 limit. The separate older `model_3800.pt` runner is not the reference successful walker.

Sources: `humanoid_project/logs/gait_diagnostics_20260902/{nominal,randomized}/{manifest.json,physics.npy,episodes.npz}`; guarded V4 `06_speed_4_0600_gate.json` and `06_speed_4_0600_eval_{47,83}/evaluation.json`.

## Why the old result differs

| Aspect | Successful original walker | Current R22 |
|---|---|---|
| Starting point | Resumed a 500-update standing policy; then 1600 walking updates | Fresh actor; no old actor or teacher; still blocked at standing |
| Standing feedback | Upright +3.5, height +2.5, both feet +1.5, stationary +1.0 | G1 core reward weights, with weak standing-specific shaping and an added aggregate positive floor |
| Action-change penalty weight | -0.01 | -0.05 |
| Actual exploration standard deviation | Inherited about 0.676; final mean 0.416 | About 1.0 across all joints |
| Observation processing | Empirical normalization; 80 actor observations including linear velocity and contacts | Fixed scaling without empirical normalization; 400 history observations; actor excludes direct linear velocity/contact state |
| Actions / drives | 22 joint targets, scale 0.35; implicit PD | 23 joint targets, scale 0.25; CAD differential constraints, torque-speed limits and target slew limits |
| Physical model | Simplified five-foot model, 38 kg; self collision disabled | Physical CAD-derived model, 34.46 kg; self collision enabled |
| Recent learning rate | Mean 0.0000441 | 0.0000100, at the adaptive floor |
| Recent total sample throughput | 55,518 environment steps/s at 6144 environments | 11,862 environment steps/s at 8192 environments |

The old TensorBoard history records its 1600-update walking stage in about **71 minutes**, after the standing warm start. The current live sampling rate is approximately **4.68 times slower**, or **78.6% lower total throughput**, despite more environments. This compares actual total environment steps per wall-clock second, not just environment counts. Startup, independent evaluations and host-RAM recycling add further time outside these sampling measurements. These are different tasks and physical models, so this is not a controlled speed benchmark.

The current pinned Unitree G1 reward bodies are not a pretrained Unitree policy. A reward port still requires a compatible action/controller contract, observations, physical model, exploration and curriculum. Similar joint names do not make the trained policies interchangeable. The archived actor has 80 inputs and 22 actions; the current actor has 400 inputs and 23 actions, so directly loading the old weights would be invalid.

## Strongest measured learning problem

Across the analyzed last 50 training updates, regular reward before flooring averages **-5.1291 units/s**. The positive-floor adjustment adds **+5.1310 units/s**, leaving only **+0.00196 units/s** of regular reward after flooring. The action-change term alone averages **-4.8607 units/s**. Exploration remains near 1.0 and the learning rate remains at 0.00001.

This is strong evidence that the reward/exploration combination leaves little useful feedback to distinguish better standing actions. It is not proof that PPO has zero gradients: PPO does not require differentiation through the reward, and terminal penalties still distinguish trajectories. It is also not evidence that reward totals certify a gait. Here the problem looks primarily like a poorly conditioned training objective, rather than a robot earning high reward while secretly walking incorrectly.

CPU formula checks using the saved original reward bodies and pinned new weights support this diagnosis. At a 12.5-degree lean, the original upright reward drops by about 2.17 units/s; the new flat-orientation penalty is only about 0.234 units/s. The expected action-change cost from independent unit-standard-deviation Gaussian innovations is -0.44 for the old 22-action penalty and -2.30 for the new 23-action penalty. This is only the innovation component; policy mean changes can increase it, as the measured -4.86 demonstrates. These tests compare isolated formulas, not full simulated policy performance.

## Independently reproduced balance-check defect

`evaluation.py` says the support rectangle is expressed in a yaw frame, but it actually rotates center of mass and feet using the full pelvis quaternion. Roll and pitch then mix vertical separation into the planar support test. The relevant physical question is the gravity projection of center of mass onto a horizontal support region, so this can falsely reject balance.

A synthetic case with COM directly above the feet and 12.5-degree pelvis pitch fails the current full-rotation test and passes the yaw-only test. The accompanying CPU audit reconstructs 181 sampled ticks for the first 16 robots of the saved 600-update test using frozen URDF inertias. Foot forward kinematics agrees with saved simulated positions within 0.000030 m. Reconstructed full-rotation support-outside fraction is 66.2%, versus 6.6% with yaw-only coordinates; the original recorded first-16 metric is 70.0%.

The reconstructed COM was not saved directly by the simulator, and the reconstruction differs from the recorded metric. This is a diagnostic sample, not a fresh 128-robot passing gate. Even after correcting the frame, **measured drift and speed still fail independently**. No promotion is justified by this audit.

## Repair priorities established by this comparison

1. Correct and regression-test the support metric's coordinate frame, including tilted-root counterexamples and raw simulator COM telemetry. Do not weaken independent drift/speed or gait requirements.
2. Restore meaningful standing feedback based on the successful old upright/height/stationary/bilateral-support structure. Calibrate exploration and action-change costs together so the aggregate reward does not remain almost entirely floored.
3. Test observation normalization and controller/action scaling in controlled experiments while retaining the CAD transmission and motor constraints. Investigate repeated host-RAM growth separately.
4. Qualify a standing checkpoint independently before a walking warm start. Then measure actual bilateral steps, foot loads, speed and frame-by-frame motion, rather than using reward as success.

These are targeted repair priorities, not a guarantee that one change solves training. No reward, gate or active source-frozen run was changed during this comparison. Current training remains running; the repeated four-hour automation is removed.

## Reproducible evidence

- [Full comparison data](comparison.json), including saved configurations, event history, checkpoint hashes, isolated reward examples and gate results.
- [Support-frame counterexample and sampled reconstruction](support_frame_audit.json).
- [Original walking versus current standing joint/contact traces](walking_vs_standing.png). The commands and robot models differ; this figure shows observed behavior, not a matched benchmark.
- CPU-only companion scripts: `physical_v29/training/analyze_old_walker_20261002.py` and `physical_v29/training/audit_support_frame_20261002.py`.

Historical checkpoints, raw recordings and logs were preserved. No second Isaac/CUDA worker, Git commit, push, training restart or shutdown was performed.

Final live refresh at 22:22:24 UTC (18:22 EDT): **724 full updates checkpointed**, still standing at 8192 environments. The supervisor has resumed after another host-RAM recycle. Latest recorded free VRAM is 1661 MiB and available RAM 14777 MiB. Automation deletion remains verified. See `live_status.json` and `live_state_snapshot.json` for the refreshed process/state evidence. The latest completed gate remains the failed 600-update standing test.
