# Five-hour training check — October 6, 22:43 UTC

Evidence saved 2026-10-06T23:04:49.366615+00:00. The same R39 full learner/supervisor 35688 is running. **3.5 m/s is still unqualified.** No reward, gain, physics, cap, actor or optimizer change entered production this check. Existing head/arm refinement continues,20 batches/update; no arm-leg trajectory is imposed. Goal remains verified5 m/s or verified4 m/s with proven native limits blocking5; computer stays on.

## Exact independent behavior

Newest completed matching pair reviewed is `402_run_350_train_131`, SHA `22d64c0388cdd035747fa721d4cfe6d0bf1cacd60dba7b1e1cc3beaba0142748`. Each nominal/pushed evaluation contains320 robots,20 seconds,32 per command. Current3.5 signed speed is 3.462/3.467m/s, lateral 0.341/0.319. Gates bind this saved actor, not newer partial learning. Nominal clones share starts: this is the existing contract, not32 independently seeded initial-state trials needed at the endpoint.

| Command m/s | Signed speed nominal / pushed | Failures nominal / pushed |
|---|---:|---|
| 0 | -0.000 / -0.000 | pass / pass |
| 0.15 | 0.103 / 0.106 | forward_net_travel / forward_net_travel |
| 0.35 | 0.331 / 0.331 | pass / pass |
| 0.65 | 0.678 / 0.678 | pass / pass |
| 1 | 0.978 / 0.974 | pass / pass |
| 1.5 | 1.487 / 1.486 | geometric_flight / geometric_flight |
| 2 | 2.006 / 2.007 | pass / pass |
| 2.5 | 2.588 / 2.592 | pass / pass |
| 3 | 3.043 / 3.042 | pass / pass |
| 3.5 | 3.462 / 3.467 | world_lateral_speed / world_lateral_speed |

Last matching actor passing every stopped/prior case through3m/s is `267_run_350_train_086`; preserve it and separately certified original349. Direction improved in some newer samples, but retention/hopping/slip remain nonmonotonic. No 3.5/4/5 or actuator-limited endpoint is established. [Exact pairs](closing_gate_review.json), [saved evidence](evidence.json).

## Actual upper motion

Inspected [current angle/target](final_upper/upper_joint_angles.png) and [velocity](final_upper/upper_joint_velocities.png) timelines, with32 robots/cmd at10 Hz and16 representatives at50 Hz, plus preserved349 reference. [Data/coordination](final_upper/upper_joint_time_audit.json) contain bilateral ranges, bias, elbow coordination, shoulder/hip phase, smoothness and saturation. Head gait oscillation, fast unequal shoulder pitch bias/amplitude and roll remain; this is not natural-motion certification. Native zero neck is torso-relative. Human comparison uses amplitude/asymmetry/coordination features from [Mirelman et al.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043); their wrist/back IMUs and morphology differ from these native encoder axes. No verified speed-matched official Unitree encoder curves were obtained; no reference curves are fabricated.

New physical test: exact actor399 actor,192 robots,six32-robot cohorts,20 seconds,truthful observations/free base/gravity/self-collision/native caps. Only diagnostic shoulder-pitch stiffness35/50/70 changed; damping1 and targets/action semantics stayed. All survive without resets,commands match,reward terms reconstruct within9.54e-7. At3.5, K35 left/right actual-minus-target phase is-175.4/-92.7deg and gains.353/.176; K70 is-89.6/-58.6 and.355/.442. K70 lateral decreases.412→.303m/s, but still exceeds the independent direction bound and does not establish all-speed retention. At1 m K70 increases ranges to47.4/44.5deg from39.5/30.0, so improved tracking is not automatically optimal appearance. **No gain candidate is deployed or reported as64/128 learning-tested.** Next compare a bounded current-state candidate to unchanged full-state control/all gates. [Response/plots](arm_response_analysis.json), [physical targets](arm_response.png), [loads](arm_load_audit.json).

K70 fast shoulder peak efforts27.7/25.8Nm are below declared71.25Nm peak, and shoulders have zero driving-speed clips at native5 ms. Continuous assumptions remain unbenchvalidated. A sparse K70 left-elbow clip is not a sustained locomotion ceiling. [All200 Hz motors](arm_response/substep_motor_audit.json). Isaac's [actuator documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html) distinguishes control gains from caps; tuning these coefficients is not raising the native limits.

Actual Isaac head-link orientation now matches URDF FK to7.34e-5deg maximum. Native-forward world-heading RMS atK35/3.5 is12.21deg, p9521.88, versus native neck4.01deg in that cohort. This closes the earlier FK/Isaac parity gap; the original CAD face/camera optical axis is still uncalibrated, and cloned cohorts do not establish population gaze. [Head parity](actual_head_axis_parity.json), [world-axis plot](actual_head_world_axis.png).

## Gradients and rejected-candidate landing diagnosis

Captured four genuine4,096-environment PPO minibatches from exact actor399 full state,seed97,without changing the update. Saved actual advantages,actions,Gaussian parameters,teacher labels and incoming mean derivatives. Moving-score CPU reproduction cosine>=.99998; CPU/GPU clipped-surrogate boundary effects remain. Prior mean-network norm1.58–2.19 versus exact-current-command PPO16.61–34.87; cosine ranges-0.145→+.556. Upper auxiliary/PPO cosine also changes sign. These finite mean-network samples before global clipping/Adam/exploration updates do not establish globally opposed gradients or justify arbitrary weight changes. Per-command coverage/advantages and effective target errors are saved in [gradient analysis](current_gradient_analysis.json).

The first CPU stopped-hook subtraction assertion failed and is preserved in `gradient_reproduction_first_failure.log`. Moving incoming derivatives are separately checked; stopped PPO reporting uses the captured Gaussian analytic surrogate. Unresolved stopped-hook subtraction is a diagnostic-accounting issue, not a reproduced production defect. Do not silently present the whole decomposition as bitwise exact or infer a runtime repair.

Replay every saved50 Hz hop event from the earlier rejected fixed-frame64/128 control/candidate. Candidate128's new2.5 failure is a real paired touchdown at0.90/0.94s during startup, two impacts out of87 qualifying touches,2.299%; control has none. The unchanged verifier counts startup. Keep the candidate rejected and require independently seeded retention-preserving replication, not gate relaxation or blind repetition. [Landing attribution](prior_landing_event_analysis.json). Official [Unitree G1 configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py) remains a research/source reference, not a guarantee this robot's gait or directions converge.

## Resume, memory and timing

Saved boundary399 SHA `cbb03b2b7577f8a7210d989a34c36992681b90e0b07a3ea9b2daf70e89cb60f6`; owned supervisor suspended and eval drained60.0 seconds. Total bounded pause 167.4 seconds includes drain. Same supervisor resumed. First resumed `402_run_350_train_131` completes65 real6,144-environment updates with exact actor/critic/Adam/exploration,zero pre-updateKL/command mismatch,upper20batches,minimum GPUfree1460MiB and host5380MiB,private+9MiB,Torch+0MiB. [Resume proof](resumed_learning_verification.json).

Through saved399: 8398completed+33hash-proven recovery=8431R39 retained main,plus16936inherited. Capacity/smoke/diagnostic samples excluded. Rates44.89k active collection+PPO/38.90k worker-effective/26.36k whole-chain;3,423.4 seconds prior pauses excluded and current pause lies after that measurement endpoint. Paused400 wrapper includes current pause. Target8192,reserve300MiB,host guards/fallback and180continuous healthy TRAINING-second ramp unchanged;6144 is actual current training count,not320 eval count. One actual worker and55 frozen hashes verified. [Timing/load evidence](throughput_and_loads.json).

Twelve [diagnostic validation checks](diagnostic_validation.json) pass,including exact initial full load,four real PPO updates/CPU saved checkpoint load,physical probe,reward reconstruction,FK parity,event replay and same resume. The premature harness invocation failed before worker drain and started no simulator; its log is retained. All failed diagnostic attempts and original sources/checkpoints remain. Completed videos and unsaved Fusion work were untouched.

## Next focused work

1. Preserve the useful R39 full learner, actor267 with every prior case passing, and separately certified actor349. The current 3.5 m/s direction gate and recurring lower-speed retention failures remain unresolved. Use the new actual per-command advantages, PPO mean-network gradients, teacher coverage and startup landing attribution to select one bounded retention-preserving continuation. The earlier fixed-frame candidate128 reduced drift but added real paired 2.5 m/s startup touchdowns at 0.90/0.94s; it remains rejected. Require independently seeded replication, unchanged full-state control at 64/128 updates, all matching prior/current gates and proper versioned provenance before deployment. Do not blindly repeat that kernel, relax startup gates or change arbitrary weights.

2. Inspect the required head/shoulder/elbow angle, target and velocity timelines at every speed. The new 192-robot,20-second shoulder stiffness 35/50/70 test with damping 1 improves response, but no gain candidate has been learning-tested or deployed. At 3.5 m/s, stiffness 70 gives phases-89.6/-58.6 degrees versus-175.4/-92.7 at 35, lateral 0.303 versus 0.412 m/s and shoulder effort peaks 27.7/25.8 Nm. Next test a shoulder-only gain candidate against unchanged full-state control for 64/128 updates, including all prior, direction, flight and posture gates and an independent seed. Higher gain also enlarges slow arm swing; improved tracking alone is not natural-motion certification. Preserve forward head and all learned body state. Actual Isaac head orientation now agrees with URDF FK within 0.000074 degrees; world native-forward heading RMS is 12.21 degrees versus neck 4.01. Original face/camera optical axis remains uncalibrated. Resolve the failed stopped-hook CPU gradient subtraction as a separate diagnostic-accounting issue; moving derivatives are independently validated, and no production defect is inferred.

3. Verify the same supervisor, one actual worker,55 frozen hashes, original local ground and exact actor/critic/Adam/exploration recovery. Keep the 8192 target,300 MiB reserve, host fallback and 180 continuous healthy TRAINING-second ramps. Account for 33 retained R39 recovery updates plus 16936 inherited; exclude the 167.4-second current diagnostic pause and prior 1823.6/1599.9-second pauses from throughput. The shoulder sweep has zero shoulder driving-speed clips at 5ms; sparse stiffness 70 elbow clips do not establish a locomotion ceiling. The endpoint still requires qualified 5 m/s or qualified 4 m/s with controlled native limits proving 5 is blocked, all unchanged gates and 32 independently seeded initial-state trials per command. Preserve unsaved Fusion work and completed videos; save, pause both training and automation, and brief with the computer on only at that verified endpoint.

Same five-hour heartbeat/thread/notification preference remains active. [Decision](decision.json). No training-from-scratch reset, hybrid, production input spoof, pulse, diagnostic learned actor or cap increase is deployed.
