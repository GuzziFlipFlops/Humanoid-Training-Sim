# October 5, 17:46 UTC adaptive check — measured trajectory tradeoff; original learner resumed

At 2026-10-05T18:10:54.098103+00:00, **R37_offline_ground_full_state** is running with 1216 saved main updates and **7,361 inherited updates**. The same supervisor PID 32492 resumes from its full saved actor, critic, Adam and exploration. All 55 frozen authored hashes verify. No training source, reward, motor cap, physics, teacher, optimizer or observation change was deployed. Running and the 5 m/s endpoint remain unqualified.

## Actual independent progress

The 1,024-update pair present at the beginning of this check had zero genuine flight and failed pushed 0.35/0.65 speed retention. The 1,152-update checkpoint060 subsequently recovered every stopped/prior command through1.0 in nominal061/pushed062, but its1.5 case failed flight and world lateral speed. Retention is nonmonotonic. Passing records are preserved and never attributed to a newer actor.

Latest completed matching pair is bound to `063_run_150_train_019`, SHA `6f3b78c30a8c416cd206c8bb084d62be40f5c676ea2c46bf4c98eb72a5be7a62`. Each test contains192 robots,20 seconds,32 independent trials per command, with unchanged physical, contact, direction, gait, nominal/pushed and stopped/prior classifiers.

| Command m/s | Nominal signed speed | Pushed signed speed | Nominal/pushed pass | True flight % nominal/pushed |
|---:|---:|---:|:---:|---:|
| 0.00 | -0.000 | -0.000 | True/True | 0.0000/0.0000 |
| 0.15 | 0.224 | 0.225 | True/True | 0.0000/0.0000 |
| 0.35 | 0.443 | 0.437 | True/True | 0.0000/0.0000 |
| 0.65 | 0.740 | 0.741 | True/True | 0.0000/0.0000 |
| 1.00 | 1.017 | 1.018 | True/True | 0.0000/0.0000 |
| 1.50 | 1.405 | 1.408 | False/False | 0.0000/0.0000 |

Stopped/prior cases all pass both tests: **True**. Running case passes: **False**. At1.5 the head's combined yaw/pitch RMS is 2.93/2.94 degrees, preserving its forward-posture improvement. Natural arm motion is not certified. This check reruns the CPU-only upper-body audit of saved full-population positions/velocities; the5-degree reporting band remains diagnostic. [Exact gates](closing_matching_gates.json), [all checkpoint-bound trends](gate_trends.json), [last prior-passing actor](last_all_prior_passing_actor.json), [upper-body measurements](upper_audit.json).

## New bounded physical timing search

At checkpoint060, SHA `0d9eebbc204ed776681eda652b50a50d0db46a300c8e3efc72f15ec2e7063461`, the owned supervisor was suspended at an evaluation boundary; its worker was allowed to finish and drained before diagnostics. The search used the exact saved actor, truthful400-input history and executed-action feedback, native CAD differential/joint/motor/slew/action constraints, free base, gravity and self collision. It modified only diagnostic hip-pitch, knee-pitch and ankle-pitch actions. No head override, root support, pose write, phase write, fabricated observation or new learned actor was used.

Ten bounded parameters describe two smooth raised-cosine offset windows, their centers/widths and hip/knee/ankle amplitudes. Windows use the **observed pre-action phase**; reward phase is recorded separately. Offsets fade in after3s. They pass through the existing action clamp, soft joint limits, slew limiter and CAD pair feasibility checks, rather than bypassing them. Zero-offset controls execute the same clamped actions as the actor. Thirteen CPU formula/selection/terminal-return checks pass. These fixtures do not prove physical gait quality. [Parameter bounds](trajectory_parameters.py), [CPU checks](trajectory_cpu_tests.json).

Four successive populations at search seed71 evaluate32 trajectory cohorts with8 robots each for12s:128 cohorts including unchanged controls and elite means. Elites are selected by **complete return**, including the first terminal penalty and excluding later reset rewards. All search robots survive. Target, actual joints, forces, geometric clearances, raw/effective actions, command snapshots and every reward term are saved at50Hz. Observed/action command difference is zero. All generated offsets remain diagnostic and are absent from production.

The search's highest-return trajectories remain grounded; its maximum real geometric-flight fraction is **0.855%**, below the existing1.5% minimum. This is useful negative evidence within these bounds, not a proof that the physical robot cannot run. [Full search populations](trajectory_search/result.json), [objective/flight plot](trajectory_return_flight.png).

## Independent-seed validation and full objective costs

The best complete-return and best-flight trajectories are frozen, then retested against unchanged control with seed83:32 robots per cohort,20 seconds. All96 survive. This is a diagnostic feasibility test of offsets, not policy or running-stage qualification. Robots within a population are not exact paired-state counterfactuals.

| Diagnostic cohort | Signed speed m/s | True flight % | Complete20s return delta | Head RMS degrees |
|---|---:|---:|---:|---:|
| unchanged_control | 1.443 | 0.0000 | +0.000 | 3.02 |
| best_complete_return | 1.469 | 0.0083 | +1.949 | 3.03 |
| best_geometric_flight | 1.400 | 0.9154 | -5.310 | 3.14 |

The highest-flight trajectory reproduces about0.915% flight, with mean minimum sole clearance about1.48mm in those samples. Its longest observed run is one50Hz frame; this is not substep-certified20ms sustained flight. **All its genuine-flight samples occur outside the desired double-swing reward window**, while99.94% of intended double-swing samples retain load. Its complete return is5.31 lower than control. Average rate changes are dominated by linear tracking−0.2069, heading−0.0317, gait−0.0267 and angular tracking−0.0249, versus flight+0.00205. Joint-acceleration cost changes by only−0.00782. This observed trajectory has a timing/tracking tradeoff; it does not prove every natural running trajectory has lower reward.

The highest-return validation cohort gains1.95 complete-return units and tracks1.469m/s, but its true flight is only0.0083%. None satisfies flight, full-return and direction/head feasibility together. Every recorded full reward exactly reconstructs from its saved term sum. Action/reward phase is separated by0.033 cycles at1.65Hz, one20ms control tick. A descriptive target lag from the previous check is not a justified universal phase shift.

**Decision: reject trajectory transfer for deployment.** No candidate reaches the prerequisite physical behavior, so injecting these labels into a64/128-update learner would not be justified. No diagnostic learned actor is deployed and no new learning experiment is falsely reported as passed. Preserve the128 populations, independent validation, bounds, seeds, all telemetry and full checkpoint. [Selection](validation_selection.json), [independent physical validation](trajectory_validation/result.json), [term/timing analysis](trajectory_analysis.json), [decision](decision.json).

The next research lead is [Periodic Reward Composition](https://arxiv.org/html/2011.01387v2): phase-dependent force/velocity signals with smoothed interval boundaries. The measured phase mismatch supports a controlled timing test; the paper supplies neither a guarantee nor robot-specific numerical weights. Any new signal must pass fixed-state counterexamples, feasible normal-limit physics and a matched current-full-state64/128-update control with every retained/head/direction gate before production.

## Exact resume, memory and throughput

Resumed `063_run_150_train_019` completes64 genuine updates at8,192; actor/critic/Adam/exploration load equality all pass, maximum pre-update KL and command mismatch are zero. Minimum free GPU455MiB, host5252MiB, private growth7MiB, Torch allocation growth0MiB.

The same supervisor resumes after **342.2s** after worker drain. Nominal061's supervisor wrapper includes this pause; it must not be treated as physical evaluation speed. Completed-main rates are **46.51k active collection-plus-PPO**, **40.32k worker-effective** and **26.87k whole-chain samples/s**, excluding this diagnostic pause from whole-chain time. Whole chain includes R37 startup, capacity and gates; prior R36/R35 failures/pauses precede R37 and are not in its clock. Capacity/smoke/diagnostic frames are not main training progress. [Pause](diagnostic_pause.json), [resume](diagnostic_resume.json), [exact resumed learning](resumed_learning_verification.json), [owned observation](final_owned_observation.json).

Current operation `066_run_150_train_020` is `train` at8192 environments; an evaluation count is not PPO capacity. Next selected training count is8192. One actual worker,55 frozen hashes,8,192 target,300MiB reserve,host fallback and180 continuous healthy training-second upward ramp remain. The earlier passive200Hz motor audit found no driving-speed clipping through2.0 for that actor/experiment; isolated clips around fallen2.5 extrapolation do not establish a locomotion-relevant ceiling. This trajectory search does not add a new200Hz ceiling certification. Native limits remain unchanged.

## Records and next check

One attempted PowerShell/Python inline resume-script construction failed from quoting before creating or executing the script. It is preserved in tool evidence; the saved standalone resume script then succeeds on the exact verified supervisor identity. A read during main startup correctly found no progress file yet; subsequent actual progress is used, never startup memory as steady capacity. Earlier reports and all failed/rejected experiments remain intact. No full-body video request is pending; completed videos and unrelated unsaved Fusion work are preserved.

1. Resolve the measured physical timing and objective tradeoff before another running repair. Independent validation gives 0.915% flight, all just after the intended window, and loses 5.31 complete-return units; tracking dominates the loss. Saved flight half-cycle phases are 0.005-0.014 after wrap, versus the intended 0.43-0.50 window. Test earlier feasible leg timing and smoothed periodic force/velocity feedback using truthful recorded states, normal-limit physics and complete-return counterexamples for hopping, sliding and early exit. A one-tick clock distinction is not proof that the reward clock is wrong or that a fixed latency shift is optimal. Require a justified narrow candidate, unchanged current full-state control at 64 and 128 updates, and all nominal, pushed, prior, head and direction checks before deployment.

2. Compare matching stopped, 0.15, 0.35, 0.65, 1.0 and 1.5 m/s gates, head/arm telemetry and per-command conditioned-teacher coverage. Prior retention regressed at 1,024 and recovered at 1,152 updates; preserve passing actors and current posture rather than claiming monotonic recovery. Diagnose recurring speed or lateral drift. Clipping-equivalence, running-row transfer and 44% exposure remain rejected absent new evidence. Keep production observations truthful; do not deploy pulses, hybrids or a tilted-head rollback.

3. Verify the same full-state learner, local ground, 55 frozen hashes, one actual worker, moving 8,192/6,144 capacity, 300 MiB reserve, host guards and 180 continuous healthy training seconds before ramps. Exclude the documented diagnostic pause from throughput. Keep native CAD caps: the earlier 200 Hz probe had no driving-speed clips through 2.0 m/s, and fallen 2.5 m/s spikes do not establish a velocity ceiling. Preserve unsaved Fusion work and completed videos.

Keep the same adaptive three-hour heartbeat active. At verified actual5m/s or a causally established relevant velocity ceiling, save,pause training and this automation,brief the user and leave the computer on.


<!-- final_liveness_1746 -->
Closing verification at 2026-10-05T18:15:17.705583+00:00, using owned snapshot 2026-10-05T18:15:17.157514+00:00 and progress through 2026-10-05T18:15:15.354898+00:00: same supervisor and one actual worker, 55 frozen hashes verified. Operation `066_run_150_train_020` has 62 observed genuine main updates at 8,192, with zero pre-update KL and command mismatch; free GPU 455 MiB and host available 5320 MiB. These are partial updates, not new qualification. Same automation ID/thread/three-hour cadence/notification preference/durable scope are verified active. History inventory: 37 physical runs, 1,542 operations, zero differing hashes and no read warnings; eight old missing manifests remain explicitly recorded. [Final evidence](final_liveness.json).
