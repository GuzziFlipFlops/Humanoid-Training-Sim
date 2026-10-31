# October 5, 07:42 EDT: retained walking and constrained take-off diagnosis

**R36 remains the same full-state learner; running is not qualified.** At the saved 960-update observation, all stopped and prior walking commands through 1.0 m/s pass matching independent nominal/pushed tests. Head RMS at the running command is 3.43/3.43 degrees. The 1.5 command travels at 1.371/1.375 m/s, but genuine flight is 0.0000%/0.0035%, below the unchanged 1.5% minimum. No whole running-stage, 5 m/s or actuator-ceiling endpoint is reached. [Decision](decision.json), [evidence](evidence.json), [owned state](final_owned_observation.json).

## Exact independent behavior

The latest complete pair is `054_run_150_gate_016_nominal` / `055_run_150_gate_016_push`, bound to `053_run_150_train_015/candidate.pt`, SHA `b323026a02a822ec89f9829956a13bd6e5f1e192efff101c92c0adb61a2868c9`. Each test has 192 robots, 20 seconds and 32 independent trials per command. All survive. A prior-case pass does not promote the running stage.

| Command m/s | Nominal / pushed signed speed | Nominal / pushed true flight | Case result |
|---|---|---|---|
| 0.00 | 0.000 / 0.000 | 0.0000% / 0.0000% | Pass |
| 0.15 | 0.229 / 0.228 | 0.0000% / 0.0000% | Pass |
| 0.35 | 0.441 / 0.437 | 0.0000% / 0.0000% | Pass |
| 0.65 | 0.718 / 0.718 | 0.0000% / 0.0000% | Pass |
| 1.00 | 1.005 / 1.005 | 0.0000% / 0.0000% | Pass |
| 1.50 | 1.371 / 1.375 | 0.0000% / 0.0035% | flight, geometric_flight |

Slow tracking is still drifting upward: 0.15 moved from 0.189 m/s after the first 64 updates to 0.229 after 960; 0.35 moved from 0.391 to 0.441. They still pass, but the passes should not be presented as perfect command tracking or indefinite stability. Earlier 832-update values were about 0.207 and 0.431. Running flight remains nearly zero across repeated complete pairs. [Latest exact gates](latest_matching_gates.json), [all gate trends](final_gate_trends.json), [per-command upper telemetry](upper_audit.json).

## Bounded physical intervention

The owned supervisor was suspended at its saved 896-update checkpoint 050 and its worker drained before a separate simulator started. Twelve cohorts, 24 robots each, ran for 20 simulated seconds. Nine experimental cohorts used smooth bilateral knee/hip/ankle shortening, varied lead time, or preceding stance extension. The other cohorts were running, walking and stopped controls. Interventions began after 3 seconds; statistics start at 5 seconds. Observations remain truthful, base free, gravity and self-collision active, and all native action, joint, slew, CAD projection and torque/speed constraints remain. There are no root impulses, pose writes, support forces, phase-input changes, trained diagnostic actors or deployed pulses.

All 288 robots survived without resets, but **no cohort produced genuine elevated-and-unloaded flight**. Some force-threshold gaps were grounded gaps. The best sampled minimum two-sole clearance was only 0.188 mm, below 1 mm. Retracting alone reduced speed to roughly 1.31–1.36 m/s; adding extension reached 1.46 m/s but remained grounded. Every running intervention had a lower complete mean reward than the unmodified running control. This grid rules out these particular interventions as a deployable repair; it does not prove take-off physically impossible or every natural running trajectory lower reward. The older preserved actor has previously demonstrated physical flight under the same caps.

[Physical cohorts and complete reward breakdown](physical_response/result.json), [synchronized 50 Hz trace](physical_response/telemetry.npz), [phase plot](phase_objective.png), [exportable plot](phase_objective.pdf).

## Effective target and reward diagnosis

A CPU replay reconstructs the actual action clamp, default-offset targets, soft joint limits, slew and CAD feasibility projection on 14,400 recorded states. Maximum difference from the deployed target is 2.98e-08 rad; no sampled target is rejected by the coupled CAD feasibility projection. Local finite differences hold each previous target fixed. Small ankle-action changes produce zero target derivative in 78.8%/88.2% of the running-control samples; knee values are 19.5%/20.7%. These are action/joint/slew dead zones, not measured policy gradients or a motor velocity ceiling. Walking also shows ankle dead zones, so their presence alone is insufficient to explain running failure.

During intended double swing, the control still has about 379.1 N of total foot load. Larger extension/retraction reduces that mean to about 157 N in one cohort but never yields true flight. Phase plots show target-to-joint response lag and timing differences; the pre-action phase and post-step reward phase are saved separately. Their one-control-tick difference must be respected in any future timing experiment.

Sixteen CPU reward checks pass: the actual v8 summed-load term reconstructs the saved term; full term sums reproduce reward rates; progressive fixed-state simultaneous unloading earns monotonically more credit; redistributing unchanged load between two intended swing feet earns no extra credit; walking and stopping have no running-unload credit; and flight credit cannot appear without physical clearance and unloading. This establishes formula consistency, not favorable long-horizon return or convergence. The diagnostic's old `mean_unload_score` helper still averages foot exponentials. **Use `term_running_unload`, not that helper, for all v8 objective conclusions.** [Target/phase analysis](physical_objective_analysis.json), [reward counterexamples](reward_phase_audit.json).

Exact CAD Jacobian/reduction motor mapping and directional finite-difference unit checks pass on the saved 50 Hz q/qd trace. In the unmodified running cohort, sampled ankle p99 speeds remain well below their caps and there are no sampled cap crossings. Diagnostic interventions can change these statistics, but no sustained locomotion-relevant ceiling is established. Five-millisecond substep attribution remains untested. [Motor audit](motor_speed_audit.json).

The primary-source lead is [Siekmann et al., Periodic Reward Composition](https://arxiv.org/html/2011.01387v2). It combines phase-dependent foot-force and foot-velocity terms and smooths interval boundaries to support learning. Its Cassie controller and training contract differ from this robot. Smoothed timing or constraint-aware exploration is a hypothesis to test on the current truthful-state controller, not a reason to copy numerical weights or claim guaranteed convergence.

## Resume, resources and preservation

The same owned supervisor PID 19116 resumed after 431.7 seconds. The first resumed main block 053 completed 64 actual PPO updates at 8,192 environments, with exact actor, critic, Adam and exploration equality to the saved 050 full checkpoint and zero maximum pre-update KL/observed-action command mismatch. No source, reward, observation/action semantics, geometry, contact, motor cap, optimizer state or actor was reset. This actual completed block verifies continuation; no extra changed-source PPO smoke is needed because no runtime change was deployed. [Pause receipt](diagnostic_pause.json), [resume receipt](diagnostic_resume.json), [resumed learning](resumed_learning_verification.json).

Completed-main throughput is 46.68k active collection-plus-PPO, 40.39k main-worker-effective and 25.88k whole-chain samples/s after excluding this pause. The denominators differ. Main memory minima are GPU 479 MiB and host available 5457 MiB. The resumed block has minimum GPU 540 MiB and host 5584 MiB, private growth 9 MiB and no Torch allocation growth. Eval051 wrapper time includes the diagnostic pause and is not physics throughput. One actual CUDA worker and all 49 frozen authored hashes verify. Target 8,192, 300 MiB reserve, host fallback and 180 healthy-training-second upward ramps remain. [Rates and block evidence](final_throughput.json).

R36 preserves another 5,825 inherited main updates. Head-forward progress and walking teachers are retained; arm telemetry is recorded, but natural arm motion is not certified. Full-body recordings are already fulfilled and no new recording is pending. The initial CPU analysis repeatedly decompressed the NPZ and was stopped using its exact process identity; its initial logs and the file-lock move error are recorded. Caching bounded arrays corrected only this diagnostic. Raw physical evidence and production training were untouched.

## Decision and next check

Keep this same learner running while preparing a justified bounded comparison. Do not raise caps or deploy the unsuccessful pulses. Do not repeat rejected running-only row imitation or the 44% running-exposure change without new evidence. The next check must measure a candidate's actual flight, direction, all prior-speed tracking and head/arm behavior against unchanged full-state control; a local dead-zone statistic does not establish a successful repair.

1. Audit running-only effective ankle/knee target exploration and local dead zones on current truthful neutral-head states. Compare a bounded training-only constraint-aware candidate against unchanged full-state control at 64 and 128 updates, with all nominal/pushed prior/head gates, before any deployment. No production action/observation rewrite or raised caps.
2. Resolve pre-action versus reward phase and measured target-to-joint/sole response delay; test feasible normal-limit timing and complete objective return. The tested bilateral pulses were all grounded and lower reward. Do not treat their failure as proof of a mechanical impossibility or arbitrarily increase a flight weight.
3. Compare fresh matched flight, stopped/prior speed and upper telemetry; preserve improved actors. Slow tracking is drifting upward despite still passing: 0.15 rose from 0.189 after 64 to 0.229 after 960 updates; 0.35 from 0.391 to 0.441. Inspect exact per-command prior target/gradient coverage and broad tracking-kernel sensitivity if this continues. Verify 8192 memory, full-state fallback, 180-second healthy ramp and throughput; keep rejected running-row/exposure variants rejected without new evidence.

Update this same adaptive three-hour heartbeat, preserving cadence, thread, notification preference, durable qualification scope and computer-on pause-and-brief endpoint. No endpoint is proven.


## Closing exact-gate correction: 2026-10-05T12:17:43.515439+00:00

The latest matching nominal 060/pushed 061 pair on the 1088-update checkpoint 059 does **not** retain every earlier speed. Stopped, 0.15 and 1.0 pass both. Nominal 0.35 fails speed at0.448m/s, and pushed 0.65 fails speed at0.745m/s. Their counterpart tests pass. All robots survive. The 1.5 command gives1.355/1.359m/s with 0%/0% true flight; head RMS4.02/4.04 degrees. The previous 1024-update checkpoint 056 passed all earlier commands in matching 057/058 and is preserved with its exact SHA/gates. These qualifications belong to 056, not 059. [Latest exact failures](closing_matching_gates.json), [last prior-passing actor](last_all_prior_passing_actor.json).

The same R36 learner remains active with one actual CUDA worker and 49 hashes verified. The owned snapshot caught062 startup with no PPO progress yet; the63 iteration counter quoted in the first closing metadata was from completed059, not062 learning. Snapshot free GPU 5,591 MiB is startup capacity, not steady 8,192 training memory. Completed-main memory/throughput and exact resumed learning are recorded separately above. No sources, weights, physical caps, actor/critic/Adam/exploration or inference changed. The earlier closing prose assumed earlier-speed passes and is explicitly corrected here; raw gate results were preserved throughout. Failed diagnostic closing selection/progress reads are preserved in tool evidence and fixed by checkpoint-bound completed-pair selection and labeled fallback progress.

Next priority is per-command conditioned-target/gradient and tracking diagnosis, alongside feasible take-off timing. A new candidate requires an unchanged full-state control, 64/128-update comparison and all matching gates before deployment. Preserve current posture, 056, the qualified walking teacher and 038 flight actor. Keep the same adaptive three-hour check active, target8192/reserve300/180healthy ramp and computer-on endpoint. Running, 5 m/s and an actuator ceiling remain unqualified.


Final liveness: R36 completed 1,152 main updates and its same supervisor is actively running the 192-robot pushed evaluation064. One actual worker is verified. This is evaluation count, not PPO count; completed main blocks use8,192. The last complete pair discussed above remains checkpoint059; no newer qualification is inferred. Same automation is verified active and the CPU history inventory has36 physical runs/1458 operations, zero differing hashes and zero read warnings. See final_liveness.json.
