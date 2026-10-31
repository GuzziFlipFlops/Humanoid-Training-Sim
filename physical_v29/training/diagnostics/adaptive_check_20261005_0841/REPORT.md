# October 5, 04:41 EDT check: posture-aware walking retention

R36 is a full-state continuation of R35 at its saved 4,352-update checkpoint, with another 1,473 inherited R34 main updates preserved. A reproduced head-feedback mismatch in the old walking teacher is corrected only in its training targets. Actor, critic, Adam, exploration and adaptive learning rate are retained; no old tilted-head actor or diagnostic learned actor is deployed. **Running remains unqualified.** [Decision](decision.json), [deployment](deployment.json), [complete evidence](evidence.json).

## Production behavior before repair

Fresh initial status at08:42 UTC showed one actual Isaac/CUDA worker, all47 frozen hashes and6144 environments. The final complete pre-pause pair211/212 belongs to checkpoint210 at4288 new R35 updates, not to the later drained checkpoint213. All192 robots survived each20-second test, with32 trials per command. Stopped,0.15 and1.0 cases passed both;0.35 and0.65 oversped. At1.5command the signed speed was1.426/1.429m/s with zero genuine flight in both, below the unchanged1.5% minimum. Running head RMS was2.958/2.954degrees, compared with51.71 before upper refinement. Earlier moving commands retained larger head error; natural arm motion is not certified. [Exact trends](gate_trends.json), [upper audit](upper_audit.json).

Supervisor45868 was suspended at an owned saved boundary, worker214 drained, and checkpoint213 SHA `ec62b406f83b5a6f78a9b4979ccedfe4200b41bd1abc7c4d8cb35037ccd71c91` preserved. The diagnostic pause lasted3731.7seconds, including controlled physical tests and several matched PPO comparisons. R35 sources, checkpoints and logs remain intact. Eval214's supervisor wrapper includes this pause; its worker result is the appropriate physics timing record. [Pause receipt](diagnostic_pause.json).

## Reproduced teacher dependence

Five cohorts per command compared current R35, original teacher, a physically neutral head with actual feedback, neutral head with original simultaneous head feedback, and a compact context. Each cohort had32 robots, normal gravity/free base/self collision and unchanged CAD motor limits. At0.35/0.65/1.0, neutral physical head with actual feedback oversped to0.815/1.254/1.657m/s; original head histories restored0.436/0.733/1.053. At1.5, neutral head plus actual feedback turned70.10degrees with1.405m/s absolute lateral travel. Original head feedback restored0.876degrees and0.109m/s lateral while actual head remained near1.8degrees, with1.814% flight after the transient. This demonstrates learned feedback dependence, not absence of mechanical coupling. All observation rewrites/head overrides are diagnostic and absent from production. [Causal physics](prior_causal_response/result.json).

Constant context overspeeds lower commands and is rejected. A64-bin periodic head context derived from original teacher trajectories is interpolated using each history frame's saved phase. Neutral-head physical teacher replay produces signed0.434/0.729/1.051/1.485m/s at0.35/0.65/1.0/1.5, with1.781% genuine flight at1.5. The separate0.15 experiment reproduces0.454m/s with actual neutral-head feedback versus0.226 with original history; periodic context restores0.225m/s with0.95degrees actual head RMS. These validate diagnostic teacher dynamics, not a new inference controller. [Phase test](phase_causal_response/result.json), [0.15 phase test](015_phase_causal_response/result.json), [visually reviewed comparison](phase_causal_comparison.png).

Teacher queries on truthful CURRENT R35 states reconstruct current actor actions within6.2e-6 and show substantial lower-target differences from teacher head conditioning. Nominal joint-target error is before CAD projection, slew and torque dynamics. Labels on every future learner state are not certified by a successful teacher replay. [Target/gradient audit](teacher_target_audit.json).

## Three tested learning variants

All variants start from the exact213 full checkpoint,4096 environments, seed71, the existing learning-rate ceiling and identical reward/physics/PPO settings. Their independent nominal/pushed tests use192 robots for20seconds. No diagnostic learned checkpoint is deployed.

1. **Running-only final-body-row imitation:**19 CPU checks, same-actor physical parity and actual PPO/load tests pass. After64updates it increases flight from control0/0.0069% to0.490/0.576%, still below1.5%. After128updates it returns to0flight and0.65 still overspeeds. Reject this fragile running transfer. [64-update comparison](matched_comparison.json), [128-update comparison](matched_extension_comparison.json).
2. **Running imitation plus three conditioned prior commands:**0.35/0.65/1.0 improve but unconditioned0.15 overspeeds near0.268m/s. Running still has no qualifying flight. Reject this broader variant. [Partial-prior result](partial_prior_comparison.json).
3. **All four walking prior targets conditioned, no new running imitation:** the existing0.25 MSE strength,23-output denominator, stopped teacher and head/arm auxiliary are retained. Teacher-only head context changes targets on exact0.15/0.35/0.65/1.0 cohorts with five matching command frames, valid phase and latest actual head within10degrees. Student observations remain truthful. Shared-trunk effects of the original prior gradient and ordinary PPO remain possible; all gates stay mandatory. This variant passes stopped and every prior walking case in both tests, improves head error and is selected for exact full-state continuation.

| Command | Control nominal / pushed m/s | Corrected nominal / pushed m/s | Corrected pair |
|---|---|---|---|
| 0.00 | -0.000 / -0.000 | -0.000 / -0.000 | Pass |
| 0.15 | 0.192 / 0.194 | 0.180 / 0.181 | Pass |
| 0.35 | 0.403 / 0.399 | 0.382 / 0.377 | Pass |
| 0.65 | 0.747 / 0.748 | 0.682 / 0.680 | Pass |
| 1.00 | 1.086 / 1.089 | 0.995 / 0.996 | Pass |
| 1.50 | 1.346 / 1.357 | 1.420 / 1.419 | Flight fails |

All robots survive. Corrected moving head RMS is about3.1–3.5degrees versus control4.4–5.5degrees at prior speeds. Running geometric flight is0/0.0174%, still below1.5%; a walking-retention repair is not learned running. One64-update comparison does not establish long-term stability. [Matched source-bound comparison](all_prior_comparison.json).

## Validation and exact continuation

27 CPU checks cover input ownership, unchanged inference mean, exact MSE formula, no direct upper-row teacher copying, mixed-history/invalid-phase/nonneutral exclusions and unchanged stopped/running derivatives.12 migration checks reject bad checkpoint/source/signature/evidence/state/qualification declarations. Actual64-env4-update reload is exact for actor/critic/Adam/exploration, with zero pre-update KL and command mismatch. The selected-source, same-actor192-robot20-second replay is bitwise equal in all39 saved arrays and every case. [CPU checks](cpu_all_prior_tests.json), [migration checks](migration_cpu_tests.json), [full-state smoke](all_prior_full_state_smoke.json), [parity](all_prior_physical_parity.json).

Three frozen runtime files change: upper_body_refinement.py and new teacher_head_context.py/json.44 protected authored files, including worker, rewards, classifier, geometry/mass/contact, CAD Jacobians/motor caps, commands/history/actions and iteration4 execution, are byte-identical. The supervisor gains strict source/checkpoint/evidence-pinned migration and inventory coverage. The prior's derivative formula is unchanged, but its numerical targets/gradients are intentionally changed; inherited header wording must not be read as numeric gradient parity. The old teacher's qualification does not qualify conditioned labels or the current learner.

R36 starts at the original213 checkpoint, not at a diagnostic A/B checkpoint. Fresh capacity/PPO/load checks and old teacher requalification precede new main learning. All current and retention gates are repeated. [Protected inventory](all_prior_source_inventory.json), [strict continuation proof](teacher_context_continuation_proof.json).

## Resources, research and preserved failures

R35 completed-main throughput is43534 active collection-plus-PPO,37489 main-worker-effective and24478 full-chain samples/s before this pause, excluding earlier owned pauses. These denominators differ; diagnostic A/B samples are excluded. [Timing evidence](throughput.json).

Pre-pause6144 PPO left about1.7GiB GPU free. The old ramp required2211MiB to add2048 environments and correctly excluded eval memory from its hold. New capacity is measured afresh. Target8192,300MiB reserve, host fallback and180 continuous healthy live-training seconds remain. Exact CAD50Hz audit finds current ankle p99 below75% of cap with isolated peaks; diagnostic teacher has rare sampled right-ankle crossings. No sustained locomotion-relevant ceiling is established; substep attribution is untested. [Motor audit](motor_speed_audit.json).

[Ross, Gordon and Bagnell's primary DAgger paper](https://proceedings.mlr.press/v15/ross11a.html) describes the observation-distribution problem in sequential imitation and motivates testing labels under learner-induced states. The selected repair is a local training-target adaptation, not a complete DAgger implementation or a convergence guarantee.

Failed preparation/launch attempts remain saved: phase script was initially launched before preparation finished and no simulator started; an older trace lacked a requested time array; a completed probe PID was stale; one patch context and one inline quoting attempt failed before runtime. The raw0.15 causal result retains a cloned scope annotation listing earlier commands; its group command fields, script and telemetry all show0.15. This is a metadata erratum, not altered telemetry. An extra poll of an already completed tool session returned unknown session; completion artifacts were verified. No failed diagnostic actor enters production.

Full-CAD video requests were already fulfilled; no repeat or new GUI capture is claimed here. Preserve unrelated unsaved Fusion work. No5m/s or causally proven actuator-ceiling endpoint is reached. The same adaptive3h heartbeat remains active and the computer stays on. Next compare full matched prior retention, posture and flight; reject fragile transfer instead of changing arbitrary reward weights. Fresh main-learning verification and automation receipts follow below.


## Fresh deployed main evidence

Snapshot 2026-10-05T10:07:24.730744+00:00: 21 genuine main updates at8192, exact213 full actor/critic/Adam/exploration, zero KL/command mismatch, 94455 conditioned teacher samples, min GPU551MiB andhost6408MiB. One actual CUDA worker and49 hashes verified. Startup/capacity updates are excluded. These observations do not qualify running or current posture. [Main evidence](new_main_evidence.json), [full owned observation](final_owned_observation.json).


## First complete deployed block and independent pair

At 2026-10-05T10:13:30.497336+00:00, R36 has completed 64 genuine main updates at 8,192 environments. Exact actor, critic, Adam and exploration load from R35 checkpoint 213 is verified. All pre-update KL and observed/action command differences are zero. Teacher conditioning processed 290,015 samples. Minimum free GPU memory is 551 MiB and host available is 6355 MiB; private memory grows 7 MiB and Torch allocation grows 0 MiB. Active collection-plus-PPO throughput is 46651, main-worker-effective throughput is 40376, and worker-internal throughput is 45288 samples/s. These are block measurements, not whole-chain rates.

The exact saved actor from block 011 is evaluated in matching 192-robot, 20-second nominal 012 and pushed 013 tests, with at least 32 independent trials per command:

| Command | Nominal / pushed signed m/s | Nominal / pushed true flight | Matching pair |
|---|---|---|---|
| 0.00 | -0.000 / -0.000 | 0.0000% / 0.0000% | Pass |
| 0.15 | 0.189 / 0.189 | 0.0000% / 0.0000% | Pass |
| 0.35 | 0.391 / 0.387 | 0.0000% / 0.0000% | Pass |
| 0.65 | 0.673 / 0.675 | 0.0000% / 0.0000% | Pass |
| 1.00 | 0.988 / 0.988 | 0.0000% / 0.0000% | Pass |
| 1.50 | 1.370 / 1.374 | 0.0000% / 0.0000% | flight, geometric_flight |

One actual CUDA worker and all 49 frozen hashes verify. The subsequent operation is `014_run_150_train_002`. Preserve this learned state and judge later changes against these exact gates. The same adaptive three-hour automation is verified active; no 5 m/s or actuator-ceiling endpoint is established. [Complete deployed verification](deployed_gate_verification.json), [fresh owned snapshot](final_owned_observation.json), [automation verification](automation_verified.json).


Final metadata check 2026-10-05T10:16:12.546891+00:00: block `014_run_150_train_002` is training at 8192 with 39 observed updates, GPU free 551 MiB and host available 6479 MiB. The current checkpoint/hashes and matching gate references in NEXT_CHECK are refreshed. The same automation is verified ACTIVE with its three-hour cadence, thread, durable scope and notification preference preserved. History inventory contains 36 physical runs and 1409 operations, zero differing hashes and zero read warnings; eight historical missing manifest artifacts remain explicitly recorded. [Final metadata](final_metadata_verification.json).
