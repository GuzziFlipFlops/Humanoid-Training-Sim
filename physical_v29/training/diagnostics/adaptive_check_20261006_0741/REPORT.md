# Five-hour training check — 2026-10-06T08:29:47.914798+00:00

R37 independently qualified **3 m/s**, then encountered an artificial episode termination at3.5. The saved full learner is continued as **R39_commanded_travel_teacher_certificate**; no policy or optimizer is started over. Current3.5m/s remains unqualified because of direction and prior-skill failures. Training is healthy at the owned observation; read fresh ACTIVE_RUN/state for later progress.

## Independent progress and reproduced failure

Exact R37checkpoint349, matching350/351, passed every nominal/pushed stopped/prior gate through3m/s with signed2.9979/2.9982m/s. Later3.5 tests had zero end-of-rollout survival. `travel_guard_attribution.json` reproduces24 representative terminal events over12 exact gate pairs: all were ONLY `excessive_drift`, while upright near18s. Frozen environment terminated total displacement above60m regardless of command.3.5m/s for20s exceeds that cap. These events are **false physical-fall classifications**, not evidence of an actuator ceiling. The earlier user-directed schedule review remains historical.

The official [Unitree G1 velocity configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py) uses timeout, root-height and orientation termination. Our60m absolute-travel guard was a local addition; copied numerical reward bodies alone did not prevent this fault. The correction retains60m of excess travel margin beyond integrated actual planar action-command travel. The budget accumulates across command switches, is idempotent per control tick and resets only selected environments. Low height, tilt, invalid differential, nonfinite and nonfoot fault predicates remain. Every independent direction, speed, flight, contact and prior gate remains unchanged.

Corrected current459 uses320 robots,32 trials per command and20s, matching nominal/pushed. At3.5 all survive, signed3.530/3.535m/s, true geometric flight38.743/38.597%; **world lateral speed0.401/0.362m/s still fails**. Retained0.15 fails forward net travel,1.5 flight is1.323/1.427% below1.5%,2.5 fails foot slip. Stopped0.35/0.65/1.0/2.0/3.0 cases pass. `physical_validation.json` saves every case. These are baseline459 findings, never qualifications of later actors. Teacher349 freshly passes every prior gate through3 on corrected source.

The corrected termination/done semantics intentionally change episode endings and any downstream terminal-penalty/return effects after the former60m boundary. Numerical reward bodies and weights are preserved; the complete training trajectory is not asserted identical after that boundary.

## Repair, validation and continuation

79 CPU guard fixtures pass, covering commanded speeds through5m/s, switches, stops, row resets, idempotence and original physical faults. Fifteen negative migration fixtures reject wrong schema/checkpoints/hash, unprotected reward/gate/geometry changes, fabricated qualification and lost recovery. `physical_parity.json` records117 bitwise array comparisons: current459 is identical before each old false termination; nominal and pushed teacher349 are identical for their full20s. No trajectories, observation rewrites, head overrides or hybrids are deployed. A matched learning experiment is unnecessary for this deterministic fault predicate: unchanged-actor physical counterfactual/parity directly reproduces the error and isolates its correction; future behavioral refinements still require matched learning control.

Source-signature changes are explicitly declared. Only baseenvironment and worker loading provenance change inside the runtime tree; the supervisor gains a strict proof verifier and fresh teacher requalification.51 other runtime files are identical, including all CAD geometry/mass/contact, native caps, numerical reward kernels/weights, verifier,400-input23-output observations/actions, PPO, teachers/head refinement and iteration4 execution. The proof restricts old-signature loads to original459 and trusted349, verifies literal inverse patches and protected hashes, and binds CPU/physics/reload evidence. No generic source mismatch is accepted. Original459 SHA75b0a3ea651e5bb9872404b9df8ac0686eb38a24f04e3cf639ad90425ad92a08 supplies the exact actor,critic,Adam and trained exploration.16936 actual earlier main updates are preserved:9517 completed R37+58 verified recovered+7361 inherited. Recovery069→070 provenance is retained. Smoke/capacity actors are discarded.

The first simulator attempt rejected changed source signatures; a second PPO smoke correctly rejected the old-signature moving teacher after fresh physical requalification. Both failures/logs/verified owned cleanup are preserved. Narrow explicit checkpoint and teacher migration was then tested:4 genuine64-env PPO updates, zero pre-updateKL/command mismatch, all actor/critic/Adam/exploration tensors equal before training, and reload passes. The4s reload is a load smoke, **not** a qualifying gate. A diagnostic manifest-reader initially used the wrong supervisor path; corrected mapping matches the existing CPU recorder. No training evidence was overwritten.

R38 fresh3m teacher tests passed, but certificate lookup required a _gate_ filename and stopped before main learning. The failed frozen run is preserved. R39 changes only that supervisor filename label; a CPU regression reproduces the old lookup error and validates the new pattern and real diagnostic certificate. Fresh physics/full-load/capacity/teacher gates are repeated before production learning. No diagnostic actor is deployed.

R37 was suspended at the saved459 boundary and drained, then retired after healthy R38 source freeze/physics/full-load/reload, before its later pre-main certificate lookup failure. R39 subsequently repeated startup and requalification. Old learning-to-new supervisor gap1029.1s and suspension-until-retirement1230.1s are separately saved in `handoff_completion.json`; new startup/capacity clocks are separate. No unsaved user work was touched. At the earlier 08:29 observation, new completed main updates=0, stage=run_350, operation=011_run_350_train_001, operation count=8192, selected next count=8192. One actual CUDA worker and55 frozen hashes verify. Read `new_main_evidence.json` for actual update/load/memory rates; startup/capacity is not qualification. R39 freshly requalifies trusted349; the actual learning actor must still pass its own current and prior gates.

## Required upper joint-angle timelines

`corrected_upper` contains actual head yaw/pitch, shoulder pitch/**roll**, elbows and hips, physical target plots, separate velocity plots, SVG, CSV and JSON. Both corrected459 and fresh349 exact pairs are analyzed:32 robots per command at10Hz and16 selected representatives per gate at50Hz. Terminal/reset rows are excluded. Four metric fixtures pass. Both plots were visually inspected. An inefficient repeated compressed-NPZ read during diagnostic CSV export was replaced by one immutable array read; cached rerun finishes in seconds. This changes no training runtime.

At459/1m/s shoulder pitch ranges are37.51/27.68deg and ipsilateral phases177/153deg. At3m/s phases89/-177deg are markedly unequal; at3.5 ranges9.41/7.21deg and phases102/111deg show weak counter-swing despite alternating hips. Representative left shoulder is biased back relative to the right, with smaller pitch swings and synchronous roll tendencies at fast speeds. Head RMS is4.06deg at1,2.56 at2.5,2.88 at3 and3.78 at3.5; native yaw shows gait-linked oscillation. This is **not natural-arm certification**. Head native zero describes torso-relative alignment, not world gaze. Physical target/joint RMS upper error at3.5 is7.69deg; raw upper actions outside the clamp are0% in the selected trace, so raw clipping alone does not explain those arms. Target/slew and physical response still matter. Velocity spikes and smoothness statistics are descriptive50Hz sample diagnostics, not imported human thresholds.10Hz population statistics cannot diagnose faster jitter.

The [Mirelman et al. human study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) measured wrist/lower-back IMUs, arm amplitude, symmetry and jerk under different walking conditions. Our comparison is feature-based, with mapped native signs/canted hip axes and different morphology. No verified comparable-speed Unitree G1/H1 encoder trajectories were obtained; rest/reward configs and dance motions are not substituted. Exact human/Unitree angle-curve similarity remains unmeasured. Preserve learned head/legs and test any later arm refinement against matched control.

## Resources and native motor evidence

Reaudited R37 retained-main rates:44.17k active collection+PPO,38.23k worker-effective,25.62k whole-chain samples/s.679.7s of earlier diagnostic pauses are excluded; current measurement ends at suspension. These denominators differ. `throughput_and_loads.json` binds all completed/recovered samples and exact receiver proof; all sampled main pre-updateKL and action-command mismatches are zero. New R39 rates are separately saved, with discarded capacity trials excluded from learned progress.

`motor_speed_audit.json` maps all320 robots' saved10Hz physical q/qd through exact pose-dependent CAD Jacobians/reductions; directional finite-difference checks pass. At3.5 left ankle motor2 p99 is0.803cap, sampled at/above-cap fraction0.45%, maximum1.75cap; right ankle also has sparse crossings. These may include passive/inertial motion and **do not prove sustained driving clipping, torque limitation or a velocity ceiling**. No fresh5ms ceiling attribution was performed this check. A real high-speed plateau requires phase-specific native substep speed/torque evidence before the4m/s fallback. Target8192,300MiB reserve,host fallback,one worker,no stereo and180continuous healthy **training** seconds before ramps remain. Evaluation/startup free memory is not steady PPO capacity.

## Next focused work and automation

1. Compare newer exact nominal/pushed3.5 direction and all prior gates against corrected459. Preserve certified349; investigate real lateral/heading behavior and retained0.15 net travel,1.5 flight and2.5 slip if they stall. Do not classify the repaired displacement guard as solved locomotion.
2. Repeat/reset-segment upper actual-angle/target/velocity, bilateral coordination and arm-leg phase audits. Quantify effective upper targets, response and auxiliary/PPO gradients before a narrow matched-control full-state refinement; no head rollback, observation spoof or policy reset.
3. Verify55 hashes,one owned worker,full-state fallback/sample accounting,moving6144/8192 memory and healthy ramps. If native limits appear, attribute actual5ms motor speed/torque driving saturation rather than sparse spikes. No rejected candidate is revived without evidence.

The same heartbeat remains ACTIVE every FIVE hours, same thread/notification/GPT-6.1 Sol max instruction, with dated focus refreshed through the native update tool. [Official schedule guidance](https://learn.chatgpt.com/docs/automations) supports updating an existing scheduled task. Endpoint stays verified5m/s, or verified4m/s only with proven native limits blocking5, every matching nominal/pushed/stopped/prior gate and32 independent trials per command. At endpoint save,pause owned training and automation,brief; **computer stays on**. No endpoint is proven now.

Earlier 08:29 main-learning accounting: 0 completed updates plus 18 observed current partial PPO updates. Partial active rates/memory/load proof are in new_main_evidence.json; no partial block is called completed or qualified.


## Closing verification — 2026-10-06T08:43:23.754762+00:00

The saved actor014 after128 new main updates is now the latest reviewed actor, with exact matching015 nominal/016 pushed tests. All320 robots survive20s,32 trials per command. The repaired termination removes a false failure; direction and retention remain unresolved.3.5 signed speed is3.575/3.582m/s, but lateral motion still fails. Prior1.5 flight and2.5 hopping recover;0.15 net travel, nominal1.0 support and both3.0 lateral fail. Qualification is not inferred from speed or survival.

| Command m/s | Nominal signed | Pushed signed | Nominal failure | Pushed failure |
|---:|---:|---:|---|---|
| 0 | -0.000 | -0.000 | Pass | Pass |
| 0.15 | 0.093 | 0.095 | forward_net_travel | forward_net_travel |
| 0.35 | 0.324 | 0.324 | Pass | Pass |
| 0.65 | 0.644 | 0.645 | Pass | Pass |
| 1 | 0.982 | 0.978 | walking_support | Pass |
| 1.5 | 1.438 | 1.435 | Pass | Pass |
| 2 | 1.980 | 1.980 | Pass | Pass |
| 2.5 | 2.571 | 2.572 | Pass | Pass |
| 3 | 2.997 | 2.995 | world_lateral_speed | world_lateral_speed |
| 3.5 | 3.575 | 3.582 | lateral_speed, world_lateral_speed | lateral_speed, world_lateral_speed |


The first64-update results remain in `closing_gate_review.json` and `closing_upper`. The newest128-update pair and mandatory per-speed actual head/arm angle-target-velocity audit are in `final_gate_review.json` and `final_upper`. Both latest plots were visually inspected: asymmetric shoulder pitch bias, small fast pitch swing, larger roll motion, bent elbows and gait-linked head oscillation remain alongside alternating hips. HeadRMS at3.5 is3.241deg; this does not certify human-like arms or world gaze. No comparable-speed mapped Unitree encoder reference was obtained.

Latest liveness: sameR39 supervisor35688, one actual worker,55 hashes verified; block017 trains at8192.128 completed plus15 observed partial updates are separate. Two completed blocks load exact actor/critic/Adam/exploration, with zero pre-updateKL/action-command mismatch. Rates49.05k active/42.65k worker-effective/16.86k whole-chain samples/s; startup/capacity/fresh gates lower the separate whole-chain rate. Minimum main GPU402MiB, host roughly5GiB; memory stable. Target8192/reserve300/host guards/fallback/180healthy-training-second ramp retained.

Timeline correction to the earlier handoff: R37 was retired after healthy **R38** startup, before R38's later pre-main certificate filename failure. R39 then repeated startup and teacher checks. Saved-boundary-to-R39 supervisor1742.3s and first-reported genuine PPO2347.5s include diagnosis and repeated startup/capacity. There is no current silent pause. The CPU final-review command initially duplicated its relative output root; retry with absolute paths succeeds. This reporting failure changed no simulator source or training state.

The final same-thread native automation update verifies ACTIVE/FIVE-hour cadence and identical notification preference/durable endpoint. Next focus is reproduced lateral direction and prior retention, then upper target/phase diagnostics and native substep attribution if needed. The current full learner remains running; neither5m/s nor the conditional4m/s finish is proven.

Final CPU closing snapshot: 192 completed main updates; 018_run_350_gate_004_nominal is a 320-robot independent evaluation, distinct from8192 PPO capacity. Results beyond reviewed128-update actor014 are not asserted qualified. Retrospective inventory refreshed: 39 physical runs, 1965 operations, zero differing frozen hashes and zero read warnings; 8 old missing manifests remain documented. See final_close_liveness.json.
