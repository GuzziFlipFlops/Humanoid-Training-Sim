# Five-hour training check — October 9, 2026, 05:47 UTC

The same full R40 learner continues. It reaches about 3.48 m/s at the 3.5 command, but the reviewed checkpoint still fails direction and several retained skills. Arm fine-tuning remains active. The new controlled experiments identify a closed-loop arm/leg coordination tradeoff; they do not supply a safe production repair or establish an actuator ceiling.

## Exact gates and decision

Keep the same full R40 learner and existing upper fine-tuning. Current3.5 cases passed at908/917 but reviewed920 regresses direction and prior retention. Controlled teacher/current upper transfers do not supply a safe repair; no new production source,reward,gain,cap,actor or optimizer change is justified. Preserve useful learned state and exact passing actor evidence.

Fully reviewed actor: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\920_run_350_train_304\candidate.pt`, SHA256 `1529b067974a7553d5bbfdde178058a24b9615b231c200c9bc3a08ac256b28db`; matching nominal921/pushed922 have320 robots,20 seconds and32 trials per command. All survive. Existing classifiers are unchanged. Nominal clones do not establish the independently generated initial-state diversity required at the final endpoint.

|Command m/s|Nominal signed|Pushed signed|Geometric flight nominal/pushed %|Nominal/pushed failures|
|---:|---:|---:|---:|---|
|0|-0.0000|-0.0000|0.000/0.000|pass / pass|
|0.15|0.1586|0.1612|0.000/0.000|forward_net_travel / forward_net_travel|
|0.35|0.3790|0.3786|0.000/0.000|bilateral_stance / pass|
|0.65|0.7436|0.7431|0.000/0.000|pass / pass|
|1|1.0210|1.0190|0.000/0.010|pass / pass|
|1.5|1.5286|1.5264|0.438/0.274|flight, geometric_flight / flight, geometric_flight|
|2|2.0869|2.0454|6.226/6.420|world_lateral_speed / world_lateral_speed|
|2.5|2.5334|2.5322|10.524/11.410|world_lateral_speed / world_lateral_speed|
|3|2.9536|2.9534|23.608/23.358|world_lateral_speed / world_lateral_speed|
|3.5|3.4788|3.4839|38.146/38.146|world_lateral_speed / world_lateral_speed|

Initial908 and later917 passed both3.5 current-speed cases;920 regresses direction. Earlier-speed retention remains nonmonotonic. Preserve those useful exact pairs,605 current-case evidence,267's earlier all-prior pair and separately certified349. Current-speed success alone cannot promote the whole stage. The owned snapshot has completed newer923 but its matching pair is incomplete at that observation; no923 gate or upper-motion qualification is inferred.

[Exact gate trends](gate_trends.json) compare current behavior with605/794/797/833/839/908/914/917/920. [Synchronized COM/contact/action audit](closing_com_contact_audit.json) verifies finite50Hz representative COM, foot forces, joints, physical targets and actions, zero command mismatch and no resets on reviewed920. Detailed mass-weighted COM coordinates are representative, while saved population metrics retain the existing full-population support classifier.

## Qualified teacher body and current upper controller

Physical actor remains exact917, SHA256 `58ec1bf8f971433a3bd21e20cc82b5ad0f8d0b74aa27355a0ce76e4b3de0e746`, across both experiments. Qualified349 is exact SHA256 `18e19ec3b558922a5927136d2b1b51fc3b68dc44a22df252e5630fce84c00a3e` and freshly certified under the current runtime through3 m/s. The newer920 full learner is preserved for resume; pinning917 for controlled diagnosis is not a production rollback.

The first experiment has1152 robots over20 seconds,36 cohorts, seeds4401/4411 and32 individually seeded small q/qd states per cohort, paired between controller branches. Its72-robot2-second smoke and13 CPU ownership/selector/projection checks pass first. Each controller reads its own actual400-input feedback/history. Diagnostic compositions split the23 outputs into body and upper rows; production inference never receives a composite. Context-conditioned variants modify only the explicitly identified diagnostic teacher input, while the current controller receives truthful observations. Teacher is never invoked above its3 m/s certificate; the3.5 sentinel uses current only.

All robots survive, commands match, effective-target reconstruction is exact and complete reward terms reconstruct within9.54e-7. Free base, gravity, self-collision,35/1 gains, geometry and native motor caps remain. There is no runtime root support, pose override, donor body-action replay or production observation substitution. Truthful3→0.15/1.5 transitions at6 seconds reproduce the same unfavorable tradeoffs; histories are not reset or fabricated at switches.

|Seed|Command|Controller composition|Signed m/s|Abs lateral m/s|True flight %|Complete reward rate|
|---:|---:|---|---:|---:|---:|---:|
|4401|0.15|current|0.1513|0.0845|0.0000|14.1818|
|4401|0.15|teacher|0.1842|0.0878|0.0000|14.2260|
|4401|0.15|teacher_body_current_upper|0.2847|0.1110|0.0000|13.8751|
|4401|0.15|current_body_teacher_upper|0.0289|0.0713|0.0000|13.8542|
|4401|0.15|context_teacher|0.1549|0.0752|0.0000|13.9420|
|4401|0.15|context_teacher_body_current_upper|0.2462|0.0773|0.0000|13.8914|
|4401|1.5|current|1.4969|0.0655|0.0000|14.3838|
|4401|1.5|teacher|1.4689|0.0741|1.8036|14.5470|
|4401|1.5|teacher_body_current_upper|1.5559|0.1256|0.9365|14.1312|
|4401|1.5|current_body_teacher_upper|1.4213|0.0488|0.9261|14.5389|
|4411|0.15|current|0.1513|0.0845|0.0000|14.1808|
|4411|0.15|teacher|0.1844|0.0876|0.0000|14.2255|
|4411|0.15|teacher_body_current_upper|0.2847|0.1110|0.0000|13.8753|
|4411|0.15|current_body_teacher_upper|0.0287|0.0713|0.0000|13.8533|
|4411|0.15|context_teacher|0.1547|0.0752|0.0000|13.9413|
|4411|0.15|context_teacher_body_current_upper|0.2460|0.0773|0.0000|13.8918|
|4411|1.5|current|1.4970|0.0656|0.0000|14.3840|
|4411|1.5|teacher|1.4691|0.0741|1.8313|14.5478|
|4411|1.5|teacher_body_current_upper|1.5560|0.1256|0.9469|14.1307|
|4411|1.5|current_body_teacher_upper|1.4212|0.0488|0.9122|14.5386|

Qualified349 still produces about1.81% genuine1.5 flight. Teacher body with current whole upper drops to about0.94% and loses complete return; at0.15 it overspeeds to about0.285m/s. Existing head-context conditioning does not make that whole-upper transfer safe. The inverse current body/teacher whole upper produces only about0.92% flight. These observations reject direct deployment of these tested compositions, not all future transfer approaches.

![Actual compatibility under own-state feedback](teacher_compatibility.png)

[Population and paired-state analysis](teacher_compatibility_analysis.json) saves per-robot differences,5th/95th percentiles, both seeds and transition windows. [Teacher effective target/history analysis](teacher_target_history_analysis.json) projects through the unchanged action clamp, joint bounds, slew and CAD map, with16 pre-action phase bins and next20ms sole response. Those instantaneous differences are not proof of an optimal delay, a safe gradient or a production numerical fault.

## Head versus arms: controlled component split

The second experiment has704 robots,22 cohorts,20 seconds, seeds4421/4431 and32 individually seeded paired states per cohort. A44-robot2-second smoke and9 CPU checks pass. Physical actor917 remains pinned, but the original supervisor resumes newer920 exactly afterward. All704 survive with exact target/command/reward checks.

`teacher_current_head` keeps teacher body and arms, selecting only current head yaw/pitch. `teacher_current_arms` keeps teacher body and head, selecting only the six current shoulder/elbow rows. `teacher_current_upper` selects both. Every branch remains in its own closed loop with actual feedback.

|Seed|Command|Controller composition|Signed m/s|Abs lateral m/s|True flight %|Complete reward rate|
|---:|---:|---|---:|---:|---:|---:|
|4421|0.15|current|0.1512|0.0845|0.0000|14.1812|
|4421|0.15|teacher|0.1842|0.0877|0.0000|14.2259|
|4421|0.15|teacher_body_current_upper|0.2846|0.1110|0.0000|13.8753|
|4421|0.15|teacher_current_head|0.1838|0.0878|0.0000|14.2444|
|4421|0.15|teacher_current_arms|0.2849|0.1079|0.0000|13.8826|
|4421|1.5|current|1.4969|0.0655|0.0000|14.3838|
|4421|1.5|teacher|1.4690|0.0741|1.8313|14.5475|
|4421|1.5|teacher_body_current_upper|1.5560|0.1257|0.9469|14.1305|
|4421|1.5|teacher_current_head|1.4657|0.0691|1.4775|14.5479|
|4421|1.5|teacher_current_arms|1.5551|0.1333|1.0856|14.1277|
|4431|0.15|current|0.1513|0.0845|0.0000|14.1807|
|4431|0.15|teacher|0.1844|0.0876|0.0000|14.2255|
|4431|0.15|teacher_body_current_upper|0.2847|0.1110|0.0000|13.8754|
|4431|0.15|teacher_current_head|0.1839|0.0878|0.0000|14.2444|
|4431|0.15|teacher_current_arms|0.2849|0.1078|0.0000|13.8834|
|4431|1.5|current|1.4969|0.0655|0.0000|14.3836|
|4431|1.5|teacher|1.4690|0.0738|1.8070|14.5479|
|4431|1.5|teacher_body_current_upper|1.5560|0.1251|0.9538|14.1312|
|4431|1.5|teacher_current_head|1.4658|0.0694|1.5018|14.5475|
|4431|1.5|teacher_current_arms|1.5553|0.1330|1.0648|14.1278|

Current arms on teacher body/head explain most0.15 overspeed and a larger1.5 flight loss. Current head alone preserves slow speed but gives1.478/1.502% flight, so it does not pass both seeds even as a diagnostic flight test. Arms-only current gives1.065/1.086%, below1.5%. Head effects are not absent. These controlled component changes include both physical dynamics and the lower controller's response to changed arm/head feedback; they do not isolate a purely mechanical cause. Student body with teacher arms-only while retaining current head has not yet been tested. No diagnostic composition enters production.

![Separated head and arm controller effects](head_arm_factorial/teacher_compatibility.png)

[Split population analysis](head_arm_factorial/teacher_compatibility_analysis.json) and [split effective targets](head_arm_factorial/teacher_target_history_analysis.json) preserve both seeds, actual angles, targets, velocities, root/contact/COM data and reset identities. This check claims no new64/128 candidate learning comparison.

## Motor limits and actual PPO derivatives

[917 native motor attribution](current_substep_motor_attribution.json) records4000 actual5ms calls at200Hz for the explicitly captured current, teacher and whole-upper compatibility cohorts. CAD Jacobians/reductions, requested/applied motor torque and directional available torque are preserved. All recorded driving-clip fractions and maximum durations are zero at0.15/1.5/3.5. Ankles can exceed the speed cap while braking; this is not sustained driving saturation. Available torque follows the frozen user-assumed flat-current/hard-speed BLDC and linear servo envelopes, not measured hardware voltage/current/thermal curves. The head-only/arms-only modes do not inherit uncaptured full-population motor attribution.

Reviewed920 sparse10Hz [nominal](closing_motor_nominal.json) and [pushed](closing_motor_pushed.json) audits use the actual interleaved case labels. At3.5 right ankle motor2 is at/above cap in2.22/2.12% of samples, with p99 about1.39/1.40 times cap. These different-actor sparse samples cannot inherit917's native driving/braking attribution. No persistent relevant ceiling or4/5 endpoint is established; no cap is raised.

Four genuine4096-environment diagnostic PPO updates load exact917 actor, critic, Adam and trained exploration, with zero pre-update KL/command mismatch and all three auxiliaries20 batches/update. The diagnostic learned actor is excluded from main progress and never deployed. [Actual gradient analysis](current_gradient_analysis.json) captures incoming derivatives before standing/moving hooks: prior norms2.301–2.598, full PPO21.315–59.691, upper2.279–2.502. Running/prior and upper/PPO cosines have both signs. Standing subtraction reproduces within5.71e-10 and moving mean-score cosine is approximately1. These are pre-global-clipping/Adam contributions, not universal opposition or causal repair evidence.

## Required upper angles, targets and velocities over time

All14 initial908 and14 reviewed920 plots were visually inspected, with hashes and notes. The [full audit](final_upper/upper_joint_time_audit.json) and CSV data include every command and both tests: head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets, velocities, bilateral coordination, ipsilateral arm/leg phase, cadence, smoothness and raw-action clamp diagnostics. Population sampling is32 robots per command at10Hz; high-rate traces are16 representatives per gate at50Hz. Each command's high-rate trace may contain fewer representatives;3.5 uses one. Resets/terminal rows are segmented;10Hz population statistics cannot certify fast jitter.

Reviewed920 nominal3.5 head RMS4.381deg,p957.002deg. Shoulder pitch median per-robot p95-p05 ranges11.74/10.25deg; slow means-17.52/5.71deg. Hip cycle2.667Hz, ipsilateral shoulder phases140.6/155.4deg. Representative upper physical target/joint RMS8.51deg; raw upper outputs outside clamp0.069%. Neither the5deg reporting band nor clamp statistics replace locomotion gates.

![Actual angles and physical targets](final_upper/upper_joint_angles.png)

![Actual velocities](final_upper/upper_joint_velocities.png)

![Slow bias versus faster residual](final_upper/upper_bias_decomposition.png)

![Commanded and measured slow bias](final_upper/upper_target_bias.png)

Opposite held fast shoulder biases, unequal roll/elbow motion and gait-locked neck oscillation remain. At2.5/3m/s physical target biases reverse slowly; total range is not periodic arm swing. The0.5Hz split excludes3-second edges, uses post-warmup time2.02s and is descriptive. Existing upper-only row refinement remains active20 batches/update; it has not certified natural arm motion.

Primary [Collins/Adamczyk/Kuo2009](https://pubmed.ncbi.nlm.nih.gov/19640879/) Figure6 contains published human shoulder/elbow angle and torque time plots. Its reported whole-body mechanical effects motivate component experiments; they do not prove this robot's cause. Exact speed, axes, rest poses and morphology mapping, and usable numerical reference curves remain unverified. [The human IMU study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) measures wrist/back motion, not mapped shoulder encoders. [Official Unitree code](https://github.com/unitreerobotics/unitree_rl_mjlab) has not supplied verified speed-matched encoder curves. Retrieval failures are preserved in [reference review](primary_motion_reference_review.json); no curve is fabricated. Comparison remains feature-based, and no human/Unitree likeness or calibrated optical gaze is claimed. Native head zero is torso-relative; historical Isaac/FK parity does not calibrate the CAD face/camera axis.

## Failed external attempts, safe resume and measured progress

The first external smoke stopped before rollout because it incorrectly required the original349 embedded reward signature to equal the newer corrected travel-guard runtime signature. The correction verifies exact349 SHA, matching asset/observation contract and complete fresh current-runtime certificate/gates, while keeping the current student's strict embedded signature. There is no generic signature bypass. Original executed source and failure logs remain in attempt01_signature_metadata; same supervisor resumed after126.955s. A stale smoke-result path was caught and canceled before the next saved-boundary pause; attempt02_preboundary_lookup never stopped the learner. A later CPU plot-receipt helper first used the wrong image location, then corrected the receipt path; the inspected plots were unchanged. These are diagnostic bookkeeping failures, not inferred production faults.

The compatibility pause lasts476.254s including108.801s worker drain. The component pause lasts371.378s including106.870s drain. One actual CUDA worker runs at a time. Resumed920 loads917 exactly and resumed923 loads920 exactly; both finish64 real main updates at8192. Both teachers and upper auxiliaries remain20 batches/update; every pre-update KL and observed/action command mismatch is zero. The original supervisor39000 is running; no silent pause or experimental controller remains. [Physical and recovery validation](physical_validation.json) passes60 cross-evidence checks, plus153/102 detailed physical-analysis checks,13/9 CPU fixtures and the physical/PPO smokes.

Since the previous check's recorded endpoint: 1,728 retained main updates and339,738,624 samples over9066.7s active,11075.1s worker and16619.9s chain excluding documented suspensions. Interval rates37.47k/30.68k/20.44k samples/s. Current-run totals3,022,454,784 samples,19,609 retained updates,plus27507 inherited. Full lineage has47,116 actual main updates. Capacity/smoke/diagnostic learning and32 unsaved interruptedR39 updates are excluded; partial progress is separate.

Current-run rates35.50k active/29.47k worker-effective/18.78k chain; denominators differ. 20901.6s documented pauses are excluded only through each timing endpoint. An older untimestamped short bookkeeping pause remains conservatively included. Last genuine main923 at8192 keeps at least1454MiB GPU and6228MiB host free,private+10MiB,Torch allocation+0MiB. All55 frozen hashes and53 protected authored worker files verify. Target8192,reserve300,host guards/fallback and180 continuous healthy TRAINING seconds before ramps remain. Evaluation memory is not training capacity. Fusion stays closed with queued work preserved; videos are fulfilled.

## Next focused tests and endpoint

1. Preserve the current full learner, the separately passing 3.5 m/s pairs 908/917 and 605, actor 267 with all prior cases passing, and separately certified 349. Compare fresh 3.5 m/s direction, 0.15 m/s net travel, 1.5 m/s genuine/geometric flight and every earlier gate. Reviewed 920 failed 2/2.5/3 m/s lateral direction; newer 923 recovers all three in both tests. Both 0.35 m/s stance and 0.65 m/s speed now fail, alongside the persistent 0.15 path and 1.5 flight failures. Preserve that actor-specific recovery and audit these fresh slow cases too. Two-seed component tests on exact 917 show that teacher legs with current arms overspeed at 0.15 and lose flight at 1.5; head changes also affect flight. This identifies closed-loop coordination, without isolating mechanical coupling. Next physically test the current body with qualified 349 arms only while retaining the CURRENT head, under each controller's truthful feedback. Resolve phase-specific effective leg targets, sole unloading and torso response on settled and transitioning histories. The inverse whole-upper test reached only about 0.9% flight; the arms-only inverse is untested. Never use 349 labels above its 3 m/s certificate. Select a command-local, state-conditioned arm-row or coupled transfer only after physical validation, exact current full-state 64/128-update controls, two learning seeds, all unchanged gates and independently seeded initial-state replication. Keep prior reweighting and rejected mean/gain/context/course candidates rejected without new evidence.
2. Every check inspect actual head yaw/pitch, shoulder pitch/roll and elbow angle, physical target and velocity timelines at every speed, including bilateral amplitude, arm-leg phase and noncancelling slow bias. Reviewed 920 has 4.38-degree head RMS and 11.74/10.25-degree shoulder pitch ranges, with opposite held fast biases and slow 2.5/3 m/s reversals. The component split attributes most slow overspeed and the larger 1.5 m/s flight loss to the current arm controller, but current head alone yields 1.478/1.502% flight and is not qualified. Do not deploy diagnostic composites or infer a purely mechanical cause. Preserve upper refinement at 20 batches/update, forward head, 35/1 gains, legs, both teachers and full learned state. Collins 2009 Figure 6 provides human shoulder/elbow time plots, but numerical curves and exact speed, axes, rest poses and morphology mapping remain unverified. Mapped Unitree encoder curves and calibrated optical gaze remain unavailable. Do not fabricate curves, substitute production observations, raise caps or restart from scratch.
3. Verify supervisor 39000, one actual worker, 55 frozen hashes, 53 protected authored files, local ground, exact actor/critic/Adam/exploration and both teachers. Count 27,507 inherited updates plus fresh completed and 50 hash-proven R40 recoveries; exclude 32 unsaved R39 updates and all smoke, capacity and diagnostic samples. Exclude the 126.955-second failed-metadata, 476.254-second compatibility and 371.378-second component-test pauses only through each timing endpoint. The canceled preboundary lookup never paused training. Resumed 920/923 load exact 917/920 respectively, with zero KL/command mismatch and all three auxiliaries at 20 batches/update. Keep target 8192, reserve 300 MiB, host guards/fallback and 180 continuous healthy TRAINING seconds; Fusion stays closed. Recorded 917 native 5 ms motor samples have zero driving clips and braking overspeed. The later 920 10 Hz ankle cap crossings cannot inherit that attribution. Only qualified 5 m/s, or qualified 4 m/s with causal native limits proving 5 blocked, ends training and automation; every endpoint needs all unchanged gates and 32 independently seeded initial-state trials per command.

Same ACTIVE five-hour heartbeat/thread/notification preference. Only actual≥4.95m/s with every unchanged nominal/pushed/stopped/prior gate and32 independently generated initial-state trials per command, or actual≥3.96m/s under those same requirements plus causal native limits proving5 blocked, ends training. At that endpoint save exact full state/hashes/gates/telemetry, pause owned training/prevent restart, pause this automation and brief with the computer on.

## Closing newer observation

Closing owned observation 2026-10-09T06:54:13.012311+00:00: the same R40 supervisor 39000 has one actual worker and 55 verified hashes. It retains 19,623 completed + 50 recovered = 19,673 current-run main updates, plus 27,507 inherited. Newer exact actor 923, matching nominal 924/pushed 925, reaches 3.527/3.532 m/s signed at command 3.5 with 0.180/0.165 m/s lateral drift; both direction gates still fail. Both 2/2.5/3 m/s cases recover all checks; preserve that improvement separately from 920. Both 0.15 net-travel, 0.35 bilateral-stance, 0.65 speed and 1.5 flight/geometric-flight tests fail. All robots survive. Full all-command upper timeline QA remains bound to 920, while native 5 ms teacher/component diagnostics remain bound to 917; no newer posture or motor qualification is inferred. Completed block 926 has 64 genuine updates at 8192, loads exact 923 actor/critic/Adam/exploration, and has zero pre-update KL/command mismatch with all three auxiliaries at 20 batches/update. Its minimum free GPU/host memory is 1454/6452 MiB, private growth 10 MiB and Torch allocation growth 0 MiB. Retained rates are 35.51k active / 29.48k worker-effective / 18.79k whole-chain samples/s. No production source, reward, gain, cap, actor or optimizer reset and no diagnostic controller deployed. Same active five-hour endpoint; computer on. Read closing_newer_pair.json, closeout_verification.json and fresh ACTIVE_RUN/state.

[Newer exact gate pair](closing_newer_pair.json), [fresh recovery evidence](liveness_learning_verification.json), [automation verification](automation_verification.json) and [CPU inventory](inventory_verification.json).
