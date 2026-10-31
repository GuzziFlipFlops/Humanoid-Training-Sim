# Five-hour humanoid check — October 9, 2026, 15:48 UTC

The sustained stall is a failure to retain all skills together. This check freshly requalifies an earlier neutral-head checkpoint, measures its arm/body compatibility, and tests a new coupled teacher target on the current full learner. The candidate improves 1.5 m/s flight in both learning seeds, but introduces direction, retention and neck-motion costs. It is rejected for immediate deployment. The original full learner has resumed and completed a genuine 64-update block.

## Exact production actor and gates

The owned snapshot at 2026-10-09T18:23:43.255727+00:00 verifies supervisor 39000, one actual CUDA worker, all 55 frozen hashes and 53 protected authored worker files. R40 retains 22,567 completed plus 50 hash-proven recovered main updates = 22,617, in addition to 27,507 inherited updates. Separate partial updates at that snapshot: 0.

The fully reviewed actor is **1061_run_350_train_351**, SHA256 `d9970618093e7df8248453a9aa7d8e196a3dd2a03980c8ce37ec6fc436418e4d`. Its matching nominal/pushed tests each use 320 robots for 20 seconds, with 32 trials per command. All survive. Both 3.5 m/s current cases pass, at **3.544/3.548 m/s signed forward speed** and **0.100/0.106 m/s lateral speed**. The stage remains unqualified because both tests fail retained 0.15 m/s net travel, 1.5 m/s geometric flight and 2.5 m/s foot slip.

|Command m/s|Signed nominal/pushed m/s|Genuine flight N/P %|Nominal / pushed failures|
|---:|---:|---:|---|
|0|-0.0001/-0.0001|0.000/0.000|pass / pass|
|0.15|0.0784/0.0804|0.000/0.000|forward_net_travel / forward_net_travel|
|0.35|0.3198/0.3188|0.000/0.000|pass / pass|
|0.65|0.6576/0.6600|0.000/0.000|pass / pass|
|1|1.0280/1.0247|0.000/0.007|pass / pass|
|1.5|1.4805/1.4787|0.455/0.493|geometric_flight / geometric_flight|
|2|2.0615/2.0499|5.191/5.392|pass / pass|
|2.5|2.5954/2.5872|12.715/13.354|foot_slip / foot_slip|
|3|3.0465/3.0459|27.479/27.229|pass / pass|
|3.5|3.5440/3.5480|40.594/40.500|pass / pass|

The latest completed main block is **1064_run_350_train_352**. It loads exact 1061 actor, critic, Adam and exploration state; all pre-update KL and command mismatch checks are zero. Its newer gate pair is separate from the fully plotted 1061 actor. Saved qualification and posture findings never transfer to partial or newer actors.

Actor 1058 separately passes both current-speed cases at 3.511/3.517 m/s, with the same three retained failures. Preserve these useful actors, the freshly requalified 267 and separately certified 349. Ordinary nominal clones do not establish the independent initial-state diversity required at the final endpoint.

[Exact gates](closing_gate_review.json), [full-state loads and retained samples](final_learning_verification.json), [owned processes and hashes](final_cpu/live_handoff.json).

## Earlier checkpoint and truthful physical response

Earlier neutral-head actor **267**, SHA256 `20a133fdd4843118d816172239201a387ca947b38952fb38f094a156afa0a320`, freshly passes every stopped and retained-speed gate through 3 m/s under current original physics. Both ordinary nominal/pushed tests and separate tests with 32 individually generated initial states per command pass. Ordinary 1.5 m/s genuine flight is **2.208%/2.135%**. This certificate supplies no labels above 3 m/s.

The 640-robot, 20-second physical comparison uses two seeds, 32 paired initial states per controller/command, truthful feedback, a free base, gravity, self-collision, original gains and native limits. Every robot survives. At 1.5 m/s:

- Current actor 1055 gives about **0.387%/0.399%** genuine flight.
- Actor 267 body and arms with the current head gives **2.397% in both seeds**, with higher complete mean reward than current.
- Actor 267 body with the current whole upper controller gives **1.340%/1.348%**, with lower complete reward.

The body/arms plus current-head diagnostic also tracks about 0.187 m/s at the 0.15 command. Fresh actor 1058 repeats this bounded physical support before learning. These closed-loop component tests identify a coordination lead; they do not isolate mechanical coupling. Every composite remains diagnostic and outside production.

Recorded native 5 ms requested/applied torque and CAD motor-speed attribution excludes terminal/reset samples. All tested 0.15/1.5 cohorts have **zero driving-speed clips**. This evidence does not establish a limit at 3.5, 4 or 5 m/s. The preceding check's explicit 4/4.5/5 tests failed gait/direction with zero driving clips and remain bound to their earlier actor.

[Fresh certificate](neutral_teacher_certificate.json), [complete physical response](neutral_teacher_response_analysis.json), [same-anchor support](coupled_training_trial/matched_coupled_teacher/current_physical_support.json).

![Truthful physical compatibility](neutral_teacher_response.png)

## New training target and matched continuation

Candidate `neutral267_coupled_exact_command_candidate_v1` changes only training labels for settled exact 0.15/1.5 commands in all five history frames, valid phase and a current head within 10 degrees. Eligible truthful student states use freshly qualified 267 labels for 15 body and six arm derivatives. Both head derivatives are zero. Other commands retain original 349 labels and conditioning. The stopped teacher and upper refinement remain at 20 batches/update. Original 0.25 MSE weight, 23-output denominator and 256 selected-sample bound remain. The teacher does not enter inference. Shared hidden features can still change.

Twenty-three CPU eligibility, reset, finite-value and derivative checks pass. All 51 unchanged physical replay arrays and case reports are bitwise equal. Eight small PPO updates plus four full-state reload updates pass and are discarded. All matched branches start from the exact full 1058 checkpoint, SHA256 `48cec6d2bcffac7ba8fcbfdb18619babb73857a02f92ca6c091359dec2088a3c`.

The comparison uses 4,096 environments, learning seeds 229/241, unchanged controls at 64/128 updates, and matching 320-robot, 20-second nominal/pushed tests. Each 128-update actor also receives separate tests with 32 individually generated initial states per command. All 32 comparison operations independently validate completion, exact full-state loading, worker cleanup and reset provenance. Original production signatures do not certify the experimental training override; no diagnostic checkpoint enters production.

|Seed|Branch|Updates|1.5 genuine flight N/P %|0.15 path efficiency N/P|3.5 lateral N/P m/s|
|---:|---|---:|---:|---:|---:|
|229|control|64|0.111/0.149|0.7506/0.7467|0.2605/0.2358|
|229|control|128|0.111/0.115|0.6758/0.6640|0.2294/0.2164|
|229|candidate|64|0.438/0.500|0.8275/0.8201|0.2388/0.2294|
|229|candidate|128|0.889/0.806|0.8161/0.8049|0.2299/0.2146|
|241|control|64|0.059/0.201|0.7376/0.7321|0.1342/0.1281|
|241|control|128|0.000/0.035|0.8502/0.8408|0.1942/0.1814|
|241|candidate|64|0.896/0.861|0.7897/0.7785|0.3322/0.3206|
|241|candidate|128|0.792/0.733|0.8160/0.8054|0.2692/0.2494|

Flight gains replicate in both seeds, but every candidate remains below the **unchanged 1.5% minimum**. Slow path improvement does not replicate: seed 241 is worse than control at 128 updates. Seed 241 also loses the control's passing 3.5 direction at 64 updates and adds 3 m/s direction failures at 128. Individually varied initial states preserve these failures. Neck RMS rises in both seeds. Reject this version for immediate deployment and do not extend the identical target solely to gain flight. This result does not disprove every coupled-target refinement.

All **512 matched updates / 50,331,648 samples**, 12 smoke/reload updates and four passive diagnostic updates are excluded from main training progress. No production source, reward, gain, cap, actor or optimizer reset is deployed.

[Complete comparison](matched_coupled_teacher_review.json), [independent validation](experiment_validation.json), [target CPU checks](command_local_teacher_CPU_checks.json), [actual PPO/target gradient capture](command_local_gradient_analysis.json).

## Actual head and arm motion

The review visually inspects all 14 initial-1049 plots, two fresh-267 plots, four plots for each matched 128-update actor, and all 14 closing-1061 plots, with hash receipts. Actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets, velocities, bilateral amplitude, arm-leg phase, cadence, smoothness, saturation and noncancelling slow bias are saved by speed. Population sampling is 32 robots per command at 10 Hz. Sixteen selected representatives per gate have 50 Hz traces, with fewer representatives at some commands; resets and terminal segments are retained. Population sampling at 10 Hz cannot certify fast jitter.

At exact 1061/3.5 m/s, native neck RMS is **3.534 degrees**, p95 **5.219 degrees**. Median shoulder pitch ranges are **11.37/5.52 degrees**. Opposite held shoulder bias, unequal fast swing/roll/elbows and gait-linked head motion persist. Slower 2.5/3 m/s bias drift is separated from cyclic motion. For 1061, the displayed 2.5 means stay on opposite sides; earlier actors' sign reversals are not attributed to it. Existing upper fine-tuning continues at 20 batches/update and gains 35/1.

Four isolated, fixed-state target-only 1e-4 SGD probes slightly change head means through shared features despite zero direct head derivative. Restricting the derivative to the final 21 non-head output rows leaves fixed-state head means exactly unchanged. This local result does not prove the cause of the observed physical neck regression, protect all gait outputs, or certify a learning repair. Selected native world head-axis traces and neck RMS are recorded separately. Known Isaac/FK parity does not calibrate the original CAD face/camera axis.

![Actual angles and physical targets](final_upper/upper_joint_angles.png)

![Actual velocities](final_upper/upper_joint_velocities.png)

![Slow bias and faster residual](final_upper/upper_bias_decomposition.png)

Primary [human arm-motion research](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/) Figure 6 shows shoulder/elbow time plots from ten adults walking at 1.25 m/s. Numerical curves, comparable axes/rest poses and morphology mapping remain unverified. Reviewed official [Unitree motion assets](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/main/src/assets/motions) include dance CSVs, with no verified speed-matched locomotion encoder reference obtained here. Human/Unitree likeness and calibrated optical gaze remain uncertified. [Sequential imitation research](https://proceedings.mlr.press/v15/ross11a.html) motivates checking student-state coverage; this PPO auxiliary is not DAgger and inherits no formal guarantee.

[All four actors' population motion](upper_matched_comparison.json), [shared-feature probe](shared_target_head_effect.json), [native world-axis comparison](native_head_axis_comparison.json), [reference availability](updated_motion_reference_review.json), [final visual notes](final_upper_QA_notes.json).

## Preserved failures and safe recovery

The first CPU certificate reader imported `pxr` outside SimulationApp after completed physical tests. Literal gate-version reading repaired the interface while retaining strict validation. A float32 autograd fixture differed by 4.66e-10 due to arithmetic order; declared atol 1e-9 / rtol 1e-6 resolves the fixture without changing a gate. Failed logs remain.

The CPU reviewer falsely reported a head-index mismatch using URDF metadata order. The experimental hook correctly used observed Isaac order: head yaw/pitch rows 11/16 and history columns 20/25, 43/48 and 66/71. The unnecessary interrupted candidate evaluation, failed first driver and safe original resume are preserved. Five CPU/runtime/context checks pass. Production sources and the experimental derivative helper are unchanged. The remainder reuses only completed exact checkpoints and cleanup receipts from the same 1058 anchor, while separately preserving newer production 1061. [Review correction](joint_order_review_repair.json).

The final CPU validator expected a training-target annotation from reset-only diverse evaluations that execute the protected original worker. It now validates their actual interface, protected hash, argv and unique finite initial states. Ordinary wrapper evaluations still require the explicit `applied=false` annotation. All 32 operations pass independently; the failed review log remains. No runtime change or extra pause results. [Interface correction](diverse_validator_interface_repair.json).

Four saved pauses last **361.844, 585.866, 2082.649, 3528.374 seconds**, including drain. Each safely resumes the same supervisor and has a completed main receiver proving exact actor, critic, Adam and exploration loading. Evaluation wrappers spanning pauses do not count as training throughput. Exclusions apply only through each measurement endpoint.

## Actual samples, time and resources

R40 retained main totals are **22,617 updates / 3,613,851,648 samples**. Active collection/PPO time is 100868.0 seconds; worker-wrapper time is 121748.9 seconds. Since the preceding 13:24 check, it adds **1,216 retained updates / 239,075,328 samples**, over 6364.9 active, 7773.7 worker and 11519.6 whole-chain seconds excluding documented pauses. Incremental rates are 37.56k / 30.75k / 20.75k samples/s. Whole retained-chain rates are 35.83k / 29.68k / 19.05k. These denominators differ.

Latest genuine main block 1064 completes 64 updates at 8,192 environments, with minimum free GPU/host memory **1400/5269 MiB**, private growth 9 MiB and Torch allocation growth 0 MiB. All three auxiliaries remain at 20 batches/update. The 32 unsaved R39 updates, capacity, smoke and all diagnostic learning are excluded.

Synchronized action/target/joint/COM/contact data are finite and command aligned. Sparse 1061 motor crossings sampled at 10 Hz do not establish sustained driving saturation or inherit the earlier 0.15/1.5 substep attribution. [COM/contact audit](closing_com_contact_audit.json), [nominal CAD motor audit](closing_motor_nominal.json), [pushed CAD motor audit](closing_motor_pushed.json).

The 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy training seconds before upward ramps remain. Evaluation memory is not training capacity. Fusion stays closed, queued work and completed videos are preserved, and the computer stays on. No endpoint is established. Completion still requires all unchanged gates and 32 independently seeded states per command at measured 5 m/s (at least 4.95 under the declared 1% tolerance), or measured 4 m/s (at least 3.96) with controlled native-limit evidence proving 5 is blocked.

## Next focused tests

1. Preserve the current full learner, separately passing current-speed actors 1058/1061, the fresh neutral-head 267 all-prior certificate and certified 349. Compare fresh exact 3.5 direction, 0.15 path efficiency, 1.5 genuine flight and every retained gate. The coupled 267 target improves flight in both seeds but remains unqualified; seed 241 loses 3.5 direction at 64 and worsens slow path/adds 3 m/s direction failures at 128. Do not deploy those actors or extend the identical target solely for flight. Measure whether a narrow final-output-row derivative protects useful current-speed outputs, then compare it with the shared-feature route and unchanged control. Fixed-state head preservation does not guarantee gait retention. Any continuation needs exact full-state 64/128 controls, two learning seeds, independent initial states, every gate and versioned provenance. Keep existing weights and truthful inputs; never use 267/349 labels above their freshly verified 3 m/s certificate.

2. Inspect actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets and velocities at every speed, including bilateral amplitude, arm-leg phase and noncancelling slow bias. This coupled target has zero direct head derivative but changes shared features; an isolated 1e-4 SGD probe slightly changes fixed-state head means, whereas final-row-only derivatives leave them unchanged. Neither result proves the cause of physical neck regression or certifies a repair. Measure per-command effective targets, shared-feature/PPO/upper gradients and free-base response before another continuation, with current full-state controls, two learning seeds and all independent gates. Preserve forward head, gait, both teachers, upper refinement at 20 batches/update and 35/1 gains. Human/Unitree encoder likeness and calibrated optical gaze remain unavailable; no production hybrid, observation substitution, imposed curve or cap raise.

3. Verify supervisor 39000, one actual CUDA worker, 55 frozen hashes/53 protected authored files, local ground and exact actor/critic/Adam/exploration plus both teachers. Count 27,507 inherited updates plus completed and 50 verified R40 recoveries; exclude 32 unsaved R39 and all 512 matched, 12 smoke/reload and four passive diagnostic updates. Exclude the four saved diagnostic pauses only through each timing endpoint, including the reviewer interruption and remainder. Preserve the 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy TRAINING seconds; Fusion stays closed. This check’s native 5 ms probes cover 0.15/1.5 only and find no driving clips; later 10 Hz current-speed samples cannot inherit that attribution. Earlier 4/4.5/5 extrapolation fails gait/direction and proves no ceiling. Only qualified 5 m/s, or qualified 4 m/s with causal native limits blocking 5, ends training; every endpoint needs all unchanged gates and 32 independently seeded states per command.


<!-- adaptive_check_20261009_1548_closeout -->

Closing newer observation 2026-10-09T18:29:17.614731+00:00: the same R40 supervisor 39000 has one actual worker and all 55 frozen hashes/53 protected files. It retains 22,567 completed plus 50 recovered = 22,617 current-run main updates, plus 27,507 inherited; 44 observed partial updates in 1067 remain separate. Exact actor 1064, SHA 0ad6a1cc1c018951b82d05c67d02ca7a2663eb93845030e255a04503c60a61d0, passes both 3.5 m/s cases at 3.494/3.502 signed and 0.076/0.086 lateral m/s. Both retained 0.15 net travel, 1.5 geometric flight and 2.5 foot-slip tests still fail; 1.5 genuine flight is 0.132%/0.208%. Every other saved case passes and all survive. The stage remains unqualified. Full all-command upper plots and 10 Hz CAD motor/COM-contact audits stay bound to 1061, not 1064. This check’s native 5 ms attribution stays bound to the earlier 0.15/1.5 physical cohorts. Original 1064 loads exact 1061 actor/critic/Adam/exploration, zero KL/command mismatch and all three auxiliaries at 20 batches/update; current 1067 preserves exact 1064 loading. No experimental actor, target, reward, gain, cap or optimizer reset is deployed. Same active five-hour endpoint; computer on.

Read `closing_newer_pair.json` and `closeout_learning_verification.json` beside the dated report. Upper timeline QA remains explicitly bound to 1061.

Final owned observation 2026-10-09T18:34:59.215546+00:00: same supervisor, one actual worker and 55 unchanged hashes. Saved main count is 22,631 completed plus 50 recovered = 22,681, plus 27,507 inherited. Latest completed block 1067_run_350_train_353 has 64 genuine updates at 8,192, exact full loading and zero KL/command mismatch; all three auxiliaries remain at 20 batches/update. The reviewed gate pair remains 1064 and full upper/motor timeline QA remains 1061; no newer qualification is inferred. [Final owned receipt](final_liveness.json).
