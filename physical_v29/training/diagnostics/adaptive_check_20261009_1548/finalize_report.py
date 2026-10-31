"""Render the saved, verified evidence in readable prose; no runtime changes."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
read = lambda name: json.loads((OUT / name).read_text(encoding='utf-8-sig'))
evidence = read('evidence.json')
decision = read('decision.json')
verification = read('final_learning_verification.json')
pair = evidence['exact_pair']
upper = next(r for r in read('final_upper/upper_joint_time_audit.json')['records']
             if r['test'] == 'nominal' and r['command_m_s'] == 3.5)
total = evidence['retained_main_totals']
delta = evidence['measured_since_previous']
last = verification['blocks'][-1]
report = (OUT / 'REPORT.md').read_text(encoding='utf-8')
gate_table = report.split('|Command m/s|', 1)[1].split('\n\n', 1)[0]
experiment_table = report.split('|Seed|Branch|', 1)[1].split('\n\n', 1)[0]

text = f'''# Five-hour humanoid check — October 9, 2026, 15:48 UTC

The sustained stall is a failure to retain all skills together. This check freshly requalifies an earlier neutral-head checkpoint, measures its arm/body compatibility, and tests a new coupled teacher target on the current full learner. The candidate improves 1.5 m/s flight in both learning seeds, but introduces direction, retention and neck-motion costs. It is rejected for immediate deployment. The original full learner has resumed and completed a genuine 64-update block.

## Exact production actor and gates

The owned snapshot at {verification['owned_snapshot_utc']} verifies supervisor 39000, one actual CUDA worker, all 55 frozen hashes and 53 protected authored worker files. R40 retains {verification['completed_main_updates']:,} completed plus 50 hash-proven recovered main updates = {verification['actual_retained_main_updates']:,}, in addition to 27,507 inherited updates. Separate partial updates at that snapshot: {verification['observed_partial_main_updates']}.

The fully reviewed actor is **1061_run_350_train_351**, SHA256 `{pair['sha256']}`. Its matching nominal/pushed tests each use 320 robots for 20 seconds, with 32 trials per command. All survive. Both 3.5 m/s current cases pass, at **3.544/3.548 m/s signed forward speed** and **0.100/0.106 m/s lateral speed**. The stage remains unqualified because both tests fail retained 0.15 m/s net travel, 1.5 m/s geometric flight and 2.5 m/s foot slip.

|Command m/s|{gate_table}

The latest completed main block is **{Path(last['folder']).name}**. It loads exact 1061 actor, critic, Adam and exploration state; all pre-update KL and command mismatch checks are zero. Its newer gate pair is separate from the fully plotted 1061 actor. Saved qualification and posture findings never transfer to partial or newer actors.

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

|Seed|Branch|{experiment_table}

Flight gains replicate in both seeds, but every candidate remains below the **unchanged 1.5% minimum**. Slow path improvement does not replicate: seed 241 is worse than control at 128 updates. Seed 241 also loses the control's passing 3.5 direction at 64 updates and adds 3 m/s direction failures at 128. Individually varied initial states preserve these failures. Neck RMS rises in both seeds. Reject this version for immediate deployment and do not extend the identical target solely to gain flight. This result does not disprove every coupled-target refinement.

All **512 matched updates / 50,331,648 samples**, 12 smoke/reload updates and four passive diagnostic updates are excluded from main training progress. No production source, reward, gain, cap, actor or optimizer reset is deployed.

[Complete comparison](matched_coupled_teacher_review.json), [independent validation](experiment_validation.json), [target CPU checks](command_local_teacher_CPU_checks.json), [actual PPO/target gradient capture](command_local_gradient_analysis.json).

## Actual head and arm motion

The review visually inspects all 14 initial-1049 plots, two fresh-267 plots, four plots for each matched 128-update actor, and all 14 closing-1061 plots, with hash receipts. Actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets, velocities, bilateral amplitude, arm-leg phase, cadence, smoothness, saturation and noncancelling slow bias are saved by speed. Population sampling is 32 robots per command at 10 Hz. Sixteen selected representatives per gate have 50 Hz traces, with fewer representatives at some commands; resets and terminal segments are retained. Population sampling at 10 Hz cannot certify fast jitter.

At exact 1061/3.5 m/s, native neck RMS is **{upper['head_rms_deg']:.3f} degrees**, p95 **{upper['head_p95_deg']:.3f} degrees**. Median shoulder pitch ranges are **{upper['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{upper['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f} degrees**. Opposite held shoulder bias, unequal fast swing/roll/elbows and gait-linked head motion persist. Slower 2.5/3 m/s bias drift is separated from cyclic motion. For 1061, the displayed 2.5 means stay on opposite sides; earlier actors' sign reversals are not attributed to it. Existing upper fine-tuning continues at 20 batches/update and gains 35/1.

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

Four saved pauses last **{', '.join(f"{r['pause_seconds']:.3f}" for r in evidence['resumes'])} seconds**, including drain. Each safely resumes the same supervisor and has a completed main receiver proving exact actor, critic, Adam and exploration loading. Evaluation wrappers spanning pauses do not count as training throughput. Exclusions apply only through each measurement endpoint.

## Actual samples, time and resources

R40 retained main totals are **{total['updates']:,} updates / {total['samples']:,} samples**. Active collection/PPO time is {total['active_seconds']:.1f} seconds; worker-wrapper time is {total['worker_seconds']:.1f} seconds. Since the preceding 13:24 check, it adds **{delta['updates']:,} retained updates / {delta['samples']:,} samples**, over {delta['active_seconds']:.1f} active, {delta['worker_seconds']:.1f} worker and {delta['chain_seconds_excluding_documented_pauses']:.1f} whole-chain seconds excluding documented pauses. Incremental rates are {delta['active_samples_s']/1000:.2f}k / {delta['worker_effective_samples_s']/1000:.2f}k / {delta['chain_samples_s']/1000:.2f}k samples/s. Whole retained-chain rates are {verification['active_PPO_samples_s']/1000:.2f}k / {verification['worker_effective_samples_s']/1000:.2f}k / {verification['whole_chain_samples_s']/1000:.2f}k. These denominators differ.

Latest genuine main block 1064 completes 64 updates at 8,192 environments, with minimum free GPU/host memory **{last['minimum_free_gpu_MiB']}/{last['minimum_host_available_MiB']} MiB**, private growth {last['private_memory_growth_MiB']} MiB and Torch allocation growth {last['torch_allocation_growth_MiB']} MiB. All three auxiliaries remain at 20 batches/update. The 32 unsaved R39 updates, capacity, smoke and all diagnostic learning are excluded.

Synchronized action/target/joint/COM/contact data are finite and command aligned. Sparse 1061 motor crossings sampled at 10 Hz do not establish sustained driving saturation or inherit the earlier 0.15/1.5 substep attribution. [COM/contact audit](closing_com_contact_audit.json), [nominal CAD motor audit](closing_motor_nominal.json), [pushed CAD motor audit](closing_motor_pushed.json).

The 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy training seconds before upward ramps remain. Evaluation memory is not training capacity. Fusion stays closed, queued work and completed videos are preserved, and the computer stays on. No endpoint is established. Completion still requires all unchanged gates and 32 independently seeded states per command at measured 5 m/s (at least 4.95 under the declared 1% tolerance), or measured 4 m/s (at least 3.96) with controlled native-limit evidence proving 5 is blocked.

## Next focused tests

'''
handoff = json.loads((OUT.parents[1] / 'automation/NEXT_CHECK.json').read_text(encoding='utf-8'))
text += '\n\n'.join(f'{i + 1}. {s}' for i, s in enumerate(handoff['focused_tests'])) + '\n'
(OUT / 'REPORT.md').write_text(text, encoding='utf-8')
print('Rendered readable report from exact verified data; no runtime state changed.')
