# Five-hour check — October8 04:46 UTC

The same R40 full learner and supervisor 39000 are running at 2026-10-08T05:51:11.727972+00:00, with one actual worker and 55 verified hashes. It retains 11111 completed + 50 verified recoveries = 11161 current-run updates, plus 27507 inherited; 0 partial updates are separately unqualified. Exact actor 524_run_350_train_172, nominal/pushed: 3.5 command signed speed 3.388/3.398, lateral 0.184/0.163 m/s; failures ['world_lateral_speed']/['world_lateral_speed']. Prior failures: {'nominal': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 2.5, 'failures': ['foot_slip']}, {'command': 3.0, 'failures': ['world_lateral_speed']}], 'pushed': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 2.5, 'failures': ['foot_slip']}, {'command': 3.0, 'failures': ['world_lateral_speed']}]}. The new phase-balanced two-shoulder-row mean candidate is REJECTED for immediate deployment. No production source, reward, gain, cap or learned-state reset; no diagnostic actor deployed. The same supervisor resumed after 2782.9s. All comparison components were independently validated, but the harness exited1 at its final 3008.8/2600s guard. Required actual upper joint, target, velocity and bias timelines were inspected; head RMS 3.68 degrees. Human/Unitree motion and optical gaze remain uncertified. Main rates: 35.59k active / 29.86k worker / 18.77k whole-chain samples/s. Denominators differ and documented suspensions are excluded. Existing upper and both prior auxiliaries remain at 20 batches/update, gains 35/1 and full learned state preserved. Same ACTIVE five-hour endpoint; no5/4 or relevant ceiling proof.

## Decision and scope

The phase-balanced shoulder-mean candidate is rejected for immediate deployment. At128 updates its standard3.5m/s nominal/pushed lateral speeds are0.158/0.138m/s;nominal fails and the standard pushed case passes. Both independently seeded current cases still fail direction,with lateral0.156/0.154m/s. The0.15 net-travel and1.5 geometric-flight failures persist,and nominal2.5 still slips. At64 it added2.5 slip and3m direction failures where the matched control passed. More decisively for its intended arm goal,the representative3.5 shoulder held-bias measure is19.60deg versus16.76deg in control. The noncancelling slow RMS confirms this is a real held offset,not just an averaging artifact. Some lower-speed bias,head and128-update3m retention results improve,so the hypothesis is not declared universally ineffective. It does not demonstrate the intended current-speed upper-motion benefit with complete retention. No diagnostic actor or new gradient enters production;the existing upper refinement and original full learner continue. Actual training-bin means differ from closed-loop output means;that is a distribution/feature-response lead,not a proved runtime fault. All planned components were independently validated despite the harness final overall-time failure,and the same supervisor resumed safely.

No diagnostic checkpoint entered production. The same original R40 full learner, both teachers and existing head/arm refinement continue. The unchanged worker source inventory and native constraints remain protected; preserve original actor267's all-prior pair and separately certified original349 through3m/s. Saved gates bind their exact actors, not newer partial updates. Neither requested completion endpoint is established.

## Current exact matching pair

Checkpoint `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\524_run_350_train_172\candidate.pt`, SHA `76a01f188d18b10aa447bd7b5ea872dcb312ad666552b45cf7ef2e55851b309b`. Matching `525_run_350_gate_173_nominal` and `526_run_350_gate_173_push` use320 robots,20 seconds and32 trials per command. Repeated nominal clones are not final independently seeded diversity certification.

|Command|Nominal signed/lateral m/s|Pushed signed/lateral m/s|Result|
|---|---|---|---|
|0|-0.000/0.001|-0.000/0.019|pass both|
|0.15|0.078/0.085|0.081/0.089|forward_net_travel|
|0.35|0.324/0.083|0.325/0.086|pass both|
|0.65|0.670/0.070|0.672/0.077|pass both|
|1|0.974/0.084|0.972/0.091|pass both|
|1.5|1.445/0.072|1.444/0.079|geometric_flight|
|2|1.990/0.071|1.983/0.090|pass both|
|2.5|2.520/0.080|2.509/0.083|foot_slip|
|3|2.993/0.196|2.994/0.205|world_lateral_speed|
|3.5|3.388/0.184|3.398/0.163|world_lateral_speed|

## Tested shoulder continuation

The new experimental phase-balanced mean term detects shared opposite held shoulder pitch that the old comfortable bands allow. It uses the truthful last five frames, exact prior/frontier translation groups, eight equal-weight phase bins, at least two samples per bin, a declared5deg free mean band and the existing0.025 arm-loss scale. The additional final-layer gradient is capped at10% of actual PPO shoulder-row gradient norm. Only two final shoulder-pitch rows get a direct new derivative; existing head, roll, elbow, lower rows and shared-trunk auxiliary derivatives are untouched. Ordinary PPO, global clipping and physical coupling can still alter gait. No amplitude/phase/reference trajectory, observation rewrite, action override, support or cap raise is imposed. This is a local adaptation hypothesis, not copied human/Unitree numerical guidance.

Population phase balancing is not an individual-cycle guarantee: opposite individual biases can cancel.21 CPU fixtures cover this limitation, held versus sine sensitivity, phase/history eligibility, finite differences, effective clamp dead zones, passive control gradient equality, output-row isolation and norm cap. A failed import-path test and initial promotion-alias bookkeeping failure are preserved; neither is a production defect.

At paused exact524 SHA76a01f188d18b10aa447bd7b5ea872dcb312ad666552b45cf7ef2e55851b309b, the unchanged-actor320-robot20-second replay has all cases and51 arrays bitwise equal. Runtime seconds are separately nondeterministic. Four real64-env PPO updates and full-checkpoint reload pass with exact actor/critic/Adam/exploration load, zero pre-update KL and command mismatch. Smoke actors are discarded. Both full-state4096-env branches start the same checkpoint/exploration/critic/Adam and receive64 then128 updates at seed97. Existing upper and both priors remain20 batches/update. Every independent gate is unchanged.

|Branch|Updates|3.5 signed nominal/push|Lateral nominal/push|True flight nominal/push|Current failures|
|---|---|---|---|---|---|
|control|64|3.449/3.459|0.255/0.220|40.319/40.285%|['world_lateral_speed']/['world_lateral_speed']|
|control|128|3.396/3.403|0.154/0.141|38.181/38.319%|['world_lateral_speed']/['world_lateral_speed']|
|candidate|64|3.418/3.425|0.273/0.251|38.167/38.226%|['world_lateral_speed']/['world_lateral_speed']|
|candidate|128|3.467/3.475|0.158/0.138|39.472/39.358%|['world_lateral_speed']/[]|

Extra nominal-physics seed2301 and pushed seed1901 generate32 unique bounded q/qd states per command, identical between branches. This establishes the stated small-state diversity protocol for these diagnostic actors, not final qualification of the original learner. Original signatures do not certify the new gradient. Full results, actual gradient norms/eligibility, targets and per-command failures are in `matched_shoulder_review.json`, `upper_matched_comparison.json` and `experiment_validation.json`.

The comparison also saves noncancelling slow-component RMS and slow range: a near-zero time or population mean can hide slow reversals or opposing individual postures. `upper_training_eval_gap.json` compares actual logged minibatch phase-balanced means with final closed-loop effective output means. Different states, speed-band versus exact command and optimization times prevent a causal attribution; these discrepancies guide the next controlled response test.

## Actual upper motion

Initial518, control/candidate128 and the final original matching pair have inspected head yaw/pitch, shoulder pitch/roll and elbow actual/target/velocity timelines, bilateral coordination, ipsilateral hip phase, slow bias and saturation. Population32/cmd10Hz and16 selected50Hz representatives are distinct; resets are segmented and10Hz is insufficient to certify fast jitter. Current3.5 head RMS3.684deg; shoulder pitch ranges10.49/9.57deg and phases166.8/-145.4deg. Slow means, target bias, phase and amplitude must be interpreted together; absolute total span is not cycle swing.

The [primary human study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) supplies wrist-IMU amplitude/asymmetry/smoothness features, not morphology-matched shoulder encoder targets. The [official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab) does not provide an obtained, mapped speed-matched G1/H1 encoder curve for this comparison. None is fabricated. No human/Unitree motion certification. Native head zero is torso-relative; known Isaac/FK agreement does not calibrate the original CAD face/camera optical axis.

Saved `gate_com_contact_audit.json` retains initial518 synchronous50Hz actual COM, joints, physical targets, actions and contact/sole-force evidence, finite arrays and zero command mismatch. Moving COM offsets and conservative transformed-sole geometry remain diagnostics, not contact-pressure certification. No geometry or support-margin change was made. Earlier4285ms timing/cache parity and sparse elbow clips remain actor-specific context; no new relevant motor ceiling is established by this experiment.

## Safe continuation and resources

The same supervisor 39000 resumes after 2782.9 seconds including worker drain. Harness completion record present: False. The harness exited1 after every component finished at3008.8s because its final2600s overall guard included saved-boundary waiting/drain. The finally handler resumed the original supervisor. This is a failed operational harness exit,not a production fault or a successful exit. Independent full-state/gate/diversity evidence is validated separately. See the original harness and worker logs and `diagnostic_resume.json`. The original learner is running and no diagnostic actor is deployed. Fresh full-state recovery and actual main accounting are in `final_learning_verification.json`: 11111 completed + 50 proven recovered = 11161 retained new updates, plus 27507 inherited. The 32 unsaved interrupted R39 updates remain excluded; 0 current partial updates are not yet retained or qualified.

Completed retained-main denominators are35592 active collection+PPO,29860 worker-effective and18769 whole-chain samples/s. Exclude documented pauses only through each time endpoint; total excluded9141.8s. All capacity/smoke/diagnostic samples are excluded. Original five-foot55.5k collection-only samples/s used a different80-input22-output/implicit-PD/no-self-collision contract; this is not a matched benchmark.

Native8192 target,300MiB reserve, host guards/fallback and180 continuous healthy TRAINING seconds before upward ramp stay. Diagnostic/evaluation/startup free memory is not production steady capacity. Fusion stays closed; queued work and fulfilled full-body videos are preserved. Computer remains on. Existing five-hour heartbeat remains active until actual5m/s≥4.95 passes every matching gate, or actual4m/s≥3.96 plus all gates and controlled native-limit evidence proving5 is blocked, with32 independently seeded initial states per command.

## Next focused tests

1. Upper motion remains a priority. The new phase-balanced ensemble shoulder-mean candidate is rejected after exact64/128 controls and independent states. At128 its3.5 held-bias measure19.60deg exceeds control16.76deg,even though some slower/head/3m cases improve. Compare actual per-robot cycle and slow target means with truthful training-state coverage,hidden-feature response and per-command auxiliary/PPO gradients. The last candidate minibatch3.25-3.5band mean(-17.72,+1.08deg) differs from final exact3.5 effective outputs(-29.10,+14.63deg);these are different states/times,not a proved bug. Do not blindly raise weights/caps or repeat the same ensemble band. Any individual-cycle or state-conditioned continuation needs narrow output-row derivatives,full-state64/128 controls,all independent prior/current/head/direction/flight gates and seeded replication. Preserve forward head,legs,both teachers,35/1 gains and full learner. Inspect actual angle/target/velocity plus noncancelling slow RMS/range;do not mistake slow reversals for natural swing or fabricate Unitree encoder curves.
2. Compare newer exact original3.5 direction and all stopped/prior gates against524 and separately certified267/349. At524 signed3.388/3.398 and lateral0.184/0.163 remain failed;0.15 net travel,1.5 geometric flight,2.5 slip and3m lateral fail both. Direction improves from518 but retention is nonmonotonic. Use actual per-command advantage/teacher/effective-target and complete objective evidence to test a different bounded retention-preserving continuation before deployment. Existing course1,velocity/cycle filters,fixed-frame kernel,lateral exposure,teacher-context removal,stiffness70 and other rejected candidates stay rejected absent new evidence. No reward-identical waiting,spoof,hybrid,clock patch,tilted-head rollback or from-scratch reset.
3. Verify same39000,one actual worker,55 hashes,local ground,full actor/critic/Adam/exploration and both teachers. Keep27507 inherited plus completed and50 hash-proven R40 recoveries;32 unsaved R39 updates excluded. Exclude this2782.9s suspension only through each timing endpoint;525 nominal wrapper includes it. The comparison finished all components but exited1 at3008.8/2600s final guard;retain failed log and plan future diagnostic budgets separately from saved-boundary waiting/drain rather than falsely claiming success. Keep8192target,300MiBreserve,host guards/fallback and180 continuous healthy TRAINING seconds;Fusion stays closed. Current6144 steady GPU is about1.8GiB free,so do not force8192 from evaluation memory. Only qualified5 or qualified4 plus controlled native limits proving5 blocked ends training/automation;every endpoint needs all unchanged gates and32 independently seeded states/cmd. Use actual actor-specific5ms CAD motor velocity/torque and sustained driving saturation if a high-speed plateau appears;isolated elbow clips or falls are not a ceiling.

## Final continuation verification

{
  "recorded_utc": "2026-10-08T05:52:42.335935+00:00",
  "supervisor": 39000,
  "actual_worker": 47152,
  "worker_launcher_pids": [
    41372
  ],
  "operation": "E:\\Aditya\\robot_sim\\physical_v29\\training\\runs\\locomotion_20261007_r40_goal_budget_full_state\\529_run_350_gate_174_push",
  "status": "running",
  "sources_verified": 55,
  "resumed_completed_block": "E:\\Aditya\\robot_sim\\physical_v29\\training\\runs\\locomotion_20261007_r40_goal_budget_full_state\\527_run_350_train_173",
  "resumed_full_state_exact": true,
  "resumed_updates": 64,
  "resumed_zero_KL_and_command_mismatch": true,
  "resumed_memory": {
    "minimum_free_gpu_MiB": 1867,
    "minimum_host_available_MiB": 8423,
    "private_memory_growth_MiB": 8,
    "torch_allocation_growth_MiB": 0
  },
  "retained_main_snapshot": 11161,
  "inherited": 27507,
  "diagnostic_pause_seconds": 2782.9059574604034,
  "inventory": {
    "physical_runs": 40,
    "legacy_launches": 97,
    "original_rsl_sessions": 61,
    "cad_archive_groups": 33,
    "physical_operations": 2976,
    "missing_manifest_artifacts": 8,
    "differing_manifest_hashes": 0,
    "read_warnings": []
  },
  "native_automation_active_same_thread_5h": true,
  "production_change": false,
  "computer_on": true,
  "reviewed_matching_pair": "E:\\Aditya\\robot_sim\\physical_v29\\training\\runs\\locomotion_20261007_r40_goal_budget_full_state\\524_run_350_train_172\\candidate.pt",
  "qualification_scope": "Saved matching actor only; newer partial learning remains unqualified."
}
