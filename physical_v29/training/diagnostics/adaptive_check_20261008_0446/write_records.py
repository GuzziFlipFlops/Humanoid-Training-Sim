"""Evidence-backed final records and same-heartbeat update arguments."""
import json,hashlib,tomllib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
live=read(OUT/'final_cpu/live_handoff.json');v=read(OUT/'final_learning_verification.json');g=read(OUT/'final_gate_review.json');pair=g['latest']
d=read(OUT/'decision.json');comparison=read(OUT/'matched_shoulder_review.json');validation=read(OUT/'experiment_validation.json');resume=read(OUT/'diagnostic_resume.json')
assert d['production_change']==False and d['diagnostic_actors_deployed']==False and validation['passed']
assert live['state']['status']=='running' and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
upper=read(OUT/'final_upper/upper_joint_time_audit.json');u=next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
a,b=pair['nominal_cases'][-1],pair['pushed_cases'][-1]
summary=(f"The same R40 full learner and supervisor 39000 are running at {v['owned_snapshot_utc']}, with one actual worker and 55 verified hashes. "
    f"It retains {v['completed_main_updates']} completed + {v['hash_proven_recovered_main_updates']} verified recoveries = {v['actual_retained_main_updates']} current-run updates, plus 27507 inherited; {v['observed_partial_main_updates']} partial updates are separately unqualified. "
    f"Exact actor {Path(pair['checkpoint']).parent.name}, nominal/pushed: 3.5 command signed speed {a['signed_speed']:.3f}/{b['signed_speed']:.3f}, lateral {a['lateral_speed']:.3f}/{b['lateral_speed']:.3f} m/s; failures {a['failures']}/{b['failures']}. "
    "Prior failures: "+str({label:[dict(command=c['command'],failures=c['failures']) for c in pair[label+'_cases'][:-1] if not c['passed']] for label in ['nominal','pushed']})+'. '+
    f"The new phase-balanced two-shoulder-row mean candidate is {d['decision']}. No production source, reward, gain, cap or learned-state reset; no diagnostic actor deployed. "
    f"The same supervisor resumed after {resume['pause_seconds']:.1f}s. All comparison components were independently validated, but the harness exited1 at its final 3008.8/2600s guard. "
    f"Required actual upper joint, target, velocity and bias timelines were inspected; head RMS {u['head_rms_deg']:.2f} degrees. Human/Unitree motion and optical gaze remain uncertified. "
    f"Main rates: {v['active_PPO_samples_s']/1000:.2f}k active / {v['worker_effective_samples_s']/1000:.2f}k worker / {v['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. Denominators differ and documented suspensions are excluded. "
    "Existing upper and both prior auxiliaries remain at 20 batches/update, gains 35/1 and full learned state preserved. Same ACTIVE five-hour endpoint; no5/4 or relevant ceiling proof.")
next_tests=d['next_tests'];now=datetime.now(timezone.utc).isoformat()
evidence=dict(recorded_utc=now,summary=summary,current_owned_snapshot=str(OUT/'final_cpu/live_handoff.json'),learning_verification=str(OUT/'final_learning_verification.json'),
    exact_matching_gate_review=str(OUT/'final_gate_review.json'),current_upper=str(OUT/'final_upper/upper_joint_time_audit.json'),
    matched_comparison=str(OUT/'matched_shoulder_review.json'),upper_comparison=str(OUT/'upper_matched_comparison.json'),
    truthful_training_evaluation_gap=str(OUT/'upper_training_eval_gap.json'),
    validation=str(OUT/'experiment_validation.json'),decision=d,diagnostic_pause=resume,production_change=False)
write(OUT/'evidence.json',evidence)
tables=['|Command|Nominal signed/lateral m/s|Pushed signed/lateral m/s|Result|','|---|---|---|---|']
for n,p in zip(pair['nominal_cases'],pair['pushed_cases']):
    reason='pass both' if n['passed'] and p['passed'] else '; '.join(sorted(set(n['failures']+p['failures'])))
    tables.append(f"|{n['command']:g}|{n['signed_speed']:.3f}/{n['lateral_speed']:.3f}|{p['signed_speed']:.3f}/{p['lateral_speed']:.3f}|{reason}|")
review_lines=[]
for r in comparison['rows']:
    cs=r['nominal_cases'][-1];ps=r['pushed_cases'][-1]
    review_lines.append(f"|{r['branch']}|{r['updates']}|{cs['signed_speed']:.3f}/{ps['signed_speed']:.3f}|{cs['lateral_speed']:.3f}/{ps['lateral_speed']:.3f}|{cs['flight']*100:.3f}/{ps['flight']*100:.3f}%|{cs['failures']}/{ps['failures']}|")
report=f'''# Five-hour check — October8 04:46 UTC

{summary}

## Decision and scope

{d['explanation']}

No diagnostic checkpoint entered production. The same original R40 full learner, both teachers and existing head/arm refinement continue. The unchanged worker source inventory and native constraints remain protected; preserve original actor267's all-prior pair and separately certified original349 through3m/s. Saved gates bind their exact actors, not newer partial updates. Neither requested completion endpoint is established.

## Current exact matching pair

Checkpoint `{pair['checkpoint']}`, SHA `{pair['sha256']}`. Matching `{Path(pair['nominal']).name}` and `{Path(pair['pushed']).name}` use320 robots,20 seconds and32 trials per command. Repeated nominal clones are not final independently seeded diversity certification.

{chr(10).join(tables)}

## Tested shoulder continuation

The new experimental phase-balanced mean term detects shared opposite held shoulder pitch that the old comfortable bands allow. It uses the truthful last five frames, exact prior/frontier translation groups, eight equal-weight phase bins, at least two samples per bin, a declared5deg free mean band and the existing0.025 arm-loss scale. The additional final-layer gradient is capped at10% of actual PPO shoulder-row gradient norm. Only two final shoulder-pitch rows get a direct new derivative; existing head, roll, elbow, lower rows and shared-trunk auxiliary derivatives are untouched. Ordinary PPO, global clipping and physical coupling can still alter gait. No amplitude/phase/reference trajectory, observation rewrite, action override, support or cap raise is imposed. This is a local adaptation hypothesis, not copied human/Unitree numerical guidance.

Population phase balancing is not an individual-cycle guarantee: opposite individual biases can cancel.21 CPU fixtures cover this limitation, held versus sine sensitivity, phase/history eligibility, finite differences, effective clamp dead zones, passive control gradient equality, output-row isolation and norm cap. A failed import-path test and initial promotion-alias bookkeeping failure are preserved; neither is a production defect.

At paused exact524 SHA76a01f188d18b10aa447bd7b5ea872dcb312ad666552b45cf7ef2e55851b309b, the unchanged-actor320-robot20-second replay has all cases and{validation['physical_arrays']} arrays bitwise equal. Runtime seconds are separately nondeterministic. Four real64-env PPO updates and full-checkpoint reload pass with exact actor/critic/Adam/exploration load, zero pre-update KL and command mismatch. Smoke actors are discarded. Both full-state4096-env branches start the same checkpoint/exploration/critic/Adam and receive64 then128 updates at seed97. Existing upper and both priors remain20 batches/update. Every independent gate is unchanged.

|Branch|Updates|3.5 signed nominal/push|Lateral nominal/push|True flight nominal/push|Current failures|
|---|---|---|---|---|---|
{chr(10).join(review_lines)}

Extra nominal-physics seed2301 and pushed seed1901 generate32 unique bounded q/qd states per command, identical between branches. This establishes the stated small-state diversity protocol for these diagnostic actors, not final qualification of the original learner. Original signatures do not certify the new gradient. Full results, actual gradient norms/eligibility, targets and per-command failures are in `matched_shoulder_review.json`, `upper_matched_comparison.json` and `experiment_validation.json`.

The comparison also saves noncancelling slow-component RMS and slow range: a near-zero time or population mean can hide slow reversals or opposing individual postures. `upper_training_eval_gap.json` compares actual logged minibatch phase-balanced means with final closed-loop effective output means. Different states, speed-band versus exact command and optimization times prevent a causal attribution; these discrepancies guide the next controlled response test.

## Actual upper motion

Initial518, control/candidate128 and the final original matching pair have inspected head yaw/pitch, shoulder pitch/roll and elbow actual/target/velocity timelines, bilateral coordination, ipsilateral hip phase, slow bias and saturation. Population32/cmd10Hz and16 selected50Hz representatives are distinct; resets are segmented and10Hz is insufficient to certify fast jitter. Current3.5 head RMS{u['head_rms_deg']:.3f}deg; shoulder pitch ranges{u['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{u['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}deg and phases{u['coordination']['arms'][0]['phase_at_hip_cycle_deg']:.1f}/{u['coordination']['arms'][1]['phase_at_hip_cycle_deg']:.1f}deg. Slow means, target bias, phase and amplitude must be interpreted together; absolute total span is not cycle swing.

The [primary human study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) supplies wrist-IMU amplitude/asymmetry/smoothness features, not morphology-matched shoulder encoder targets. The [official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab) does not provide an obtained, mapped speed-matched G1/H1 encoder curve for this comparison. None is fabricated. No human/Unitree motion certification. Native head zero is torso-relative; known Isaac/FK agreement does not calibrate the original CAD face/camera optical axis.

Saved `gate_com_contact_audit.json` retains initial518 synchronous50Hz actual COM, joints, physical targets, actions and contact/sole-force evidence, finite arrays and zero command mismatch. Moving COM offsets and conservative transformed-sole geometry remain diagnostics, not contact-pressure certification. No geometry or support-margin change was made. Earlier4285ms timing/cache parity and sparse elbow clips remain actor-specific context; no new relevant motor ceiling is established by this experiment.

## Safe continuation and resources

The same supervisor 39000 resumes after {resume['pause_seconds']:.1f} seconds including worker drain. Harness completion record present: {comparison['harness_completion_record_present']}. {comparison.get('operational_status_explanation', 'Independent component validation and operational harness exit status are separate.')} See the original harness and worker logs and `diagnostic_resume.json`. The original learner is running and no diagnostic actor is deployed. Fresh full-state recovery and actual main accounting are in `final_learning_verification.json`: {v['completed_main_updates']} completed + {v['hash_proven_recovered_main_updates']} proven recovered = {v['actual_retained_main_updates']} retained new updates, plus 27507 inherited. The 32 unsaved interrupted R39 updates remain excluded; {v['observed_partial_main_updates']} current partial updates are not yet retained or qualified.

Completed retained-main denominators are{v['active_PPO_samples_s']:.0f} active collection+PPO,{v['worker_effective_samples_s']:.0f} worker-effective and{v['whole_chain_samples_s']:.0f} whole-chain samples/s. Exclude documented pauses only through each time endpoint; total excluded{v['documented_diagnostic_pause_seconds_excluded']:.1f}s. All capacity/smoke/diagnostic samples are excluded. Original five-foot55.5k collection-only samples/s used a different80-input22-output/implicit-PD/no-self-collision contract; this is not a matched benchmark.

Native8192 target,300MiB reserve, host guards/fallback and180 continuous healthy TRAINING seconds before upward ramp stay. Diagnostic/evaluation/startup free memory is not production steady capacity. Fusion stays closed; queued work and fulfilled full-body videos are preserved. Computer remains on. Existing five-hour heartbeat remains active until actual5m/s≥4.95 passes every matching gate, or actual4m/s≥3.96 plus all gates and controlled native-limit evidence proving5 is blocked, with32 independently seeded initial states per command.

## Next focused tests

{chr(10).join(f'{i+1}. {s}' for i,s in enumerate(next_tests))}
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
entry=f"\n## Five-hour check October8 04:46 UTC — {now}\n\n{summary}\n\n{d['explanation']} Report: {OUT/'REPORT.md'}. Controlled candidate, smoke, failed attempts, complete checkpoints and source hashes are preserved. No production change. Next: {' '.join(next_tests)}\n"
for p in [TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with p.open('a',encoding='utf-8') as h:h.write(entry)
marker='adaptive_check_20261008_0446_record'
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    p.write_text(f'<!-- {marker} -->\n{summary} Read {OUT/"REPORT.md"},then fresh ACTIVE_RUN/state. Earlier entries remain historical.\n<!-- /{marker} -->\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
old=read(OUT/'NEXT_CHECK_before.json')
next_record=dict(schema='adaptive_humanoid_check_v2',recorded_utc=now,automation_id='humanoid-training-research-and-repair-every-four-hours',
    active_run=v['run'],report=str(OUT/'REPORT.md'),latest_observation=v['owned_snapshot_utc'],latest_liveness_utc=v['owned_snapshot_utc'],
    highest_fully_qualified_stage='run_300',highest_fully_qualified_command_m_s=3.,current_stage=live['state']['stage'],current_stage_qualified=False,
    running_qualified=False,velocity_ceiling_proven=False,completed_main_updates=v['completed_main_updates'],new_main_updates=v['completed_main_updates'],
    recovered_main_updates=v['hash_proven_recovered_main_updates'],actual_retained_current_run_main_updates=v['actual_retained_main_updates'],
    inherited_main_updates_preserved=27507,partial_observed_main_updates=v['observed_partial_main_updates'],current_owned_operation=v['operation_path'],
    selected_actor=pair['checkpoint'],selected_actor_sha256=pair['sha256'],latest_matching_pair=pair,
    latest_liveness=str(OUT/'final_cpu/live_handoff.json'),resumed_learning_evidence=str(OUT/'final_learning_verification.json'),
    focused_tests=next_tests,next_tests=next_tests,focus=next_tests,summary=summary,decision=str(OUT/'decision.json'),
    upper_joint_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'),mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),
    mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'),upper_target_bias_audit=str(OUT/'final_upper/upper_target_bias_audit.json'),
    upper_comparison=str(OUT/'upper_matched_comparison.json'),matched_shoulder_comparison=str(OUT/'matched_shoulder_review.json'),
    upper_training_evaluation_gap=str(OUT/'upper_training_eval_gap.json'),
    synchronized_com_contact_audit=str(OUT/'gate_com_contact_audit.json'),diagnostic_resume=str(OUT/'diagnostic_resume.json'),
    retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
    resources=dict(target_environments=8192,steady_training_environments=6144,VRAM_reserve_MiB=300,host_guards=True,upward_ramp_continuous_healthy_training_seconds=180,one_actual_CUDA_worker=True,no_stereo=True),
    prior_source_and_history_scope='Preserve full R40 state and certified267/349;final nominal clones are not independent initial-state diversity.',
    recording_request=old.get('recording_request',{'status':'completed'}),fusion_user_preference=old.get('fusion_user_preference'),resource_release_report=old.get('resource_release_report'),
    native_automation_same_thread_five_hours=True,computer_on=True,production_change=False,
    endpoint_policy='Qualified5 command measured>=4.95m/s;or qualified4>=3.96m/s only with causal native limits blocking5. Every unchanged matching nominal/pushed/stopped/prior gate,32 independently seeded states/command. Save,pause training and this automation,brief,computer stays on.',
    previous_full_handoff_snapshot=str(OUT/'NEXT_CHECK_before.json'),history_inventory=str(OUT/'inventory_verification.json'))
write(TRAIN/'automation/NEXT_CHECK.json',next_record)
before=tomllib.loads((OUT/'automation_before.toml').read_text(encoding='utf-8'))
prefix=before['prompt'].split('Latest dated five-hour check',1)[0]
prompt=prefix+f"Latest dated five-hour check October8 04:46 UTC; {summary}\nRead diagnostics/adaptive_check_20261008_0446/REPORT.md,decision.json,matched_shoulder_review.json,upper_matched_comparison.json,experiment_validation.json,final_upper and final_learning_verification.json,then fresh ACTIVE_RUN/state.\n\nNext focused tests:\n"+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests))+'\n\nPreserve useful current full learner and all rejected evidence. Same ACTIVEfive-hour heartbeat/thread/notifications and durable verified5 or verified4+causal native-limit endpoint;save,pause both,brief,computer on. Fusion remains closed and queued work preserved;videos fulfilled.'
args=dict(id='humanoid-training-research-and-repair-every-four-hours',mode='update',kind='heartbeat',name=before['name'],prompt=prompt,rrule=before['rrule'],status='ACTIVE',targetThreadId=before['target_thread_id'])
if 'notification_policy' in before:args['notificationPolicy']=before['notification_policy']
write(OUT/'automation_update_args.json',args)
print(json.dumps(dict(summary=summary,report=str(OUT/'REPORT.md'),next_tests=next_tests),indent=2))
