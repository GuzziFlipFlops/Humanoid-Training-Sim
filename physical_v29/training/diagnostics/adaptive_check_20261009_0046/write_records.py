"""Save this check's evidence, chronological records and native update arguments."""
import hashlib,json,tomllib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
marker='adaptive_check_20261009_0046_record'
now=datetime.now(timezone.utc).isoformat()
live=read(OUT/'closing_cpu/live_handoff.json');v=read(OUT/'closing_learning_verification.json')
pair=read(OUT/'closing_gate_review.json')['latest']
resume=read(OUT/'diagnostic_resume.json');physical=read(OUT/'arm_target_response/result.json')
sham=read(OUT/'replay_sham_analysis.json');sham_resume=read(OUT/'replay_sham/diagnostic_resume.json')
validation=read(OUT/'physical_validation.json');decision=read(OUT/'decision.json')
coupling=read(OUT/'arm_coupling_analysis.json');target=read(OUT/'teacher_effective_target_analysis.json')
motor=read(OUT/'current_substep_motor_attribution.json');gradient=read(OUT/'current_gradient_analysis.json')
upper=read(OUT/'final_upper/upper_joint_time_audit.json');bias=read(OUT/'final_upper/upper_bias_audit.json')
before=tomllib.loads((OUT/'previous_automation.toml').read_text(encoding='utf-8'))
current_path=Path('C:/Users/rshan/.codex/automations')/before['id']/'automation.toml'
assert tomllib.loads(current_path.read_text(encoding='utf-8'))['prompt']==before['prompt']
assert v['inherited_actual_main_updates']==27507 and v['hash_proven_recovered_main_updates']==50
assert v['supervisor_pid']==39000 and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
assert validation['passed'] and not decision['deploy'] and physical['checkpoint_sha256']==resume['checkpoint_sha256']
assert read(OUT/'final_upper/audit_set.json')['all_plots_visually_inspected']
u=next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
ub=next(r for r in bias['records'] if r['test']=='nominal' and r['speed_m_s']==3.5)
assert u['sha256']==pair['sha256']
for name in ['final_motor_speed_audit.json','final_pushed_motor_speed_audit.json']:
 assert read(OUT/name)['passed'] and read(OUT/name)['source_checkpoint_sha256']==pair['sha256']
old=read(OUT.parent/'adaptive_check_20261008_1946/final_owned_learning_verification.json')
totals=lambda d:{k:sum(b[k] for b in d['blocks']) for k in ['updates','samples','active_seconds','worker_seconds']}
total=totals(v);previous=totals(old);delta={k:total[k]-previous[k] for k in total}
delta['chain_seconds_excluding_documented_pauses']=v['whole_chain_measurement_endpoint_unix']-old['whole_chain_measurement_endpoint_unix']-v['documented_diagnostic_pause_seconds_excluded']+old['documented_diagnostic_pause_seconds_excluded']
for key,denominator in [('active_samples_s','active_seconds'),('worker_effective_samples_s','worker_seconds'),('chain_samples_s','chain_seconds_excluding_documented_pauses')]:
 assert delta[denominator]>0
 delta[key]=delta['samples']/delta[denominator]
resumed=next(b for b in v['blocks'] if b['full_load_proof']['source_sha256']==resume['checkpoint_sha256'])
assert resumed['updates']==64 and resumed['all_preupdate_KL_and_command_mismatch_zero']
failed=[read(OUT/f'attempt0{i}/RECORD.json') for i in [1,2]]
last=v['blocks'][-1];a,b=pair['nominal_cases'][-1],pair['pushed_cases'][-1]
prior_failures={k:[dict(command=c['command'],failures=c['failures']) for c in pair[k+'_cases'][:-1] if not c['passed']] for k in ['nominal','pushed']}
next_tests=decision['next_tests'];assert len(next_tests)==3
summary=(f"Same R40 full learner/supervisor 39000 is running at {v['owned_snapshot_utc']}, with one actual worker and 55 verified frozen hashes. "
 f"It retains {v['completed_main_updates']:,} completed + 50 recovered = {v['actual_retained_main_updates']:,} current-run main updates, plus 27,507 inherited; {v['observed_partial_main_updates']} partial updates are separate. "
 f"Exact {Path(pair['checkpoint']).parent.name} at command 3.5 gives {a['signed_speed']:.3f}/{b['signed_speed']:.3f} signed and {a['lateral_speed']:.3f}/{b['lateral_speed']:.3f} lateral m/s; failures {a['failures']}/{b['failures']}. Prior failures: {prior_failures}. "
 "Saved gates bind that exact actor, not newer partial learning. "
 +decision['explanation']+' '+decision['diagnostic_summary']+' '
 +f"Required actual head/shoulder/elbow angle, target, velocity and slow-bias timelines were inspected at initial 827 and closing {Path(pair['checkpoint']).parent.name} across every command in both tests. Head RMS {u['head_rms_deg']:.2f} degrees; held arm biases and unequal motion remain. No human/Unitree encoder or optical-gaze certification. "
 f"Two failed probe interfaces resumed safely after 275.0/291.8 seconds; main diagnosis and sham resumed the same supervisor after {resume['pause_seconds']:.1f}/{sham_resume['pause_seconds']:.1f} seconds. "
 f"First resumed block {Path(resumed['folder']).name} completes 64 genuine updates at {resumed['environments']} with exact actor/critic/Adam/exploration, zero KL/command mismatch and all three auxiliaries at 20 batches/update. "
 f"Rates {v['active_PPO_samples_s']/1000:.2f}k active / {v['worker_effective_samples_s']/1000:.2f}k worker-effective / {v['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s; documented pauses excluded through each endpoint. "
 "No production source, reward, gain, cap, actor or optimizer reset and no diagnostic controller deployed. Original 35/1 gains, truthful 400/23 inference, teachers, upper refinement and iteration 4 remain. Same active five-hour endpoint; computer on.")
gate_table=['|Command m/s|Nominal signed|Pushed signed|Nominal / pushed failures|','|---:|---:|---:|---|']
for x,y in zip(pair['nominal_cases'],pair['pushed_cases']):
 gate_table.append(f"|{x['command']:g}|{x['signed_speed']:.4f}|{y['signed_speed']:.4f}|{', '.join(x['failures']) or 'pass'} / {', '.join(y['failures']) or 'pass'}|")
physical_table=['|Seed|Command|Mode|Survival|Signed m/s|Abs lateral m/s|True flight %|Full reward rate|','|---:|---:|---|---:|---:|---:|---:|---:|']
fmt=lambda value:'unavailable' if value is None else f'{value:.4f}'
for c in physical['groups']:
 if c['mode'] not in ['control','transition']:continue
 physical_table.append(f"|{c['seed']}|{c['command']:g}|{c['mode']}|{100*c['survival']:.1f}%|{fmt(c['signed_forward_m_s'])}|{fmt(c['absolute_lateral_m_s'])}|{100*c['genuine_flight_fraction']:.4f}|{fmt(c['mean_reward_rate'])}|")
target_table=['|Seed|Command|Window|Projected lower target RMS deg|Teacher joint/slew fraction|Context eligible|','|---:|---:|---|---:|---:|---:|']
for c in target['rows']:
 if not c['teacher_command_in_production_retention_contract'] or not c['samples']:continue
 target_table.append(f"|{c['seed']}|{c['command']:g}|{c['mode']}: {c['window']}|{c['projected_lower_target_RMS_deg']:.3f}|{100*c['teacher_lower_joint_clip_fraction']:.1f}% / {100*c['teacher_lower_slew_fraction']:.1f}%|{100*c['existing_head_context_eligibility_fraction']:.1f}%|")
g=gradient['batches']
report=f'''# Five-hour training check — October 9, 2026, 00:46 UTC

{summary}

## Decision and exact gates

{decision['explanation']}

{decision['diagnostic_summary']}

Matching pair: `{pair['checkpoint']}`, SHA256 `{pair['sha256']}`. Existing nominal/pushed 320-robot, 20-second classifiers remain unchanged. The usual nominal clones do not establish final independent initial-state diversity. The diagnostic paired states below are separate physical tests and never replace gates.

{chr(10).join(gate_table)}

This check's initial 827 pair failed 3 m/s direction; later 833 recovered that pair. Preserve each actor's evidence separately. Qualification is nonmonotonic. Preserve current full state, 605's separately passing 3.5 current-case evidence, 267's earlier all-prior pair and separately certified 349. No 3.5 stage, 4/5 endpoint or relevant actuator ceiling is inferred.

## Controlled arm/base response and truthful teacher histories

Exact diagnostic actor: `{physical['checkpoint']}`, SHA256 `{physical['checkpoint_sha256']}`. A repaired 68-robot, 2-second smoke test precedes 1,088 robots for 20 seconds: two seeds, 32 individually seeded nominal-physics q/qd states per cohort, paired between branches. All initial-state hashes and truthful command-history transitions are checked. Gravity, free base, self-collision, original 35/1 gains and native CAD limits remain. Initial q/qd writes occur only at reset.

After 5 seconds, two opposite signs of smooth native-angle 5-degree shoulder-pitch offsets are compared with unchanged control. A prescribed nonshoulder action stream tests replay; a second cohort uses its own truthful feedback. These external diagnostic controllers are never deployed or certified by the original reward signature. They cannot establish an additive separation of mechanical and learned effects.

{chr(10).join(physical_table)}

Paired per-robot differences, 5th/95th percentiles and both seeds are saved in [arm response](arm_coupling_analysis.json). Original body-action replay differences are exactly zero; actual target response and lower-policy feedback are measured rather than assumed. The plotted curves are 32-robot means at seed 4321; individual statistics are also saved.

![Actual arm and world-lateral response](arm_coupling_response.png)

The added 768-robot,20-second sham uses the same exact839 actor and two seeds,32 paired states per command/mode. Every zero-offset replay cohort falls too; all original controls survive. Joint states already diverge before5 seconds,and some slow-command zero-offset falls precede the offsets. Prescribed body-action replay is therefore rejected as mechanical isolation. Its falls cannot show that the arms prevent natural motion or establish an actuator ceiling. All replay streams and reset evidence are preserved; post-fall joint data are excluded from the plotted paired-live errors. See [replay sham](replay_sham_analysis.json).

![Explicit zero-offset replay control](replay_sham_control.png)

Original teacher 349 is loaded exactly and independently certified through 3 m/s. Passive labels reconstruct the complete existing action clamp, joint bounds, slew and CAD rejection map on current truthful histories. Student observations are never substituted. The 3.5 teacher extrapolation is explicitly outside the production retention contract and cannot supply a safe new label. Mixed histories retain four older 3.5 command frames at the switch; existing slow-walking context conditioning is excluded until its exact eligibility conditions hold.

{chr(10).join(target_table)}

[Teacher effective targets](teacher_effective_target_analysis.json) contain 16 phase bins and actual next-20-ms sole forces/clearance. The 10 Hz label sample is descriptive and does not identify a fixed physical delay or prove a gradient repair. [Physical validation](physical_validation.json) passes {validation['count']} checks, including finite synchronous COM/foot/joint/action data, complete reward sums, target parity and strict resume.

## Native motor and actual PPO evidence

Unmodified control cohorts alone record 4,000 native 5 ms calls, requested/applied motor torque, CAD Jacobian/reduction motor speeds and driving-clip flags. [Current substep attribution](current_substep_motor_attribution.json) is bound to the diagnostic actor and commands 0.15/1.5/3.5. Overspeed while braking and brief clips do not establish that limits block 5 m/s. Later closing-actor 10 Hz motor audits remain separately attributed. No native limit is raised.

Four genuine 4096-environment PPO updates pass exact actor/critic/Adam/exploration load, zero pre-update KL and action-command mismatch, and all three 20-batch auxiliaries. The diagnostic learned actor is discarded. Actual incoming derivatives are captured before standing/moving hooks; CPU reproduction independently cross-checks them. Mean-network prior norms range {min(x['prior_parameter_norm'] for x in g):.3f}–{max(x['prior_parameter_norm'] for x in g):.3f}, full PPO {min(x['all_PPO_parameter_norm'] for x in g):.3f}–{max(x['all_PPO_parameter_norm'] for x in g):.3f}, upper {min(x['upper_auxiliary_parameter_norm'] for x in g):.3f}–{max(x['upper_auxiliary_parameter_norm'] for x in g):.3f}. Running/prior cosine ranges {min(x['running_teacher_cosine'] for x in g):.3f} to {max(x['running_teacher_cosine'] for x in g):.3f}. These are pre-global-clipping/Adam contributions, not full optimizer steps or proof of universal gradient opposition. See [actual gradient analysis](current_gradient_analysis.json).

## Required upper joint angles over time

All 14 initial-827 and all 14 closing-pair angle/target/velocity/bias plots were visually inspected with saved hashes. Head yaw/pitch, shoulder pitch/roll and elbow data exist at every command. Population statistics use 32 robots per command at 10 Hz; 16 selected representatives per gate use 50 Hz. Resets and terminal samples are segmented. Sparse population traces cannot certify high-frequency jitter.

Closing exact actor head RMS {u['head_rms_deg']:.3f} degrees, p95 {u['head_p95_deg']:.3f}. Median shoulder pitch p95-p05 ranges {u['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{u['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f} degrees; slow means {ub['joints']['left_shoulder_pitch']['slow_bias_mean_deg']:.2f}/{ub['joints']['right_shoulder_pitch']['slow_bias_mean_deg']:.2f}. Hip cadence {u['coordination']['hip_cycle_hz']:.3f} Hz; ipsilateral shoulder/hip phase {u['coordination']['arms'][0]['phase_at_hip_cycle_deg']:.1f}/{u['coordination']['arms'][1]['phase_at_hip_cycle_deg']:.1f} degrees. Representative physical target/joint RMS error {u['representative_50hz']['target_joint_rms_error_deg']:.2f} degrees. The head 5-degree band is diagnostic only.

![Actual joint angles and physical targets](final_upper/upper_joint_angles.png)

![Actual joint velocities](final_upper/upper_joint_velocities.png)

![Slow bias and faster residual](final_upper/upper_bias_decomposition.png)

Held shoulder bias, unequal swing/elbow/roll and gait-linked head oscillation remain. The 0.5 Hz bias split is descriptive, excludes filter edges and does not certify natural swing. Existing upper-row refinement continues; no imposed reference curve or gain change is deployed. The primary [human arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) uses wrist/trunk IMUs rather than mapped shoulder encoders. [Official Unitree code](https://github.com/unitreerobotics/unitree_rl_mjlab) is not a verified speed-matched joint trajectory for this CAD. Comparison remains feature-based and morphology-limited; no reference curve is fabricated. Native-zero head is torso-relative. Historical Isaac/FK parity does not calibrate the CAD face/camera optical axis or certify world gaze.

## Preserved failures, safe resume and real progress

Attempt 01 failed before rollout because RSL-RL consumes nested configuration keys; separate deep copies correct the external probe. Attempt 02 failed before the first step because raw observations are a dict, while wrapped observations are TensorDict; storage-owning cloning supports both. Failed executed sources, logs and original checkpoint identities are preserved in attempt01/attempt02. Both resumed the original supervisor after 274.97 and 291.84 seconds. The first automatic attempt-02 resume status assertion failed; fresh identity/status inspection and strict retry succeeded. No production fault or training change is inferred. Nineteen CPU boundary/constraint/storage-ownership tests and the physical smoke pass before the larger experiment.

The successful bounded diagnosis suspends the same supervisor at a saved boundary, drains its worker, runs one diagnostic CUDA worker at a time and resumes it after {resume['pause_seconds']:.1f} seconds (drain {resume['worker_drain_seconds']:.1f}). The separately bounded replay-sham test pauses at the same839 checkpoint and resumes after {sham_resume['pause_seconds']:.1f} seconds (drain {sham_resume['worker_drain_seconds']:.1f}). First resumed `{Path(resumed['folder']).name}` completes 64 genuine main updates with exact full learned state and all KL/command checks zero. All later loads also verify. No silent pause remains.

Since the previous check: {delta['updates']:,} retained main updates, {delta['samples']:,} samples; {delta['active_seconds']:.1f} active seconds, {delta['worker_seconds']:.1f} worker seconds and {delta['chain_seconds_excluding_documented_pauses']:.1f} whole-chain seconds excluding newly documented suspensions. Interval rates {delta['active_samples_s']/1000:.2f}k/{delta['worker_effective_samples_s']/1000:.2f}k/{delta['chain_samples_s']/1000:.2f}k samples/s. Total current-run main samples {total['samples']:,}; retained updates {v['actual_retained_main_updates']:,} plus 27,507 inherited. Capacity/smoke/diagnostic samples and 32 unsaved interrupted R39 updates are excluded. Partial updates are separate.

Current-run rates {v['active_PPO_samples_s']/1000:.2f}k active / {v['worker_effective_samples_s']/1000:.2f}k worker-effective / {v['whole_chain_samples_s']/1000:.2f}k whole-chain; denominators differ. {v['documented_diagnostic_pause_seconds_excluded']:.1f} seconds of documented suspensions are excluded only through each measurement endpoint; an older untimestamped brief bookkeeping pause remains conservatively included. Last actual main block minimum free GPU {last['minimum_free_gpu_MiB']:.0f} MiB, host {last['minimum_host_available_MiB']:.0f} MiB, private growth {last['private_memory_growth_MiB']:.0f} MiB, Torch allocation growth {last['torch_allocation_growth_MiB']:.0f} MiB. One actual worker, 55 frozen hashes and 53 protected authored files verify. Target 8192, reserve 300 MiB, host fallback and 180 continuous healthy training seconds remain; evaluation memory never qualifies capacity. Fusion remains closed and queued work preserved. Full-body videos remain fulfilled.

## Next focused tests

{chr(10).join(f'{i+1}. {s}' for i,s in enumerate(next_tests))}

Same ACTIVE five-hour heartbeat/thread/notification preference. Endpoint remains actual ≥4.95 m/s under the declared tolerance with every unchanged gate and 32 individually seeded states per command, or actual ≥3.96 m/s with every gate and controlled native-limit evidence proving 5 m/s blocked. At that endpoint save, pause training/prevent restart, pause automation and brief with the computer on.
'''
evidence=dict(recorded_utc=now,summary=summary,decision=decision,latest_exact_pair=pair,
 owned_snapshot=str(OUT/'closing_cpu/live_handoff.json'),physical_checkpoint=resume['checkpoint'],physical_checkpoint_sha256=resume['checkpoint_sha256'],
 diagnostic_resume=resume,replay_sham_resume=sham_resume,replay_sham_analysis=sham,failed_attempts=failed,physical_validation=validation,first_resumed_main=resumed,
 mandatory_upper_actor=pair['checkpoint'],mandatory_upper_folder=str(OUT/'final_upper'),retained_main_totals=total,measured_since_previous=delta,
 production_change=False,computer_on=True)
history=[TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]
references=[Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
assert not (OUT/'REPORT.md').exists() and all(marker not in p.read_text(encoding='utf-8') for p in history+references)
write(OUT/'evidence.json',evidence);write(OUT/'failed_attempts.json',failed);(OUT/'REPORT.md').write_text(report,encoding='utf-8')
write(OUT/'resumed_learning_verification.json',dict(resume=resume,first_resumed_main_block=resumed,all_later_loads_exact=True,diagnostic_samples_excluded=True))
entry=f'\n<!-- {marker} -->\n## Five-hour check — October 9, 00:46 UTC — {now}\n\n{summary}\n\nReport: {OUT/"REPORT.md"}.\n\n'+ '\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests))+f'\n<!-- /{marker} -->\n'
for p in history:
 with p.open('a',encoding='utf-8') as f:f.write(entry)
for p in references:
 p.write_text(f'<!-- {marker} -->\n{summary}\nRead {OUT/"REPORT.md"}, then fresh ACTIVE_RUN/state. Earlier entries preserved.\n<!-- /{marker} -->\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
handoff=read(TRAIN/'automation/NEXT_CHECK.json')
historical={k:handoff.pop(k) for k in ['matched_retention_comparison','physical_retention_comparison','upper_comparison','arm_feedback_sensitivity','historical_diagnostic_paths','failed_diagnostic_attempts','final_owned_observation','final_verification','automation_verification','inventory_verification'] if k in handoff}
handoff.update(recorded_utc=now,report=str(OUT/'REPORT.md'),latest_observation=v['owned_snapshot_utc'],latest_liveness_utc=v['owned_snapshot_utc'],
 active_run=v['run'],current_stage=v['stage'],completed_main_updates=v['completed_main_updates'],new_main_updates=v['completed_main_updates'],
 recovered_main_updates=50,actual_retained_current_run_main_updates=v['actual_retained_main_updates'],partial_observed_main_updates=v['observed_partial_main_updates'],
 current_owned_operation=v['operation_path'],selected_actor=pair['checkpoint'],selected_actor_sha256=pair['sha256'],latest_matching_pair=pair,
 reviewed_speed_case_passes_both=a['passed'] and b['passed'],current_stage_qualified=False,running_qualified=False,velocity_ceiling_proven=False,
 summary=summary,focused_tests=next_tests,next_tests=next_tests,focus=next_tests,decision=str(OUT/'decision.json'),
 latest_liveness=str(OUT/'closing_cpu/live_handoff.json'),resumed_learning_evidence=str(OUT/'closing_learning_verification.json'),
 mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),upper_joint_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'),
 mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'),upper_target_bias_audit=str(OUT/'final_upper/upper_target_bias_audit.json'),
 mandatory_all_speed_traces=str(OUT/'final_upper'),mandatory_all_speed_trace_checkpoint=pair['checkpoint'],mandatory_all_speed_trace_sha256=pair['sha256'],
 arm_coupling_analysis=str(OUT/'arm_coupling_analysis.json'),teacher_effective_target_analysis=str(OUT/'teacher_effective_target_analysis.json'),
 current_gradient_analysis=str(OUT/'current_gradient_analysis.json'),synchronized_com_contact_audit=str(OUT/'final_com_contact_audit.json'),
 motor_speed_audit=str(OUT/'final_motor_speed_audit.json'),pushed_motor_speed_audit=str(OUT/'final_pushed_motor_speed_audit.json'),motor_audit_checkpoint=pair['checkpoint'],
 motor_substep_attribution=str(OUT/'current_substep_motor_attribution.json'),motor_substep_checkpoint=resume['checkpoint'],
 physical_coupling_checkpoint=resume['checkpoint'],physical_coupling_sha256=resume['checkpoint_sha256'],
 diagnostic_resume=str(OUT/'diagnostic_resume.json'),replay_sham_resume=str(OUT/'replay_sham/diagnostic_resume.json'),replay_sham_analysis=str(OUT/'replay_sham_analysis.json'),failed_diagnostic_attempts=str(OUT/'failed_attempts.json'),historical_diagnostic_paths=historical,
 previous_full_handoff_snapshot=str(OUT/'previous_NEXT_CHECK.json'),retained_main_totals=total,measured_since_previous=delta,
 gate_trends=str(OUT/'gate_trends.json'),retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
 resources=dict(last_completed_main=last,target_environments=8192,reserve_MiB=300,healthy_training_ramp_seconds=180,evaluation_memory_is_not_capacity=True),
 production_change=False,computer_on=True,history_inventory=str(OUT/'inventory.log'),automation_verified_active=False)
handoff.pop('closing_summary',None);write(TRAIN/'automation/NEXT_CHECK.json',handoff)
prefix=before['prompt'].split('Latest dated five-hour check',1)[0]
assert hashlib.sha256(prefix.encode()).hexdigest()=='46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
prompt=prefix+'Latest dated five-hour check October 9, 00:46 UTC: '+summary+'\nRead diagnostics/adaptive_check_20261009_0046/REPORT.md, decision.json, physical_validation.json, arm_coupling_analysis.json, teacher_effective_target_analysis.json, current_substep_motor_attribution.json, current_gradient_analysis.json, final_upper plots/data and closing_gate_review.json, then fresh ACTIVE_RUN/state.\n\nNext focused tests:\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests))+'\n\nPreserve the same ACTIVE five-hour heartbeat/thread/notifications and qualified 5 or qualified 4 plus causal native-limit endpoint. Save, pause both, prevent restart and brief with the computer on only at that verified endpoint. Preserve full learned state, forward head, completed videos and closed Fusion queued work.'
arguments=dict(id=before['id'],mode='update',kind='heartbeat',name=before['name'],prompt=prompt,rrule=before['rrule'],status='ACTIVE',targetThreadId=before['target_thread_id'])
if 'notification_policy' in before:arguments['notificationPolicy']=before['notification_policy']
write(OUT/'automation_update_args.json',arguments)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),reviewed_actor=Path(pair['checkpoint']).parent.name,retained_updates=v['actual_retained_main_updates'],interval=delta),indent=2))
