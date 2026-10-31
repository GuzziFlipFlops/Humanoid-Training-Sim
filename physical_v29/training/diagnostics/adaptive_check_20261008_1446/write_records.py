"""Write one reviewed chronological entry and the same native heartbeat handoff."""
import hashlib,json,tomllib
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
marker='adaptive_check_20261008_1446_record'
live=read(OUT/'final_cpu/live_handoff.json');v=read(OUT/'final_learning_verification.json')
all_trace_pair=read(OUT/'final_gate_review.json')['latest']
pair=read(OUT/'closing_gate_review.json')['latest'];decision=read(OUT/'decision.json')
comparison=read(OUT/'matched_sequence_review.json');validation=read(OUT/'experiment_validation.json')
upper=read(OUT/'closing_upper/upper_joint_time_audit.json');resume=read(OUT/'diagnostic_resume.json')
failures=read(OUT/'failed_attempts.json');previous=read(OUT/'previous_NEXT_CHECK.json')
old_learning=read(previous['resumed_learning_evidence']);before=tomllib.loads((OUT/'previous_automation.toml').read_text())
assert live['state']['status']=='running' and live['state']['supervisor_pid']==39000
assert len(live['actual_worker_pids'])==1 and len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
assert comparison['completed'] and validation['passed'] and not decision['deploy']
assert v['inherited_actual_main_updates']==27507
assert (OUT/'final_upper/visual_inspection.json').exists()
assert (OUT/'closing_upper/visual_inspection.json').exists()
assert read(OUT/'final_upper/audit_set.json')['all_plots_visually_inspected']
for name in ['final_motor_speed_audit.json','final_pushed_motor_speed_audit.json']:
    audit=read(OUT/name)
    assert audit['passed'] and audit['source_checkpoint_sha256']==all_trace_pair['sha256']
u=next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
assert Path(u['checkpoint']).resolve()==Path(pair['checkpoint']).resolve() and u['sha256']==pair['sha256']
totals=lambda d:{k:sum(b[k] for b in d['blocks']) for k in ['updates','samples','active_seconds','worker_seconds']}
total=totals(v);old_total=totals(old_learning);delta={k:total[k]-old_total[k] for k in total}
delta['chain_seconds_excluding_documented_pauses']=v['whole_chain_measurement_endpoint_unix']-old_learning['whole_chain_measurement_endpoint_unix']-v['documented_diagnostic_pause_seconds_excluded']+old_learning['documented_diagnostic_pause_seconds_excluded']
for key,denominator in [('active_samples_s','active_seconds'),('worker_effective_samples_s','worker_seconds'),('chain_samples_s','chain_seconds_excluding_documented_pauses')]:delta[key]=delta['samples']/delta[denominator]
recovered=sum(b['updates'] for b in v['blocks'] if b['hash_proven_recovered'])
last=v['blocks'][-1];now=datetime.now(timezone.utc).isoformat();a,b=pair['nominal_cases'][-1],pair['pushed_cases'][-1]
prior_failures={k:[dict(command=c['command'],failures=c['failures']) for c in pair[k+'_cases'][:-1] if not c['passed']] for k in ['nominal','pushed']}
next_tests=decision['next_tests'];assert 1<=len(next_tests)<=3
summary=(f"The same R40 full learner and supervisor 39000 are running at {v['owned_snapshot_utc']}, with one actual worker and 55 verified frozen hashes. "
    f"It retains {v['completed_main_updates']:,} completed plus {recovered} hash-proven recovered = {v['actual_retained_main_updates']:,} current-run main updates, plus 27,507 inherited; {v['observed_partial_main_updates']} partial updates are separate. "
    f"Exact actor {Path(pair['checkpoint']).parent.name}, nominal/pushed: command 3.5 reaches {a['signed_speed']:.3f}/{b['signed_speed']:.3f} m/s signed speed and {a['lateral_speed']:.3f}/{b['lateral_speed']:.3f} m/s lateral speed; failures {a['failures']}/{b['failures']}. "
    f"Prior failures: {prior_failures}. Saved gates do not qualify newer actors. {decision['explanation']} "
    f"Head RMS is {u['head_rms_deg']:.2f} degrees. Actual upper angle/target/velocity, bilateral and arm-leg coordination, and noncancelling slow-bias timelines were inspected. Human/Unitree motion and optical gaze remain uncertified. "
    f"The same supervisor resumed after {resume['pause_seconds']:.1f} seconds; both failed diagnostic attempts also resumed safely and their evidence is preserved. "
    f"First resumed block 719 completed 64 genuine updates at 8,192 environments with exact full-state load, zero KL/command mismatch, stable memory and all three auxiliaries at 20 batches/update. Current operation is {Path(v['operation_path']).name}, count {v['current_environment_count']}. "
    f"Main rates are {v['active_PPO_samples_s']/1000:.2f}k active, {v['worker_effective_samples_s']/1000:.2f}k worker-effective and {v['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. "
    "Original 35/1 gains, native caps, reward v8/verifier v6, truthful 400-input/23-output inference, full learned state and all three 20-batch auxiliaries are retained. No diagnostic actor is deployed. The same five-hour automation remains active; computer on.")
table=['| Command | Nominal signed m/s | Pushed signed m/s | Nominal / pushed failures |','|---:|---:|---:|---|']
for x,y in zip(pair['nominal_cases'],pair['pushed_cases']):table.append(f"|{x['command']:g}|{x['signed_speed']:.4f}|{y['signed_speed']:.4f}|{', '.join(x['failures']) or 'pass'} / {', '.join(y['failures']) or 'pass'}|")
from render_report import render
report=render(locals())

evidence=dict(recorded_utc=now,summary=summary,owned_snapshot=str(OUT/'final_cpu/live_handoff.json'),
    latest_exact_pair=pair,comparison=str(OUT/'matched_sequence_review.json'),validation=validation,
    decision=decision,mandatory_all_command_upper=str(OUT/'final_upper'),mandatory_all_command_upper_checkpoint=all_trace_pair,
    latest_upper=str(OUT/'closing_upper'),failed_attempts=failures,
    retained_main_totals=total,measured_since_previous=delta,diagnostic_resume=resume,
    production_change=False,computer_on=True)
assert not (OUT/'REPORT.md').exists(),'Already recorded; preserve chronological entry'
write(OUT/'evidence.json',evidence);(OUT/'REPORT.md').write_text(report,encoding='utf-8')
entry=f'\n## Five-hour check — October 8, 14:46 UTC — {now}\n\n{summary}\n\n{decision["explanation"]}\n\nReport: {OUT/"REPORT.md"}.\n\nNext focused tests:\n\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests))+'\n'
for path in [TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    assert path.exists()
    with path.open('a',encoding='utf-8') as f:f.write(entry)
for path in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    text=path.read_text(encoding='utf-8');assert marker not in text
    path.write_text(f'<!-- {marker} -->\n{summary}\nRead {OUT/"REPORT.md"}, then fresh ACTIVE_RUN/state. Earlier entries remain preserved.\n<!-- /{marker} -->\n\n'+text,encoding='utf-8')
old=read(TRAIN/'automation/NEXT_CHECK.json')
historical={k:old.pop(k) for k in ['individual_upper_cycles','upper_state_context_sensitivity','diagnostic_actor_605_upper','closeout_verification','final_owned_observation','final_verification','automation_verification','inventory_verification'] if k in old}
old.update(recorded_utc=now,report=str(OUT/'REPORT.md'),latest_observation=v['owned_snapshot_utc'],latest_liveness_utc=v['owned_snapshot_utc'],
    active_run=v['run'],current_stage=live['state']['stage'],completed_main_updates=v['completed_main_updates'],new_main_updates=v['completed_main_updates'],
    recovered_main_updates=recovered,actual_retained_current_run_main_updates=v['actual_retained_main_updates'],partial_observed_main_updates=v['observed_partial_main_updates'],
    current_owned_operation=v['operation_path'],selected_actor=pair['checkpoint'],selected_actor_sha256=pair['sha256'],latest_matching_pair=pair,
    reviewed_speed_case_passes_both=a['passed'] and b['passed'],current_stage_qualified=False,running_qualified=False,velocity_ceiling_proven=False,
    summary=summary,focused_tests=next_tests,next_tests=next_tests,focus=next_tests,decision=str(OUT/'decision.json'),
    latest_liveness=str(OUT/'final_cpu/live_handoff.json'),resumed_learning_evidence=str(OUT/'final_learning_verification.json'),
    upper_joint_time_series=str(OUT/'closing_upper/upper_joint_time_audit.json'),mandatory_upper_audit=str(OUT/'closing_upper/upper_joint_time_audit.json'),
    mandatory_upper_bias_audit=str(OUT/'closing_upper/upper_bias_audit.json'),upper_target_bias_audit=str(OUT/'closing_upper/upper_target_bias_audit.json'),
    mandatory_all_speed_traces=str(OUT/'final_upper'),upper_sequence_comparison=str(OUT/'matched_sequence_review.json'),
    mandatory_all_speed_trace_checkpoint=all_trace_pair['checkpoint'],mandatory_all_speed_trace_sha256=all_trace_pair['sha256'],
    sequence_coverage_analysis=str(OUT/'sequence_coverage_analysis.json'),historical_diagnostic_paths=historical,
    synchronized_com_contact_audit=str(OUT/'final_com_contact_audit.json'),motor_speed_audit=str(OUT/'final_motor_speed_audit.json'),
    pushed_motor_speed_audit=str(OUT/'final_pushed_motor_speed_audit.json'),motor_audit_checkpoint=all_trace_pair['checkpoint'],
    previous_actor_motor_speed_audit=str(OUT/'motor_speed_audit.json'),
    diagnostic_resume=str(OUT/'diagnostic_resume.json'),failed_diagnostic_attempts=str(OUT/'failed_attempts.json'),previous_full_handoff_snapshot=str(OUT/'previous_NEXT_CHECK.json'),
    retained_main_totals=total,measured_since_previous=delta,
    retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
    resources=dict(last_completed_main=last,target_environments=8192,reserve_MiB=300,healthy_training_ramp_seconds=180,evaluation_memory_is_not_capacity=True),
    production_change=False,computer_on=True,history_inventory=str(OUT/'inventory.log'),automation_verified_active=False)
old.pop('closing_summary',None);write(TRAIN/'automation/NEXT_CHECK.json',old)
prefix=before['prompt'].split('Latest dated five-hour check',1)[0]
assert hashlib.sha256(prefix.encode()).hexdigest()=='46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
prompt=prefix+'Latest dated five-hour check October 8, 14:46 UTC: '+summary+'\nRead diagnostics/adaptive_check_20261008_1446/REPORT.md, decision.json, matched_sequence_review.json, experiment_validation.json, sequence_coverage_analysis.json, upper_matched_comparison.json, final_upper (all commands, exact 716), closing_upper and closing_gate_review (exact 719), final motor audits (716) and final_learning_verification.json, then fresh ACTIVE_RUN/state.\n\nNext focused tests:\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests))+'\n\nKeep the same ACTIVE five-hour heartbeat, thread, notification preference and verified 5 m/s or qualified 4 m/s plus causal native-limit endpoint. At that verified endpoint save, pause both, prevent restart, brief and leave the computer on. Fusion stays closed; queued work and completed videos are preserved. Preserve the full learned state and forward head.'
arguments=dict(id=before['id'],mode='update',kind='heartbeat',name=before['name'],prompt=prompt,rrule=before['rrule'],status='ACTIVE',targetThreadId=before['target_thread_id'])
if 'notification_policy' in before:arguments['notificationPolicy']=before['notification_policy']
write(OUT/'automation_update_args.json',arguments)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),retained_updates=v['actual_retained_main_updates'],reviewed_actor=Path(pair['checkpoint']).parent.name,decision=decision['verdict'])))
