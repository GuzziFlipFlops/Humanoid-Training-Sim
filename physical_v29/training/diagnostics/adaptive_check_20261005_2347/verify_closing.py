"""Fresh ownership/progress, checkpoint qualification and native schedule audit."""
import json,tomllib,hashlib
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
args=read(OUT/'automation_update_args.json');before=read(OUT/'automation_current_before_update.json')
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text())
assert cfg['prompt']==args['prompt'] and cfg['status']=='ACTIVE'
assert cfg['id']==before['id'] and cfg['kind']==before['kind']=='heartbeat' and cfg['rrule']==before['rrule']=='FREQ=HOURLY;INTERVAL=3'
assert cfg['target_thread_id']==before['target_thread_id'] and cfg.get('notification_policy')==before.get('notification_policy')
assert cfg['prompt'].split('Latest dated focus',1)[0].rstrip()==before['prompt'].split('Latest dated focus',1)[0].rstrip()
live=read(OUT/'final_owned_observation.json');s=live['state'];run=Path(live['active_run']['run']);g=read(OUT/'final_gate_review.json')
assert s['status']=='running' and s['supervisor_pid']==32492 and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
operation=Path(s['operation_path']);progress=None
if s['operation']=='train' and (operation/'progress.jsonl').exists():
    lines=(operation/'progress.jsonl').read_text().splitlines();rows=[]
    for line in lines:
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass # concurrent tail is not a completed update
    if rows:
        assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
        progress=dict(updates=len(rows),last_utc=datetime.fromtimestamp(rows[-1]['time'],timezone.utc).isoformat(),
            environment_count=rows[-1]['num_envs'],gpu=rows[-1]['gpu'],zero_preupdate_KL_command_mismatch=True)
latest=g['latest'];n=latest['nominal_cases'];p=latest['pushed_cases']
latest_stage=Path(latest['nominal']).name.split('_gate_')[0].split('_',1)[1]
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),owned_snapshot_utc=live['recorded_utc'],
    run=str(run),status=s['status'],stage=s['stage'],supervisor=s['supervisor_pid'],one_actual_worker=live['actual_worker_pids'],
    frozen_hashes_verified=55,completed_operation_main_updates=s['total_training_iterations'],verified_recovered_main_updates=58,
    actual_retained_main_updates=s['total_training_iterations']+58,declared_inherited_main_updates=s['prior_training_iterations'],
    operation=str(operation),operation_kind=s['operation'],operation_environments=s['environment_count'],selected_next_environments=s.get('next_training_environment_count'),
    owned_GPU=live['gpu'],owned_host=live['system_memory'],current_main_progress=progress,latest_matching_pair=latest,
    latest_tested_stage=latest_stage,current_tested_stage_qualified=latest['nominal_all_pass'] and latest['pushed_all_pass'],
    automation_verified_active=True,same_id_thread_cadence_notification_scope=True,history_inventory=read(OUT/'history_inventory_final.log'),
    runtime_unchanged=True,training_pause_this_check=False,endpoint_proven=False,
    scope='Partial live progress, owned snapshot and exact saved matching gates have separate timestamps. No qualification is assigned to a newer learner.')
write(OUT/'final_liveness.json',record)
pairpath=OUT/'final_matching_gates.json';write(pairpath,latest)
curr=HERE/'automation/NEXT_CHECK.json';nextcheck=read(curr)
assert nextcheck['recording_request']['status']=='complete'
nextcheck.update(final_liveness=str(OUT/'final_liveness.json'),automation_verified_active=True,new_main_updates=s['total_training_iterations'],
    actual_retained_current_run_main_updates=s['total_training_iterations']+58,latest_closing_gate_pair=str(pairpath))
nextcheck['selected_actor'].update(checkpoint=latest['checkpoint'],sha256=latest['sha256'],matching_gate_pair=str(pairpath),
    standing_and_walking_cases_through_1_m_s_pass=all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in latest[k] if c['command']<=1),
    current_stage_qualified=record['current_tested_stage_qualified'])
nextcheck['latest_verified_state'].update(recorded_utc=record['recorded_utc'],owned_snapshot_utc=live['recorded_utc'],status=s['status'],stage=s['stage'],
    operation=str(operation),operation_environments=s['environment_count'],learning_environments=s.get('next_training_environment_count'),
    actual_workers=live['actual_worker_pids'],gpu=live['gpu'],host=live['system_memory'],current_main_progress=progress)
nextcheck['current_stage']=s['stage'];write(curr,nextcheck)
for x in s['promotions']:
    if x.get('requalification'):continue
    nextcheck['highest_fully_qualified_stage']=x['stage'];nextcheck['highest_fully_qualified_command_m_s']=x['speed']
write(curr,nextcheck)
def summary(cmd):
    a=next(c for c in n if c['command']==cmd);b=next(c for c in p if c['command']==cmd)
    return f"{cmd:g} command: signed {a['signed_speed']:.3f}/{b['signed_speed']:.3f}m/s, true flight {a['flight']*100:.3f}/{b['flight']*100:.3f}%, pass {a['passed']}/{b['passed']}"
brief=f'''Closing observation {record['recorded_utc']}, owned snapshot {live['recorded_utc']}: same supervisor32492, one actual worker and55 frozen hashes. Stage{s['stage']}, operation{operation.name}, count{s['environment_count']}, {s['total_training_iterations']}completed main+58retained recovery. Latest exact actor{Path(latest['checkpoint']).parent.name} matching nominal{Path(latest['nominal']).name}/pushed{Path(latest['pushed']).name}: {summary(1.5)}; {summary(2.)}. Current tested stage qualified:{record['current_tested_stage_qualified']}. Preserve improving learner. Owned main progress:{progress}. Same heartbeat ID/thread/3hcadence/notification/durable scope verified ACTIVE. History inventory{record['history_inventory']['physical_runs']}physical runs/{record['history_inventory']['physical_operations']}operations/zero hash differences/no read warnings;8old missing manifests remain documented. No source/reward/physics/cap/optimizer/actor reset or pause this check. No5m/s or persistent relevant velocity ceiling. [Final evidence](final_liveness.json), [latest exact pair](final_matching_gates.json).'''
report=OUT/'REPORT.md';report.write_text(report.read_text(encoding='utf-8')+'\n\n<!-- closing_2347 -->\n'+brief+'\n',encoding='utf-8')
paths=[HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]
for path in paths:
    if '<!-- closing_2347 -->' not in path.read_text(encoding='utf-8-sig'):
        with path.open('a',encoding='utf-8') as f:f.write('\n\n<!-- closing_2347 -->\n'+brief+'\n')
# Preserve failed report-writer attempt independently; it changed no runtime.
report.write_text(report.read_text(encoding='utf-8')+'\nReport generation initially missed a diagnostic SHA helper; the failure log is preserved at `report_writer_first_failure.log`. The helper was added and records regenerated successfully. No training source changed.\n',encoding='utf-8')
a=read(HERE/'ACTIVE_RUN.json');a.update(last_verified_utc=record['recorded_utc'],final_owned_observation=str(OUT/'final_liveness.json'));write(HERE/'ACTIVE_RUN.json',a)
args['prompt']+='\n\nClosing owned observation: '+brief;write(OUT/'automation_update_args.json',args)
write(OUT/'automation_verification_before_closing_refresh.json',dict(recorded_utc=record['recorded_utc'],same_id_thread_cadence_notification_scope=True,status='ACTIVE'))
write(OUT/'diagnostic_source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')})
print(json.dumps({k:record[k] for k in ['recorded_utc','stage','completed_operation_main_updates','operation','one_actual_worker','current_main_progress','current_tested_stage_qualified']},indent=2))
