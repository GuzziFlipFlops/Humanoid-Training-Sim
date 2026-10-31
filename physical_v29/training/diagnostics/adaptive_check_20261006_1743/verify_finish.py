"""Verify saved documentation, same native heartbeat and current owned liveness."""
import json,tomllib,shutil,importlib.util,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
old=tomllib.loads((OUT/'automation_before.toml').read_text());p=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
auto=tomllib.loads(p.read_text());args=read(OUT/'automation_update_arguments.json')
for k in ['id','kind','name','rrule','status','target_thread_id','notification_policy']:assert auto.get(k)==old.get(k),(k,auto.get(k),old.get(k))
assert auto['prompt']==args['prompt'] and auto['rrule']=='FREQ=HOURLY;INTERVAL=5' and auto['status']=='ACTIVE'
assert auto['prompt'].split('Latest dated five-hour check')[0]==old['prompt'].split('Latest dated five-hour check')[0]
assert auto['target_thread_id']=='01a08b12-e85a-7f73-8677-315c708cd9f7'
shutil.copy2(p,OUT/'automation_after.toml')
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:assert 'adaptive_check_20261006_1743_record' in p.read_text(encoding='utf-8')
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:assert 'adaptive_check_20261006_1743' in p.read_text(encoding='utf-8')[:3000]
report=(OUT/'REPORT.md').read_text(encoding='utf-8');links=re.findall(r'\]\(([^)]+)\)',report)
for link in links:
 if not link.startswith('http'):assert (OUT/link).exists(),link
qa=read(OUT/'plot_qa.json')
for f in qa['files']:assert f['visually_inspected'] and hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
target=OUT/'final_cpu';target.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('owned_recorder',HERE/'record_owned_observation_handoff.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=target;m.main()
live=read(target/'live_handoff.json');s=live['state'];assert s['status']=='running' and s['supervisor_pid']==35688
assert len(live['actual_worker_pids'])==1 and len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
inventory=read(OUT/'history_inventory_stdout.json');assert inventory['differing_manifest_hashes']==0 and not inventory['read_warnings']
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),same_native_heartbeat=True,active=True,cadence_hours=5,same_thread=True,same_notification_preference=True,durable_scope_unchanged=True,
 report_links_verified=len(links),inspected_plot_hashes_verified=len(qa['files']),journals_and_handoffs_saved=True,history_inventory=inventory,
 status=s['status'],stage=s['stage'],operation=s.get('operation_path'),completed_operation_main_updates=s['total_training_iterations'],
 recovered_main_updates=33,actual_retained_current_main_updates=s['total_training_iterations']+33,inherited_main_updates_preserved=16936,
 actual_worker_pids=live['actual_worker_pids'],all55_frozen_hashes_verified=True,last_reviewed_actor=Path(read(OUT/'closing_gate_review.json')['latest']['checkpoint']).parent.name,newer_actor_qualification_not_inferred=True)
(OUT/'final_liveness.json').write_text(json.dumps(record,indent=2)+'\n')
nc=read(HERE/'automation/NEXT_CHECK.json');nc.update(final_liveness=str(OUT/'final_liveness.json'),closing_liveness=str(target/'live_handoff.json'),history_inventory=inventory,automation_verified_active=str(OUT/'final_liveness.json'))
nc.update(new_main_updates=s['total_training_iterations'],actual_retained_current_run_main_updates=s['total_training_iterations']+33,latest_verified_state=s,latest_observation=str(target/'live_handoff.json'))
nc['closing_progress_snapshot'].update(recorded_utc=record['recorded_utc'],operation=record['operation'],completed_main_updates=record['completed_operation_main_updates'],actual_retained_main_updates=record['actual_retained_current_main_updates'])
nc['latest_gate_trends']=str(OUT/'closing_gate_review.json')
nc['main_count_scope']='R39 completed-operation main plus33 hash-proven retained recovery;16936 inherited actual main including58 recovered R37. Diagnostic/smoke/capacity learning excluded. Partial updates never counted as completed.'
nc['actual_retained_count_status']='Verified R39 recovered33 full checkpoint and exact receiver; latest completed-operation total plus33 is separately bound. Earlier R37 recovery remains inside16936 inherited.'
nc['historical_qualification_contract_limit']='Preserved existing nominal-condition gates; nominal clones do not establish32 independently seeded initial-state trials required at final5/conditional4.'
(HERE/'automation/NEXT_CHECK.json').write_text(json.dumps(nc,indent=2)+'\n')
print(json.dumps(record,indent=2))
