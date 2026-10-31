import json,tomllib,shutil,time
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json');args=read(OUT/'automation_update_args.json');before=read(OUT/'automation_current_before_update.json')
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text())
assert cfg['prompt']==args['prompt'] and cfg['status']=='ACTIVE' and cfg['rrule']=='FREQ=HOURLY;INTERVAL=3'
assert cfg['id']==before['id'] and cfg['kind']==before['kind']=='heartbeat' and cfg['target_thread_id']==before['target_thread_id']
assert cfg.get('notification_policy')==before.get('notification_policy')
assert cfg['prompt'].split('Latest dated focus',1)[0].rstrip()==before['prompt'].split('Latest dated focus',1)[0].rstrip()
assert live['state']['status']=='running' and live['state']['supervisor_pid']==read(OUT/'diagnostic_pause.json')['supervisor_pid']
assert len(live['actual_worker_pids'])==1 and len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
assert time.time()-live['recorded_unix']<180
operation=Path(live['state']['operation_path']);assert live['state']['operation']=='train'
rows=[json.loads(x) for x in (operation/'progress.jsonl').read_text().splitlines()];assert rows
assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
assert rows[-1]['num_envs']==8192 and rows[-1]['gpu']['free_mb']>=300 and rows[-1]['gpu']['ram_available_mb']>=2048
paths=[HERE/'ITERATION_HISTORY.md',HERE/'automation/THREE_HOUR_CHECK.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]
assert all('adaptive_check_20261005_1746' in p.read_text(encoding='utf-8-sig') for p in paths)
shutil.copy2(HERE/'diagnostics/walk_regression_20261003/live_handoff.json',OUT/'final_liveness_owned_observation.json')
verified_utc=datetime.now(timezone.utc).isoformat()
record=dict(recorded_utc=verified_utc,owned_snapshot_utc=live['recorded_utc'],progress_last_utc=datetime.fromtimestamp(rows[-1]['time'],timezone.utc).isoformat(),status='running',main=live['state']['total_training_iterations'],operation=str(operation),
 partial_current_main_updates=len(rows),environments=rows[-1]['num_envs'],current_free_GPU_MiB=rows[-1]['gpu']['free_mb'],
 current_host_available_MiB=rows[-1]['gpu']['ram_available_mb'],one_actual_worker=live['actual_worker_pids'],same_supervisor=True,
 frozen_hashes_verified=55,zero_preupdate_KL_command_mismatch=True,automation_status='ACTIVE',
 same_id_thread_cadence_notifications_scope=True,history_inventory=read(OUT/'history_inventory_final.log'),
 documents_updated=[str(p) for p in paths],scope='Fresh liveness/partial progress, not a new independent qualification; exact matching pair remains in closing_matching_gates.json')
(OUT/'final_liveness.json').write_text(json.dumps(record,indent=2)+'\n')
nextpath=HERE/'automation/NEXT_CHECK.json';n=read(nextpath);n['final_liveness']=str(OUT/'final_liveness.json');n['automation_verified_active']=True;nextpath.write_text(json.dumps(n,indent=2)+'\n')
note=f"\n\n<!-- final_liveness_1746 -->\nClosing verification at {verified_utc}, using owned snapshot {live['recorded_utc']} and progress through {record['progress_last_utc']}: same supervisor and one actual worker, 55 frozen hashes verified. Operation `{operation.name}` has {len(rows)} observed genuine main updates at 8,192, with zero pre-update KL and command mismatch; free GPU {rows[-1]['gpu']['free_mb']} MiB and host available {rows[-1]['gpu']['ram_available_mb']} MiB. These are partial updates, not new qualification. Same automation ID/thread/three-hour cadence/notification preference/durable scope are verified active. History inventory: 37 physical runs, 1,542 operations, zero differing hashes and no read warnings; eight old missing manifests remain explicitly recorded. [Final evidence](final_liveness.json).\n"
path=OUT/'REPORT.md';text=path.read_text(encoding='utf-8');text=text.split('\n\nClosing liveness at ',1)[0].split('\n\n<!-- final_liveness_1746 -->',1)[0];path.write_text(text+note,encoding='utf-8')
print(json.dumps(record,indent=2))
