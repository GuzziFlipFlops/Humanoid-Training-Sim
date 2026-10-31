"""Verify and record documentation, automation, inventory and resumed ownership."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
inventory=read(OUT/'history_inventory_receipt.json');auto=read(OUT/'automation_verification.json')
live=read(OUT/'closing_liveness/live_handoff.json');learning=read(OUT/'closing_learning_verification.json')
assert inventory['differing_manifest_hashes']==0 and not inventory['read_warnings']
assert inventory['missing_manifest_artifacts']==8 # Previously documented historical omissions.
assert auto['status']=='ACTIVE' and auto['same_thread'] and auto['five_hour_cadence'] and auto['durable_scope_endpoint_prefix_unchanged']
assert live['state']['status']=='running' and live['state']['supervisor_pid']==39000
assert len(live['actual_worker_pids'])==1 and len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
marker='<!-- adaptive_check_20261007_1344_record -->'
docs=[TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
assert all(p.read_text(encoding='utf-8-sig').count(marker)==1 for p in docs)
upper=read(OUT/'final_upper/upper_joint_time_audit.json');bias=read(OUT/'final_upper/upper_bias_audit.json');qa=read(OUT/'plot_qa.json')
for name in ['upper_joint_angles.png','upper_joint_velocities.png','upper_bias_decomposition.png']:
    f=OUT/'final_upper'/name;digest=hashlib.sha256(f.read_bytes()).hexdigest()
    assert any(x['path']==str(f) and x['sha256']==digest and x['visually_inspected'] for x in qa['files'])
assert bias['synthetic_fixtures_passed']==4 and len(bias['records'])==20
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),owned_snapshot_utc=live['recorded_utc'],same_supervisor=39000,actual_workers=live['actual_worker_pids'],frozen_hashes_verified=55,production_source_changed=False,diagnostic_actors_deployed=False,actual_retained_main_updates=learning['actual_retained_main_updates'],completed_main_updates=learning['completed_main_updates'],hash_proven_recovered_main_updates=learning['hash_proven_recovered_main_updates'],inherited_main_updates=27507,operation=live['state']['operation_path'],operation_count=live['state']['environment_count'],capacity_note='Owned operation can be evaluation. Last completed main count below is actual PPO capacity.',last_completed_main_count=learning['blocks'][-1]['environments'],all_journals_handoffs_updated_once=True,mandatory_exact257_upper_plots_inspected=True,upper_scope='Exact257 nominal258/pushed259; newer checkpoints are not certified by these timelines or gates.',inventory=inventory,automation=auto,throughput={k:learning[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},gpu=live['gpu'],host=live['system_memory'])
(OUT/'closeout_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
npath=TRAINING/'automation/NEXT_CHECK.json';n=read(npath)
n.update(closing_owned_observation=str(OUT/'closing_liveness/live_handoff.json'),closing_learning_verification=str(OUT/'closing_learning_verification.json'),closing_inventory=str(OUT/'history_inventory_receipt.json'),native_automation_verification=str(OUT/'automation_verification.json'),closeout_verification=str(OUT/'closeout_verification.json'),closing_retained_main_updates=learning['actual_retained_main_updates'],closing_recovered_main_updates=learning['hash_proven_recovered_main_updates'],current_owned_operation=live['state']['operation_path'],newer_observation_scope='Later owned operation and main accounting only. Reviewed matching gates and inspected upper timelines remain exact257; no qualification attributed to newer actors.')
npath.write_text(json.dumps(n,indent=2)+'\n')
closing=f"\nClosing ownership observation {live['recorded_utc']}: same supervisor39000, one actual worker and55 verified frozen files. Completed{learning['completed_main_updates']} +{learning['hash_proven_recovered_main_updates']} received recoveries = {learning['actual_retained_main_updates']} retained current-run main updates, plus27507 inherited. Last completed main uses{learning['blocks'][-1]['environments']} environments; current operation is{Path(live['state']['operation_path']).name} with count{live['state']['environment_count']}, which can be evaluation rather than PPO. Exact257 gates and upper QA remain the reviewed evidence; newer actors are not qualified by those results. Inventory40 physical runs,zero differing hashes/read warnings,eight historical missing manifests preserved. Same ACTIVEfive-hour heartbeat/thread/notification/durable endpoint verified. See closeout_verification.json.\n"
with (OUT/'REPORT.md').open('a',encoding='utf-8') as stream:stream.write(closing)
for p in docs[:3]:
    with p.open('a',encoding='utf-8') as stream:stream.write(closing+'Report: '+str(OUT/'REPORT.md')+'\n')
print(json.dumps({k:receipt[k] for k in ['owned_snapshot_utc','actual_workers','actual_retained_main_updates','last_completed_main_count','operation','operation_count']},indent=2))
