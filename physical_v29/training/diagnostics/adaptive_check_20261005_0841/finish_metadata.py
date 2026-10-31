import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
n=read(HERE/'automation/NEXT_CHECK.json');s=read(Path(n['active_run'])/'state.json');folder=Path(s['operation_path']);p=read(folder/'progress.json')
n['selected_actor'].update(checkpoint=s['checkpoint'],sha256=hashlib.sha256(Path(s['checkpoint']).read_bytes()).hexdigest(),
 matching_gate_pair=str(OUT/'deployed_gate_verification.json'),standing_and_walking_cases_through_1_m_s_pass=True,running_qualified=False)
n['latest_verified_state'].update(environments=p['num_envs'],completed_main_updates=s['total_training_iterations'],
 observed_current_operation_updates=p['iteration'],latest_progress_unix=p['time'])
n['observed_partial_main_updates']=p['iteration'];n['latest_gate_trends']=str(OUT/'deployed_gate_verification.json')
n['deployed_completed_main_throughput']=read(OUT/'deployed_gate_verification.json')['main_worker_effective_samples_s']
write(HERE/'automation/NEXT_CHECK.json',n)
inventory=read(OUT/'history_inventory_final.log');automation=read(OUT/'automation_verified.json')
assert inventory['differing_manifest_hashes']==0 and not inventory['read_warnings'] and automation['verified']
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=n['active_run'],checkpoint=s['checkpoint'],
 status=s['status'],stage=s['stage'],main_completed=s['total_training_iterations'],partial=p['iteration'],environments=p['num_envs'],
 free_GPU_MiB=p['gpu']['free_mb'],host_available_MiB=p['gpu']['ram_available_mb'],KL=p['ppo']['pre_update_kl'],
 command_difference=p['behavior']['action_command_difference'],automation=automation,inventory=inventory)
write(OUT/'final_metadata_verification.json',record)
with (OUT/'REPORT.md').open('a',encoding='utf-8') as f:
 f.write(f"\n\nFinal metadata check {record['recorded_utc']}: block `{folder.name}` is training at {p['num_envs']} with {p['iteration']} observed updates, GPU free {p['gpu']['free_mb']} MiB and host available {p['gpu']['ram_available_mb']} MiB. The current checkpoint/hashes and matching gate references in NEXT_CHECK are refreshed. The same automation is verified ACTIVE with its three-hour cadence, thread, durable scope and notification preference preserved. History inventory contains {inventory['physical_runs']} physical runs and {inventory['physical_operations']} operations, zero differing hashes and zero read warnings; eight historical missing manifest artifacts remain explicitly recorded. [Final metadata](final_metadata_verification.json).\n")
print(json.dumps(record,indent=2))
