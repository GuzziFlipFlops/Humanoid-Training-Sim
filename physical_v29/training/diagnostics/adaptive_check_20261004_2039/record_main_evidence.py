"""Summarize actual partial main updates without mixing throughput denominators."""
import json
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
s=json.loads((OUT/'replacement_status.json').read_text());run=Path(s['run']);p=run/s['current_operation']
rows=[json.loads(x) for x in (p/'progress.jsonl').read_text().splitlines()]
steps=sum(x['num_envs']*24 for x in rows);seconds=sum(x['collect_seconds']+x['learn_seconds'] for x in rows)
v={'recorded_utc':datetime.now(timezone.utc).isoformat(),'operation':p.name,'observed_updates':len(rows),
   'main_completed_block_updates_at_snapshot':s['completed_main_updates'],
   'maximum_pre_update_KL':max(x['ppo']['pre_update_kl'] for x in rows),
   'maximum_action_command_difference':max(x['behavior']['action_command_difference'] for x in rows),
   'minimum_free_gpu_mb':min(x['gpu']['free_mb'] for x in rows),
   'minimum_host_available_mb':min(x['gpu']['ram_available_mb'] for x in rows),
   'private_growth_mb':rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb'],
   'torch_allocation_growth_mb':rows[-1]['gpu']['torch_allocated_mb']-rows[0]['gpu']['torch_allocated_mb'],
   'active_samples_s':steps/seconds,'prior_retention_batches':rows[-1]['locomotion_retention']['batches'],
   'load_proof':s['full_load_proof'],
   'scope':'Partial main block, active collection-plus-PPO only; no worker-effective/full-chain speed or independent qualification inferred'}
assert v['maximum_pre_update_KL']==0 and v['maximum_action_command_difference']==0 and v['prior_retention_batches']>0
(OUT/'new_main_evidence.json').write_text(json.dumps(v,indent=2)+'\n')
args=json.loads((OUT/'automation_update_args.json').read_text())
extra=' Fresh deployed partial main evidence: {observed_updates} observed8192 updates, exact full load, zero maximum pre-update KL/command mismatch, qualified prior20 batches/update; minGPU{minimum_free_gpu_mb}MiB,host{minimum_host_available_mb}MiB,private growth{private_growth_mb}MiB,Torch allocation growth{torch_allocation_growth_mb}MiB,about{active_samples_s:.0f} active samples/s. Read new_main_evidence.json; this is not running qualification or completed-block throughput.'.format(**v)
extra+=' The take-off pulse tradeoff is quantified in physical_cost_tradeoff.json: tracking-0.0685,acceleration-0.0379,height-0.0184,flight+0.00418; if plateaued test full objective and phase/target timing before another focused repair.'
args['prompt']=args['prompt'].split(' Fresh deployed partial main evidence:')[0]+extra
(OUT/'automation_update_args.json').write_text(json.dumps(args,indent=2)+'\n')
next_path=OUT.parents[1]/'automation/NEXT_CHECK.json';n=json.loads(next_path.read_text());n['partial_main_evidence']=v
next_path.write_text(json.dumps(n,indent=2)+'\n')
report=OUT/'REPORT.md';report.write_text(report.read_text()+'\n\n'+extra+'\n')
print(json.dumps(v,indent=2))
