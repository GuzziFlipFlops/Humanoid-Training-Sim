import json,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
out=Path(__file__).resolve().parent;train=out.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
live=read(out/'initial_owned_observation.json');s=live['state'];run=Path(live['active_run']['run']);ctx=read(out/'resume_context_cpu.json')
blocks=[]
for a in s['attempts']:
 f=Path(a['folder'])
 if a['operation']!='train' or not re.search(r'_run_\d+_train_\d+',f.name) or not a['completed']:continue
 rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()];load=read(f/'full_load_proof.json')
 assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
 assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 blocks.append(dict(folder=str(f),updates=len(rows),samples=sum(r['num_envs']*24 for r in rows),active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows),worker_seconds=a['seconds'],min_gpu_MiB=min(r['gpu']['free_mb'] for r in rows),min_host_MiB=min(r['gpu']['ram_available_mb'] for r in rows)))
assert sum(x['updates'] for x in blocks)==s['total_training_iterations']
for recovery in ctx['recovered_evidence']:
 f=Path(recovery['checkpoint']).parent;rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
 a=next(a for a in s['attempts'] if Path(a['folder'])==f)
 blocks.append(dict(folder=str(f),updates=len(rows),samples=sum(r['num_envs']*24 for r in rows),active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows),worker_seconds=a['seconds'],hash_proven_recovered=True))
pauses=[]
for p in train.glob('diagnostics/*/diagnostic_resume.json'):
 before=p.parent/'diagnostic_pause.json'
 if before.exists() and Path(read(before)['run']).resolve()==run.resolve():pauses.append(dict(receipt=str(p),seconds=read(p)['pause_seconds']))
endpoint=s['updated_unix'];samples=sum(x['samples'] for x in blocks);failed=Path(s['operation_path']);lost=[json.loads(x) for x in (failed/'progress.jsonl').read_text().splitlines()]
deadtime=live['recorded_unix']-endpoint
value=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),completed_operation_main_updates=s['total_training_iterations'],verified_recovered_main_updates=ctx['verified_recovered_main_updates'],retained_R39_main_updates=sum(x['updates'] for x in blocks),inherited_prior_main_updates=s['prior_training_iterations'],total_inherited_R40=context_total if (context_total:=ctx['inherited_main_updates']) else None,retained_samples=samples,active_collection_PPO_samples_s=samples/sum(x['active_seconds'] for x in blocks),worker_effective_samples_s=samples/sum(x['worker_seconds'] for x in blocks),whole_chain_to_failure_samples_s=samples/(endpoint-s['started_unix']-sum(x['seconds'] for x in pauses)),excluded_pauses=pauses,excluded_pause_seconds=sum(x['seconds'] for x in pauses),failed_unsaved_updates=len(lost),failed_unsaved_samples=sum(r['num_envs']*24 for r in lost),failed_block_active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in lost),failure_unix=endpoint,downtime_before_check_seconds=deadtime,all_retained_main_KL_and_command_mismatch_zero=True,block_accounting=blocks,scope='Retained main samples only. Capacity, smoke, diagnostic learning and interrupted32-update block excluded. Whole-chain endpoint is failure, includes interrupted work/detection; wall pauses excluded. Later stopped downtime and diagnostic comparison are separate, not hidden in this rate.')
(out/'throughput_and_loads.json').write_text(json.dumps(value,indent=2))
upper=read(out/'initial_upper/upper_joint_time_audit.json');v=[]
for r in upper['records']:
 if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s'] in [1.,2.5,3.,3.5]:
  v.append(dict(command=r['command_m_s'],head_rms_deg=r['head_rms_deg'],head_p95_deg=r['head_p95_deg'],shoulder_ranges_deg=[r['joints'][side+'_shoulder_pitch']['median_per_robot_p95_p05_range_deg'] for side in ['left','right']],coordination=r['coordination'],representative=r['representative_50hz']))
(out/'upper_summary.json').write_text(json.dumps(v,indent=2))
print(json.dumps({k:v for k,v in value.items() if k not in ['block_accounting','excluded_pauses']},indent=2))
