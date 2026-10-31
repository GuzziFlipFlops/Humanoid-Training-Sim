"""Final CPU accounting and preservation only; cannot start or resume learning."""
import ast, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;T=OUT.parents[1];R=T/'runs/locomotion_20261007_r40_goal_budget_full_state'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
state=read(R/'state.json');assert state['status']=='stopped_by_user_budget'
attempts=state['attempts'];receivers=[Path(a['folder']) for a in attempts]
blocks=[]
for a in attempts:
 f=Path(a['folder'])
 if a['operation']!='train' or '_run_' not in f.name or '_train_' not in f.name:continue
 recovered=not a['completed'];received=None
 if recovered:
  cp=f/'memory_recovery.pt'
  if not a.get('memory_failure') or not cp.exists():continue
  digest=sha(cp)
  received=next((p for p in receivers if (p/'full_load_proof.json').exists() and read(p/'full_load_proof.json')['source_sha256']==digest),None)
  if received is None:continue
  proof=read(received/'full_load_proof.json');assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
  assert Path(read(received/'run_config.json')['arguments']['checkpoint']).resolve()==cp.resolve()
 rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
 proof=read(f/'full_load_proof.json');assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
 assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
 blocks.append(dict(folder=str(f),recovered=recovered,receiver=str(received) if received else None,updates=len(rows),samples=sum(r['num_envs']*24 for r in rows),active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows),worker_seconds=a['seconds'],first_time=rows[0]['time'],last_time=rows[-1]['time'],minimum_free_gpu_MiB=min(r['gpu']['free_mb'] for r in rows),minimum_host_available_MiB=min(r['gpu']['ram_available_mb'] for r in rows)))
assert sum(b['updates'] for b in blocks if not b['recovered'])==26023
assert sum(b['updates'] for b in blocks if b['recovered'])==50
endpoint=max(b['last_time'] for b in blocks)
# The previous audit's explicit documented suspension paths plus the substantial iteration.
old=OUT.parent/'adaptive_check_20261010_0149'
tree=ast.parse((old/'verify_recovery.py').read_text())
paths=[]
for node in ast.walk(tree):
 if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id in ('resume_path','failure_resume'):
  paths.extend(eval(compile(ast.Expression(node.iter),'<known local paths>','eval'),{'OUT':old}))
paths.append(OUT.parent/'substantial_iteration_20261010_0405/diagnostic_resume.json')
intervals=[]
for p in paths:
 if not p.exists():continue
 d=read(p);lo=max(state['started_unix'],d['suspended_unix']);hi=min(endpoint,d['resumed_unix'])
 if hi>lo:intervals.append(dict(path=str(p),start=lo,end=hi,seconds=hi-lo))
intervals.sort(key=lambda d:d['start'])
assert all(intervals[i]['end']<=intervals[i+1]['start'] for i in range(len(intervals)-1))
samples=sum(b['samples'] for b in blocks);excluded=sum(d['seconds'] for d in intervals)
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),measurement_endpoint_utc=datetime.fromtimestamp(endpoint,timezone.utc).isoformat(),completed_main_updates=26023,hash_proven_recovered_main_updates=50,current_retained_main_updates=26073,inherited_verified_main_updates=27507,total_retained_updates_including_inheritance=53580,completed_main_samples=sum(b['samples'] for b in blocks if not b['recovered']),recovered_samples=sum(b['samples'] for b in blocks if b['recovered']),total_current_run_retained_samples=samples,active_PPO_samples_s=samples/sum(b['active_seconds'] for b in blocks),worker_effective_samples_s=samples/sum(b['worker_seconds'] for b in blocks),whole_chain_samples_s=samples/(endpoint-state['started_unix']-excluded),documented_pause_seconds_excluded=excluded,intervals=intervals,blocks=blocks,last_main=blocks[-1],scope='All completed and exactly received recovery main samples. Active=collection+PPO; worker includes operation overhead; whole chain includes startup, capacity and gates, excludes only documented suspensions through endpoint. Unmeasured brief bookkeeping pause remains included. Final evaluations/capture, capacity/smoke/diagnostic updates and unsaved R39 32 updates excluded. No learning after saved stop.')
write(OUT/'final_timing.json',record)
save=OUT/'preserved_actors_and_teachers';save.mkdir(exist_ok=True)
info=read(OUT/'full_state_save_receipt.json')['infos'];cert=read(info['locomotion_prior_certificate'])
paths=[('standing_teacher',Path(info['standing_prior'])),('qualified_349_teacher',Path(cert['checkpoint'])),('selected_1064_video_actor',R/'1064_run_350_train_352/candidate.pt')]
refs=[]
for name,p in paths:
 target=save/(name+'.pt');digest=sha(p)
 if not target.exists():shutil.copy2(p,target)
 assert sha(target)==digest
 refs.append(dict(role=name,source=str(p),saved=str(target),sha256=digest))
write(save/'receipt.json',dict(actors=refs,qualification_certificate=cert,no_reset=True,no_learning=True))
print(json.dumps({k:v for k,v in record.items() if k not in ('blocks','intervals','last_main')},indent=2))
