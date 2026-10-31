"""CPU-only closing evidence and exact resumed-main verification."""
import json,importlib.util,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
target=OUT/'closing_cpu';target.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('owned_recorder',HERE/'record_owned_observation_handoff.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=target;m.main()
live=read(target/'live_handoff.json');s=live['state'];pause=read(OUT/'diagnostic_pause.json')
assert s['status']=='running' and s['supervisor_pid']==pause['supervisor_pid']
assert all(live['frozen_source_integrity'].values()) and len(live['actual_worker_pids'])==1
with (OUT/'closing_review_stdout.log').open('w',encoding='utf-8') as f:
 subprocess.run([sys.executable,str(OUT/'review_gates.py'),'--snapshot','closing_cpu/live_handoff.json','--output','closing_gate_review.json'],stdout=f,check=True)
g=read(OUT/'closing_gate_review.json');old=read(OUT/'initial_gate_review.json')
for p in g['promotions']:
 if p.get('requalification'):
  match=[x for x in old['promotions'] if x['stage']==p['stage'] and 'matching_pair' in x]
  if match:p['matching_pair']=match[0]['matching_pair']
(OUT/'closing_gate_review.json').write_text(json.dumps(g,indent=2)+'\n')
if '--verify-resume' in sys.argv:
 run=Path(pause['run']);blocks=[]
 for p in sorted(run.glob('*_train_*')):
  if not (p/'full_load_proof.json').exists() or not (p/'result.json').exists():continue
  load=read(p/'full_load_proof.json')
  if load.get('source_sha256')==pause['checkpoint_sha256'] and read(p/'result.json').get('completed'):blocks.append(p)
 assert len(blocks)==1,blocks
 p=blocks[0];load=read(p/'full_load_proof.json');rows=[json.loads(x) for x in (p/'progress.jsonl').read_text().splitlines()]
 assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
 assert all(x['ppo']['pre_update_kl']==0 and x['behavior']['action_command_difference']==0 for x in rows)
 assert all(x['upper_refinement']['batches']==20 for x in rows)
 gpu=[x['gpu'] for x in rows];resume=read(OUT/'diagnostic_resume.json')
 result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),same_supervisor=True,supervisor_pid=s['supervisor_pid'],
  block=str(p),completed_genuine_updates=len(rows),num_envs=rows[0]['num_envs'],full_load_proof=load,
  zero_maximum_preupdate_KL_and_command_mismatch=True,upper_auxiliary_batches_per_update=20,
  minimum_free_gpu_MiB=min(x['free_mb'] for x in gpu),minimum_host_available_MiB=min(x['ram_available_mb'] for x in gpu),
  private_memory_growth_MiB=gpu[-1]['process_private_mb']-gpu[0]['process_private_mb'],torch_allocation_growth_MiB=gpu[-1]['torch_allocated_mb']-gpu[0]['torch_allocated_mb'],
  active_PPO_samples_s=sum(x['num_envs']*24 for x in rows)/sum(x['collect_seconds']+x['learn_seconds'] for x in rows),
  frozen_hashes_verified=len(live['frozen_source_integrity']),actual_worker_pids=live['actual_worker_pids'],pause_seconds_excluded=resume['pause_seconds'],diagnostic_actors_deployed=False)
 (OUT/'resumed_learning_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
else:print(json.dumps(dict(status=s['status'],stage=s['stage'],updates=s['total_training_iterations'],latest_pair=Path(g['latest']['checkpoint']).parent.name)))
