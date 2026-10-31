"""Sequential, bounded full-state course-command test; no production changes."""
import json,hashlib,subprocess,sys,time
from pathlib import Path
import psutil
out=Path(__file__).resolve().parent;train=out/'control_source/physical_v29/training'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
boundary=read(out/'stopped_boundary.json');cp=boundary['checkpoint']
assert hashlib.sha256(Path(cp).read_bytes()).hexdigest()==boundary['checkpoint_sha256']
args=read(Path(cp).parent/'run_config.json')['arguments'];root=out/'matched_heading';root.mkdir(exist_ok=True)
teacher=[]
for k in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:teacher+=['--'+k,str(args[k])]
manifest=dict(same_starting_checkpoint=cp,checkpoint_sha256=boundary['checkpoint_sha256'],
 candidate='For x>3, add unit-gain fixed-frame velocity-course feedback to existing half-gain heading yaw command, clamp +/-0.2rad/s',
 prior_commands_through3_bitwise_unchanged=True,teacher_context_unchanged=True,truthful_student_observations=True,
 reward_weights_physics_actions_gates_upper_PPO_unchanged=True,original_signature_does_not_certify_training_override=True,
 num_envs=4096,updates=[64,128],training_seed=97,evaluation_seeds=[901,1301,1901],
 nominal_fixed_reset_seed_not_independent_initial_state_diversity=True,deployed=False)
(root/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
started=time.time()
def launch(argv,folder,timeout=380):
 if (folder/'result.json').exists():assert read(folder/'result.json')['completed'];return
 assert not any(any(Path(x).name in ['worker_unitree.py','heading_candidate.py','diverse_heading_eval.py'] for x in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
 folder.mkdir(exist_ok=True)
 with (folder/'stdout.log').open('w',encoding='utf-8') as so,(folder/'stderr.log').open('w',encoding='utf-8') as se:
  proc=subprocess.Popen([sys.executable,'-u','-X','utf8']+argv,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
  ident=psutil.Process(proc.pid);(folder/'process.json').write_text(json.dumps(dict(pid=proc.pid,create_time=ident.create_time(),arguments=argv),indent=2))
  try:
   end=time.monotonic()+timeout
   while proc.poll() is None and not (folder/'result.json').exists():
    assert time.monotonic()<end,'Diagnostic timeout';time.sleep(2)
   if proc.poll() is None:
    assert read(folder/'result.json')['completed']
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
     owned=ident.children(recursive=True)+[ident]
     for p in owned:
      if p.is_running():p.terminate()
     psutil.wait_procs(owned,timeout=20)
     (folder/'exit_cleanup.json').write_text(json.dumps(dict(reason='Kit shutdown after completed saved result',owned_pid=ident.pid,only_after_completed_result=True)))
   code=proc.poll()
  except BaseException:
   owned=ident.children(recursive=True)+[ident]
   for p in owned:
    if p.is_running():p.terminate()
   psutil.wait_procs(owned,timeout=20);raise
 assert (code==0 or (folder/'exit_cleanup.json').exists()) and read(folder/'result.json')['completed'],(str(folder),code)
 print(json.dumps(dict(completed=folder.name,elapsed=time.time()-started)),flush=True)
common=['--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300']
wrapper=out/'heading_candidate.py'
smoke=root/'candidate_smoke'
launch([str(wrapper),'--mode','train','--output',str(smoke),'--checkpoint',cp,'--num_envs','64','--iterations','4','--learning_rate','.0001','--seed','97']+teacher+common,smoke)
launch([str(wrapper),'--mode','eval','--output',str(root/'candidate_reload'),'--checkpoint',str(smoke/'candidate.pt'),'--num_envs','64','--duration','4','--seed','97']+common,root/'candidate_reload')
for branch in ['control','candidate']:
 checkpoint=cp;script=train/'worker_unitree.py' if branch=='control' else wrapper
 for update in [64,128]:
  f=root/f'{branch}_{update:03d}_train'
  launch([str(script),'--mode','train','--output',str(f),'--checkpoint',checkpoint,'--num_envs','4096','--iterations','64','--learning_rate','.0001','--seed','97']+teacher+common,f)
  checkpoint=str(f/'candidate.pt');load=read(f/'full_load_proof.json')
  assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
  rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
  assert len(rows)==64 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
  for suffix,seed,perturb in [('nominal',901,False),('push',1301,True)]+([('push_seed1901',1901,True)] if update==128 else []):
   g=root/f'{branch}_{update:03d}_{suffix}'
   launch([str(script),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320','--duration','20','--seed',str(seed)]+common+(['--perturb'] if perturb else []),g)
  if update==128:
   g=root/f'{branch}_128_diverse_nominal'
   launch([str(out/'diverse_heading_eval.py'),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320','--duration','20','--seed','2301']+common+(['--candidate-course'] if branch=='candidate' else []),g)
   receipt=read(g/'initial_diversity.json');assert all(n==32 for n in receipt['unique_initial_states_by_command'].values())
  assert time.time()-started<2500,'Matched diagnostic overall deadline exceeded'
(root/'completion.json').write_text(json.dumps(dict(completed=True,seconds=time.time()-started,smoke_actors_discarded=True,diagnostic_actors_deployed=False),indent=2))
