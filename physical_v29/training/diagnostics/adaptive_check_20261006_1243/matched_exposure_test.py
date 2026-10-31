"""Bounded sequential control/candidate PPO with exact independent gate pairs."""
import json,hashlib,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause=read(OUT/'diagnostic_pause.json');s=psutil.Process(pause['supervisor_pid'])
assert s.create_time()==pause['supervisor_create_time'] and s.status()==psutil.STATUS_STOPPED
assert hashlib.sha256(Path(pause['checkpoint']).read_bytes()).hexdigest()==pause['checkpoint_sha256']
args=read(Path(pause['checkpoint']).parent/'run_config.json')['arguments']
root=OUT/'matched_exposure';root.mkdir(exist_ok=True)
manifest=dict(seed=71,num_envs=4096,iterations_per_block=64,blocks=[64,128],same_starting_checkpoint=pause['checkpoint'],source_signature_unchanged=True,truthful_commands=True,unchanged_gates=True,candidate='Uniform +/-0.1m/s lateral command only on sampled frontier x>3m/s at stage3.5. Existing stopped/prior exposure, reward, PPO, upper/teacher gradients, native caps and inference are unchanged. Extra sampling changes subsequent random draw stream.',wrapper_sha256=hashlib.sha256((OUT/'lateral_exposure_candidate.py').read_bytes()).hexdigest(),deployed=False)
(root/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
def launch(a,folder,timeout=320):
 if (folder/'result.json').exists():
  assert read(folder/'result.json')['completed']
  print(json.dumps(dict(reused_completed=str(folder))),flush=True)
  return
 folder.mkdir(exist_ok=True)
 for p in psutil.process_iter(['pid','cmdline']):
  c=p.info['cmdline'] or []
  assert not any(Path(x).name in ['worker_unitree.py','capture_current_gradients.py','steering_probe.py','lateral_exposure_candidate.py'] for x in c),'Existing CUDA worker'
 with (folder/'stdout.log').open('w',encoding='utf-8') as o,(folder/'stderr.log').open('w',encoding='utf-8') as e:
  p=subprocess.Popen([sys.executable,'-u','-X','utf8']+a,stdout=o,stderr=e,creationflags=subprocess.CREATE_NO_WINDOW)
  (folder/'process.json').write_text(json.dumps(dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),command=a),indent=2))
  code=p.wait(timeout=timeout)
 assert code==0,(folder,code)
 assert read(folder/'result.json')['completed'],folder
 print(json.dumps(dict(completed=str(folder),code=code)),flush=True)
teacher_args=[]
for k in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:teacher_args+=['--'+k,str(args[k])]
start=time.time()
for branch in ['control','candidate']:
 cp=pause['checkpoint']
 for updates in [64,128]:
  f=root/f'{branch}_{updates:03d}_train'
  script=TRAIN/'worker_unitree.py' if branch=='control' else OUT/'lateral_exposure_candidate.py'
  common=['--headless','--device','cuda:0','--speed','3.5','--seed','71','--reserve_mb','300']
  launch([str(script),'--mode','train','--output',str(f),'--checkpoint',cp,'--num_envs','4096','--iterations','64','--learning_rate','.0001']+teacher_args+common,f)
  cp=str(f/'candidate.pt')
  load=read(f/'full_load_proof.json')
  assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
  progress=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
  assert len(progress)==64 and all(r['ppo']['pre_update_kl']==0. and r['behavior']['action_command_difference']==0. for r in progress)
  for perturb in [False,True]:
   g=root/(f'{branch}_{updates:03d}_'+('push' if perturb else 'nominal'))
   launch([str(TRAIN/'worker_unitree.py'),'--mode','eval','--output',str(g),'--checkpoint',cp,'--num_envs','320','--duration','20']+common+(['--perturb'] if perturb else []),g)
  if time.time()-start>1600:raise TimeoutError('Bounded diagnostic learning budget exceeded')
(root/'completion.json').write_text(json.dumps(dict(completed=True,seconds=time.time()-start,diagnostic_actors_deployed=False),indent=2))
