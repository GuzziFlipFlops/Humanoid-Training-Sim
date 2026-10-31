"""Bounded physical/PPO tests while the owned production worker is drained."""
import hashlib,json,subprocess,sys,time,psutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
NEW=OUT/'validation_source/physical_v29/training'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run']);state=read(run/'state.json')
sup=psutil.Process(pause['supervisor_pid'])
assert sup.create_time()==pause['supervisor_create_time'] and sup.status()==psutil.STATUS_STOPPED
assert state['checkpoint']==pause['checkpoint'] and sha(state['checkpoint'])==pause['checkpoint_sha256']
sys.path.insert(0,str(NEW))
from standing_retention import qualification_certificate
from curriculum_provenance import training_reward_signature
signature=training_reward_signature(NEW)
cfg=read(Path(pause['completed_eval'])/'run_config.json')
teacher=state['standing_retention_prior']
certificate=qualification_certificate(run,teacher['checkpoint'],1.,signature,cfg['gate_version'])
write(OUT/'prior_certificate.json',certificate)
for name,mode,count,iterations,duration,checkpoint,seed,perturb in [
 ('running_fullstate_smoke','train',64,8,4.,pause['checkpoint'],71,False),
 ('checkpoint_reload','eval',192,0,20.,str(OUT/'running_fullstate_smoke/candidate.pt'),901,False),
 ('same_actor_nominal','eval',192,0,20.,pause['checkpoint'],901,False)]:
 assert not (OUT/name).exists(),'Preserve attempts'
 for p in psutil.process_iter(['cmdline']):
  assert not any(Path(x).name=='worker_unitree.py' for x in p.info['cmdline'] or []),'Another Isaac worker'
 args=[sys.executable,'-u','-X','utf8',str(NEW/'worker_unitree.py'),'--headless','--device','cuda:0',
  '--mode',mode,'--num_envs',str(count),'--speed','1.5','--iterations',str(iterations),
  '--duration',str(duration),'--seed',str(seed),'--learning_rate','0.0001',
  '--checkpoint',checkpoint,'--output',str(OUT/name),'--reserve_mb','300']
 if mode=='train':args += ['--standing_prior',teacher['checkpoint'],'--standing_prior_sha256',teacher['sha256'],
  '--locomotion_prior_certificate',str(OUT/'prior_certificate.json')]
 folder=OUT/name;folder.mkdir()
 with (folder/'stdout.log').open('w') as log,(folder/'stderr.log').open('w') as err:
  proc=subprocess.Popen(args,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  write(folder/'command.json',dict(argv=args,pid=proc.pid,create_time=psutil.Process(proc.pid).create_time()))
  print(json.dumps(dict(started=name,pid=proc.pid)),flush=True)
  code=proc.wait(timeout=600)
 assert code==0 and (folder/'result.json').is_file(),name
 r=read(folder/'result.json');assert r['completed'],r
 print(json.dumps(dict(completed=name,seconds=r['seconds'])),flush=True)
print('VALIDATION COMPLETE',flush=True)
