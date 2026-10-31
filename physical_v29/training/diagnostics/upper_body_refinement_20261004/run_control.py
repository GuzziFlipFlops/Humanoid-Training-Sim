"""Same saved policy, seed, eight PPO updates without the proposed refinement."""
import json,subprocess,sys,psutil
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run']);state=read(run/'state.json')
worker=run/'source/physical_v29/training/worker_unitree.py';teacher=state['standing_retention_prior']
for name,mode,checkpoint,count,it in [('control_smoke','train',pause['checkpoint'],64,8),
 ('control_reload','eval',str(OUT/'control_smoke/candidate.pt'),192,0)]:
 assert not (OUT/name).exists()
 assert not any(any(Path(a).name=='worker_unitree.py' for a in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
 args=[sys.executable,'-u','-X','utf8',str(worker),'--headless','--device','cuda:0',
  '--mode',mode,'--output',str(OUT/name),'--checkpoint',checkpoint,'--num_envs',str(count),
  '--iterations',str(it),'--speed','1.5','--seed',str(71 if mode=='train' else 901),
  '--duration','20','--learning_rate','0.0001','--reserve_mb','300']
 if mode=='train':args+=['--standing_prior',teacher['checkpoint'],'--standing_prior_sha256',teacher['sha256'],
  '--locomotion_prior_certificate',str(OUT/'prior_certificate.json')]
 folder=OUT/name;folder.mkdir()
 with (folder/'stdout.log').open('w') as log,(folder/'stderr.log').open('w') as err:
  p=subprocess.Popen(args,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  (folder/'command.json').write_text(json.dumps(dict(argv=args,pid=p.pid,create_time=psutil.Process(p.pid).create_time()),indent=2))
  print(json.dumps(dict(started=name,pid=p.pid)),flush=True);code=p.wait(timeout=600)
 assert code==0 and read(folder/'result.json')['completed']
 print(json.dumps(dict(completed=name)),flush=True)
