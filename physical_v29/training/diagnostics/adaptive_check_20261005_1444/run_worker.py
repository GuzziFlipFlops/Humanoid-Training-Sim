"""One explicitly owned diagnostic Isaac worker at a time, unique evidence path."""
import sys,json,subprocess,time,hashlib,argparse
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--folder',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--mode',choices=['eval','train'],required=True)
p.add_argument('--environments',type=int,default=192);p.add_argument('--iterations',type=int,default=64)
p.add_argument('--seed',type=int,default=901);p.add_argument('--perturb',action='store_true');a=p.parse_args()
for proc in psutil.process_iter(['cmdline']):
 args=proc.info['cmdline'] or []
 assert not any(Path(x).name in ['worker_unitree.py','supervisor_unitree.py','inspect_ground.py'] for x in args),'Another owned simulator/supervisor exists'
folder=OUT/a.folder;assert not folder.exists();folder.mkdir()
cp=Path(a.checkpoint);source=Path(a.source)
cmd=[sys.executable,'-u','-X','utf8',str(source/'training/worker_unitree.py'),'--headless','--device','cuda:0',
 '--output',str(folder),'--checkpoint',str(cp),'--mode',a.mode,'--speed','1.5','--num_envs',str(a.environments),'--reserve_mb','300','--seed',str(a.seed)]
if a.mode=='eval':cmd+=['--duration','20']+(['--perturb'] if a.perturb else [])
else:
 prior=json.loads((cp.parent/'standing_retention.json').read_text())
 cert=cp.parent/'qualified_prior_certificate.json'
 if not cert.exists():
  # This certifies the frozen teacher, never the diagnostic student. A direct
  # worker serializes it inside locomotion_retention.json rather than making
  # the supervisor's separate convenience file.
  metadata=json.loads((cp.parent/'locomotion_retention.json').read_text())
  certificate=metadata['certificate']
  assert certificate['checkpoint_sha256']==prior['sha256']
  assert Path(certificate['checkpoint']).resolve()==Path(prior['teacher']).resolve()
  cert=folder/'qualified_teacher_launch_certificate.json'
  cert.write_text(json.dumps(certificate,indent=2)+'\n')
 cmd+=['--iterations',str(a.iterations),'--learning_rate','0.0001','--standing_prior',prior['teacher'],
       '--standing_prior_sha256',prior['sha256'],'--locomotion_prior_certificate',str(cert)]
(folder/'launch.json').write_text(json.dumps(cmd,indent=2)+'\n')
start=time.time()
with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
result=dict(returncode=r.returncode,seconds=time.time()-start,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
(folder/'wrapper_result.json').write_text(json.dumps(result,indent=2)+'\n')
assert r.returncode==0,result
assert json.loads((folder/'result.json').read_text())['completed']
print(json.dumps(dict(folder=str(folder),**result)))
