import json,subprocess,sys,time,os,hashlib
from pathlib import Path
import psutil,numpy as np
OUT=Path(__file__).resolve().parent;T=OUT.parents[1];R=T/'runs/locomotion_20261007_r40_goal_budget_full_state'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
assert read(R/'state.json')['status']=='stopped_by_user_budget'
latest=Path(read(OUT/'stop_receipt.json')['checkpoint']);selected=R/'1064_run_350_train_352/candidate.pt'
jobs=[('passive_parity',latest,901,False,False,True),('selected_diverse_nominal',selected,5011,False,True,True),('selected_diverse_pushed',selected,5039,True,True,False),('latest_diverse_nominal',latest,5011,False,True,False),('latest_diverse_pushed',latest,5039,True,True,False)]
for label,cp,seed,push,diverse,native in jobs:
 folder=OUT/label
 if (folder/'completion_receipt.json').exists():continue
 folder.mkdir(exist_ok=True)
 assert not psutil.pid_exists(39000),'owned supervisor must stay retired'
 assert not any((p.info['name'] or '').lower().startswith('python') and any(Path(a).name in ['eval_entry.py','worker_unitree.py','record_motor_gui.py'] for a in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline','name'])),'Existing Isaac/CUDA worker; do not overlap'
 env=os.environ.copy();env['FINAL_DIVERSE']=str(int(diverse));env['FINAL_NATIVE']=str(int(native));env['PYTHONUTF8']='1'
 command=[sys.executable,'-u','-X','utf8',str(OUT/'eval_entry.py'),'--headless','--device','cuda:0','--mode','eval','--num_envs','320','--speed','3.5','--duration','20','--output',str(folder),'--checkpoint',str(cp),'--seed',str(seed),'--reserve_mb','300']+(['--perturb'] if push else [])
 with (folder/'stdout.log').open('w') as so,(folder/'stderr.log').open('w') as se:
  p=subprocess.Popen(command,stdout=so,stderr=se,env=env,creationflags=subprocess.CREATE_NO_WINDOW);created=psutil.Process(p.pid).create_time();start=time.time()
  (folder/'process.json').write_text(json.dumps(dict(pid=p.pid,created=created,command=command)))
  deadline=time.monotonic()+450;completed_at=None
  while p.poll() is None:
   failed=(folder/'failure.json').exists()
   complete=(folder/'result.json').exists() and (not native or (folder/'motor_summary.json').exists())
   if complete and completed_at is None:completed_at=time.monotonic()
   if failed or time.monotonic()>deadline or (completed_at is not None and time.monotonic()-completed_at>20):
    ident=psutil.Process(p.pid);assert ident.create_time()==created
    tree=ident.children(recursive=True)+[ident]
    for owned in tree:
     try:owned.kill()
     except psutil.NoSuchProcess:pass
    psutil.wait_procs(tree,timeout=15);p.wait(timeout=30)
    if failed or not complete:raise RuntimeError(label+' worker failed or exceeded export timeout')
    break
   time.sleep(1)
 assert (folder/'result.json').exists() and read(folder/'result.json')['completed'] and not (folder/'failure.json').exists(),label
 if native:assert (folder/'motor_summary.json').exists(),label+' native export missing'
 receipt=dict(completed=True,wall_seconds=time.time()-start,checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),diverse=diverse,perturb=push,seed=seed,exit_code=p.returncode,no_learning=True)
 (folder/'completion_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
 if label=='passive_parity':
  original=R/'1227_run_350_gate_407_nominal'
  with np.load(original/'telemetry.npz') as a,np.load(folder/'telemetry.npz') as b:
   arrays={k:np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind in 'fc' else np.array_equal(a[k],b[k]) for k in a.files}
  assert all(arrays.values()) and read(original/'evaluation.json')==read(folder/'evaluation.json'),'passive replay parity'
  (OUT/'passive_capture_parity.json').write_text(json.dumps(dict(passed=True,arrays=arrays,checkpoint_sha256=receipt['sha256']),indent=2))
(OUT/'evaluation_complete.json').write_text(json.dumps(dict(completed=True,jobs=[j[0] for j in jobs],no_learning=True)))
