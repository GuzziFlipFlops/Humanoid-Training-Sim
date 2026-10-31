import json,subprocess,sys,time,hashlib
from pathlib import Path
import psutil
out=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
deadline=time.time()+180
while not (out/'diagnostic_pause.json').exists():
 assert time.time()<deadline,'Saved-boundary worker has not drained; no diagnostic may start'
 time.sleep(1)
pause=read(out/'diagnostic_pause.json');sup=psutil.Process(pause['supervisor_pid'])
assert sup.create_time()==pause['supervisor_create_time'] and sup.status()==psutil.STATUS_STOPPED
config=read(Path(pause['checkpoint']).parent/'run_config.json')['arguments']
assert hashlib.sha256(Path(pause['checkpoint']).read_bytes()).hexdigest()==pause['checkpoint_sha256']
args=[str(out/'capture_current_gradients.py'),'--headless','--device','cuda:0','--mode','train','--output',str(out/'gradient_smoke'),
 '--checkpoint',pause['checkpoint'],'--num_envs','4096','--speed','3.5','--iterations','4','--learning_rate','.0001','--seed','97','--reserve_mb','300']
for key in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:
 args+=['--'+key,str(config[key])]
jobs=[('gradient',args,300),('arm_probe',[str(out/'arm_response_probe.py'),'--headless','--device','cuda:0'],300)]
started=time.time()
try:
 for name,args,timeout in jobs:
  with (out/(name+'_stdout.log')).open('w',encoding='utf-8') as so,(out/(name+'_stderr.log')).open('w',encoding='utf-8') as se:
   proc=subprocess.Popen([sys.executable,'-u','-X','utf8']+args,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
   (out/(name+'_process.json')).write_text(json.dumps(dict(pid=proc.pid,create_time=psutil.Process(proc.pid).create_time(),arguments=args),indent=2))
   try:code=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    parent=psutil.Process(proc.pid);owned=parent.children(recursive=True)+[parent]
    for child in owned:
     if child.is_running():child.terminate()
    psutil.wait_procs(owned,timeout=20)
    raise
  assert code==0,(name,code)
  print(json.dumps(dict(completed=name,elapsed=time.time()-started)),flush=True)
 (out/'bounded_diagnostics_completion.json').write_text(json.dumps(dict(completed=True,seconds=time.time()-started,production_changed=False,diagnostic_actors_deployed=False),indent=2))
except BaseException as e:
 (out/'bounded_diagnostics_failure.json').write_text(json.dumps(dict(error=repr(e),seconds=time.time()-started),indent=2));raise
finally:
 # Never leave production paused after a diagnostic fault.
 code=subprocess.call([sys.executable,'-u','-X','utf8',str(out/'resume_after_diagnosis.py')])
 assert code==0,'Resume requires immediate owned-process investigation'
