import json,time,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
import psutil
OUT=Path(__file__).resolve().parent; TRAIN=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def output(p):
 c=p.cmdline();return Path(c[c.index('--output')+1]).resolve()
active=read(TRAIN/'ACTIVE_RUN.json');run=Path(active['run']);state=read(run/'state.json')
s=psutil.Process(state['supervisor_pid']);identity=s.create_time()
assert output(s)==run.resolve() and any(Path(a).name=='supervisor_unitree.py' for a in s.cmdline())
deadline=time.monotonic()+600
while True:
 state=read(run/'state.json')
 if state.get('operation')=='eval' and Path(state['checkpoint']).is_file():
  try:
   w=psutil.Process(state['child_pid']);assert output(w)==Path(state['operation_path']).resolve()
   assert any(Path(a).name=='worker_unitree.py' for a in w.cmdline());break
  except psutil.NoSuchProcess:pass
 assert time.monotonic()<deadline,'saved boundary unavailable'
 time.sleep(1)
assert s.create_time()==identity;s.suspend()
write(OUT/'saved_boundary_state.json',state)
owned=[w]+w.children(recursive=True); ids={p.pid:p.create_time() for p in owned}
receipt=dict(started_utc=datetime.now(timezone.utc).isoformat(),run=str(run),supervisor_pid=s.pid,supervisor_create_time=identity,supervisor_cmdline=s.cmdline(),checkpoint=state['checkpoint'],checkpoint_sha256=hashlib.sha256(Path(state['checkpoint']).read_bytes()).hexdigest(),owned_worker_ids=ids,automatic_restart_authorized=False,classification='user budget stop, not locomotion success or motor ceiling')
write(run/'NO_AUTOMATIC_RESTART.json',receipt);write(OUT/'stop_in_progress.json',receipt)
print(json.dumps(receipt),flush=True)
deadline=time.monotonic()+240
while True:
 live=[]
 for p in owned:
  try:
   if p.is_running() and p.create_time()==ids[p.pid]:live.append(p.pid)
  except psutil.NoSuchProcess:pass
 if not live:break
 assert time.monotonic()<deadline,'evaluation drain timed out; supervisor remains suspended'
 time.sleep(1)
receipt['drained_evaluation']=read(Path(state['operation_path'])/'result.json')
assert receipt['drained_evaluation']['completed']
assert s.create_time()==identity and s.status()==psutil.STATUS_STOPPED
s.kill();s.wait(timeout=30)
state.update(status='stopped_by_user_budget',child_pid=None,updated_unix=time.time(),stop_receipt=str(OUT/'stop_receipt.json'),resume_requires_new_user_authorization=True)
active.update(status=state['status'],checkpoint=state['checkpoint'],checkpoint_sha256=receipt['checkpoint_sha256'],stop_receipt=str(OUT/'stop_receipt.json'),resume_requires_new_user_authorization=True)
write(run/'state.json',state);write(TRAIN/'ACTIVE_RUN.json',active)
receipt.update(stopped_utc=datetime.now(timezone.utc).isoformat(),supervisor_retired=True,owned_worker_drained=True,computer_left_on=True)
write(OUT/'stop_receipt.json',receipt);print(json.dumps(receipt),flush=True)
