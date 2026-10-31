"""Owned saved-boundary pause with failure cleanup; immutable diagnostic copy."""
import hashlib,json,time,shutil
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def output(p):
 a=p.cmdline();return Path(a[a.index('--output')+1]).resolve()
active=read(HERE/'ACTIVE_RUN.json');run=Path(active['run'])
assert run.name=='locomotion_20261007_r40_goal_budget_full_state'
state=read(run/'state.json');supervisor=psutil.Process(state['supervisor_pid']);identity=supervisor.create_time()
assert output(supervisor)==run.resolve() and any(Path(a).name=='supervisor_unitree.py' for a in supervisor.cmdline())
src=run/'source/physical_v29';dst=OUT/'control_source/physical_v29'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={p.relative_to(src).as_posix():sha(p) for p in src.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
assert all(sha(dst/n)==h for n,h in files.items())
write(OUT/'protected_source_copy.json',dict(source=str(src),protected_hashes=files,files=len(files)))
deadline=time.monotonic()+600
while True:
 state=read(run/'state.json');assert state['status']=='running'
 if state['operation']=='eval' and Path(state['checkpoint']).is_file():
  try:
   worker=psutil.Process(state['child_pid']);operation=Path(state['operation_path'])
   assert output(worker)==operation.resolve() and any(Path(a).name=='worker_unitree.py' for a in worker.cmdline())
   break
  except psutil.NoSuchProcess:pass
 assert time.monotonic()<deadline,'No saved boundary; preserve active PPO'
 time.sleep(2)
owned=[worker]+worker.children(recursive=True);identities={p.pid:p.create_time() for p in owned}
assert supervisor.create_time()==identity
supervisor.suspend();suspended=time.time();write(OUT/'state_before_diagnosis.json',state)
print(json.dumps(dict(supervisor_suspended=supervisor.pid,checkpoint=state['checkpoint'],draining=str(operation))),flush=True)
try:
 deadline=time.monotonic()+240
 while any(p.is_running() and p.create_time()==identities[p.pid] for p in owned):
  assert time.monotonic()<deadline,'Eval failed to drain'
  time.sleep(2)
 assert (operation/'result.json').exists() and read(operation/'result.json')['completed']
 assert not any('worker_unitree.py' in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
 audit=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),paused_unix=time.time(),suspended_unix=suspended,
  run=str(run),supervisor_pid=supervisor.pid,supervisor_create_time=identity,checkpoint=state['checkpoint'],checkpoint_sha256=sha(state['checkpoint']),
  completed_eval=str(operation),worker_drained=True,scope='Bounded truthful retention exposure and native motor attribution, exact-full-state controls; no diagnostic actor deployed')
 state.update(status='paused_for_controlled_diagnosis',child_pid=None,updated_unix=time.time(),pause_reason=audit['scope'])
 write(run/'state.json',state);active['status']=state['status'];write(HERE/'ACTIVE_RUN.json',active)
 write(OUT/'diagnostic_pause.json',audit);write(OUT/'stopped_boundary.json',audit)
 print(json.dumps(audit),flush=True)
except BaseException:
 supervisor.resume();raise
