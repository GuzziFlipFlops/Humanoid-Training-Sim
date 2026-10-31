"""Suspend only the owned supervisor at a saved boundary; drain its eval worker."""
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def output(p):
    a=p.cmdline();return Path(a[a.index('--output')+1]).resolve()
active=read(HERE/'ACTIVE_RUN.json');run=Path(active['run'])
assert run.name=='locomotion_20261004_r34_bilateral_swing_unloading'
state=read(run/'state.json');supervisor=psutil.Process(state['supervisor_pid']);identity=supervisor.create_time()
assert output(supervisor)==run.resolve()
assert any(Path(a).name=='supervisor_unitree.py' for a in supervisor.cmdline())
deadline=time.monotonic()+900
while True:
    state=read(run/'state.json');assert state['status']=='running'
    if state['operation']=='eval' and Path(state['checkpoint']).is_file():
        try:
            worker=psutil.Process(state['child_pid']);operation=Path(state['operation_path'])
            assert output(worker)==operation.resolve()
            assert any(Path(a).name=='worker_unitree.py' for a in worker.cmdline())
            break
        except psutil.NoSuchProcess:pass
    assert time.monotonic()<deadline,'No saved boundary; preserve active PPO'
    time.sleep(2)
owned=[worker]+worker.children(recursive=True);identities={p.pid:p.create_time() for p in owned}
assert supervisor.create_time()==identity
supervisor.suspend()
write(OUT/'state_before_video_pause.json',state)
print(json.dumps({'supervisor_suspended':supervisor.pid,'saved_checkpoint':state['checkpoint'],'draining_eval':str(operation)}),flush=True)
deadline=time.monotonic()+300
while any(p.is_running() and p.create_time()==identities[p.pid] for p in owned):
    assert time.monotonic()<deadline,'Eval failed to drain; do not start recorder'
    time.sleep(2)
assert (operation/'result.json').exists(),'Eval did not complete'
assert not any(p.is_running() for p in owned)
audit={'recorded_utc':datetime.now(timezone.utc).isoformat(),'paused_unix':time.time(),'run':str(run),
    'supervisor_pid':supervisor.pid,'supervisor_create_time':identity,'checkpoint':state['checkpoint'],
    'checkpoint_sha256':hashlib.sha256(Path(state['checkpoint']).read_bytes()).hexdigest(),
    'completed_eval':str(operation),'worker_drained':True,'scope':'User-requested temporary pause for GUI videos; same supervisor will resume'}
state.update(status='paused_for_video_capture',child_pid=None,updated_unix=time.time(),pause_reason='User requested GUI walking videos, then resume')
write(run/'state.json',state);active['status']='paused_for_video_capture';write(HERE/'ACTIVE_RUN.json',active)
write(OUT/'video_pause.json',audit);print(json.dumps(audit,indent=2),flush=True)

