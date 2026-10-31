"""Stop only the verified supervisor at a saved eval boundary; drain its child."""
import datetime, hashlib, json, sys, time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
HERE=OUT.parents[1]
sys.path.insert(0,str(HERE))
from status_io import atomic_json
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def output(p):
    a=p.cmdline();return Path(a[a.index('--output')+1]).resolve()
active=read(HERE/'ACTIVE_RUN.json');run=Path(active['run'])
assert run.name=='locomotion_20261004_r31_exact_command_rehearsal'
initial=read(run/'state.json');supervisor=psutil.Process(initial['supervisor_pid'])
identity=supervisor.create_time()
assert output(supervisor)==run.resolve()
assert any(Path(a).name=='supervisor_unitree.py' for a in supervisor.cmdline())
deadline=time.monotonic()+900
while True:
    assert supervisor.is_running() and supervisor.create_time()==identity
    state=read(run/'state.json')
    assert state['status']=='running' and state['stage']=='run_150','Stage progressed; re-assess'
    operation=Path(state['operation_path'])
    if state['operation']=='eval':
        checkpoint=Path(state['checkpoint']);assert checkpoint.exists()
        try:
            worker=psutil.Process(state['child_pid'])
            assert output(worker)==operation.resolve()
            assert any(Path(a).name=='worker_unitree.py' for a in worker.cmdline())
            break
        except psutil.NoSuchProcess:pass
    assert time.monotonic()<deadline,'No saved boundary; do not interrupt PPO'
    time.sleep(2)
owned=[worker]+worker.children(recursive=True)
identities={p.pid:p.create_time() for p in owned}
atomic_json(OUT/'state_before_drain.json',state)
supervisor.kill();supervisor.wait(timeout=10)
print(json.dumps({'supervisor_stopped':supervisor.pid,'saved_checkpoint':str(checkpoint),'eval_left_running':str(operation)}),flush=True)
deadline=time.monotonic()+600;result_seen=None
while True:
    alive=[p for p in owned if p.is_running() and p.create_time()==identities[p.pid]]
    if not alive:break
    if (operation/'result.json').exists() or (operation/'failure.json').exists():
        result_seen=result_seen or time.monotonic()
        if time.monotonic()-result_seen>30:
            for p in reversed(alive):
                try:p.kill()
                except psutil.NoSuchProcess:pass
            psutil.wait_procs(alive,timeout=10);break
    assert time.monotonic()<deadline,'Owned worker did not drain; do not start another'
    time.sleep(2)
assert not any(p.is_running() and p.create_time()==identities[p.pid] for p in owned)
state=read(run/'state.json');assert state['checkpoint']==str(checkpoint)
state.update(status='superseded_for_diagnostic_repair',child_pid=None,updated_unix=time.time(),
    pause_reason='Running lacks geometric flight after repeated blocks; preserved checkpoint for normal-limit take-off intervention')
atomic_json(run/'state.json',state)
active.update(status='diagnostic_boundary',note=state['pause_reason']);atomic_json(HERE/'ACTIVE_RUN.json',active)
audit=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=str(run),
    checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    updates=state['total_training_iterations'],operation=str(operation),
    owned_identities=identities,all_owned_workers_drained=True,
    eval_completed=(operation/'result.json').exists())
atomic_json(OUT/'drain.json',audit);print(json.dumps(audit,indent=2),flush=True)
