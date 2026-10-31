"""Drain a saved R30 candidate only after another nominal slow-tracking failure."""
import datetime
import json
import time
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT/'physical_v29/training/runs/locomotion_20261004_r30_exact_graph_execution'
OUT = Path(__file__).parent

def read(path): return json.loads(path.read_text())
def output(process):
    args = process.cmdline()
    return Path(args[args.index('--output') + 1]).resolve()

initial = read(RUN/'state.json')
supervisor = psutil.Process(initial['supervisor_pid'])
identity = supervisor.create_time()
assert output(supervisor) == RUN.resolve()
assert any(Path(a).name == 'supervisor_unitree.py' for a in supervisor.cmdline())
deadline = time.monotonic() + 900
while True:
    assert supervisor.is_running() and supervisor.create_time() == identity
    state = read(RUN/'state.json')
    assert state['status'] == 'running' and state['stage'] == 'walk_065', 'Stage progressed; preserve improving run and re-assess'
    operation = Path(state.get('operation_path', RUN))
    if state.get('operation') == 'eval' and operation.name.endswith('_push'):
        nominal_suffix = operation.name.split('_', 1)[1][:-4]+'nominal'
        nominal_candidates = sorted(RUN.glob('*_'+nominal_suffix+'/result.json'))
        nominal_path = nominal_candidates[-1] if nominal_candidates else operation/'missing_nominal.json'
        if nominal_path.exists():
            nominal = read(nominal_path)
            slow = next(c for c in nominal['cases'] if abs(c['statistics']['command_m_s']-.15) < 1e-5)
            if not slow['passed'] and any(f in slow['failures'] for f in ['speed', 'world_forward_speed']):
                try:
                    worker = psutil.Process(state['child_pid'])
                    assert output(worker) == operation.resolve()
                    assert any(Path(a).name == 'worker_unitree.py' for a in worker.cmdline())
                    break
                except psutil.NoSuchProcess:
                    pass
    assert time.monotonic() < deadline, 'No matching failure boundary; do not interrupt learning'
    time.sleep(3)

owned = [worker] + worker.children(recursive=True)
owned_ids = {p.pid: p.create_time() for p in owned}
(OUT/'r30_state_before_rehearsal.json').write_text(json.dumps(state, indent=2))
supervisor.kill()
supervisor.wait(timeout=10)
print(json.dumps(dict(supervisor_stopped=supervisor.pid, evaluation_left_running=str(operation))), flush=True)
deadline = time.monotonic() + 600
result_seen = None
while True:
    alive = [p for p in owned if p.is_running() and p.create_time() == owned_ids[p.pid]]
    if not alive: break
    if (operation/'result.json').exists() or (operation/'failure.json').exists():
        result_seen = result_seen or time.monotonic()
        if time.monotonic()-result_seen > 20:
            for p in reversed(alive):
                try: p.kill()
                except psutil.NoSuchProcess: pass
            psutil.wait_procs(alive, timeout=10)
            break
    assert time.monotonic() < deadline, 'Drain timed out; do not start replacement worker'
    time.sleep(3)

assert not any(p.is_running() and p.create_time() == owned_ids[p.pid] for p in owned)
state = read(RUN/'state.json')
state.update(status='superseded_for_diagnostic_repair', child_pid=None,
             pause_reason='Repeated slow-command overspeed; training-only exact-command rehearsal preserving the full learned checkpoint',
             updated_unix=time.time())
from sys import path
path.insert(0, str(ROOT/'physical_v29/training'))
from status_io import atomic_json
atomic_json(RUN/'state.json', state)
result = read(operation/'result.json') if (operation/'result.json').exists() else None
audit = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             supervisor_pid=supervisor.pid, owned_worker_identities=owned_ids,
             operation=str(operation), nominal_slow_failure=slow,
             evaluation_completed=bool(result and result.get('completed')),
             checkpoint=state['checkpoint'], all_owned_workers_drained=True,
             total_training_iterations=state['total_training_iterations'])
(OUT/'r30_drain.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2), flush=True)
