"""Stop only the verified R28 supervisor; let its current evaluation finish."""
import datetime
import json
import time
from pathlib import Path

import psutil

BASE = Path(__file__).resolve().parents[4]
TRAINING = BASE / 'physical_v29/training'
RUN = TRAINING / 'runs/locomotion_20261004_r28_heading_alignment_cache_fixed'
DIAG = Path(__file__).resolve().parent


def output_path(command):
    return Path(command[command.index('--output') + 1]).resolve()


state = json.loads((RUN / 'state.json').read_text())
assert state['operation'] == 'eval', 'Drain at an evaluation boundary; preserve all training updates'
supervisor = psutil.Process(state['supervisor_pid'])
assert any('supervisor_unitree.py' in x for x in supervisor.cmdline())
assert output_path(supervisor.cmdline()) == RUN.resolve()
operation = Path(state['operation_path'])
worker = psutil.Process(state['child_pid'])
assert output_path(worker.cmdline()) == operation.resolve()
assert any('worker_unitree.py' in x for x in worker.cmdline())
owned = [worker] + worker.children(recursive=True)
owned_identity = {p.pid: p.create_time() for p in owned}
before = json.loads((RUN / 'state.json').read_text())
(DIAG / 'r28_state_before_capacity_maintenance.json').write_text(json.dumps(before, indent=2))
supervisor.kill()
supervisor.wait(timeout=10)
print(json.dumps(dict(supervisor_stopped=supervisor.pid, evaluation_left_running=str(operation))), flush=True)
deadline = time.monotonic() + 600
result_seen = None
while True:
    alive = [p for p in owned if p.is_running() and p.create_time() == owned_identity[p.pid]]
    if not alive:
        break
    if (operation / 'result.json').exists() or (operation / 'failure.json').exists():
        result_seen = result_seen or time.monotonic()
        if time.monotonic() - result_seen > 20:
            for process in reversed(alive):
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=10)
            break
    assert time.monotonic() < deadline, 'Evaluation drain timed out; no replacement worker was started'
    time.sleep(3)
assert not any(p.is_running() and p.create_time() == owned_identity[p.pid] for p in owned)
state = json.loads((RUN / 'state.json').read_text())
state.update(status='superseded_for_diagnostic_repair', child_pid=None,
             pause_reason='Host-RAM capacity scheduling repair; full runtime bytes and checkpoint preserved',
             updated_unix=time.time())
(RUN / 'state.json').write_text(json.dumps(state, indent=2))
result = json.loads((operation / 'result.json').read_text()) if (operation / 'result.json').exists() else None
audit = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             supervisor_pid=supervisor.pid, owned_worker_ids=owned_identity,
             operation=str(operation), evaluation_completed=bool(result and result.get('completed')),
             evaluation_passed=None if result is None else result.get('passed'),
             checkpoint=state['checkpoint'], all_owned_workers_drained=True,
             total_training_iterations=state['total_training_iterations'])
(DIAG / 'r28_capacity_maintenance_drain.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2), flush=True)
