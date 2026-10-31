"""Pause only the owned supervisor; preserve its in-flight worker until save."""
import datetime
import json
from pathlib import Path
import time

import psutil

BASE = Path(__file__).resolve().parents[4]
DIAG = Path(__file__).resolve().parent
pointer = json.loads((BASE / 'physical_v29/training/ACTIVE_RUN.json').read_text())
run = Path(pointer['run']).resolve()
state_path = run / 'state.json'
state = json.loads(state_path.read_text())
assert state['status'] == 'running', state['status']
assert state['operation'] in ('train', 'eval'), state['operation']

def output_path(process):
    command = process.cmdline()
    return Path(command[command.index('--output') + 1]).resolve(), command

supervisor = psutil.Process(state['supervisor_pid'])
path, command = output_path(supervisor)
assert path == run and any('supervisor_unitree.py' in a for a in command)
worker = psutil.Process(state['child_pid'])
operation = Path(state['operation_path']).resolve()
path, command = output_path(worker)
assert path == operation and any('worker_unitree.py' in a for a in command)
owned = [worker] + worker.children(recursive=True)
identities = [dict(pid=p.pid, created=p.create_time(), command=p.cmdline()) for p in owned]
(DIAG / 'state_before_pause.json').write_text(json.dumps(state, indent=2))
supervisor.kill()
supervisor.wait(timeout=10)
assert not supervisor.is_running()
state.update(status='draining_for_performance_review', updated_unix=time.time(),
             pause_reason='User requested stopping recurring checks and proposed an optimization pause; save current block before profiling')
state_path.write_text(json.dumps(state, indent=2))
audit = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             run=str(run), operation=str(operation), mode=state['operation'],
             supervisor_pid=supervisor.pid, supervisor_stopped=True,
             owned_worker_identities=identities, prior_iterations=state['total_training_iterations'],
             prior_checkpoint=state['checkpoint'], scope='Only the owned supervisor was stopped. Worker is finishing its checkpoint normally.')
(DIAG / 'pause_request.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
