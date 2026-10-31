"""Record the authorized R25 stop after its last worker has saved and exited."""
import json
import time
from pathlib import Path
import psutil
from status_io import atomic_json

HERE=Path(__file__).parent
RUN=HERE/'runs/locomotion_20261003_r25_owned_observations'
OUT=HERE/'diagnostics/r25_run_review_20261003_1919'

for proc in psutil.process_iter(['name','cmdline']):
    args=proc.info['cmdline'] or []
    assert not any(Path(a).name in ('worker_unitree.py','supervisor_unitree.py') for a in args), 'A training process is still running'
state=json.loads((RUN/'state.json').read_text())
folder=Path(state['operation_path'])
result=json.loads((folder/'result.json').read_text())
assert result['completed'] and Path(result['checkpoint']).is_file()
already_recorded=any(Path(a['folder'])==folder for a in state['attempts'])
if not already_recorded:
    state['total_training_iterations']+=result['iterations']
    state['attempts'].append(dict(operation='train',folder=str(folder),environments=result['num_envs'],
        speed=state['stage_speed'],completed=True,memory_failure=False,failure=None,
        worker_seconds=result['seconds'],scope='Final worker drained after supervisor stop; parent wall time unavailable'))
state.update(status='superseded_for_heading_repair',child_pid=None,checkpoint=result['checkpoint'],
    stopped_unix=time.time(),updated_unix=time.time(),
    stop_reason='Reproduced reset-frame direction omission and host-versus-VRAM ramp accounting fault; actor preserved for fresh-source requalification')
atomic_json(RUN/'state.json',state)
atomic_json(OUT/'drained_run.json',dict(run=str(RUN),state=state,last_worker_result=result,
    selected_actor=str(RUN/'155_run_150_train_003_part_06/candidate.pt'),
    selection_reason='Latest completed matching gate pair156/157 retained every old walking/standing case; running remains unqualified. All newer checkpointed training also preserved.'))
print(json.dumps(dict(status=state['status'],full_training_iterations=state['total_training_iterations'],checkpoint=state['checkpoint'])))
