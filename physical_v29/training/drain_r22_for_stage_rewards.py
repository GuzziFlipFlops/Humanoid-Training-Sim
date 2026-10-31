"""Stop the verified R22 supervisor, then let its worker save at a boundary."""
import json, time
from pathlib import Path
import psutil
from status_io import atomic_json

HERE = Path(__file__).resolve().parent
active = json.loads((HERE/'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
run = Path(active['run'])
assert run.name == 'locomotion_20261002_r22_unitree_nominal'
state = json.loads((run/'state.json').read_text())
atomic_json(run/'state_before_stage_reward_repair.json', state)
folder = Path(state['operation_path'])
supervisor = psutil.Process(state['supervisor_pid'])
assert str(run) in ' '.join(supervisor.cmdline()) and 'supervisor_unitree.py' in ' '.join(supervisor.cmdline())
supervisors = [supervisor]
parent = supervisor.parent()
if parent and 'supervisor_unitree.py' in ' '.join(parent.cmdline()) and str(run) in ' '.join(parent.cmdline()):
    supervisors.append(parent)
child = psutil.Process(state['child_pid'])
assert str(run/'source/physical_v29/training/worker_unitree.py') in ' '.join(child.cmdline())
owned = [(p, p.create_time()) for p in [child]+child.children(recursive=True)]
for p in supervisors:
    p.terminate()
psutil.wait_procs(supervisors, timeout=8)
state.update(status='draining_for_stage_reward_repair', updated_unix=time.time())
atomic_json(run/'state.json', state)
print('Draining owned worker '+str(folder), flush=True)
deadline = time.time()+600
while not (folder/'result.json').exists() and not (folder/'failure.json').exists():
    if not child.is_running() or time.time()>deadline:
        break
    time.sleep(2)
result = json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
_, remaining = psutil.wait_procs([p for p,_ in owned], timeout=12)
for p in remaining:
    try:
        original = next(t for q,t in owned if q.pid == p.pid)
        if p.create_time() == original:
            p.terminate()
    except psutil.NoSuchProcess:
        pass
psutil.wait_procs(remaining, timeout=10)
state['child_pid'] = None
if result and result.get('completed') and state['operation']=='train':
    state['checkpoint'] = result['checkpoint']
    state['total_training_iterations'] += result['iterations']
state['attempts'].append(dict(operation=state['operation'],folder=str(folder),
    completed=bool(result and result.get('completed')),finished_by_repair=True))
state.update(status='superseded_for_stage_reward_repair',updated_unix=time.time())
atomic_json(run/'state.json',state)
atomic_json(run/'stage_reward_repair_drain.json',dict(result=result,checkpoint=state['checkpoint'],
    full_updates=state['total_training_iterations'],all_owned_workers_stopped=True,
    reason='User-authorized reward repair; old checkpoints and logs preserved'))
print('Stopped; preserved '+str(state['checkpoint']),flush=True)
