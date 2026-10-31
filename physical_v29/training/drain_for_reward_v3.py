"""Drain the verified active chain before the authorized reward repair."""
import json, time, subprocess
from pathlib import Path
import psutil
from status_io import atomic_json

here = Path(__file__).resolve().parent
active = json.loads((here/'ACTIVE_RUN.json').read_text())
run = Path(active['run'])
assert run.name == 'locomotion_20261001_r10_auto_vram_duration'
state = json.loads((run/'state.json').read_text())
folder = Path(state['operation_path'])
child = psutil.Process(state['child_pid'])
assert str(run/'source/physical_v29/training/worker.py') in ' '.join(child.cmdline())
owned = [child] + child.children(recursive=True)
parents = []
for pid in {active['supervisor_pid'], state['supervisor_pid']}:
    p = psutil.Process(pid)
    assert str(run) in ' '.join(p.cmdline()) and 'supervisor.py' in ' '.join(p.cmdline())
    parents.append(p)
for p in parents: p.terminate()
psutil.wait_procs(parents, timeout=10)
state.update(status='draining_for_reward_v3', updated_unix=time.time())
atomic_json(run/'state.json', state)
print('Draining '+str(folder), flush=True)
deadline = time.time()+600
while not (folder/'result.json').exists() and not (folder/'failure.json').exists():
    if not child.is_running() or time.time()>deadline: break
    time.sleep(3)
result = json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
_, remaining = psutil.wait_procs(owned, timeout=12)
for p in remaining:
    try: p.terminate()
    except psutil.NoSuchProcess: pass
psutil.wait_procs(remaining, timeout=10)
state['child_pid'] = None
if result and result.get('completed') and state['operation']=='train':
    state['checkpoint'] = result['checkpoint']
    state['total_training_iterations'] += result['iterations']
state['attempts'].append(dict(operation=state['operation'], folder=str(folder),
    environments=state['environment_count'], speed=state['stage_speed'],
    completed=bool(result and result.get('completed')), finished_by_repair=True))
state.update(status='superseded_for_reward_v3', updated_unix=time.time())
atomic_json(run/'state.json', state)
print('Stopped; preserved '+str(state['checkpoint']), flush=True)
