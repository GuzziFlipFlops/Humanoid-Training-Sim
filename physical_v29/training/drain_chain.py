"""Finish an owned worker and preserve its artifacts before a source migration."""
import argparse, json, time
from pathlib import Path
import psutil
from status_io import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected_run', required=True)
    parser.add_argument('--reason', required=True)
    parser.add_argument('--wait_seconds', type=int, default=600)
    parser.add_argument('--worker_name', choices=['worker.py','worker_unitree.py'], default='worker.py')
    parser.add_argument('--supervisor_name', choices=['supervisor.py','supervisor_unitree.py'], default='supervisor.py')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    active = json.loads((here/'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
    run = Path(active['run']).resolve()
    assert run == Path(args.expected_run).resolve()
    state = json.loads((run/'state.json').read_text())
    folder = Path(state['operation_path'])
    assert folder.resolve().is_relative_to(run)
    child = psutil.Process(state['child_pid'])
    assert str(run/'source/physical_v29/training'/args.worker_name) in ' '.join(child.cmdline())
    owned = [child] + child.children(recursive=True)
    parents = []
    for pid in {active['supervisor_pid'], state['supervisor_pid']}:
        parent = psutil.Process(pid)
        assert str(run) in ' '.join(parent.cmdline()) and args.supervisor_name in ' '.join(parent.cmdline())
        parents.append(parent)
    for parent in parents: parent.terminate()
    psutil.wait_procs(parents, timeout=10)
    state.update(status='draining_for_diagnostic_repair', updated_unix=time.time(), repair_reason=args.reason)
    atomic_json(run/'state.json', state)
    print('Draining '+str(folder), flush=True)
    deadline = time.time()+args.wait_seconds
    while not (folder/'result.json').exists() and not (folder/'failure.json').exists():
        if not child.is_running() or time.time() > deadline: break
        time.sleep(3)
    result = json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
    _, remaining = psutil.wait_procs(owned, timeout=12)
    for p in remaining:
        try: p.terminate()
        except psutil.NoSuchProcess: pass
    psutil.wait_procs(remaining, timeout=10)
    state['child_pid'] = None
    if result and result.get('completed') and state['operation'] == 'train':
        state['checkpoint'] = result['checkpoint']
        state['total_training_iterations'] += result['iterations']
    state['attempts'].append(dict(operation=state['operation'], folder=str(folder),
        environments=state['environment_count'], speed=state.get('stage_speed',json.loads((folder/'run_config.json').read_text())['arguments']['speed']),
        completed=bool(result and result.get('completed')), finished_by_repair=True))
    state.update(status='superseded_for_diagnostic_repair', updated_unix=time.time())
    atomic_json(run/'state.json', state)
    print('Stopped; preserved '+str(state['checkpoint']), flush=True)


if __name__ == '__main__': main()
