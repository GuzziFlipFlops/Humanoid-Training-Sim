"""Preserve a verified chain and drain its current worker before a repair."""
import argparse, json, time
from pathlib import Path
import psutil
from status_io import atomic_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expected_run',required=True)
    parser.add_argument('--reason',required=True)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    active=json.loads((here/'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
    run=Path(active['run']);assert run.name==args.expected_run
    state=json.loads((run/'state.json').read_text())
    assert state['status']=='running'
    atomic_json(run/'state_before_research_repair_20261003.json',state)
    sup=psutil.Process(state['supervisor_pid'])
    assert 'supervisor_unitree.py' in ' '.join(sup.cmdline()) and str(run) in ' '.join(sup.cmdline())
    supervisors=[sup];parent=sup.parent()
    if parent and 'supervisor_unitree.py' in ' '.join(parent.cmdline()) and str(run) in ' '.join(parent.cmdline()):supervisors.append(parent)
    folder=Path(state['operation_path']);child=psutil.Process(state['child_pid'])
    assert str(run/'source/physical_v29/training/worker_unitree.py') in ' '.join(child.cmdline())
    owned=[(p,p.create_time()) for p in [child]+child.children(recursive=True)]
    for p in supervisors:p.terminate()
    psutil.wait_procs(supervisors,timeout=8)
    state.update(status='draining_for_research_repair',updated_unix=time.time());atomic_json(run/'state.json',state)
    print('Draining current owned worker: '+str(folder),flush=True)
    deadline=time.time()+600
    while not (folder/'result.json').exists() and not (folder/'failure.json').exists():
        if not child.is_running() or time.time()>deadline:break
        time.sleep(2)
    result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
    _,remaining=psutil.wait_procs([p for p,_ in owned],timeout=12)
    for p in remaining:
        try:
            if p.create_time()==next(t for q,t in owned if q.pid==p.pid):p.terminate()
        except psutil.NoSuchProcess:pass
    _,remaining=psutil.wait_procs(remaining,timeout=10);assert not remaining,'Owned worker did not stop'
    state['child_pid']=None
    if result and result.get('completed') and state['operation']=='train':
        if '_train_' in folder.name:
            state['checkpoint']=result['checkpoint'];state['total_training_iterations']+=result['iterations']
        else:state['diagnostic_checkpoint']=result['checkpoint']
    state['attempts'].append(dict(operation=state['operation'],folder=str(folder),completed=bool(result and result.get('completed')),finished_by_repair=True))
    state.update(status='superseded_for_research_repair',updated_unix=time.time());atomic_json(run/'state.json',state)
    atomic_json(run/'research_repair_drain_20261003.json',dict(result=result,checkpoint=state['checkpoint'],
        full_updates=state['total_training_iterations'],all_owned_workers_stopped=True,reason=args.reason))
    print(json.dumps(dict(stopped=True,checkpoint=state['checkpoint'],saved_updates=state['total_training_iterations'])),flush=True)

if __name__=='__main__':main()
