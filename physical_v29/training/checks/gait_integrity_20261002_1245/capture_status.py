"""Read-only evidence capture; no simulator or CUDA tensor allocation."""
import datetime,json,subprocess,time
from pathlib import Path
import psutil

HERE=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


if __name__=='__main__':
    pointer=read(HERE/'ACTIVE_RUN.json');run=Path(pointer['run']);state=read(run/'state.json')
    processes=[]
    for p in psutil.process_iter(['pid','ppid','name','cmdline','memory_info']):
        command=' '.join(p.info['cmdline'] or [])
        if 'python' in (p.info['name'] or '').lower() and ('training/worker.py' in command.replace('\\','/') or 'training/supervisor.py' in command.replace('\\','/')):
            processes.append(dict(pid=p.pid,parent=p.info['ppid'],command=command,
                matches_active_run=str(run) in command,rss_mb=round(p.info['memory_info'].rss/2**20,1)))
    worker=state.get('child_pid');supervisor=state.get('supervisor_pid')
    verified_supervisor=any(p['pid']==supervisor and p['matches_active_run'] and 'supervisor.py' in p['command'] for p in processes)
    verified_worker=any(p['pid']==worker and p['matches_active_run'] and 'worker.py' in p['command'] for p in processes)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    used,free,util=map(int,gpu.splitlines()[0].split(','))
    ram=psutil.virtual_memory();operation=Path(state['operation_path']) if state.get('operation_path') else None
    progress=read(operation/'progress.json') if operation else None
    prior=HERE/'runs/locomotion_20261002_r15_motion_discovery'
    previous=read(prior/'state.json')
    report=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),active_pointer=pointer,
        state=state,state_age_s=time.time()-state['updated_unix'],verified_supervisor=verified_supervisor,
        verified_worker=verified_worker,processes=processes,
        gpu=dict(used_mb=used,free_mb=free,utilization=util),ram=dict(available_mb=round(ram.available/2**20),used_percent=ram.percent),
        progress=progress,progress_age_s=time.time()-progress['time'] if progress else None,
        benchmarks=read(run/'benchmarks.json'),gate_migration=read(run/'gate_migration.json'),
        reward_migration=read(run/'reward_migration.json'),latest_gate=read(run/'latest_gate.json'),
        prior=dict(run=str(prior),checkpoint=previous['checkpoint'],updates=previous['total_training_iterations']+previous.get('prior_training_iterations',0),
            latest_gate=read(prior/'latest_gate.json')),
        completion=dict(entire_chain_passed=state['status']=='all_requested_stages_passed',shutdown_issued=False))
    output=Path(__file__).with_name('status.json');output.write_text(json.dumps(report,indent=2))
    compact=dict(captured_utc=report['captured_utc'],run=str(run),status=state['status'],stage=state['stage'],operation=state.get('operation'),
        environments=state['environment_count'],next_training_environments=state.get('next_training_environment_count'),
        verified_supervisor=verified_supervisor,verified_worker=verified_worker,gpu=report['gpu'],ram=report['ram'],
        iteration=progress.get('iteration') if progress else None,progress_age_s=report['progress_age_s'],
        behavior=progress.get('behavior') if progress else None,completed_updates=state['total_training_iterations'],
        retention_training=progress.get('retention') if progress else None,
        inherited_updates=state.get('prior_training_iterations'),qualified_stages=[p['stage'] for p in state['promotions']],
        attempts=[dict(operation=a['operation'],environments=a['environments'],completed=a['completed']) for a in state['attempts'][-3:]])
    print(json.dumps(compact,indent=2))
