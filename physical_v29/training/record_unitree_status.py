"""Save verified handoff process/progress evidence without changing training."""
from pathlib import Path
import datetime, json, hashlib, time, subprocess
import psutil

HERE=Path(__file__).parent
active=json.loads((HERE/'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
run=Path(active['run'])
state=json.loads((run/'state.json').read_text(encoding='utf-8'))
assert state['status']=='running' and state['operation']=='train'
assert state['stage']!='preflight','Record full training, not the capacity smoke'
supervisor=psutil.Process(state['supervisor_pid'])
assert str(run).lower() in ' '.join(supervisor.cmdline()).lower()
worker=psutil.Process(state['child_pid'])
assert 'worker_unitree.py' in ' '.join(worker.cmdline())
assert str(run).lower() in ' '.join(worker.cmdline()).lower()
descendants=[]
for proc in [worker]+worker.children(recursive=True):
    descendants.append(dict(pid=proc.pid,parent_pid=proc.ppid(),name=proc.name(),command=proc.cmdline()))
progress_path=Path(state['operation_path'])/'progress.json'
progress=json.loads(progress_path.read_text(encoding='utf-8'))
assert time.time()-progress['time']<60,'Stale progress'
assert state['environment_count']==progress['num_envs']
assert progress['num_envs'] in [8192,6144,4096,3072,2048,1024,512]
assert progress['gpu']['free_mb']>=300
if run.name=='locomotion_20261002_r22_unitree_nominal':
    assert progress['policy']['nominal_pose_max_abs_drift_rad']==0.
hashes={}
for name in ['environment_unitree.py','unitree_reward_port.py','worker_unitree.py']:
    workspace=HERE/name
    frozen=run/'source/physical_v29/training'/name
    assert workspace.read_bytes()==frozen.read_bytes(),f'Frozen code differs: {name}'
    hashes[name]=hashlib.sha256(frozen.read_bytes()).hexdigest()
gates=[dict(folder=str(p.parent),evaluation=json.loads(p.read_text())) for p in sorted(run.glob('*/evaluation.json'))]
record=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    active=active,state=state,progress=progress,supervisor_process=dict(pid=supervisor.pid,command=supervisor.cmdline()),
    worker_process_tree=descendants,frozen_source_hashes=hashes,
    independent_evaluations=[dict(folder=g['folder'],passed=g['evaluation']['passed'],duration_s=g['evaluation']['duration_s'],cases=g['evaluation']['cases']) for g in gates],
    status_scope='Verified running full PPO; qualification is only the saved independent gates, not this dated process/progress snapshot')
output=HERE/('diagnostics/unitree_termination_20261002/handoff_status.json' if run.name in ['locomotion_20261002_r21_unitree_lifetime','locomotion_20261002_r22_unitree_nominal'] else 'diagnostics/unitree_port_20261002/handoff_status.json')
output.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(dict(status=state['status'],stage=state['stage'],environments=progress['num_envs'],
     saved_updates=state['total_training_iterations'],current_worker_iteration=progress['iteration'],
     free_vram_mib=progress['gpu']['free_mb'],available_ram_mib=progress['gpu']['ram_available_mb'],
     checked_at_utc=record['checked_at_utc'],evidence=str(output)),indent=2))
