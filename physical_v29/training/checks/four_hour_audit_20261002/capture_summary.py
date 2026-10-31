"""Reproduce the four-hour audit from preserved files; no simulation process."""
import json, math, time, sys, subprocess
from pathlib import Path
import numpy as np
import psutil

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parents[1]
prior=TRAIN/'runs/locomotion_20261002_r14_memory_continuation'
run=TRAIN/'runs/locomotion_20261002_r15_motion_discovery'
old=json.loads((prior/'state.json').read_text())
state=json.loads((run/'state.json').read_text())
gates=[]
for folder in sorted(prior.glob('*gate_*')):
    if not (folder/'evaluation.json').exists():continue
    report=json.loads((folder/'evaluation.json').read_text())
    gates.append(dict(source=str(folder/'evaluation.json'),perturb=report['perturb'],passed=report['passed'],
        cases=[dict(c['statistics'],failures=c['failures']) for c in report['cases']]))
training=[a for a in old['attempts'] if a['operation']=='train' and '_train_' in Path(a['folder']).name]
steps=sum(a.get('environments',0)*json.loads((Path(a['folder'])/'result.json').read_text()).get('iterations',0)*24
    for a in training if a.get('completed') and (Path(a['folder'])/'result.json').exists())
def owned_duration(attempt):
    if 'seconds' in attempt:return attempt['seconds']
    folder=Path(attempt['folder']);start=json.loads((folder/'command.json').read_text())['started_unix']
    return max(0.,(folder/'result.json').stat().st_mtime-start)
time_train=sum(owned_duration(a) for a in training)
rows=[]
csv_path=HERE/'measured_joint_ranges.csv'
if csv_path.exists():
    import csv
    rows=list(csv.DictReader(csv_path.open()))
bound=[dict(speed=v,maximum_free_guidance=(2*.8+.35+.25)*2.5/(.9+.5*v)+2.)
       for v in [.081,.12,.15,.35,1.,1.5,2.5]]
summary=dict(recorded_unix=time.time(),prior_run=str(prior),active_run=str(run),
    prior_status=old['status'],preserved_checkpoint=old['checkpoint'],
    preserved_updates=old['total_training_iterations']+old.get('prior_training_iterations',0),
    completed_local_updates=old['total_training_iterations'],resource_recycles=len(old.get('resource_recycles',[])),
    measured_training_environment_steps=steps,owned_training_seconds=time_train,
    effective_training_environment_steps_per_second=steps/max(time_train,1.),
    note='Measured training time includes worker startup/recycling; gates, benchmarks and the prior 64 updates are separate.',
    gates=gates,old_posture_maximum_per_second=.65+.4+.7,
    new_stationary_motion_deficit_per_second=-1.8,new_posture_plus_deficit_upper_bound=-.05,
    finite_discovery_bound=bound,fall_event_cost=8.,
    cpu_validation=dict(tests=102,passed=True,command='env_isaaclab/Scripts/python.exe -m unittest discover -s physical_v29/training -p test*.py'),
    current_runtime=dict(status=state['status'],operation=state.get('operation'),stage=state['stage'],
        total_training_iterations=state['total_training_iterations'],prior_training_iterations=state.get('prior_training_iterations'),
        environment_count=state.get('environment_count'),gpu=state.get('gpu'),system_memory=state.get('system_memory'),
        progress=state.get('progress'),promotions=state.get('promotions'),updated_unix=state['updated_unix']),
    checkpoint_migration=json.loads((run/'001_reward_ppo_smoke/warm_start.json').read_text()),
    saved_joint_audit=str(HERE/'diagnosis.json'),measured_joint_ranges=str(csv_path),
    shutdown=False,shutdown_reason='Walking and all subsequent running gates have not passed.',
    limitations=['Reward repairs have not yet qualified walking.','Underlying allocation growth is contained, not identified.',
        'Exact saved CAD axes do not validate provisional contact geometry, inertia or balance authority.'])
owned=[]
for proc in psutil.process_iter(['name','cmdline','pid','ppid','create_time']):
    try:
        info=proc.info;command=' '.join(info['cmdline'] or []).replace('\\','/').lower()
        if 'python' in (info['name'] or '').lower() and str(run).replace('\\','/').lower() in command:
            mem=proc.memory_info()
            owned.append(dict(pid=info['pid'],parent_pid=info['ppid'],created_unix=info['create_time'],
                command=info['cmdline'],rss_mb=round(mem.rss/2**20),private_mb=round(getattr(mem,'private',mem.rss)/2**20)))
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
summary['verified_process_snapshot']=owned
summary['state_age_seconds']=time.time()-state['updated_unix']
host=psutil.virtual_memory()
summary['current_host_memory']=dict(available_mb=round(host.available/2**20),total_mb=round(host.total/2**20),percent=host.percent)
try:
    summary['current_gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True,timeout=5).strip()
except (OSError,subprocess.SubprocessError) as exc:summary['current_gpu_error']=str(exc)
for name in ['benchmarks.json','reward_migration.json','latest_gate.json']:
    if (run/name).exists():summary[name]=json.loads((run/name).read_text())
if (HERE/'optimizer_continuation.json').exists():summary['optimizer_continuation']=json.loads((HERE/'optimizer_continuation.json').read_text())
(HERE/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(summary=str(HERE/'summary.json'),preserved_updates=summary['preserved_updates'],
    current_status=state['status'],current_operation=state.get('operation'),current_environment_count=state.get('environment_count'))))
