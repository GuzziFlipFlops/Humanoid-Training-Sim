"""Recover the stopped r6 chain while Fusion remains open. Local single GPU."""
import json,subprocess,time
from pathlib import Path
from status_io import atomic_json
here=Path(__file__).resolve().parent
previous=here/'runs/locomotion_20261001_r6_measured_steps'
state=json.loads((previous/'state.json').read_text())
assert state['status']=='failed' and state['child_pid'] is None
assert (previous/'009_walk_015_train_002/memory_recovery.pt').exists()
run=here/'runs/locomotion_20261001_r8_fusion_memory_recovery'
run.mkdir(exist_ok=True)
assert not (run/'state.json').exists()
python=here.parents[1]/'env_isaaclab/Scripts/python.exe'
command=[str(python),'-u','-X','utf8',str(here/'supervisor.py'),'--output',str(run),'--reserve_mb','300','--max_hours','12','--iterations_per_block','25','--validated_checks',str(here/'checks/reward_repair_20261001/reused_checks'),'--prior_run',str(previous),'--initial_checkpoint',state['promotions'][0]['checkpoint'],'--train_envs','3072','--resume_run',str(previous)]
with (run/'supervisor_stdout.log').open('w',encoding='utf-8') as stdout,(run/'supervisor_stderr.log').open('w',encoding='utf-8') as stderr:
 proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,cwd=str(here.parents[1]),creationflags=subprocess.CREATE_NO_WINDOW)
atomic_json(run/'launch.json',dict(command=command,pid=proc.pid,time=time.time(),reason='Fusion pushed 4096 below the 300MB reserve; retain frozen worker/asset/rewards, completed stand gates and failed walk attempt'))
atomic_json(here/'ACTIVE_RUN.json',dict(run=str(run),supervisor_pid=proc.pid,started_unix=time.time(),prior_run=str(previous),note='Memory recovery during concurrent Fusion work; stage gates still required'))
print(json.dumps(dict(run=str(run),pid=proc.pid)))
