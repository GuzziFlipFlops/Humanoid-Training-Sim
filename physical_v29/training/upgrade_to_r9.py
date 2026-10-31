"""Drain exactly the owned r8 worker, then replace only its supervisor."""
import csv,json,subprocess,time,traceback
from pathlib import Path
import psutil
from status_io import atomic_json
from vram_ramp import VramRamp

HERE=Path(__file__).resolve().parent
PRIOR=HERE/'runs/locomotion_20261001_r8_fusion_memory_recovery'
RUN=HERE/'runs/locomotion_20261001_r9_auto_vram'

def read(p): return json.loads(p.read_text())
def report(**fields): atomic_json(RUN/'upgrade.json',dict(time=time.time(),**fields))

def main():
 RUN.mkdir(exist_ok=True)
 assert not (RUN/'state.json').exists(),'Never overwrite a run'
 active=read(HERE/'ACTIVE_RUN.json');assert Path(active['run'])==PRIOR
 before=read(PRIOR/'state.json');assert before['status']=='running'
 parents=[]
 for pid in {before['supervisor_pid'],active['supervisor_pid']}:
  proc=psutil.Process(pid);command=' '.join(proc.cmdline())
  assert str(HERE/'supervisor.py') in command and str(PRIOR) in command,(pid,command)
  parents.append(proc)
 child=psutil.Process(before['child_pid']);child_created=child.create_time()
 assert str(PRIOR/'source/physical_v29/training/worker.py') in ' '.join(child.cmdline())
 # Kill only the two verified supervisor processes, not their worker tree.
 for proc in parents: proc.terminate()
 psutil.wait_procs(parents,timeout=10)
 assert not any(p.is_running() and p.status()!=psutil.STATUS_ZOMBIE for p in parents)
 state=read(PRIOR/'state.json');assert state['child_pid']==child.pid
 folder=Path(state['operation_path']);started=time.time()
 state.update(status='draining_for_supervisor_upgrade',upgrade_to=str(RUN));atomic_json(PRIOR/'state.json',state)
 report(status='draining_owned_worker',pid=child.pid,operation=state['operation'],folder=str(folder))
 while not (folder/'result.json').exists() and not (folder/'failure.json').exists():
  if not child.is_running(): break
  if time.time()-started>900: raise RuntimeError('Owned worker did not drain in 900s')
  report(status='draining_owned_worker',pid=child.pid,operation=state['operation'],folder=str(folder))
  time.sleep(3)
 result=read(folder/'result.json') if (folder/'result.json').exists() else None
 failure=read(folder/'failure.json') if (folder/'failure.json').exists() else None
 # Kit can hang after writing the completed checkpoint. Only the verified PID tree.
 _,alive=psutil.wait_procs([child],timeout=12)
 if alive:
  assert child.create_time()==child_created
  subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,stdout=subprocess.DEVNULL)
 state['child_pid']=None
 if result and result.get('completed') and state['operation']=='train':
  assert Path(result['checkpoint']).exists()
  state['checkpoint']=result['checkpoint'];state['total_training_iterations']+=result['iterations']
 state['attempts'].append(dict(operation=state['operation'],folder=str(folder),environments=state['environment_count'],speed=state['stage_speed'],completed=bool(result and result.get('completed')),failure=failure,finished_by_upgrade=True))
 state.update(status='stopped_for_supervisor_upgrade',updated_unix=time.time())
 atomic_json(PRIOR/'state.json',state)
 # Replay real allocated training samples to retain an already-qualified 3-minute ramp.
 train=next((a for a in reversed(state['attempts']) if a['operation']=='train' and a.get('completed')),None)
 count=train['environments'] if train else 3072
 ramp=VramRamp();evidence=None
 if train:
  tf=Path(train['folder']);progress=[json.loads(x) for x in (tf/'progress.jsonl').read_text().splitlines()]
  first=next((x['time'] for x in progress if x['iteration']>=1),float('inf'))
  # Match this operation by its command's start and last progress timestamp.
  end=progress[-1]['time']+3
  for row in csv.DictReader((PRIOR/'gpu.csv').open()):
   t=float(row['unix'])
   if row['operation']=='train' and int(row['environments'])==count and first<=t<=end:
    evidence=ramp.observe(t,count,float(row['free_mb']))
 target=ramp.consume()
 next_count=target or count
 # Recheck current free memory before using the previous qualified decision.
 if target:
  from supervisor import gpu
  if gpu().get('free_mb',0)<evidence['required_free_mb']:next_count=count
 command=[str(HERE.parents[1]/'env_isaaclab/Scripts/python.exe'),'-u','-X','utf8',str(HERE/'supervisor.py'),
          '--output',str(RUN),'--reserve_mb','300','--max_hours','18','--iterations_per_block','25',
          '--validated_checks',str(HERE/'checks/reward_repair_20261001/reused_checks'),
          '--prior_run',str(PRIOR),'--initial_checkpoint',state['promotions'][0]['checkpoint'],
          '--train_envs',str(next_count),'--resume_run',str(PRIOR)]
 with (RUN/'supervisor_stdout.log').open('w',encoding='utf-8') as out,(RUN/'supervisor_stderr.log').open('w',encoding='utf-8') as err:
  proc=subprocess.Popen(command,stdout=out,stderr=err,cwd=str(HERE.parents[1]),creationflags=subprocess.CREATE_NO_WINDOW)
 atomic_json(RUN/'launch.json',dict(command=command,pid=proc.pid,time=time.time(),ramp_from_prior=evidence,initial_envs=next_count))
 atomic_json(HERE/'ACTIVE_RUN.json',dict(run=str(RUN),supervisor_pid=proc.pid,started_unix=time.time(),prior_run=str(PRIOR),note='Automatic three-minute VRAM ramp enabled; 300MB reserve and stage gates retained'))
 report(status='new_supervisor_started',pid=proc.pid,initial_envs=next_count,ramp_evidence=evidence)

if __name__=='__main__':
 try:main()
 except BaseException:
  report(status='failed',error=traceback.format_exc());raise
