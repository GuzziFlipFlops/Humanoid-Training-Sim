"""Local, persistent, single-GPU stand/walk/run chain with independent gates.

No policy is promoted from its reward, exit code or an assisted rollout. A failed
new command continues training; loss of a prior command blocks promotion while
the candidate keeps learning. The last fully qualified policy stays preserved.
Memory failures lower the count. Three minutes of measured training headroom
allow one upward step at a checkpoint boundary. No cloud logging/publication.
"""
import argparse,json,time,os,subprocess,sys,hashlib,csv,statistics,traceback,shutil,psutil
from pathlib import Path
from status_io import atomic_json
from vram_ramp import VramRamp,estimate_increment,training_iterations
HERE=Path(__file__).resolve().parent
PYTHON=HERE.parents[1]/'env_isaaclab/Scripts/python.exe'
STAGES=[('stand',0.),('walk_015',.15),('walk_035',.35),('walk_065',.65),('walk_100',1.),('run_150',1.5),('run_200',2.),('run_250',2.5)]
ENV_FALLBACK=[8192,6144,4096,3072,2048,1024,512]

def atomic(path,data):
 atomic_json(path,data)

def gpu():
 try:
  x=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()[0]
  u,f,v=map(int,x.split(','));return {'used_mb':u,'free_mb':f,'utilization':v}
 except Exception:return {}

def is_memory_failure(folder):
 text=''.join((folder/n).read_text(errors='replace') for n in ['stderr.log','failure.json'] if (folder/n).exists()).lower()
 return any(k in text for k in ['out of memory','cuda error: memory','vram_reserve','bad allocation','failed to allocate'])

def select_environment_count(eligible,requested=None):
 assert eligible,'No benchmark preserves the VRAM reserve'
 if requested is not None:
  allowed=[b['environments'] for b in eligible if b['environments']<=requested]
  if not allowed:
   assert requested in ENV_FALLBACK and requested<min(b['environments'] for b in eligible)
   return requested # conservatively lower than measured capacity; worker guard still applies
  return max(allowed)
 best=max(b['environment_steps_per_second'] for b in eligible)
 return max(b['environments'] for b in eligible if b['environment_steps_per_second']>=best*.8)

class Chain:
 def __init__(self,root,reserve,hours,chunk,validated_checks='',prior_run='',initial_checkpoint='',train_envs=None,resume_run=''):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
  assert not (self.root/'state.json').exists(),'Use a fresh chain directory; preserve older attempts'
  self.start=time.time();self.deadline=self.start+hours*3600;self.reserve=reserve;self.chunk=chunk
  self.state=dict(status='starting',supervisor_pid=os.getpid(),started_unix=self.start,reserve_mb=reserve,stage='preflight',environment_count=None,checkpoint=None,promotions=[],attempts=[],total_training_iterations=0)
  self.count=0;self.write()
  self.validated_checks=Path(validated_checks) if validated_checks else None
  self.prior_run=Path(prior_run) if prior_run else None
  self.initial_checkpoint=initial_checkpoint
  self.train_envs=train_envs
  self.resume_run=Path(resume_run) if resume_run else None
  self.ramp=VramRamp(reserve_mb=reserve)
  self.step_seconds=None;self.step_count=None
  # Worker reads this frozen source tree, rather than future workspace edits.
  snapshot=self.root/'source/physical_v29';(snapshot/'training/assets').mkdir(parents=True)
  (snapshot/'assets').mkdir()
  for n in ['environment.py','worker.py','evaluation.py','evaluation_steps.py','walking_rewards.py','reference_geometry.py','mass_estimate.json','status_io.py']:
   shutil.copy2(HERE/n,snapshot/'training'/n)
  for n in ['robot_training.urdf','motion_reference.npz']:
   shutil.copy2(HERE/'assets'/n,snapshot/'training/assets'/n)
  for n in ['hardware_actuators.py','cad_differential.py']:
   shutil.copy2(HERE.parent/n,snapshot/n)
  for n in ['actuator_limits.json','differential_geometry.json','export_metadata.json']:
   shutil.copy2(HERE.parent/'assets'/n,snapshot/'assets'/n)
  self.worker=snapshot/'training/worker.py'
  frozen={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['environment.py','worker.py','evaluation.py','evaluation_steps.py','supervisor.py','status_io.py','walking_rewards.py','reference_geometry.py','mass_estimate.json','assets/robot_training.urdf','assets/motion_reference.npz']}
  for n in ['actuator_limits.json','differential_geometry.json']:
   frozen['physical_assets/'+n]=hashlib.sha256((HERE.parent/'assets'/n).read_bytes()).hexdigest()
  for n in ['hardware_actuators.py','cad_differential.py','assets/export_metadata.json']:
   frozen['physical_v29/'+n]=hashlib.sha256((HERE.parent/n).read_bytes()).hexdigest()
  frozen['vram_ramp.py']=hashlib.sha256((HERE/'vram_ramp.py').read_bytes()).hexdigest()
  for name in ['supervisor.py','vram_ramp.py']:shutil.copy2(HERE/name,self.root/name)
  atomic(self.root/'chain_config.json',dict(stages=STAGES,reserve_mb=reserve,max_hours=hours,iterations_per_block=chunk,train_environment_fallback=ENV_FALLBACK,automatic_upward_ramp=dict(hold_seconds=180,max_envs=8192,one_tier_at_a_time=True,checkpoint_boundary=True,training_samples_only=True),eval_envs=128,evaluation_seconds=20,seeds=[901,1301],sources_sha256=frozen,no_cameras=True,local_only=True,resume_run=str(self.resume_run) if self.resume_run else None))
 def write(self):
  self.state['updated_unix']=time.time();atomic(self.root/'state.json',self.state)
 def child(self,mode,n,speed=0.,checkpoint='',iterations=None,perturb=False,seed=71,label=''):
  self.count+=1;folder=self.root/f'{self.count:03d}_{label or mode}';folder.mkdir()
  command=[str(PYTHON),'-u','-X','utf8',str(self.worker),'--headless','--device','cuda:0','--mode',mode,'--num_envs',str(n),'--speed',str(speed),'--output',str(folder),'--reserve_mb',str(self.reserve),'--seed',str(seed)]
  if checkpoint:command+=['--checkpoint',str(checkpoint)]
  if mode=='train':command+=['--iterations',str(iterations or self.chunk)]
  if perturb:command+=['--perturb']
  atomic(folder/'command.json',{'argv':command,'started_unix':time.time()})
  with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
   p=subprocess.Popen(command,stdout=stdout,stderr=stderr,cwd=str(HERE.parents[1]),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
   self.state.pop('progress',None)
   self.state.update(status='running',child_pid=p.pid,operation=mode,environment_count=n,operation_path=str(folder));self.write()
   started=time.time();result=None;failure=None;last_progress=None
   with (self.root/'gpu.csv').open('a',newline='') as f:
    w=csv.writer(f)
    if f.tell()==0:w.writerow(['unix','operation','pid','environments','used_mb','free_mb','utilization'])
    while p.poll() is None:
     m=gpu();w.writerow([time.time(),mode,p.pid,n,m.get('used_mb'),m.get('free_mb'),m.get('utilization')]);f.flush()
     self.state['gpu']=m
     host=psutil.virtual_memory()
     self.state['system_memory']=dict(available_mb=round(host.available/2**20),percent_used=host.percent)
     if (folder/'progress.json').exists():
      try:last_progress=json.loads((folder/'progress.json').read_text());self.state['progress']=last_progress
      except (ValueError,PermissionError):pass
     if mode=='train':
      fresh=last_progress is not None and time.time()-last_progress['time']<60 and last_progress['iteration']>=1
      if fresh:
       self.step_seconds=last_progress['collect_seconds']+last_progress['learn_seconds'];self.step_count=n
       self.state['last_training_step_seconds']=self.step_seconds
      self.state['vram_ramp']=self.ramp.observe(time.monotonic(),n,m.get('free_mb'),fresh)
     self.write()
     if (folder/'result.json').exists():
      result=json.loads((folder/'result.json').read_text());break
     if (folder/'failure.json').exists():
      failure=json.loads((folder/'failure.json').read_text());break
     # A failed startup or frozen update never masquerades as progress.
     latest=last_progress.get('time',started) if last_progress else started
     if time.time()-latest>600 or time.time()>self.deadline:
      failure={'error':'NO_PROGRESS_600_SECONDS' if time.time()<self.deadline else 'CHAIN_TIME_LIMIT'};atomic(folder/'supervisor_failure.json',failure);break
     time.sleep(3)
   # Kit sometimes hangs after writing evidence. Stop only this owned tree.
   if p.poll() is None:
    try:p.wait(timeout=12)
    except subprocess.TimeoutExpired:
     if os.name=='nt':subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
     else:p.terminate()
   if result is None and (folder/'result.json').exists():result=json.loads((folder/'result.json').read_text())
   if failure is None and (folder/'failure.json').exists():failure=json.loads((folder/'failure.json').read_text())
  item=dict(operation=mode,folder=str(folder),environments=n,speed=speed,seconds=time.time()-started,completed=bool(result and result.get('completed')),failure=failure,memory_failure=is_memory_failure(folder))
  self.state['attempts'].append(item);self.state['child_pid']=None;self.write()
  return result,item,folder
 def checked(self,*a,**kw):
  r,item,folder=self.child(*a,**kw)
  if not r or not r.get('completed'):raise RuntimeError(f'Worker failed; evidence: {folder}, {item}')
  return r,folder
 def gates(self,speed,checkpoint,label):
  reports=[]
  for perturb,seed in [(False,901),(True,1301)]:
   r,f=self.checked('eval',128,speed,checkpoint,perturb=perturb,seed=seed,label=label+('_push' if perturb else '_nominal'))
   reports.append(r)
   if speed==0 and not r['passed']:break # no prior command yet; nominal failure already prevents promotion
  return reports
 def run(self):
  if self.validated_checks:
   reused={}
   for name in ['preflight_07','ppo_smoke_01','reload_01']:
    p=self.validated_checks/name;r=json.loads((p/'result.json').read_text());cfg=json.loads((p/'run_config.json').read_text())
    assert r['completed'] and cfg['urdf_sha256']==hashlib.sha256((HERE/'assets/robot_training.urdf').read_bytes()).hexdigest()
    if name=='preflight_07':assert r['passed']
    reused[name]={'folder':str(p),'result':r,'config':cfg}
   atomic(self.root/'reused_validation.json',reused)
   # The asset preflight remains valid, but changed reward/worker code always
   # receives a fresh small PPO/load smoke test in this run.
   r,folder=self.checked('train',64,.15,self.initial_checkpoint,iterations=4,label='reward_ppo_smoke')
   self.checked('eval',64,.15,checkpoint=r['checkpoint'],label='reward_checkpoint_reload')
  else:
   r,_=self.checked('preflight',32,label='physics_preflight');assert r['passed']
   r,folder=self.checked('train',64,.15,self.initial_checkpoint,iterations=4,label='ppo_smoke');smoke=r['checkpoint']
   self.checked('eval',64,.15,checkpoint=smoke,label='checkpoint_reload')
  # Benchmark actual PPO updates including contact physics and CAD motor mapping.
  benchmarks=[]
  if self.prior_run:
   for n in ['environment.py','worker.py','evaluation.py','evaluation_steps.py','walking_rewards.py','reference_geometry.py','mass_estimate.json','assets/robot_training.urdf','assets/motion_reference.npz']:
    assert (self.prior_run/'source/physical_v29/training'/n).read_bytes()==(HERE/n).read_bytes(),f'Cannot reuse benchmark: {n} changed'
   benchmarks=json.loads((self.prior_run/'benchmarks.json').read_text())
   atomic(self.root/'benchmark_provenance.json',dict(prior_run=str(self.prior_run),reason='Identical frozen worker, environment, rewards, PPO, contact asset and actuator models'))
   atomic(self.root/'benchmarks.json',benchmarks)
  for n in ([] if benchmarks else [4096,6144,8192]):
   r,item,f=self.child('train',n,.15,self.initial_checkpoint,iterations=4,label=f'benchmark_{n}')
   if r and r.get('completed'):
    lines=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
    warm=lines[1:] or lines
    rate=n*24/statistics.mean(x['collect_seconds']+x['learn_seconds'] for x in warm)
    minfree=min(x['gpu'].get('free_mb',0) for x in lines)
    benchmarks.append(dict(environments=n,environment_steps_per_second=rate,minimum_free_mb=minfree,folder=str(f),eligible=minfree>=self.reserve))
   elif not item['memory_failure']:raise RuntimeError(f'Non-memory benchmark failure: {f}')
   atomic(self.root/'benchmarks.json',benchmarks)
  eligible=[b for b in benchmarks if b['eligible']];assert eligible,'No benchmark preserves 300 MB VRAM'
  # Prefer user's high count when throughput is close, otherwise select measured fastest.
  self.n=select_environment_count(eligible,self.train_envs)
  self.ramp.mib_per_env=estimate_increment(benchmarks)
  self.state['next_training_environment_count']=self.n
  if self.train_envs is not None:
   self.state['environment_count_reason']='Explicit requested count' if self.n==self.train_envs else 'Benchmark memory fallback from requested count'
   if self.n!=self.train_envs:self.state['memory_fallback']=dict(environments=self.n,requested=self.train_envs,phase='capacity_benchmark',reason='Requested count did not preserve the VRAM reserve')
  if self.initial_checkpoint:
   checkpoint=self.initial_checkpoint
   self.state['initial_checkpoint_scope']='Unqualified standing prior; start learning stand before promotion tests'
  else:
   baseline,f=self.checked('eval',128,label='initial_stand_nominal');checkpoint=baseline['checkpoint']
  trusted=None;stage_start=0
  if self.resume_run:
   previous=json.loads((self.resume_run/'state.json').read_text())
   assert previous['status'] in ('failed','stopped_for_supervisor_upgrade') and previous.get('child_pid') is None,'Only a stopped chain can be resumed'
   # Verify all copied policy, dynamics and hardware bytes, not only names.
   prior_source=self.resume_run/'source/physical_v29'
   for path in (self.root/'source/physical_v29').rglob('*'):
    if path.is_file() and path.suffix!='.pyc':
     relative=path.relative_to(self.root/'source/physical_v29')
     assert (prior_source/relative).read_bytes()==path.read_bytes(),f'Resume source changed: {relative}'
   self.state['promotions']=previous['promotions'][:]
   previous_train=next((a for a in reversed(previous['attempts']) if a['operation']=='train' and a.get('completed')),None)
   if previous_train:
    timing=json.loads((Path(previous_train['folder'])/'progress.json').read_text())
    self.step_seconds=timing['collect_seconds']+timing['learn_seconds'];self.step_count=timing['num_envs']
   trusted=previous['promotions'][-1]['checkpoint'] if previous['promotions'] else None
   stage_start=next(i for i,(s,_) in enumerate(STAGES) if s==previous['stage'])
   recovery=Path(previous['operation_path'])/'memory_recovery.pt'
   checkpoint=str(recovery) if recovery.exists() else previous['checkpoint']
   assert checkpoint and Path(checkpoint).exists()
   self.state.update(resumed_from=str(self.resume_run),resume_checkpoint=checkpoint,prior_training_iterations=previous['total_training_iterations']+previous.get('prior_training_iterations',0),initial_checkpoint_scope='Resumed stopped chain; all frozen policy/physics sources identical; current stage hard gates re-run before learning/promotion')
   self.write()
  for stage,speed in STAGES[stage_start:]:
   self.state.update(stage=stage,stage_speed=speed,checkpoint=checkpoint);self.write()
   attempt=0
   while time.time()<self.deadline:
    attempt+=1
    reports=self.gates(speed,checkpoint,f'{stage}_gate_{attempt:03d}')
    passed=bool(reports) and all(r['passed'] for r in reports)
    if reports:atomic(self.root/'latest_gate.json',dict(stage=stage,checkpoint=checkpoint,reports=reports,passed=passed))
    if passed:
     trusted=checkpoint;self.state['promotions'].append(dict(stage=stage,speed=speed,checkpoint=checkpoint,time=time.time(),gate_attempt=attempt));self.write();break
    retention=all(c['passed'] for r in (reports or []) for c in r['cases'] if c['statistics']['command_m_s']<speed-1e-5)
    # A learning candidate is not the deployed/trusted policy. Rolling it back
    # every 25-33 updates repeatedly erased all walking learning (r10). Keep
    # optimizing it, including full standing rehearsal. ALL earlier-command
    # gates still have to pass before trusted is replaced or a stage advances.
    self.state['retention_passed']=retention
    self.state['trusted_checkpoint']=trusted
    self.state['candidate_qualified']=False
    if trusted and not retention:
     self.state['retention_failures']=self.state.get('retention_failures',0)+1
     self.state['retention_action']='Continue unqualified candidate; preserve trusted policy and block promotion'
    self.write()
    while True:
     predicted=self.step_seconds*self.n/self.step_count if self.step_seconds is not None else None
     iterations=training_iterations(self.chunk,predicted,self.n,self.ramp.hold_seconds)
     self.state['planned_training_iterations']=iterations;self.write()
     r,item,f=self.child('train',self.n,speed,checkpoint,iterations=iterations,seed=71+attempt,label=f'{stage}_train_{attempt:03d}')
     if r and r.get('completed'):
      checkpoint=r['checkpoint'];self.state['checkpoint']=checkpoint;self.state['total_training_iterations']+=r['iterations']
      target=self.ramp.consume()
      if target:
       atomic(f/'environment_ramp.json',dict(from_envs=self.n,to_envs=target,reason='180 continuous seconds with capacity for the increment plus 300MB reserve',evidence=self.state.get('vram_ramp'),applies='Next training worker; gates still run at 128'))
       self.n=target;self.state['environment_count_reason']='Automatic three-minute VRAM ramp at checkpoint boundary'
      self.state['next_training_environment_count']=self.n;self.write();break
     if not item['memory_failure']:raise RuntimeError(f'Non-memory training failure: {f}')
     available=[n for n in ENV_FALLBACK if n<self.n]
     if not available:raise RuntimeError('Memory exhausted even at512; preserved latest checkpoint')
     recovery=f/'memory_recovery.pt'
     if recovery.exists():checkpoint=str(recovery)
     self.ramp.memory_failure(time.monotonic())
     self.n=max(available);self.state['memory_fallback']=dict(environments=self.n,source=str(f),checkpoint=checkpoint);self.state['next_training_environment_count']=self.n;self.write()
   else:
    self.state['status']='time_limit_current_stage_unpassed';self.write();return
  self.state.update(status='all_requested_stages_passed',checkpoint=trusted,completed_unix=time.time());self.write()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--reserve_mb',type=int,default=300);p.add_argument('--max_hours',type=float,default=18);p.add_argument('--iterations_per_block',type=int,default=150);p.add_argument('--validated_checks',default='');p.add_argument('--prior_run',default='');p.add_argument('--initial_checkpoint',default='');p.add_argument('--train_envs',type=int,choices=ENV_FALLBACK);p.add_argument('--resume_run',default='')
 args=p.parse_args();c=Chain(args.output,args.reserve_mb,args.max_hours,args.iterations_per_block,args.validated_checks,args.prior_run,args.initial_checkpoint,args.train_envs,args.resume_run)
 try:c.run()
 except BaseException:
  c.state.update(status='failed',error=traceback.format_exc());c.write();traceback.print_exc();sys.exit(1)
