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
from curriculum_provenance import evaluation_speeds
HERE=Path(__file__).resolve().parent
PYTHON=HERE.parents[1]/'env_isaaclab/Scripts/python.exe'
STAGES=[('stand',0.),('walk_035',.35),('walk_065',.65),('walk_100',1.),('run_150',1.5),('run_200',2.),('run_250',2.5),('run_300',3.),('run_350',3.5),('run_400',4.),('run_450',4.5),('run_500',5.)]
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
 return any(k in text for k in ['out of memory','cuda error: memory','vram_reserve','host_ram_pressure','bad allocation','failed to allocate'])

def memory_failure_resource(folder):
 text=''.join((folder/n).read_text(errors='replace') for n in ['stderr.log','failure.json'] if (folder/n).exists()).lower()
 return 'host' if 'host_ram_pressure' in text and not any(k in text for k in ['vram_reserve','cuda error: memory','cuda out of memory']) else 'vram'

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

def benchmark_capacity_eligible(benchmark,reserve_mb):
 """A checkpoint saved for resource recovery is not sustained capacity."""
 expected=benchmark.get('expected_updates',8 if benchmark['environments']==4096 else 24)
 return (benchmark['minimum_free_mb']>=reserve_mb
         and benchmark.get('minimum_host_available_mb',0)>=2048
         and not benchmark.get('host_ram_recycled',False)
         and benchmark.get('updates',0)>=expected)

def fallback_after_recycle(count,result,ramp,now):
 """Continue the saved optimizer at a lower count, with a fresh ramp hold."""
 assert result.get('recycled') and result.get('checkpoint'),'Missing saved resource checkpoint'
 lower=[n for n in ENV_FALLBACK if n<count]
 if not lower:raise RuntimeError('Host memory pressure even at512; latest checkpoint preserved')
 # ResourceRecycle currently has only host-pressure reasons; CUDA reserve
 # failures are separate failed attempts. Do not inflate the VRAM estimate.
 reason=result['recycle_reason']
 assert reason in ('host_ram_below_2048MiB','worker_commit_above_host_budget'),reason
 ramp.memory_failure(now,'host')
 return max(lower)

def verify_resource_migration(prior,new_source,optimization_proof='',curriculum_proof='',retention_proof=''):
 """Only maintenance changes may continue a candidate and its qualification."""
 if retention_proof:
  assert not optimization_proof and not curriculum_proof
  if json.loads(Path(retention_proof).read_text()).get('schema')=='posture_conditioned_walking_prior_v1':
   return verify_teacher_context_migration(prior,new_source,retention_proof)
  if json.loads(Path(retention_proof).read_text()).get('schema')=='upper_body_output_refinement_v1':
   from upper_body_refinement import verify_upper_body_migration
   return verify_upper_body_migration(prior,new_source,retention_proof)
  from standing_retention import verify_retention_migration
  return verify_retention_migration(prior,new_source,retention_proof)
 if curriculum_proof:
  assert not optimization_proof
  from curriculum_provenance import verify_curriculum_migration
  return verify_curriculum_migration(prior,new_source,curriculum_proof)
 import ast
 previous=json.loads((prior/'state.json').read_text())
 if optimization_proof:
  assert previous['status'] in ('paused_for_performance_review','superseded_for_optimization') and previous.get('child_pid') is None
 else:
  assert previous['status']=='superseded_for_diagnostic_repair' and previous.get('child_pid') is None
 old_source=prior/'source/physical_v29'
 def sources(root):
  return {p.relative_to(root).as_posix():p for p in root.rglob('*')
          if p.is_file() and p.suffix!='.pyc'
          and p.relative_to(root).parts[:3]!=('training','assets','usd')}
 # USDs/configs are generated by UrdfConverter inside this cache directory;
 # their absolute paths differ in a new run. URDF and converter runtime bytes
 # above the cache are protected and fresh physical/load smoke follows.
 old_files,new_files=sources(old_source),sources(new_source)
 if optimization_proof:
  # Explicit execution-only continuation. No generalized source exception:
  # all three changed implementation files and every saved test are pinned.
  proof_path=Path(optimization_proof).resolve();proof=json.loads(proof_path.read_text())
  assert proof['schema']=='exact_execution_optimization_v1' and proof['selected_iteration']=='04'
  assert Path(proof['prior_run']).resolve()==prior.resolve()
  assert hashlib.sha256(Path(previous['checkpoint']).read_bytes()).hexdigest()==proof['checkpoint_sha256']
  permitted={'cad_differential.py','hardware_actuators.py','owned_cuda_graph.py'}
  assert set(proof['implementation_hashes'])==permitted
  differences={n for n in old_files.keys()|new_files.keys()
               if n not in old_files or n not in new_files or old_files[n].read_bytes()!=new_files[n].read_bytes()}
  assert differences==permitted,('Non-execution source changed',differences)
  for name,digest in proof['implementation_hashes'].items():
   assert hashlib.sha256(new_files[name].read_bytes()).hexdigest()==digest,name
  required={'math_cuda','actuator_parity','physical_parity_4096','mixed_replay_parity','ppo_4096','ppo_6144','source_audit'}
  assert set(proof['tests'])==required
  for name,record in proof['tests'].items():
   path=(proof_path.parent/record['path']).resolve()
   assert path.is_relative_to(proof_path.parent),name
   assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],name
   test=json.loads(path.read_text())
   if name in ('physical_parity_4096','mixed_replay_parity'):assert test['all_bitwise_equal'],name
   elif name.startswith('ppo_'):
    assert test['completed'] and test['updates']>=8 and test['full_checkpoint_reload'] and test['optimizer_state_saved'],name
    assert all(r['ppo']['pre_update_kl']<1e-5 and r['memory']['vram_free_mb']>=300 and r['memory']['host_available_mb']>=2048 for r in test['rows']),name
   else:assert test['passed'],name
   if name=='math_cuda':
    assert test['device']=='cuda:0' and test['source_sha256']==proof['implementation_hashes']['cad_differential.py']
  assert not proof.get('qualifies_locomotion',True),'Execution tests do not qualify walking'
  return previous
 assert old_files.keys()==new_files.keys(),'Runtime source inventory changed'
 for relative,path in new_files.items():
  assert old_files[relative].read_bytes()==path.read_bytes(),f'Runtime source changed: {relative}'
 if (new_source/'training/standing_retention.py').exists():
  assert (old_source/'training/standing_retention.py').exists(),'Standing retention protocol changed; use explicit learning repair'
  assert (old_source/'training/standing_retention.py').read_bytes()==(new_source/'training/standing_retention.py').read_bytes(),'Standing retention loss changed'
 for path in ['training/environment.py','training/evaluation.py','training/evaluation_steps.py',
              'training/walking_rewards.py','training/reference_geometry.py','training/mass_estimate.json',
              'training/assets/robot_training.urdf','training/assets/motion_reference.npz',
              'hardware_actuators.py','cad_differential.py','assets/actuator_limits.json',
              'assets/differential_geometry.json','assets/export_metadata.json']:
  assert (old_source/path).read_bytes()==(new_source/path).read_bytes(),path
 def agent_ast(path):
  tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='agent_config')
  return ast.dump(node,include_attributes=False)
 assert agent_ast(old_source/'training/worker.py')==agent_ast(new_source/'training/worker.py'),'PPO config changed'
 assert previous['checkpoint'] and Path(previous['checkpoint']).exists()
 return previous

def verify_teacher_context_migration(prior,new_source,proof_path):
 """Pin a tested teacher-target correction; never alter student inference."""
 from curriculum_provenance import training_reward_signature
 prior=Path(prior);new_source=Path(new_source);proof_path=Path(proof_path).resolve()
 proof=json.loads(proof_path.read_text());state=json.loads((prior/'state.json').read_text())
 assert proof['schema']=='posture_conditioned_walking_prior_v1'
 assert state['status']=='superseded_for_diagnostic_repair' and state.get('child_pid') is None
 assert Path(proof['prior_run']).resolve()==prior.resolve()
 assert Path(proof['checkpoint']).resolve()==Path(state['checkpoint']).resolve()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert sha(__file__)==proof['supervisor_sha256']
 assert sha(state['checkpoint'])==proof['checkpoint_sha256']
 def inventory(root):
  return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()
          and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
 old,new=inventory(prior/'source/physical_v29'),inventory(new_source)
 changed={k for k in old.keys()|new.keys() if k not in old or k not in new or old[k].read_bytes()!=new[k].read_bytes()}
 allowed={'training/upper_body_refinement.py','training/teacher_head_context.py','training/teacher_head_context.json'}
 assert changed==allowed and set(proof['implementation_hashes'])==allowed,changed
 for k,h in proof['implementation_hashes'].items():assert sha(new[k])==h
 assert training_reward_signature(new_source/'training')==proof['reward_signature']
 assert training_reward_signature(prior/'source/physical_v29/training')==proof['reward_signature']
 assert proof['actor_critic_Adam_exploration_retained'] and proof['student_observations_truthful']
 assert not proof['inference_teacher'] and not proof['qualifies_running']
 assert proof['conditioned_commands']==[.15,.35,.65,1.] and proof['same_stage']==state['stage']
 for key in ['cpu_tests','physical_validation','matched_validation','full_state_smoke']:
  record=proof[key];p=(proof_path.parent/record['path']).resolve()
  assert p.is_relative_to(proof_path.parent) and sha(p)==record['sha256']
  assert json.loads(p.read_text())['passed'],key
 return state

def verify_gate_upgrade(old_path,new_path):
 """Allow this explicit tightening only; every existing gate is preserved."""
 import ast,copy
 old=ast.parse(old_path.read_text());new=ast.parse(new_path.read_text())
 def version(tree):
  return next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GATE_VERSION' for t in n.targets))
 def gate(tree):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gate_case')
 assert version(old)=='physical_landing_v2' and version(new)=='physical_landing_v3','Only the audited v2 to v3 gate upgrade is allowed'
 expected=copy.deepcopy(gate(old))
 branch=next(n for n in expected.body if isinstance(n,ast.If) and ast.unparse(n.test)=='speed == 0')
 branch.orelse+=ast.parse("checks['alternating_landings']=stats.get('same_foot_fraction',1.)<.02\nif speed<=1.:checks['walking_support']=stats['flight_fraction']<.02").body
 assert ast.dump(expected,include_attributes=False)==ast.dump(gate(new),include_attributes=False),'Existing independent gate thresholds or logic changed'
 return dict(from_version=version(old),to_version=version(new),existing_gate_logic_preserved=True,
  added_checks=['alternating_landings','walking_support'],qualification_reused=False,
  old_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest())

def verify_reward_migration(prior,new_source,gate_repair=False):
 """Keep learned actor, but never carry qualification across reward repairs.

 Assets, contacts, actuator limits, reference, observations, action mapping,
 resets, randomization and PPO must be unchanged. The independent verifier is
 unchanged unless an explicit, tested v2-to-v3 gate repair is requested.
 The only environment exceptions are reward logic and its discovery state.
 """
 import ast,copy
 previous=json.loads((prior/'state.json').read_text())
 assert previous['status']=='superseded_for_diagnostic_repair' and previous.get('child_pid') is None
 assert previous['checkpoint'] and Path(previous['checkpoint']).exists()
 assert previous['promotions'] and previous['stage'] in dict(STAGES),'No qualified prior stage'
 old_source=prior/'source/physical_v29'
 if gate_repair:verify_gate_upgrade(old_source/'training/evaluation.py',new_source/'training/evaluation.py')
 for path in ([] if gate_repair else ['training/evaluation.py','training/evaluation_steps.py'])+['training/reference_geometry.py',
              'training/mass_estimate.json','training/assets/robot_training.urdf','training/assets/motion_reference.npz',
              'hardware_actuators.py','cad_differential.py','assets/actuator_limits.json',
              'assets/differential_geometry.json','assets/export_metadata.json']:
  assert (old_source/path).read_bytes()==(new_source/path).read_bytes(),path
 def protected_environment(path):
  tree=ast.parse(path.read_text());protected={}
  for node in tree.body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
    node=copy.deepcopy(node)
    if isinstance(node,ast.ClassDef) and node.name=='LocomotionEnv':
     node.body=[n for n in node.body if not (isinstance(n,ast.FunctionDef) and n.name in ['_get_rewards','read_reward_audit'])]
     for method in node.body:
      if isinstance(method,ast.FunctionDef) and method.name in ['__init__','_reset_idx']:
       method.body=[n for n in method.body if not any(isinstance(a,ast.Attribute) and a.attr in ['step_discovery','standing_anchor'] for a in ast.walk(n))]
    protected[node.name]=ast.dump(node,include_attributes=False)
   elif isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id in ['FRAMES','HISTORY','BOUNDED_MOTION'] for n in node.targets):
    protected[ast.unparse(node.targets[0])]=ast.dump(node,include_attributes=False)
  return protected
 assert protected_environment(old_source/'training/environment.py')==protected_environment(new_source/'training/environment.py'),'Non-reward environment code changed'
 def agent_ast(path):
  tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='agent_config')
  return ast.dump(node,include_attributes=False)
 assert agent_ast(old_source/'training/worker.py')==agent_ast(new_source/'training/worker.py'),'PPO config changed'
 return previous

class Chain:
 def __init__(self,root,reserve,hours,chunk,validated_checks='',prior_run='',initial_checkpoint='',train_envs=None,resume_run='',migrate_run='',reward_migrate_run='',gate_repair=False,actor_warm_start=False,learning_rate=.001,optimization_proof='',curriculum_proof='',retention_proof=''):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
  assert not (self.root/'state.json').exists(),'Use a fresh chain directory; preserve older attempts'
  self.start=time.time();self.deadline=self.start+hours*3600;self.reserve=reserve;self.chunk=chunk
  self.state=dict(status='starting',supervisor_pid=os.getpid(),started_unix=self.start,reserve_mb=reserve,stage='preflight',environment_count=None,checkpoint=None,promotions=[],attempts=[],total_training_iterations=0)
  self.count=0;self.write()
  self.validated_checks=Path(validated_checks) if validated_checks else None
  self.prior_run=Path(prior_run) if prior_run else None
  self.initial_checkpoint=initial_checkpoint
  self.actor_warm_start=actor_warm_start
  self.learning_rate=learning_rate
  self.optimization_proof=optimization_proof
  self.curriculum_proof=curriculum_proof
  self.retention_proof=retention_proof
  if retention_proof:
   proof=json.loads(Path(retention_proof).read_text())
   assert hashlib.sha256((HERE/'supervisor_unitree.py').read_bytes()).hexdigest()==proof['supervisor_sha256']
  assert not retention_proof or (migrate_run and not curriculum_proof and not optimization_proof)
  assert not curriculum_proof or (migrate_run and not optimization_proof),'Curriculum proof requires its own full checkpoint continuation'
  assert not optimization_proof or migrate_run,'Execution proof requires an explicit continued run'
  self.state['learning_rate_initial']=learning_rate
  self.train_envs=train_envs
  self.resume_run=Path(resume_run) if resume_run else None
  self.migrate_run=Path(migrate_run) if migrate_run else None
  self.reward_migrate_run=Path(reward_migrate_run) if reward_migrate_run else None
  self.gate_repair=gate_repair
  self.standing_prior=None;self.standing_prior_sha256=None
  assert not (reward_migrate_run or validated_checks or gate_repair), 'Reward changes require fresh actor transfer and qualification'
  assert not prior_run or (migrate_run and Path(prior_run).resolve()==Path(migrate_run).resolve()), 'Benchmark reuse requires the same audited resource-only migration'
  if migrate_run:
   assert initial_checkpoint and not actor_warm_start,'Resource maintenance requires full checkpoint continuation, without actor-only transfer'
  else:
   assert bool(initial_checkpoint)==bool(actor_warm_start),'An initial checkpoint requires explicit actor-only transfer'
  if initial_checkpoint:assert Path(initial_checkpoint).is_file(),'Selected checkpoint is missing'
  assert not gate_repair or self.reward_migrate_run,'Gate repair requires fresh reward-migration qualification'
  assert sum(bool(p) for p in [self.migrate_run,self.resume_run,self.reward_migrate_run])<=1,'Choose one continuation mode'
  self.ramp=VramRamp(reserve_mb=reserve)
  self.step_seconds=None;self.step_count=None
  # Worker reads this frozen source tree, rather than future workspace edits.
  snapshot=self.root/'source/physical_v29';(snapshot/'training/assets').mkdir(parents=True)
  (snapshot/'assets').mkdir()
  for n in ['environment_unitree.py','worker_unitree.py','unitree_reward_port.py','stage_rewards.py','support_geometry.py','reward_warm_start.py','heading_control.py','environment.py','worker.py','evaluation.py','evaluation_steps.py','memory_guard.py','walking_rewards.py','standing_retention.py','ppo_update_audit.py','reference_geometry.py','mass_estimate.json','status_io.py','command_curriculum.py','curriculum_provenance.py','upper_body_refinement.py','teacher_head_context.py','teacher_head_context.json']:
   shutil.copy2(HERE/n,snapshot/'training'/n)
  shutil.copytree(HERE/'unitree_vendor',snapshot/'training/unitree_vendor')
  for n in ['robot_training.urdf','motion_reference.npz']:
   shutil.copy2(HERE/'assets'/n,snapshot/'training/assets'/n)
  for n in ['hardware_actuators.py','cad_differential.py','owned_cuda_graph.py']:
   shutil.copy2(HERE.parent/n,snapshot/n)
  for n in ['actuator_limits.json','differential_geometry.json','export_metadata.json']:
   shutil.copy2(HERE.parent/'assets'/n,snapshot/'assets'/n)
  self.worker=snapshot/'training/worker_unitree.py'
  if self.migrate_run:
   previous=verify_resource_migration(self.migrate_run,snapshot,self.optimization_proof,self.curriculum_proof,self.retention_proof)
   assert str(Path(self.initial_checkpoint).resolve())==str(Path(previous['checkpoint']).resolve()),'Migration smoke must use the continued candidate'
   if previous['promotions'] and previous['promotions'][0]['speed']==0.:
    self.standing_prior=previous['promotions'][0]['checkpoint']
    self.standing_prior_sha256=hashlib.sha256(Path(self.standing_prior).read_bytes()).hexdigest()
  if self.reward_migrate_run:
   previous=verify_reward_migration(self.reward_migrate_run,snapshot,self.gate_repair)
   self.standing_prior=previous['promotions'][0]['checkpoint']
   if Path(self.standing_prior).exists():self.standing_prior_sha256=hashlib.sha256(Path(self.standing_prior).read_bytes()).hexdigest()
   assert Path(self.initial_checkpoint).resolve()==Path(previous['checkpoint']).resolve(),'Reward migration smoke must use the learned candidate'
   if self.gate_repair:atomic(self.root/'gate_migration.json',verify_gate_upgrade(self.reward_migrate_run/'source/physical_v29/training/evaluation.py',snapshot/'training/evaluation.py'))
  frozen={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['environment_unitree.py','worker_unitree.py','unitree_reward_port.py','stage_rewards.py','support_geometry.py','reward_warm_start.py','heading_control.py','environment.py','worker.py','evaluation.py','evaluation_steps.py','memory_guard.py','supervisor_unitree.py','status_io.py','walking_rewards.py','standing_retention.py','ppo_update_audit.py','reference_geometry.py','mass_estimate.json','assets/robot_training.urdf','assets/motion_reference.npz','command_curriculum.py','curriculum_provenance.py']}
  frozen['upper_body_refinement.py']=hashlib.sha256((HERE/'upper_body_refinement.py').read_bytes()).hexdigest()
  for name in ['teacher_head_context.py','teacher_head_context.json']:
   frozen[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
  for path in (HERE/'unitree_vendor').rglob('*'):
   if path.is_file() and path.suffix!='.pyc':frozen['unitree_vendor/'+path.relative_to(HERE/'unitree_vendor').as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
  for n in ['actuator_limits.json','differential_geometry.json']:
   frozen['physical_assets/'+n]=hashlib.sha256((HERE.parent/'assets'/n).read_bytes()).hexdigest()
  for n in ['hardware_actuators.py','cad_differential.py','owned_cuda_graph.py','assets/export_metadata.json']:
   frozen['physical_v29/'+n]=hashlib.sha256((HERE.parent/n).read_bytes()).hexdigest()
  frozen['vram_ramp.py']=hashlib.sha256((HERE/'vram_ramp.py').read_bytes()).hexdigest()
  for name in ['supervisor_unitree.py','vram_ramp.py']:shutil.copy2(HERE/name,self.root/name)
  atomic(self.root/'chain_config.json',dict(stages=STAGES,reserve_mb=reserve,max_hours=hours,iterations_per_block=chunk,train_environment_fallback=ENV_FALLBACK,automatic_upward_ramp=dict(hold_seconds=180,max_envs=8192,one_tier_at_a_time=True,checkpoint_boundary=True,training_samples_only=True),host_memory_maintenance=dict(minimum_available_mb=2048,maximum_worker_commit_to_host_ram=1.15,checkpoint_recycling=True),eval_envs=128,eval_minimum_trials_per_command=32,evaluation_seconds=20,seeds=[901,1301],sources_sha256=frozen,no_cameras=True,local_only=True,resume_run=str(self.resume_run) if self.resume_run else None,migrate_run=str(self.migrate_run) if self.migrate_run else None,reward_migrate_run=str(self.reward_migrate_run) if self.reward_migrate_run else None,actor_warm_start=actor_warm_start,initial_actor_checkpoint=initial_checkpoint or None,qualification_reused=False,optimization_proof=optimization_proof or None,curriculum_proof=curriculum_proof or None,retention_proof=retention_proof or None))
 def write(self):
  self.state['updated_unix']=time.time();atomic(self.root/'state.json',self.state)
 def child(self,mode,n,speed=0.,checkpoint='',iterations=None,perturb=False,seed=71,label='',standing_prior=''):
  self.count+=1;folder=self.root/f'{self.count:03d}_{label or mode}';folder.mkdir()
  command=[str(PYTHON),'-u','-X','utf8',str(self.worker),'--headless','--device','cuda:0','--mode',mode,'--num_envs',str(n),'--speed',str(speed),'--output',str(folder),'--reserve_mb',str(self.reserve),'--seed',str(seed)]
  if checkpoint:
   command+=['--checkpoint',str(checkpoint)]
   if self.actor_warm_start and str(checkpoint)==self.initial_checkpoint:command+=['--actor_warm_start']
  if mode=='train':command+=['--iterations',str(iterations or self.chunk),'--learning_rate',str(self.learning_rate)]
  if mode=='train' and speed>0 and (standing_prior or self.standing_prior):
   prior=standing_prior or self.standing_prior
   digest=hashlib.sha256(Path(prior).read_bytes()).hexdigest()
   if not standing_prior:assert digest==self.standing_prior_sha256,'Qualified standing teacher bytes changed'
   command+=['--standing_prior',str(prior),'--standing_prior_sha256',digest]
  if mode=='train' and speed>=1.5:
   qualified=[p for p in self.state['promotions'] if .1<p['speed']<speed]
   if qualified:
    from standing_retention import qualification_certificate
    from curriculum_provenance import training_reward_signature
    import ast
    GATE_VERSION=next(ast.literal_eval(n.value) for n in ast.parse((HERE/'evaluation.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GATE_VERSION' for t in n.targets))
    prior=max(qualified,key=lambda p:p['speed'])
    prior_run=Path(prior['checkpoint']).parent.parent
    receipt=qualification_certificate(prior_run,prior['checkpoint'],prior['speed'],training_reward_signature(HERE),GATE_VERSION)
    certificate=folder/'qualified_prior_certificate.json';atomic(certificate,receipt)
    command+=['--locomotion_prior_certificate',str(certificate)]
  if mode=='eval' and label=='checkpoint_reload':command+=['--duration','4'] # load smoke only, never a promotion gate
  if mode=='eval' and label=='phase_retention_reload':command+=['--duration','4']
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
      self.state['vram_ramp']=self.ramp.observe(time.monotonic(),n,m.get('free_mb'),fresh,round(host.available/2**20))
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
   n=max(128,32*len(evaluation_speeds(speed)))
   r,f=self.checked('eval',n,speed,checkpoint,perturb=perturb,seed=seed,label=label+('_push' if perturb else '_nominal'))
   reports.append(r)
   if speed==0 and not r['passed']:break # no prior command yet; nominal failure already prevents promotion
  return reports
 def run(self):
  if self.migrate_run:
   previous=verify_resource_migration(self.migrate_run,self.root/'source/physical_v29',self.optimization_proof,self.curriculum_proof,self.retention_proof)
   if self.optimization_proof or self.curriculum_proof or self.retention_proof:
    preflight,_=self.checked('preflight',32,label='physics_preflight')
    assert preflight['passed'],'Optimized execution physical preflight failed'
   else:
    preflight=self.migrate_run/'001_physics_preflight/result.json'
    assert json.loads(preflight.read_text())['passed'],'Identical-runtime physical preflight is missing'
    atomic(self.root/'reused_physics_preflight.json',dict(source=str(preflight),
     runtime_inventory_and_bytes_identical=True,scope='Resource-only supervisor maintenance; fresh PPO and load smoke follows'))
   self.actor_warm_start=False
   speed=previous.get('stage_speed',dict(STAGES)[previous['stage']])
   r,folder=self.checked('train',64,speed,self.initial_checkpoint,iterations=4,label='resource_ppo_smoke')
   self.checked('eval',64,speed,checkpoint=r['checkpoint'],label='checkpoint_reload')
  elif self.validated_checks:
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
   r,folder=self.checked('train',64,0.,self.initial_checkpoint,iterations=4,label='ppo_smoke');smoke=r['checkpoint']
   self.checked('eval',64,0.,checkpoint=smoke,label='checkpoint_reload')
   if self.actor_warm_start:
    # Exercise the added phase objective and stopped-row gradient on real
    # physics before capacity/full-chain training. Its actor is discarded;
    # neither this teacher use nor a four-second replay reuses qualification.
    phase,_=self.checked('train',64,.35,self.initial_checkpoint,iterations=4,
        standing_prior=self.initial_checkpoint,label='phase_retention_smoke')
    self.checked('eval',64,.35,checkpoint=phase['checkpoint'],label='phase_retention_reload')
  # Benchmark actual PPO updates including contact physics and CAD motor mapping.
  benchmarks=[]
  if self.prior_run:
   for n in ['environment_unitree.py','worker_unitree.py','unitree_reward_port.py','environment.py','worker.py','evaluation.py','evaluation_steps.py','walking_rewards.py','reference_geometry.py','mass_estimate.json','assets/robot_training.urdf','assets/motion_reference.npz']:
    assert (self.prior_run/'source/physical_v29/training'/n).read_bytes()==(HERE/n).read_bytes(),f'Cannot reuse benchmark: {n} changed'
   benchmarks=json.loads((self.prior_run/'benchmarks.json').read_text())
   for benchmark in benchmarks:
    benchmark['prior_eligible']=benchmark.get('eligible')
    benchmark['eligible']=benchmark_capacity_eligible(benchmark,self.reserve)
   atomic(self.root/'benchmark_provenance.json',dict(prior_run=str(self.prior_run),reason='Identical frozen worker, environment, rewards, PPO, contact asset and actuator models'))
   atomic(self.root/'benchmarks.json',benchmarks)
  capacity_upper=None
  for n in ([] if benchmarks else [4096,6144,8192,3072,2048,1024,512]):
   if n<4096 and any(b['eligible'] for b in benchmarks):break
   if capacity_upper is not None and n>=capacity_upper:continue
   r,item,f=self.child('train',n,0.,self.initial_checkpoint,iterations=8 if n==4096 else 24,label=f'benchmark_{n}')
   if r and r.get('completed'):
    lines=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
    warm=lines[1:] or lines
    rate=n*24/statistics.mean(x['collect_seconds']+x['learn_seconds'] for x in warm)
    minfree=min(x['gpu'].get('free_mb',0) for x in lines)
    updates=int(r['iterations'])
    benchmark=dict(environments=n,environment_steps_per_second=rate,minimum_free_mb=minfree,folder=str(f),
      updates=updates,wall_seconds=item['seconds'],effective_environment_steps_per_second=n*24*updates/item['seconds'],
      host_ram_recycled=bool(r.get('recycled')),minimum_host_available_mb=min(x['gpu'].get('ram_available_mb',0) for x in lines),expected_updates=8 if n==4096 else 24)
    benchmark['eligible']=benchmark_capacity_eligible(benchmark,self.reserve)
    benchmarks.append(benchmark)
   elif not item['memory_failure']:raise RuntimeError(f'Non-memory benchmark failure: {f}')
   else:
    capacity_upper=n
    self.state.setdefault('capacity_memory_failures',[]).append(dict(environments=n,folder=str(f),resource=memory_failure_resource(f)))
   atomic(self.root/'benchmarks.json',benchmarks)
  eligible=[b for b in benchmarks if b['eligible']];assert eligible,'No complete benchmark preserves both host RAM and VRAM reserves'
  # Prefer user's high count when throughput is close, otherwise select measured fastest.
  self.n=select_environment_count(eligible,self.train_envs)
  self.ramp.mib_per_env=estimate_increment(benchmarks)
  self.state['next_training_environment_count']=self.n
  if self.train_envs is not None:
   self.state['environment_count_reason']='Explicit requested count' if self.n==self.train_envs else 'Benchmark memory fallback from requested count'
   if self.n!=self.train_envs:self.state['memory_fallback']=dict(environments=self.n,requested=self.train_envs,phase='capacity_benchmark',reason='Requested count did not complete capacity testing with healthy host RAM and the VRAM reserve')
  if self.actor_warm_start:
   transfer=json.loads((folder/'warm_start.json').read_text())
   checkpoint=transfer['baseline_checkpoint']
   assert Path(checkpoint).is_file(),'Transferred actor baseline is missing'
   self.state['initial_checkpoint_scope']='Audited compatible v29 actor preserved before smoke updates, v4 trained exploration retained, fresh critic/Adam; real four-update smoke/reload tested separately; no qualification reused'
   self.state['initial_actor_checkpoint']=self.initial_checkpoint
  elif self.initial_checkpoint:
   checkpoint=self.initial_checkpoint
   self.state['initial_checkpoint_scope']='Unqualified standing prior; start learning stand before promotion tests'
  else:
   # Start from this run's freshly trained/reloaded smoke. Its first 128-robot
   # standing gate remains mandatory; avoid a duplicate identical baseline gate.
   checkpoint=smoke
   self.state['initial_checkpoint_scope']='Fresh matching-source four-update smoke; unqualified; mandatory standing gate follows'
  trusted=None;stage_start=0
  if self.reward_migrate_run:
   previous=verify_reward_migration(self.reward_migrate_run,self.root/'source/physical_v29',self.gate_repair)
   prior=previous['promotions'][-1];trusted=prior['checkpoint']
   reports=self.gates(prior['speed'],trusted,'reward_migration_prior')
   passed=bool(reports) and all(r['passed'] for r in reports)
   import ast
   def reward_version(path):
    tree=ast.parse(path.read_text());return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REWARD_VERSION' for t in n.targets))
   changed_reward=reward_version(self.reward_migrate_run/'source/physical_v29/training/walking_rewards.py')!=reward_version(self.root/'source/physical_v29/training/walking_rewards.py')
   atomic(self.root/'reward_migration.json',dict(source=str(self.reward_migrate_run),candidate=checkpoint,
    trusted_prior=trusted,prior_reports=reports,prior_passed=passed,actor_preserved=True,
    critic_optimizer_reset_for_new_reward=changed_reward,qualification_reused=False,
    standing_prior_training_regularization=dict(checkpoint=self.standing_prior,sha256=self.standing_prior_sha256,weight=.25,inference_teacher=False)))
   if passed:
    self.state['promotions']=[dict(p,checkpoint=trusted,requalified_unix=time.time(),requalification='reward_migration_prior') for p in previous['promotions']]
    stage_start=next(i for i,(name,_) in enumerate(STAGES) if name==previous['stage'])
   else:
    # Preserve the candidate, but failed old-skill checks cannot be reused.
    # Relearn from the prior at standing rather than exiting the chain.
    self.state['preserved_unqualified_candidate']=checkpoint;checkpoint=trusted;trusted=None
   self.state.update(reward_migrated_from=str(self.reward_migrate_run),
    prior_training_iterations=previous['total_training_iterations']+previous.get('prior_training_iterations',0),
    initial_checkpoint_scope='Learned actor retained; value/Adam reset only for different rewards, prior qualification independently rechecked; stopped-command standing prior regularizes training only')
   self.write()
  if self.migrate_run:
   previous=verify_resource_migration(self.migrate_run,self.root/'source/physical_v29',self.optimization_proof,self.curriculum_proof,self.retention_proof)
   checkpoint=previous['checkpoint'];assert checkpoint and Path(checkpoint).exists()
   self.actor_warm_start=False # warm-start smokes are discarded; continue the full saved optimizer
   if self.retention_proof:
    baseline=previous['promotions'][-1]
    reports=self.gates(baseline['speed'],baseline['checkpoint'],'retention_teacher_requalification')
    assert reports and all(r['passed'] for r in reports),'Qualified teacher failed fresh retained skill verification'
    for p in previous['promotions']:p['requalification']='fresh_matching_prior_retention_teacher'
   self.state['promotions']=previous['promotions'][:]
   trusted=previous['promotions'][-1]['checkpoint'] if previous['promotions'] else None
   if previous['promotions'] and previous['promotions'][0]['speed']==0.:
    self.standing_prior=previous['promotions'][0]['checkpoint']
    self.standing_prior_sha256=hashlib.sha256(Path(self.standing_prior).read_bytes()).hexdigest()
    self.state['standing_retention_prior']=dict(checkpoint=self.standing_prior,sha256=self.standing_prior_sha256,training_only=True,weight=.25)
   stage_start=next(i for i,(name,_) in enumerate(STAGES) if name==previous['stage'])
   self.state.update(migrated_from=str(self.migrate_run),prior_training_iterations=previous['total_training_iterations']+previous.get('prior_training_iterations',0),
    initial_checkpoint_scope='Audited full actor/critic/Adam continuation; '+('training-only exact-command rehearsal, original reward/observation/action signature and prior gates preserved; ' if self.curriculum_proof else 'qualified prior-command gradient repair; ' if self.retention_proof else 'resource maintenance; ')+'all current and retention gates re-run')
   atomic(self.root/'resource_migration.json',dict(source=str(self.migrate_run),checkpoint=checkpoint,physics_reward_and_PPO_algorithm_config_unchanged=True,training_gradient_repair=bool(self.retention_proof),retention_proof=self.retention_proof or None,prior_speed_gates_unchanged=True,training_commands_changed=bool(self.curriculum_proof),curriculum_proof=self.curriculum_proof or None))
   self.write()
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
     trusted=checkpoint
     if speed==0.:
      self.standing_prior=checkpoint;self.standing_prior_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
      self.state['standing_retention_prior']=dict(checkpoint=checkpoint,sha256=self.standing_prior_sha256,training_only=True,weight=.25)
     self.state['promotions'].append(dict(stage=stage,speed=speed,checkpoint=checkpoint,time=time.time(),gate_attempt=attempt));self.write();break
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
    predicted=self.step_seconds*self.n/self.step_count if self.step_seconds is not None else None
    planned=training_iterations(self.chunk,predicted,self.n,self.ramp.hold_seconds);completed_here=0;part=0
    while True:
     iterations=max(1,planned-completed_here);part+=1
     self.state['planned_training_iterations']=iterations;self.write()
     r,item,f=self.child('train',self.n,speed,checkpoint,iterations=iterations,seed=71+attempt+part-1,label=f'{stage}_train_{attempt:03d}'+(f'_part_{part:02d}' if part>1 else ''))
     if r and r.get('completed'):
      checkpoint=r['checkpoint'];self.state['checkpoint']=checkpoint;self.state['total_training_iterations']+=r['iterations']
      completed_here+=r['iterations']
      if r.get('recycled'):
       previous_count=self.n
       self.n=fallback_after_recycle(self.n,r,self.ramp,time.monotonic())
       self.state.setdefault('resource_recycles',[]).append(dict(source=str(f),checkpoint=checkpoint,reason=r['recycle_reason'],
        completed_iterations=completed_here,planned_iterations=planned,environments=previous_count,next_environments=self.n,gpu_and_host=r['gpu']))
       self.state.update(next_training_environment_count=self.n,last_memory_failure_resource='host',
        memory_fallback=dict(environments=self.n,from_environments=previous_count,source=str(f),checkpoint=checkpoint,reason=r['recycle_reason']),
        environment_count_reason='Host RAM fallback after completed checkpoint recovery')
       self.write()
       if completed_here<planned:continue
       break
      target=self.ramp.consume()
      if target:
       atomic(f/'environment_ramp.json',dict(from_envs=self.n,to_envs=target,reason='180 continuous seconds with capacity for the increment plus 300MB reserve',evidence=self.state.get('vram_ramp'),applies='Next training worker; gates still run at 128'))
       self.n=target;self.state['environment_count_reason']='Automatic three-minute VRAM ramp at checkpoint boundary'
      self.state['next_training_environment_count']=self.n;self.write();break
     if not item['memory_failure']:raise RuntimeError(f'Non-memory training failure: {f}')
     available=[n for n in ENV_FALLBACK if n<self.n]
     if not available:raise RuntimeError('Memory exhausted even at512; preserved latest checkpoint')
     recovery=f/'memory_recovery.pt'
     if not recovery.exists():recovery=f/'resource_recovery.pt'
     if recovery.exists():checkpoint=str(recovery)
     resource=memory_failure_resource(f)
     self.ramp.memory_failure(time.monotonic(),resource)
     self.state['last_memory_failure_resource']=resource
     self.n=max(available);self.state['memory_fallback']=dict(environments=self.n,source=str(f),checkpoint=checkpoint);self.state['next_training_environment_count']=self.n;self.write()
   else:
    self.state['status']='time_limit_current_stage_unpassed';self.write();return
  self.state.update(status='all_requested_stages_passed',checkpoint=trusted,completed_unix=time.time());self.write()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--retention_proof',default='');p.add_argument('--output',required=True);p.add_argument('--reserve_mb',type=int,default=300);p.add_argument('--max_hours',type=float,default=18);p.add_argument('--iterations_per_block',type=int,default=150);p.add_argument('--validated_checks',default='');p.add_argument('--prior_run',default='');p.add_argument('--initial_checkpoint',default='');p.add_argument('--train_envs',type=int,choices=ENV_FALLBACK);p.add_argument('--resume_run',default='');p.add_argument('--migrate_run',default='');p.add_argument('--reward_migrate_run',default='');p.add_argument('--gate_repair',action='store_true');p.add_argument('--actor_warm_start',action='store_true');p.add_argument('--learning_rate',type=float,default=.001);p.add_argument('--optimization_proof',default='');p.add_argument('--curriculum_proof',default='')
 args=p.parse_args();c=Chain(args.output,args.reserve_mb,args.max_hours,args.iterations_per_block,args.validated_checks,args.prior_run,args.initial_checkpoint,args.train_envs,args.resume_run,args.migrate_run,args.reward_migrate_run,args.gate_repair,args.actor_warm_start,args.learning_rate,args.optimization_proof,args.curriculum_proof,args.retention_proof)
 try:c.run()
 except BaseException:
  c.state.update(status='failed',error=traceback.format_exc());c.write();traceback.print_exc();sys.exit(1)
