"""One Isaac process for conversion, preflight, PPO or independent evaluation.

Only result.json confirms completion. Failures print tracebacks even if Kit
returns zero. The supervisor owns promotion, memory fallback and shutdown.
"""
import argparse,json,hashlib,time,os,sys,traceback,copy,subprocess
from pathlib import Path
from status_io import atomic_json
from curriculum_provenance import training_reward_signature
from command_curriculum import TRAINING_COMMAND_VERSION,command_coverage
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['preflight','train','eval','play'],required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--checkpoint',default='')
parser.add_argument('--actor_warm_start',action='store_true')
parser.add_argument('--num_envs',type=int,default=64)
parser.add_argument('--speed',type=float,default=0.)
parser.add_argument('--iterations',type=int,default=100)
parser.add_argument('--learning_rate',type=float,default=.001)
parser.add_argument('--seed',type=int,default=71)
parser.add_argument('--duration',type=float,default=20.)
parser.add_argument('--perturb',action='store_true')
parser.add_argument('--reserve_mb',type=int,default=300)
parser.add_argument('--standing_prior',default='')
parser.add_argument('--standing_prior_sha256',default='')
parser.add_argument('--locomotion_prior_certificate',default='')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args)
import torch,numpy as np,psutil,gc
torch.set_num_threads(4)
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment_unitree import LocomotionCfg,LocomotionEnv,META,MASS
from evaluation import evaluate,GATE_VERSION
from stage_rewards import REWARD_VERSION
from reward_warm_start import validate_actor_transfer,INITIAL_EXPLORATION_STD
from memory_guard import recycle_reason,ResourceRecycle,restored_learning_rate
from standing_retention import StandingRetention,LEARNER_VERSION,validate_teacher,validate_qualification
from upper_body_refinement import UpperBodyRefinement,LocomotionOnlyRetention,UPPER_REFINEMENT_VERSION
from ppo_update_audit import PpoUpdateAudit
from heading_control import COMMAND_VERSION

CONTRACT='physical_v29_history5_unitree_defaultoffset_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reward_signature():
 return training_reward_signature(HERE)

def agent_config():
 return dict(num_steps_per_env=24,save_interval=50,logger='tensorboard',check_for_nan=True,
   obs_groups={'actor':['policy'],'critic':['critic']},
   actor=dict(class_name='MLPModel',hidden_dims=[512,256,128],activation='elu',obs_normalization=False,
      distribution_cfg=dict(class_name='GaussianDistribution',init_std=INITIAL_EXPLORATION_STD,std_type='log')),
   critic=dict(class_name='MLPModel',hidden_dims=[512,256,128],activation='elu',obs_normalization=False),
   algorithm=dict(class_name='PPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,
     entropy_coef=.01,num_learning_epochs=5,num_mini_batches=4,learning_rate=args.learning_rate,
     schedule='adaptive',gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.,rnd_cfg=None,symmetry_cfg=None))

def memory():
 try:
  line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()[0]
  used,free,util=[int(x.strip()) for x in line.split(',')]
  host=psutil.virtual_memory();proc=psutil.Process().memory_info()
  return dict(used_mb=used,free_mb=free,utilization=util,ram_available_mb=round(host.available/2**20),ram_total_mb=round(host.total/2**20),
   process_rss_mb=round(proc.rss/2**20),process_private_mb=round(getattr(proc,'private',proc.rss)/2**20),
   torch_allocated_mb=round(torch.cuda.memory_allocated()/2**20),torch_reserved_mb=round(torch.cuda.memory_reserved()/2**20))
 except Exception:return {}

def main():
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise FileExistsError('Completed output cannot be overwritten')
 converter=UrdfConverter(UrdfConverterCfg(asset_path=str(HERE/'assets/robot_training.urdf'),
  usd_dir=str(HERE/'assets/usd'),usd_file_name='robot_training.usd',fix_base=False,root_link_name='pelvis',
  merge_fixed_joints=True,make_instanceable=True,collision_from_visuals=False,self_collision=True,
  joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
 cfg=LocomotionCfg();cfg.scene.num_envs=args.num_envs;cfg.scene.env_spacing=3.0;cfg.robot_usd=converter.usd_path
 cfg.sim.device=args.device or 'cuda:0';cfg.stage_speed=args.speed;cfg.evaluation=args.mode in ['preflight','eval','play'];cfg.perturb=args.perturb;cfg.seed=args.seed
 if args.mode in ['eval','play']:cfg.episode_length_s=args.duration+10.
 # Keep buffer allocations suitable for this 8GB GPU, with overflow treated as failure.
 cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity=2**22
 cfg.sim.physx.gpu_total_aggregate_pairs_capacity=2**20
 cfg.sim.physx.gpu_max_rigid_contact_count=2**20
 cfg.sim.physx.gpu_max_rigid_patch_count=2**17
 cfg.sim.physx.gpu_heap_capacity=2**26
 cfg.sim.physx.gpu_collision_stack_size=2**25
 (out/'run_config.json').write_text(json.dumps(dict(arguments=vars(args),contract=CONTRACT,reward_version=REWARD_VERSION,gate_version=GATE_VERSION,reference_runtime='unitree_defaultoffset_v1',joints=[j['name'] for j in META['joints']],mass_kg=MASS['total_mass_kg'],urdf_sha256=sha(HERE/'assets/robot_training.urdf'),actuator_sha256=sha(HERE.parent/'assets/actuator_limits.json'),motion_sha256=sha(HERE/'assets/motion_reference.npz'),reference='Pinned Unitree G1 reward bodies and nominal-pose offset action; no synthetic IK trajectory',gravity=True,fixed_base=False,self_collision=True,physics_dt=.005,control_dt=.020),indent=2))
 envraw=LocomotionEnv(cfg);envraw.initialize_randomization();env=RslRlVecEnvWrapper(envraw,clip_actions=3.)
 (out/'unitree_reward_manifest.json').write_text(json.dumps(envraw.reward_port.manifest(),indent=2))
 assert not args.locomotion_prior_certificate or args.mode=='train','Locomotion prior is training-only'
 assert not args.standing_prior or (args.mode=='train' and args.speed>0.),'Standing retention is training-only during locomotion'
 print('[V29] ENVIRONMENT_READY '+json.dumps({'num_envs':args.num_envs,'mass_kg':MASS['total_mass_kg'],'joints':envraw.robot.joint_names,'gpu':memory()}),flush=True)
 start=time.monotonic()
 if args.mode=='preflight':
  print('[V29] RESET '+json.dumps(dict(q=envraw.robot.data.joint_pos[0].tolist(),root=envraw.robot.data.root_state_w[0].tolist(),feet=envraw.robot.data.body_link_pos_w[0,envraw.feet].tolist())),flush=True)
  # Analytic motion-reference reachability and simultaneous differential combinations.
  differential=[]
  for key,ids,model in envraw.actuator.pairs:
   q=envraw.clips[:,:,ids].reshape(-1,2);_,_,valid,errors=model.state(q)
   assert valid.all(),f'{key} has {(~valid).sum().item()} invalid reference targets'
   angles=torch.tensor([[-.10,-.06],[-.10,.06],[.10,-.06],[.10,.06],[0,0]],device=envraw.device)
   _,J,v,err=model.state(angles);assert v.all(),f'{key} simultaneous pitch/roll combinations unreachable'
   desired=torch.tensor([[4.,2.]],device=envraw.device).expand(5,-1)
   tau,state=model.limit(angles,torch.zeros_like(angles),desired,30 if key=='waist' else 45,400*2*torch.pi/60,.95)
   assert state['motor_applied'].abs().max()<=(30 if key=='waist' else 45)+1e-4
   differential.append(dict(pair=key,reference_samples=len(q),combined_angle_samples=angles.cpu().tolist(),maximum_closure_error_mm=float(errors.max())))
  zero=torch.zeros((args.num_envs,23),device=envraw.device);support=[];heights=[];tilts=[];fall_count=0;invalid=0
  for tick in range(200):
   _,_,done,_=env.step(zero);m=envraw.telemetry()
   fall_count+=int(done.sum());invalid+=int(m['invalid'].sum())
   if tick>=100:
    support.append(float(m['foot_force'].sum(1).mean()));heights.append(float(m['height'].mean()));tilts.append(float(m['tilt'].mean()))
   if tick%100==0 or tick in [1,2,5,10,20,40]:
    print('[V29] PREFLIGHT '+json.dumps(dict(tick=tick,seconds=time.monotonic()-start,falls=fall_count,invalid=invalid,height=float(m['height'].mean()),support=float(m['foot_force'].sum(1).mean()),tilt=float(m['tilt'].mean()),nonfoot=float(m['nonfoot'].max()),joints=envraw.robot.data.joint_pos[0].tolist())),flush=True)
   if tick==1:
    print('[V29] CONTACTS '+json.dumps(dict(zip(envraw.contacts.body_names,envraw.contacts.data.net_forces_w[0].norm(dim=-1).tolist()))),flush=True)
  avg_support=float(np.mean(support));assert .75*envraw.total_weight<avg_support<1.30*envraw.total_weight,(avg_support,envraw.total_weight)
  assert invalid==0,invalid
  assert np.isfinite(heights).all() and np.isfinite(tilts).all()
  result=dict(completed=True,passed=True,scope='physics/contact/transmission preflight, NOT a learned standing gate',seconds=time.monotonic()-start,num_envs=args.num_envs,differentials=differential,mean_support_N=avg_support,expected_weight_N=envraw.total_weight,mean_height_m=float(np.mean(heights)),mean_tilt_deg=float(np.rad2deg(np.mean(tilts))),zero_action_falls=fall_count,invalid=invalid,gpu=memory())
 else:
  config=agent_config();(out/'agent.json').write_text(json.dumps(config,indent=2))
  runner=OnPolicyRunner(env,copy.deepcopy(config),log_dir=str(out) if args.mode=='train' else None,device=envraw.device)
  if args.mode!='train':runner.logger.init_logging_writer()
  else:runner.logger.writer=None # learn() initializes it; early local actor save does not upload
  original_save=runner.save
  def save_with_provenance(path,infos=None):
   details=dict(contract=CONTRACT,asset_sha256=sha(HERE/'assets/robot_training.urdf'),motion_sha256=sha(HERE/'assets/motion_reference.npz'),actuator_sha256=sha(HERE.parent/'assets/actuator_limits.json'),reward_version=REWARD_VERSION,reward_signature=reward_signature(),command_version=COMMAND_VERSION,speed=args.speed,source=args.checkpoint,learner_version=LEARNER_VERSION if args.standing_prior else 'plain_ppo',standing_prior=args.standing_prior,standing_prior_sha256=sha(args.standing_prior) if args.standing_prior else None)
   details['locomotion_prior_certificate']=args.locomotion_prior_certificate or None
   details['training_command_version']=TRAINING_COMMAND_VERSION
   details['upper_refinement_version']=UPPER_REFINEMENT_VERSION
   details['prior_output_scope']='Torso/legs only; upper posture has a separate training-only output loss'
   details.update(infos or {})
   return original_save(path,infos=details)
  runner.save=save_with_provenance
  if args.checkpoint:
   saved=torch.load(args.checkpoint,weights_only=False,map_location=envraw.device)
   assert saved.get('infos',{}).get('contract')==CONTRACT,'Checkpoint incompatible: never load old 22-axis policy'
   if args.actor_warm_start:
    assert args.mode=='train','Actor transfer is a training-only initialization'
    transfer=validate_actor_transfer(saved,HERE)
   else:
    assert saved['infos'].get('reward_signature')==reward_signature(),'Reward/observation/action source changed; use explicit audited actor transfer'
   assert saved['infos']['asset_sha256']==sha(HERE/'assets/robot_training.urdf'),'Asset changed since checkpoint'
   same_reward=saved['infos'].get('reward_version','legacy_v1')==REWARD_VERSION
   for key,path in [('motion_sha256',HERE/'assets/motion_reference.npz'),('actuator_sha256',HERE.parent/'assets/actuator_limits.json')]:
    if key in saved['infos']:assert saved['infos'][key]==sha(path),f'{key} changed since checkpoint'
   runner.load(args.checkpoint,load_cfg={'actor':True,'critic':same_reward and not args.actor_warm_start,'optimizer':args.mode=='train' and same_reward and not args.actor_warm_start,'iteration':False})
   actor_same=all(torch.equal(value,runner.alg.actor.state_dict()[key]) for key,value in saved['actor_state_dict'].items())
   assert actor_same,'Warm start changed actor tensors'
   if not args.actor_warm_start and same_reward:
    critic_same=all(torch.equal(v,runner.alg.critic.state_dict()[k]) for k,v in saved['critic_state_dict'].items())
    assert critic_same,'Full load changed critic tensors'
    optimizer_same=None
    if args.mode=='train':
     loaded=runner.alg.optimizer.state_dict();original=saved['optimizer_state_dict']
     optimizer_same=loaded['param_groups']==original['param_groups'] and loaded['state'].keys()==original['state'].keys()
     optimizer_same=optimizer_same and all(torch.equal(v.detach().cpu(),loaded['state'][k][field].detach().cpu()) if torch.is_tensor(v) else v==loaded['state'][k][field] for k,state in original['state'].items() for field,v in state.items())
     assert optimizer_same,'Full load changed Adam state'
    atomic_json(out/'full_load_proof.json',dict(actor_equal=True,critic_equal=critic_same,optimizer_equal=optimizer_same,
      source_sha256=sha(args.checkpoint),exploration_equal=True,learning_rate_restoration_follows=True))
   if args.actor_warm_start:
    assert not runner.alg.optimizer.state,'A changed objective must start with fresh optimizer moments'
    transfer.update(source=args.checkpoint,source_sha256=sha(args.checkpoint),new_reward=REWARD_VERSION,
       actor_tensors_equal_before_exploration_reset=actor_same)
    if not transfer['preserve_trained_exploration']:
     with torch.no_grad():runner.alg.actor.distribution.log_std_param.fill_(np.log(INITIAL_EXPLORATION_STD))
    # Qualify the preserved mean before the smoke's fresh-critic updates.
    baseline=str(out/'transferred_actor.pt')
    runner.save(baseline,infos={'actor_transfer_baseline':True})
    transfer['baseline_checkpoint']=baseline
    (out/'warm_start.json').write_text(json.dumps(transfer,indent=2))
   elif not same_reward:
    # Values and Adam moments from the old reward are not valid estimates here.
    # Keep the standing mean/normalizer; retain modest exploration for learning.
    if args.mode=='train':
     with torch.no_grad():runner.alg.actor.distribution.log_std_param.clamp_(min=np.log(.12))
    (out/'warm_start.json').write_text(json.dumps(dict(source=args.checkpoint,source_reward=saved['infos'].get('reward_version','legacy_v1'),new_reward=REWARD_VERSION,actor_tensors_equal_before_exploration_floor=actor_same,critic_loaded=False,optimizer_loaded=False,training_exploration_floor=.12 if args.mode=='train' else None),indent=2))
   del saved
   if args.mode=='train':
    runner.alg.learning_rate=restored_learning_rate(runner.alg.optimizer,args.learning_rate)
  else:
   # A trainable zero-residual mean begins at the physically checked standing prior.
   torch.nn.init.zeros_(runner.alg.actor.mlp[-1].weight);torch.nn.init.zeros_(runner.alg.actor.mlp[-1].bias)
  if args.mode=='train':
   retention=None;locomotion_retention=None
   joint_names=list(envraw.robot.joint_names)
   update_audit=PpoUpdateAudit(runner.alg)
   if args.standing_prior:
    assert sha(args.standing_prior)==args.standing_prior_sha256,'Standing teacher bytes changed'
    teacher=copy.deepcopy(runner.alg.actor)
    teacher_saved=torch.load(args.standing_prior,weights_only=False,map_location='cpu')
    missing_teacher_metadata=validate_teacher(teacher_saved['infos'],CONTRACT,
     {key:sha(path) for key,path in [('asset_sha256',HERE/'assets/robot_training.urdf'),('motion_sha256',HERE/'assets/motion_reference.npz'),('actuator_sha256',HERE.parent/'assets/actuator_limits.json')]})
    teacher.load_state_dict(teacher_saved['actor_state_dict']);del teacher_saved
    retention=LocomotionOnlyRetention(runner.alg.actor,teacher,joint_names=joint_names)
    (out/'standing_retention.json').write_text(json.dumps(dict(teacher=args.standing_prior,sha256=args.standing_prior_sha256,weight=.25,max_samples=256,learner_version=LEARNER_VERSION,training_only=True,legacy_metadata_fields_absent=missing_teacher_metadata,independent_current_asset_requalification_required=True),indent=2))
   if args.locomotion_prior_certificate:
    assert args.speed>=1.5,'Prior locomotion regularization is reserved for running'
    certificate=json.loads(Path(args.locomotion_prior_certificate).read_text())
    maximum=validate_qualification(certificate,certificate['checkpoint'],reward_signature(),GATE_VERSION)
    assert .1<maximum<args.speed,'Teacher may cover only qualified prior commands'
    teacher=copy.deepcopy(runner.alg.actor)
    teacher_saved=torch.load(certificate['checkpoint'],weights_only=False,map_location='cpu')
    assert teacher_saved['infos']['reward_signature']==reward_signature(),'Teacher reward/runtime differs'
    validate_teacher(teacher_saved['infos'],CONTRACT,
     {key:sha(path) for key,path in [('asset_sha256',HERE/'assets/robot_training.urdf'),('motion_sha256',HERE/'assets/motion_reference.npz'),('actuator_sha256',HERE.parent/'assets/actuator_limits.json')]})
    teacher.load_state_dict(teacher_saved['actor_state_dict']);del teacher_saved
    locomotion_retention=LocomotionOnlyRetention(runner.alg.actor,teacher,command_range=(.1,maximum),joint_names=joint_names)
    atomic_json(out/'locomotion_retention.json',dict(certificate=certificate,weight=.25,max_samples=256,
     learner_version=LEARNER_VERSION,training_only=True,inference_teacher=False))
   upper_refinement=UpperBodyRefinement(runner.alg.actor,joint_names)
   atomic_json(out/'upper_refinement.json',upper_refinement.read_audit())
   # Continuous heartbeat and GPU reserve checks at each PPO update.
   original_log=runner.logger.log
   def log_progress(**kwargs):
    # Collect cyclic Python/TensorDict objects between updates, never while
    # physics or autograd is executing. Record whether cleanup changes growth.
    collected=gc.collect() if kwargs['it']%4==0 else 0
    with torch.no_grad():runner.alg.actor.distribution.log_std_param.clamp_(np.log(.04),np.log(.5 if args.speed==0. else .7))
    original_log(**kwargs)
    # Small diagnostic sample; keep runtime telemetry bounded on the 8GB GPU.
    with torch.no_grad():
     raw=envraw.history[:16].reshape(-1,400)
     norm=runner.alg.actor.obs_normalizer(raw).reshape(-1,80).abs()
     groups={'angular':(0,3),'gravity':(3,6),'command':(6,9),'position':(9,32),'velocity':(32,55),'action_history':(55,78),'phase':(78,80)}
     normalized_p99={k:float(torch.quantile(norm[:,lo:hi].reshape(-1),.99)) for k,(lo,hi) in groups.items()}
     policy_debug=dict(normalized_observation_abs_p99=normalized_p99,
      exploration_std=runner.alg.actor.distribution.log_std_param.exp().cpu().tolist(),learning_rate=runner.alg.learning_rate,
      nominal_pose_max_abs_drift_rad=float((envraw.stand-envraw.nominal_stand_snapshot).abs().max()),
      scope='Last 16 environments and 5 history frames; a diagnostic sample, not a promotion gate')
     assert policy_debug['nominal_pose_max_abs_drift_rad']==0.,'Randomized reset corrupted the nominal action/observation origin'
    state=dict(iteration=kwargs['it'],num_envs=args.num_envs,speed=args.speed,time=time.time(),gpu=memory(),collect_seconds=kwargs['collect_time'],learn_seconds=kwargs['learn_time'],reward_version=REWARD_VERSION,behavior=envraw.read_reward_audit(),joints=envraw.read_joint_audit(),policy=policy_debug,ppo=update_audit.latest,retention=retention.read_audit() if retention else None,garbage_collected=collected)
    state['locomotion_retention']=locomotion_retention.read_audit() if locomotion_retention else None
    state['upper_refinement']=upper_refinement.read_audit()
    state['upper_refinement']['actual_joint_mean_rad']={name:float(envraw.robot.data.joint_pos[:,joint_names.index(name)].mean()) for name in upper_refinement.names if name in ['head_yaw','head_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']}
    state['command_coverage']=command_coverage(envraw.commands,args.speed)
    atomic_json(out/'progress.json',state)
    with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
    if state['gpu'].get('free_mb',99999)<args.reserve_mb:
     runner.save(str(out/'memory_recovery.pt'),infos={'contract':CONTRACT,'asset_sha256':sha(HERE/'assets/robot_training.urdf'),'motion_sha256':sha(HERE/'assets/motion_reference.npz'),'actuator_sha256':sha(HERE.parent/'assets/actuator_limits.json'),'reward_version':REWARD_VERSION,'speed':args.speed})
     raise MemoryError('VRAM_RESERVE_BELOW_300MB')
    reason=recycle_reason(state['gpu'])
    if reason:
     runner.save(str(out/'resource_recovery.pt'),infos={'iterations':kwargs['it']+1,'resource_recycle':reason})
     if kwargs['it']<3:raise MemoryError('HOST_RAM_PRESSURE_AT_WORKER_STARTUP')
     raise ResourceRecycle(kwargs['it'],reason,state['gpu'])
   runner.logger.log=log_progress
   try:
    runner.learn(args.iterations,init_at_random_ep_len=True)
   except ResourceRecycle as recycle:
    runner.logger.stop_logging_writer()
    result=dict(completed=True,checkpoint=str(out/'resource_recovery.pt'),iterations=recycle.iterations,
     recycled=True,recycle_reason=recycle.reason,seconds=time.monotonic()-start,num_envs=args.num_envs,gpu=recycle.memory)
   else:
    runner.save(str(out/'candidate.pt'),infos={'contract':CONTRACT,'asset_sha256':sha(HERE/'assets/robot_training.urdf'),'motion_sha256':sha(HERE/'assets/motion_reference.npz'),'actuator_sha256':sha(HERE.parent/'assets/actuator_limits.json'),'reward_version':REWARD_VERSION,'speed':args.speed,'source':args.checkpoint,'iterations':args.iterations})
    result=dict(completed=True,checkpoint=str(out/'candidate.pt'),iterations=args.iterations,seconds=time.monotonic()-start,num_envs=args.num_envs,gpu=memory())
  else:
   policy=runner.get_inference_policy(envraw.device)
   result=evaluate(env,policy,args.speed,args.duration,args.seed,args.perturb,out,play=args.mode=='play')
   if not args.checkpoint:
    runner.save(str(out/'initial_policy.pt'),infos={'contract':CONTRACT,'asset_sha256':sha(HERE/'assets/robot_training.urdf'),'motion_sha256':sha(HERE/'assets/motion_reference.npz'),'actuator_sha256':sha(HERE.parent/'assets/actuator_limits.json'),'reward_version':REWARD_VERSION,'speed':0.})
    result['checkpoint']=str(out/'initial_policy.pt')
   result['completed']=True;result['seconds']=time.monotonic()-start
 (out/'result.json').write_text(json.dumps(result,indent=2))
 print('[V29] COMPLETE '+json.dumps(result),flush=True)
 env.close()

if __name__=='__main__':
 status=0
 try:main()
 except BaseException as ex:
  status=2 if isinstance(ex,MemoryError) else 1
  traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
  Path(args.output).mkdir(parents=True,exist_ok=True)
  (Path(args.output)/'failure.json').write_text(json.dumps({'error':repr(ex),'time':time.time(),'status':status},indent=2))
 finally:
  # Kit close occasionally hangs on this host. Supervisor owns the completed PID.
  app.app.close()
 raise SystemExit(status)
