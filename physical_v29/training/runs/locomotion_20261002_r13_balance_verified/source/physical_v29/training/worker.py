"""One Isaac process for conversion, preflight, PPO or independent evaluation.

Only result.json confirms completion. Failures print tracebacks even if Kit
returns zero. The supervisor owns promotion, memory fallback and shutdown.
"""
import argparse,json,hashlib,time,os,sys,traceback,copy,subprocess
from pathlib import Path
from status_io import atomic_json
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['preflight','train','eval','play'],required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--checkpoint',default='')
parser.add_argument('--num_envs',type=int,default=64)
parser.add_argument('--speed',type=float,default=0.)
parser.add_argument('--iterations',type=int,default=100)
parser.add_argument('--learning_rate',type=float,default=.0002)
parser.add_argument('--seed',type=int,default=71)
parser.add_argument('--duration',type=float,default=20.)
parser.add_argument('--perturb',action='store_true')
parser.add_argument('--reserve_mb',type=int,default=300)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args)
import torch,numpy as np,psutil
torch.set_num_threads(4)
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment import LocomotionCfg,LocomotionEnv,META,MASS
from evaluation import evaluate,GATE_VERSION
from walking_rewards import REWARD_VERSION

CONTRACT='physical_v29_history4_ik_residual_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def agent_config():
 return dict(num_steps_per_env=24,save_interval=50,logger='tensorboard',check_for_nan=True,
   obs_groups={'actor':['policy'],'critic':['critic']},
   actor=dict(class_name='MLPModel',hidden_dims=[512,256,128],activation='elu',obs_normalization=True,
      distribution_cfg=dict(class_name='GaussianDistribution',init_std=.30,std_type='log')),
   critic=dict(class_name='MLPModel',hidden_dims=[512,256,128],activation='elu',obs_normalization=True),
   algorithm=dict(class_name='PPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,
     entropy_coef=.003,num_learning_epochs=4,num_mini_batches=8,learning_rate=args.learning_rate,
     schedule='adaptive',gamma=.995,lam=.95,desired_kl=.008,max_grad_norm=1.,rnd_cfg=None,symmetry_cfg=None))

def memory():
 try:
  line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()[0]
  used,free,util=[int(x.strip()) for x in line.split(',')]
  host=psutil.virtual_memory();proc=psutil.Process().memory_info()
  return dict(used_mb=used,free_mb=free,utilization=util,ram_available_mb=round(host.available/2**20),
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
 (out/'run_config.json').write_text(json.dumps(dict(arguments=vars(args),contract=CONTRACT,reward_version=REWARD_VERSION,gate_version=GATE_VERSION,reference_runtime='controller_envelope_v2',joints=[j['name'] for j in META['joints']],mass_kg=MASS['total_mass_kg'],urdf_sha256=sha(HERE/'assets/robot_training.urdf'),actuator_sha256=sha(HERE.parent/'assets/actuator_limits.json'),motion_sha256=sha(HERE/'assets/motion_reference.npz'),reference='CAD IK gait guidance, not a human mocap or AMP policy',gravity=True,fixed_base=False,self_collision=True,physics_dt=.005,control_dt=.020),indent=2))
 envraw=LocomotionEnv(cfg);envraw.initialize_randomization();env=RslRlVecEnvWrapper(envraw,clip_actions=3.)
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
  original_save=runner.save
  def save_with_provenance(path,infos=None):
   details=dict(contract=CONTRACT,asset_sha256=sha(HERE/'assets/robot_training.urdf'),motion_sha256=sha(HERE/'assets/motion_reference.npz'),actuator_sha256=sha(HERE.parent/'assets/actuator_limits.json'),reward_version=REWARD_VERSION,speed=args.speed,source=args.checkpoint)
   details.update(infos or {})
   return original_save(path,infos=details)
  runner.save=save_with_provenance
  if args.checkpoint:
   saved=torch.load(args.checkpoint,weights_only=False,map_location=envraw.device)
   assert saved.get('infos',{}).get('contract')==CONTRACT,'Checkpoint incompatible: never load old 22-axis policy'
   assert saved['infos']['asset_sha256']==sha(HERE/'assets/robot_training.urdf'),'Asset changed since checkpoint'
   same_reward=saved['infos'].get('reward_version','legacy_v1')==REWARD_VERSION
   for key,path in [('motion_sha256',HERE/'assets/motion_reference.npz'),('actuator_sha256',HERE.parent/'assets/actuator_limits.json')]:
    if key in saved['infos']:assert saved['infos'][key]==sha(path),f'{key} changed since checkpoint'
   runner.load(args.checkpoint,load_cfg={'actor':True,'critic':same_reward,'optimizer':args.mode=='train' and same_reward,'iteration':False})
   actor_same=all(torch.equal(value,runner.alg.actor.state_dict()[key]) for key,value in saved['actor_state_dict'].items())
   assert actor_same,'Warm start changed actor tensors'
   if not same_reward:
    # Values and Adam moments from the old reward are not valid estimates here.
    # Keep the standing mean/normalizer; retain modest exploration for learning.
    if args.mode=='train':
     with torch.no_grad():runner.alg.actor.distribution.log_std_param.clamp_(min=np.log(.12))
    (out/'warm_start.json').write_text(json.dumps(dict(source=args.checkpoint,source_reward=saved['infos'].get('reward_version','legacy_v1'),new_reward=REWARD_VERSION,actor_tensors_equal_before_exploration_floor=actor_same,critic_loaded=False,optimizer_loaded=False,training_exploration_floor=.12 if args.mode=='train' else None),indent=2))
   del saved
   if args.mode=='train':
    runner.alg.learning_rate=min(runner.alg.learning_rate,args.learning_rate)
    for group in runner.alg.optimizer.param_groups:group['lr']=runner.alg.learning_rate
  else:
   # A trainable zero-residual mean begins at the physically checked standing prior.
   torch.nn.init.zeros_(runner.alg.actor.mlp[-1].weight);torch.nn.init.zeros_(runner.alg.actor.mlp[-1].bias)
  if args.mode=='train':
   # Continuous heartbeat and GPU reserve checks at each PPO update.
   original_log=runner.logger.log
   def log_progress(**kwargs):
    with torch.no_grad():runner.alg.actor.distribution.log_std_param.clamp_(np.log(.04),np.log(.65))
    original_log(**kwargs)
    # Small diagnostic sample; keep runtime telemetry bounded on the 8GB GPU.
    with torch.no_grad():
     raw=envraw.history[:16].reshape(-1,320)
     norm=runner.alg.actor.obs_normalizer(raw).reshape(-1,80).abs()
     groups={'angular':(0,3),'gravity':(3,6),'command':(6,9),'position':(9,32),'velocity':(32,55),'action_history':(55,78),'phase':(78,80)}
     normalized_p99={k:float(torch.quantile(norm[:,lo:hi].reshape(-1),.99)) for k,(lo,hi) in groups.items()}
     policy_debug=dict(normalized_observation_abs_p99=normalized_p99,
      exploration_std=runner.alg.actor.distribution.log_std_param.exp().cpu().tolist(),learning_rate=runner.alg.learning_rate,
      scope='Last 16 environments and 4 history frames; a diagnostic sample, not a promotion gate')
    state=dict(iteration=kwargs['it'],num_envs=args.num_envs,speed=args.speed,time=time.time(),gpu=memory(),collect_seconds=kwargs['collect_time'],learn_seconds=kwargs['learn_time'],reward_version=REWARD_VERSION,behavior=envraw.read_reward_audit(),joints=envraw.read_joint_audit(),policy=policy_debug)
    atomic_json(out/'progress.json',state)
    with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
    if state['gpu'].get('free_mb',99999)<args.reserve_mb:
     runner.save(str(out/'memory_recovery.pt'),infos={'contract':CONTRACT,'asset_sha256':sha(HERE/'assets/robot_training.urdf'),'motion_sha256':sha(HERE/'assets/motion_reference.npz'),'actuator_sha256':sha(HERE.parent/'assets/actuator_limits.json'),'reward_version':REWARD_VERSION,'speed':args.speed})
     raise MemoryError('VRAM_RESERVE_BELOW_300MB')
   runner.logger.log=log_progress
   runner.learn(args.iterations,init_at_random_ep_len=True)
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
