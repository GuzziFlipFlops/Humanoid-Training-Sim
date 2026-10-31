"""One bounded training block or matched evaluation. The supervisor owns promotion."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'source'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--mode',choices=['train','eval'],required=True)
parser.add_argument('--stage',choices=['index','balance','transition','targets','torso','speed'],default='index')
parser.add_argument('--ceiling',type=float,default=1.8)
parser.add_argument('--scale',type=float,default=.45)
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--reference',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--num_envs',type=int,default=8192)
parser.add_argument('--iterations',type=int,default=200)
parser.add_argument('--learning_rate',type=float,default=5e-5)
parser.add_argument('--seed',type=int,default=47)
parser.add_argument('--perturb',action='store_true')
parser.add_argument('--resume_optimizer',action='store_true')
parser.add_argument('--replicas',type=int,default=24)
parser.add_argument('--duration',type=float,default=20.)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from rsl_rl.runners import OnPolicyRunner
from humanoid_tasks.curriculum.agents.rsl_rl_cfg import HumanoidRunFastPPORunnerCfg
from candidate import make_cfg, ANKLES
from evaluation import evaluate
import humanoid_tasks


def main():
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise FileExistsError('Refusing to overwrite a completed block')
    count = args.num_envs if args.mode=='train' else args.replicas*6
    cfg=make_cfg(args.stage,args.ceiling,args.scale,count)
    cfg.seed=args.seed
    cfg.sim.device=args.device or 'cuda:0'
    if args.mode=='eval':
        cfg.episode_length_s=args.duration+5.
        cfg.events.push=None
        cfg.events.torso_mass=None; cfg.events.torso_com=None; cfg.events.actuator_gains=None
        cfg.events.physics_material.params.update(static_friction_range=(.95,.95),dynamic_friction_range=(.8,.8),
                                                  restitution_range=(0.,0.),num_buckets=1)
        cfg.events.reset_base.params['pose_range']={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
        cfg.events.reset_base.params['velocity_range']={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
        cfg.events.reset_joints.params.update(position_range=(1.,1.),velocity_range=(0.,0.))
        cfg.observations.policy.enable_corruption=False
        cfg.commands.base_velocity.resampling_time_range=(1000.,1000.)
    cfg.log_dir=str(out)
    version=metadata.version('rsl-rl-lib')
    agent=HumanoidRunFastPPORunnerCfg()
    agent.experiment_name='guarded_run_v3'
    agent.save_interval=50
    agent.algorithm.learning_rate=args.learning_rate
    agent.algorithm.schedule='fixed'
    agent.algorithm.entropy_coef=.002
    agent.algorithm.clip_param=.10
    agent.seed=args.seed
    agent=handle_deprecated_rsl_rl_cfg(agent,version)
    agent.obs_groups={'actor':['policy'],'critic':['policy']}
    dump_yaml(str(out/'env.yaml'),cfg); dump_yaml(str(out/'agent.yaml'),agent)
    env=RslRlVecEnvWrapper(gym.make('Isaac-Humanoid5ft-NaturalRun-v0',cfg=cfg),clip_actions=agent.clip_actions)
    raw=env.unwrapped
    term=raw.reward_manager.get_term_cfg('ankle_differential').func
    assert [raw.scene['robot'].joint_names[i] for i in term.ids]==ANKLES
    # Check the blend at the important endpoints in the actual loaded simulator.
    # This catches an unintended reward change to <=1 m/s walking before PPO starts.
    from candidate import BlendedRunReward
    command=raw.command_manager.get_term('base_velocity')
    saved_command=command.vel_command_b.clone()
    for name in raw.reward_manager.active_terms:
        term_cfg=raw.reward_manager.get_term_cfg(name)
        if isinstance(term_cfg.func,BlendedRunReward):
            blend=term_cfg.func
            for speed,endpoint in ((1.,blend.walk),(2.,blend.run)):
                command.vel_command_b[:]=0.; command.vel_command_b[:,0]=speed
                expected=endpoint.func(raw,**endpoint.params)*endpoint.weight
                actual=blend(raw,name)
                torch.testing.assert_close(actual,expected)
    command.vel_command_b.copy_(saved_command)
    print('[GUARDED] reward blend endpoints verified (unchanged walk contributions at <=1 m/s)',flush=True)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out) if args.mode=='train' else None,device=raw.device)
    ckpt=handle_deprecated_rsl_rl_checkpoint(args.checkpoint,version)
    runner.load(ckpt,load_cfg={'actor':True,'critic':True,'optimizer':args.resume_optimizer,'iteration':False})
    runner.current_learning_iteration=0
    print('[GUARDED] checkpoint loaded; ordered ankle indices verified; '+json.dumps(vars(args)),flush=True)
    start=time.monotonic()
    if args.mode=='train':
        runner.learn(args.iterations,init_at_random_ep_len=True)
        runner.save(str(out/'candidate.pt'),infos={'candidate_version':'guarded_run_v3','stage':args.stage,
                    'action_scale':args.scale,'source':args.checkpoint,'iterations_this_block':args.iterations})
        result=dict(completed=True,checkpoint=str(out/'candidate.pt'),iterations=args.iterations,
                    seconds=time.monotonic()-start,num_envs=count,action_scale=args.scale,stage=args.stage)
        if hasattr(raw, '_guarded_stance'):
            samples, valid, ungated, gated = raw._guarded_stance.diagnostic_totals.tolist()
            result['stance_signal'] = dict(valid_fraction=valid/max(samples,1.),
                ungated_mean_cost=ungated/max(samples,1.), mean_cost=gated/max(samples,1.),
                weighted_mean_cost=-.60*gated/max(samples,1.))
            assert valid > 0 and gated > 0, 'Balance shaping never activated; do not run an ineffective chain'
    else:
        reference=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=raw.device)
        reference.load(handle_deprecated_rsl_rl_checkpoint(args.reference,version))
        report=evaluate(env,{'reference':reference.get_inference_policy(raw.device),
                             'candidate':runner.get_inference_policy(raw.device)},args.scale,args.ceiling,out,
                        args.seed,args.perturb,args.duration,args.replicas)
        result=dict(completed=True,evaluation=str(out/'evaluation.json'),seconds=time.monotonic()-start)
    result['source_sha256']=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('[GUARDED] COMPLETE '+json.dumps(result),flush=True)
    env.close()


if __name__=='__main__':
    status=0
    try:
        main()
    except BaseException:
        status=1
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    finally:
        app.app.close()
    raise SystemExit(status)
