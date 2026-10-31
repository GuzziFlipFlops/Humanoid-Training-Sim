"""One bounded recovery train/evaluation block with source and checkpoint records."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'eval', 'hold'], required=True)
parser.add_argument('--stage', choices=['discovery', 'refine'], default='discovery')
parser.add_argument('--num_envs', type=int, default=8192)
parser.add_argument('--iterations', type=int, default=200)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--group', type=int, default=0)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seed', type=int, default=47)
parser.add_argument('--perturb', action='store_true')
parser.add_argument('--learning_rate', type=float, default=3e-4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from humanoid_tasks.curriculum.agents.rsl_rl_cfg import HumanoidBasePPORunnerCfg
from candidate import make_cfg, HERE as CANDIDATE_HERE
from evaluation import audit
from reward_core import Limits


def main():
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists(): raise FileExistsError('Completed blocks cannot be overwritten')
    cfg = make_cfg(args.num_envs, args.stage, args.groups, split=int(args.mode != 'train'))
    cfg.seed = args.seed
    cfg.sim.device = args.device or 'cuda:0'
    cfg.log_dir = str(out)
    if args.mode != 'train':
        cfg.events.physics_material.params.update(static_friction_range=(.95, .95),
            dynamic_friction_range=(.8, .8), num_buckets=1)
    agent = HumanoidBasePPORunnerCfg()
    agent.experiment_name = 'guarded_getup_v1'
    agent.algorithm.learning_rate = args.learning_rate
    agent.algorithm.gamma = .9985  # retain useful value across a seven-second hold
    agent.algorithm.lam = .95
    agent.algorithm.schedule = 'adaptive'
    agent.algorithm.entropy_coef = .008 if args.stage == 'discovery' else .003
    agent.save_interval = 50
    agent.num_steps_per_env = 24
    agent.seed = args.seed
    agent = handle_deprecated_rsl_rl_cfg(agent, metadata.version('rsl-rl-lib'))
    agent.obs_groups = {'actor': ['policy'], 'critic': ['critic']}
    dump_yaml(str(out / 'env.yaml'), cfg)
    dump_yaml(str(out / 'agent.yaml'), agent)
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg), clip_actions=agent.clip_actions)
    raw = env.unwrapped
    try:
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(out) if args.mode == 'train' else None, device=raw.device)
        if args.checkpoint:
            stored = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            infos = stored.get('infos') or {}
            if infos.get('candidate_version') != 'guarded_getup_v1' or infos.get('action_scale') != .7:
                raise RuntimeError('Refusing a different policy schema/action scale')
            runner.load(args.checkpoint, load_cfg={'actor': True, 'critic': True, 'optimizer': True, 'iteration': False})
        runner.current_learning_iteration = 0
        start = time.monotonic()
        print(f'[GETUP] {args.mode}; {args.num_envs} envs, {args.iterations} iterations, groups={args.groups}; new recovery policy', flush=True)
        if args.mode == 'train':
            runner.learn(args.iterations, init_at_random_ep_len=False)
            checkpoint = out / 'candidate.pt'
            runner.save(str(checkpoint), infos={'candidate_version': 'guarded_getup_v1', 'action_scale': .7,
                'stage': args.stage, 'groups': args.groups, 'source': args.checkpoint,
                'iterations_this_block': args.iterations})
            counts = raw._recovery_reward.counts.cpu().tolist()
            result = dict(completed=True, checkpoint=str(checkpoint), seconds=time.monotonic()-start,
                          iterations=args.iterations, num_envs=args.num_envs, episode_counts=counts,
                          reward_totals=raw._recovery_reward.totals.cpu().tolist(),
                          action_scale=.7, stage=args.stage, groups=args.groups)
        else:
            if not args.checkpoint and args.mode != 'hold': raise ValueError('Evaluation requires a checkpoint')
            raw._forced_group = args.group
            env.reset()
            policy = runner.get_inference_policy(raw.device)
            obs = env.get_observations()
            hold_action = (raw.scene['robot'].data.joint_pos - raw.scene['robot'].data.default_joint_pos) / .7
            term = raw._recovery_reward
            term.capture = True
            records = {}
            for step in range(round(25. / raw.step_dt)):
                if args.perturb and step == round(16. / raw.step_dt):
                    velocity = raw.scene['robot'].data.root_vel_w.clone()
                    velocity[:, 1] += .10
                    raw.scene['robot'].write_root_velocity_to_sim(velocity)
                with torch.inference_mode():
                    action = hold_action if args.mode == 'hold' else policy(obs)
                    if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite inference')
                    obs, _, done, _ = env.step(action)
                    policy.reset(done)
                for name, value in term.captured.items(): records.setdefault(name, []).append(value)
            trace = {k: np.stack(v) for k, v in records.items()}
            report = audit(trace, raw.step_dt)
            report.update(group=args.group, seed=args.seed, perturb=args.perturb, thresholds=asdict(Limits()))
            np.savez_compressed(out / 'telemetry.npz', **trace)
            (out / 'evaluation.json').write_text(json.dumps(report, indent=2))
            result = dict(completed=True, evaluation=str(out/'evaluation.json'),
                          seconds=time.monotonic()-start, report=report)
        result['checkpoint_input_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        print('[GETUP] COMPLETE ' + json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    status = 0
    try:
        main()
    except BaseException:
        status = 1
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        app.app.close()
    raise SystemExit(status)
