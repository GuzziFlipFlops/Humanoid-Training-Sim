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
from inference_helpers import push_laterally


def main():
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists(): raise FileExistsError('Completed blocks cannot be overwritten')
    cfg = make_cfg(args.num_envs, args.stage, args.groups, split=int(args.mode != 'train'))
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    (out / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=2))
    cfg.seed = args.seed
    cfg.sim.device = args.device or 'cuda:0'
    cfg.log_dir = str(out)
    if args.mode != 'train':
        cfg.events.physics_material.params.update(static_friction_range=(.95, .95),
            dynamic_friction_range=(.8, .8), num_buckets=1)
    agent = HumanoidBasePPORunnerCfg()
    agent.experiment_name = 'guarded_getup_v3'
    agent.algorithm.learning_rate = args.learning_rate
    agent.algorithm.gamma = .9985  # retain useful value across a seven-second hold
    agent.algorithm.lam = .95
    agent.algorithm.schedule = 'adaptive'
    agent.algorithm.entropy_coef = .001 if args.stage == 'discovery' else .0005
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
            version = infos.get('candidate_version')
            if version not in ('guarded_getup_v2', 'guarded_getup_v3') or infos.get('action_scale') != .7:
                raise RuntimeError('Refusing a different policy schema/action scale')
            # Preserve the learned actor and its normalizer. A changed reward
            # requires a fresh value estimator/optimizer on the v1 migration.
            continuing = version == 'guarded_getup_v3'
            runner.load(args.checkpoint, load_cfg={'actor': True, 'critic': continuing,
                'optimizer': continuing, 'iteration': False})
            if not continuing:
                with torch.no_grad(): runner.alg.actor.distribution.std_param.fill_(.45)
        def bound_noise(optimizer, positional, keyword):
            with torch.no_grad(): runner.alg.actor.distribution.std_param.clamp_(.15, .60)
        runner.alg.optimizer.register_step_post_hook(bound_noise)
        runner.current_learning_iteration = 0
        start = time.monotonic()
        print(f'[GETUP] {args.mode}; {args.num_envs} envs, {args.iterations} iterations, groups={args.groups}; source={args.checkpoint}', flush=True)
        if args.mode == 'train':
            runner.learn(args.iterations, init_at_random_ep_len=False)
            checkpoint = out / 'candidate.pt'
            runner.save(str(checkpoint), infos={'candidate_version': 'guarded_getup_v3', 'action_scale': .7,
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
            # Feasibility-only ankle IMU feedback, not recovery training help.
            # Measured settled q includes gravity deflection; using it as a new
            # PD target would add a second deflection and tip the robot.
            hold_action = torch.zeros_like(raw.scene['robot'].data.joint_pos)
            ankle_pitch = [raw.scene['robot'].joint_names.index(s+'_ankle_pitch') for s in ('left','right')]
            ankle_roll = [raw.scene['robot'].joint_names.index(s+'_ankle_roll') for s in ('left','right')]
            term = raw._recovery_reward
            term.capture = True
            records = {}
            for step in range(round(25. / raw.step_dt)):
                if args.perturb and step == round(16. / raw.step_dt):
                    push_laterally(raw.scene['robot'], .10)
                    print('[GETUP] perturbation applied at 16 s', flush=True)
                with torch.inference_mode():
                    if args.mode == 'hold':
                        gravity = raw.scene['robot'].data.projected_gravity_b
                        omega = raw.scene['robot'].data.root_ang_vel_b
                        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
                        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
                        hold_action[:, ankle_pitch] = ((.65*pitch + .16*omega[:, 1]).clamp(-.18,.18)/.7)[:,None]
                        hold_action[:, ankle_roll] = ((.65*roll + .16*omega[:, 0]).clamp(-.12,.12)/.7)[:,None]
                    action = hold_action if args.mode == 'hold' else policy(obs)
                    if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite inference')
                    obs, _, done, _ = env.step(action)
                    policy.reset(done)
                for name, value in term.captured.items(): records.setdefault(name, []).append(value)
                if (step+1) % 250 == 0:
                    print(f'[GETUP] evaluation {step+1}/1250 steps; mean height={term.captured["height"].mean():.3f} m', flush=True)
            trace = {k: np.stack(v) for k, v in records.items()}
            report = audit(trace, raw.step_dt)
            report.update(group=args.group, seed=args.seed, perturb=args.perturb, thresholds=asdict(Limits(progress_budget=4., failure_penalty=8.)))
            np.savez_compressed(out / 'telemetry.npz', **trace)
            (out / 'evaluation.json').write_text(json.dumps(report, indent=2))
            result = dict(completed=True, evaluation=str(out/'evaluation.json'),
                          seconds=time.monotonic()-start, report=report)
        result['checkpoint_input_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None
        result['source_hashes'] = source_hashes
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
