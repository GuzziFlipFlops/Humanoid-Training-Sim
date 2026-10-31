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
parser.add_argument('--stage', choices=['hold', 'discovery', 'refine'], default='discovery')
parser.add_argument('--num_envs', type=int, default=8192)
parser.add_argument('--iterations', type=int, default=200)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--group', type=int, default=0)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seed', type=int, default=47)
parser.add_argument('--perturb', action='store_true')
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--assist', type=float, default=0.)
parser.add_argument('--eval_seconds', type=float, default=25.)
parser.add_argument('--forced_group', type=int, default=-1)
parser.add_argument('--start_mode', choices=['lying', 'crouch', 'deep_crouch',
                    *(f'tilt{angle}' for angle in (1, 2, 3, 4, *range(5, 90, 5))),
                    'tucked_supine'], default='lying')
parser.add_argument('--disable_self_collisions', action='store_true')
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

TRUSTED_BOOTSTRAP_SHA256 = {
    # Natural standing model and the user-approved walker (zero command).
    'f87d0eb0851ebbabb7ac6c17b26cb596a6f674f86e298d33dbb24a95e4fe9851',
    'dbddd7222f591d8d43d5e13d679a8ec4f9deb251dd864a15f78e9f88dcf6e174',
}

# The archived standing policy learned to remain upright, but its old task did
# not penalize yaw rate and allowed large unnecessary head/arm/yaw motions.
# Neutralizing only these output rows preserves the learned sagittal and ankle
# balance controller while removing the known reward exploit deterministically.
STAND_NEUTRAL_JOINTS = (
    'left_hip_yaw', 'right_hip_yaw', 'waist_yaw',
    'head_yaw', 'head_pitch',
    'left_shoulder_pitch', 'right_shoulder_pitch',
    'left_shoulder_roll', 'right_shoulder_roll',
    'left_elbow_pitch', 'right_elbow_pitch',
)


def main():
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists(): raise FileExistsError('Completed blocks cannot be overwritten')
    if args.mode != 'train' and args.assist != 0. and args.eval_seconds >= 25.:
        raise ValueError('Strict 25-second evaluation must be unassisted')
    cfg = make_cfg(args.num_envs, args.stage, args.groups, split=int(args.mode != 'train'),
                   assist=args.assist, forced_group=args.forced_group,
                   self_collisions=not args.disable_self_collisions,
                   start_mode=args.start_mode)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    (out / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=2))
    cfg.seed = args.seed
    cfg.sim.device = args.device or 'cuda:0'
    cfg.log_dir = str(out)
    if args.mode != 'train':
        cfg.events.physics_material.params.update(static_friction_range=(.95, .95),
            dynamic_friction_range=(.8, .8), num_buckets=1)
    agent = HumanoidBasePPORunnerCfg()
    agent.experiment_name = 'guarded_getup_v13_stable_reverse'
    # v5's 0.18 exploration with a .25-rad action scale was too small to
    # discover recovery.  v6 restores the isolated recovery scale and broad
    # exploration used by the earlier moving candidates.
    agent.algorithm.learning_rate = min(args.learning_rate, 3e-4)
    agent.algorithm.gamma = .9985  # retain useful value across a seven-second hold
    agent.algorithm.lam = .95
    agent.algorithm.schedule = 'adaptive'
    agent.algorithm.desired_kl = .01
    agent.algorithm.clip_param = .20
    agent.algorithm.num_learning_epochs = 5
    agent.algorithm.entropy_coef = .005
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
            allowed_versions = {
                'guarded_getup_v11_rise_only',
                'guarded_getup_v12_reverse_curriculum',
                'guarded_getup_v13_stable_reverse',
            }
            if (infos.get('candidate_version') not in allowed_versions
                    or infos.get('action_contract') != 'rise_only_full_range_v1'):
                raise RuntimeError('Refusing a checkpoint outside the compatible v11-v13 full-range contract')
            same_reward = infos.get('candidate_version') == 'guarded_getup_v13_stable_reverse'
            # V13 changes the objective from transient height to stable dwell.
            # Preserve the moving actor and its normalizer, but never import a
            # critic or optimizer trained against the obsolete v11/v12 return.
            runner.load(args.checkpoint, load_cfg={'actor': True, 'critic': same_reward,
                'optimizer': same_reward, 'iteration': False})
        elif args.mode == 'train':
            # A corrected-axis recovery network starts from scratch.  Loading a
            # locomotion/standing actor would mismatch the four-frame observation
            # contract and repeat the invalid v8 migration.
            with torch.no_grad():
                # The action is normalized across each joint's full URDF
                # range.  The old 0.45 std produced roughly one-radian knee
                # jumps and 80-93% unsafe resets even at a five-degree start.
                # Reverse curriculum needs local exploration around the
                # already-successful neutral controller, not random poses.
                runner.alg.actor.distribution.std_param.fill_(.20)
                actor_head = list(runner.alg.actor.mlp.children())[-1]
                actor_head.weight.zero_()
                actor_head.bias.zero_()
                print('[GETUP] zero-mean initialization preserves the verified crouch-to-stand controller', flush=True)
        # Unlike v10, rise-only discovery trains the complete actor and its
        # observation normalizer.  There is no output-head trust region or
        # standing-controller preservation constraint in this experiment.
        print('[GETUP] full 22-joint actor/normalizer trainable; full limit-aware action range', flush=True)
        runner.current_learning_iteration = 0
        start = time.monotonic()
        print(f'[GETUP] {args.mode}; {args.num_envs} envs, {args.iterations} iterations, groups={args.groups}; forced_group={args.forced_group}; assist={args.assist:.2f}; source={args.checkpoint}', flush=True)
        if args.mode == 'train':
            runner.learn(args.iterations, init_at_random_ep_len=False)
            checkpoint = out / 'candidate.pt'
            reward_diagnostics = {}
            for name, values in raw.reward_manager._episode_sums.items():
                values = values.detach()
                finite = bool(torch.isfinite(values).all())
                nonzero = int(torch.count_nonzero(values).item())
                reward_diagnostics[name] = {
                    'finite': finite,
                    'nonzero_envs': nonzero,
                    'mean_abs': float(values.abs().mean()),
                    'max_abs': float(values.abs().max()),
                }
            intended = ('height', 'upward_velocity') if args.stage == 'discovery' else (
                'height_band', 'upward_velocity_below_stand', 'upright',
                'two_foot_support', 'quiet_stand')
            invalid = [] if args.iterations == 0 else [name for name in intended
                       if name not in reward_diagnostics
                       or not reward_diagnostics[name]['finite']
                       or reward_diagnostics[name]['nonzero_envs'] == 0]
            if invalid:
                raise RuntimeError(f'Intended reward terms were nonfinite or identically zero: {invalid}')
            print('[GETUP] reward diagnostics ' + json.dumps(reward_diagnostics), flush=True)
            runner.save(str(checkpoint), infos={'candidate_version': 'guarded_getup_v13_stable_reverse',
                'action_contract': 'rise_only_full_range_v1',
                'stage': args.stage, 'groups': args.groups, 'source': args.checkpoint,
                'iterations_this_block': args.iterations, 'assist_fraction': args.assist,
                'forced_group': args.forced_group, 'start_mode': args.start_mode})
            counts = raw._recovery_reward.counts.cpu().tolist()
            result = dict(completed=True, checkpoint=str(checkpoint), seconds=time.monotonic()-start,
                          iterations=args.iterations, num_envs=args.num_envs, episode_counts=counts,
                          reward_totals=raw._recovery_reward.totals.cpu().tolist(),
                           action_contract='rise_only_full_range_v1', stage=args.stage, groups=args.groups,
                           assist_fraction=args.assist, forced_group=args.forced_group,
                           start_mode=args.start_mode, reward_diagnostics=reward_diagnostics,
                          final_mean_assist_n=float(getattr(raw, '_assist_force_z', torch.zeros(1, device=raw.device)).mean()))
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
            action_term = raw.action_manager.get_term('joint_pos')
            ankle_pitch = [raw.scene['robot'].joint_names.index(s+'_ankle_pitch') for s in ('left','right')]
            ankle_roll = [raw.scene['robot'].joint_names.index(s+'_ankle_roll') for s in ('left','right')]
            term = raw._recovery_reward
            term.capture = True
            records = {}
            eval_steps = round(args.eval_seconds / raw.step_dt)
            for step in range(eval_steps):
                if args.perturb and step == round(16. / raw.step_dt):
                    push_laterally(raw.scene['robot'], .10)
                    print('[GETUP] perturbation applied at 16 s', flush=True)
                with torch.inference_mode():
                    if args.mode == 'hold':
                        gravity = raw.scene['robot'].data.projected_gravity_b
                        omega = raw.scene['robot'].data.root_ang_vel_b
                        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
                        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
                        target = raw.scene['robot'].data.default_joint_pos.clone()
                        target[:, ankle_pitch] += ((.65*pitch + .16*omega[:, 1]).clamp(-.18,.18))[:,None]
                        target[:, ankle_roll] += ((.65*roll + .16*omega[:, 0]).clamp(-.12,.12))[:,None]
                        hold_action.copy_(action_term.normalized_for_target(target))
                    action = hold_action if args.mode == 'hold' else policy(obs)
                    if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite inference')
                    obs, _, done, _ = env.step(action)
                    policy.reset(done)
                    # Isaac's RewardManager intentionally skips zero-weight
                    # terms. Execute the zero-return telemetry term explicitly
                    # during evaluation; its value is not added to reward.
                    term(raw)
                for name, value in term.captured.items(): records.setdefault(name, []).append(value)
                if (step+1) % 250 == 0:
                    print(f'[GETUP] evaluation {step+1}/{eval_steps} steps; mean height={term.captured["height"].mean():.3f} m', flush=True)
            trace = {k: np.stack(v) for k, v in records.items()}
            report = audit(trace, raw.step_dt)
            report.update(group=args.group, seed=args.seed, perturb=args.perturb,
                          assist_fraction=args.assist, eval_seconds=args.eval_seconds,
                          thresholds=asdict(Limits(progress_budget=4., failure_penalty=0.)))
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
