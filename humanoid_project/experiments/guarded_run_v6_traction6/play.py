"""Read-only playback of a guarded v4 checkpoint at its saved action scale."""
import argparse
import importlib.metadata as metadata
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'source'))
DEFAULT = ROOT / 'logs/guarded_run_v4_20260903_0305/06_speed_4_0600_8192/candidate.pt'

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, default=DEFAULT)
parser.add_argument('--speed', type=float, default=4.)
parser.add_argument('--steps', type=int, default=0, help='0: run until the GUI is closed')
parser.add_argument('--real-time', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint.is_file():
    parser.error(f'Checkpoint does not exist: {args.checkpoint}')
if not math.isfinite(args.speed) or not 0. <= args.speed <= 4. or args.steps < 0:
    parser.error('Choose speed 0 through 4 m/s and nonnegative steps.')
launcher = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from rsl_rl.runners import OnPolicyRunner
from humanoid_tasks.curriculum.agents.rsl_rl_cfg import HumanoidRunFastPPORunnerCfg
from candidate import make_cfg
import humanoid_tasks


def main():
    # Local, user-owned checkpoint. Preserve its learned action/normalization convention.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    info = checkpoint.get('infos') or {}
    if info.get('candidate_version') != 'guarded_run_v4' or 'action_scale' not in info:
        raise ValueError('This viewer requires a v4 checkpoint with saved action_scale metadata.')
    scale = float(info['action_scale'])
    del checkpoint
    cfg = make_cfg('speed', 4., scale, 1)
    cfg.sim.device = args.device or 'cuda:0'
    cfg.seed = 47
    cfg.episode_length_s = 60.
    cfg.events.push = None
    cfg.events.torso_mass = None
    cfg.events.torso_com = None
    cfg.events.actuator_gains = None
    cfg.events.physics_material.params.update(static_friction_range=(.95, .95),
        dynamic_friction_range=(.8, .8), restitution_range=(0., 0.), num_buckets=1)
    for name in ('pose_range', 'velocity_range'):
        cfg.events.reset_base.params[name] = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.events.reset_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    cfg.observations.policy.enable_corruption = False
    # MixedSpeeds ignores a fixed range: replace the sampler, not only the range.
    command = cfg.commands.base_velocity
    command.class_type = UniformVelocityCommand
    command.ranges.lin_vel_x = (args.speed, args.speed)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    command.heading_command = False
    command.rel_standing_envs = 1. if args.speed == 0. else 0.
    command.resampling_time_range = (1000., 1000.)
    cfg.viewer.origin_type = 'asset_root'
    cfg.viewer.asset_name = 'robot'
    cfg.viewer.eye = (3., 3., 1.3)
    cfg.viewer.lookat = (0., 0., .1)
    version = metadata.version('rsl-rl-lib')
    agent = handle_deprecated_rsl_rl_cfg(HumanoidRunFastPPORunnerCfg(), version)
    agent.obs_groups = {'actor': ['policy'], 'critic': ['policy']}
    env = RslRlVecEnvWrapper(gym.make('Isaac-Humanoid5ft-NaturalRun-v0', cfg=cfg), clip_actions=agent.clip_actions)
    try:
        raw = env.unwrapped
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=raw.device)
        runner.load(handle_deprecated_rsl_rl_checkpoint(str(args.checkpoint), version),
                    load_cfg={'actor': True, 'critic': True, 'optimizer': False, 'iteration': False})
        policy = runner.get_inference_policy(raw.device)
        obs = env.get_observations()
        print(f'[PLAY] Loaded {args.checkpoint}\n[PLAY] Fixed {args.speed:.2f} m/s; native scale {scale}; no pushes.', flush=True)
        count = falls = 0
        while launcher.app.is_running() and (not args.steps or count < args.steps):
            start = time.monotonic()
            with torch.inference_mode():
                action = policy(obs)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite policy action')
                obs, _, dones, _ = env.step(action)
                policy.reset(dones)
            count += 1
            falls += int(raw.reset_terminated.sum().item())
            if count % max(1, round(1. / raw.step_dt)) == 0:
                vx = raw.scene['robot'].data.root_lin_vel_b[0, 0].item()
                actual_cmd = raw.command_manager.get_command('base_velocity')[0, 0].item()
                print(f'[PLAY] {count * raw.step_dt:.1f}s | command {actual_cmd:.2f} | actual {vx:.2f} m/s | falls {falls}', flush=True)
            if args.real_time:
                time.sleep(max(0., raw.step_dt - (time.monotonic() - start)))
        print(f'[PLAY] Completed {count} inference steps; falls {falls}. No training or checkpoint exports.', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        launcher.app.close()
