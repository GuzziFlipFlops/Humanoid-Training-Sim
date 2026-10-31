"""Read-only policy rollout: per-leg timing and sampled vertical-load balance."""
import argparse
import csv
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Isaac-Humanoid5ft-NaturalRun-v0')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--speed', type=float, default=1.8)
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
launcher = AppLauncher(args)

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils.hydra import hydra_task_config
import humanoid_tasks


@hydra_task_config(args.task, 'rsl_rl_cfg_entry_point')
def main(cfg, agent):
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum = None
    cfg.events.push = None
    cfg.episode_length_s = 24.0
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = (args.speed, args.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.0
    command.debug_vis = False
    version = metadata.version('rsl-rl-lib')
    agent = handle_deprecated_rsl_rl_cfg(agent, version)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(handle_deprecated_rsl_rl_checkpoint(args.checkpoint, version))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    raw = env.unwrapped
    robot = raw.scene['robot']
    sensor = raw.scene.sensors['contact_forces']
    names = ['left_foot', 'right_foot']
    foot_ids = [sensor.find_bodies(name)[0][0] for name in names]
    device = raw.device
    load = torch.zeros((args.num_envs, 2), device=device)
    contact_sum = torch.zeros_like(load)
    stance_sum = torch.zeros_like(load)
    stance_n = torch.zeros_like(load)
    air_sum = torch.zeros_like(load)
    air_n = torch.zeros_like(load)
    support = torch.zeros(3, device=device)
    speed_sum = torch.zeros(args.num_envs, device=device)
    speed_error = torch.zeros_like(speed_sum)
    counts = torch.zeros_like(speed_sum)
    ever_fallen = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dt = float(raw.step_dt)
    obs = env.get_observations()
    with output.with_suffix('.csv').open('w', newline='') as stream, torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(['sim_time_s', 'env0_episode_time_s', 'env0_vx_mps', 'env0_left_fz_n', 'env0_right_fz_n', 'env0_left_contact', 'env0_right_contact', 'env0_done'])
        for step in range(round(20 / dt)):
            obs, _, dones, extras = env.step(policy(obs))
            if hasattr(policy, 'reset'):
                policy.reset(dones)
            timeouts = extras.get('time_outs', torch.zeros_like(dones))
            ever_fallen |= dones.bool() & ~timeouts.bool()
            fz = sensor.data.net_forces_w[:, foot_ids, 2].clamp_min(0)
            contact = sensor.data.current_contact_time[:, foot_ids] > 0
            # Exclude startup and reset transients from timing/load estimates.
            valid = (raw.episode_length_buf * dt > 2.0) & ~dones.bool()
            load += fz * valid[:, None] * dt
            contact_sum += contact * valid[:, None]
            touchdown = sensor.compute_first_contact(dt)[:, foot_ids] & valid[:, None]
            lift = sensor.compute_first_air(dt)[:, foot_ids] & valid[:, None]
            air_sum += sensor.data.last_air_time[:, foot_ids] * touchdown
            air_n += touchdown
            stance_sum += sensor.data.last_contact_time[:, foot_ids] * lift
            stance_n += lift
            n_support = contact.sum(dim=1)
            for n in range(3):
                support[n] += ((n_support == n) & valid).sum()
            vx = robot.data.root_lin_vel_b[:, 0]
            speed_sum += vx * valid
            speed_error += (vx - args.speed).abs() * valid
            counts += valid
            writer.writerow([(step + 1) * dt, float(raw.episode_length_buf[0]) * dt, float(vx[0]), float(fz[0, 0]), float(fz[0, 1]), bool(contact[0, 0]), bool(contact[0, 1]), bool(dones[0])])
    mean_stance = stance_sum / stance_n.clamp_min(1)
    mean_air = air_sum / air_n.clamp_min(1)
    fractions = contact_sum / counts[:, None].clamp_min(1)
    report = {
        'checkpoint': args.checkpoint, 'task': args.task, 'target_mps': args.speed,
        'num_envs': args.num_envs, 'duration_s': 20, 'seed': 42,
        'pushes': False, 'observation_noise': False,
        'note': 'New rollout, not synchronized to uploaded video. Startup/reset first 2s excluded. Vertical load sampled at policy rate: diagnostic only, not impact-load certification.',
        'ever_fallen_fraction': float(ever_fallen.float().mean()),
        'mean_forward_speed_mps': float((speed_sum / counts.clamp_min(1)).mean()),
        'speed_mae_mps': float((speed_error / counts.clamp_min(1)).mean()),
        'left_right_contact_fraction': fractions.mean(dim=0).tolist(),
        'left_right_completed_stance_s': mean_stance.mean(dim=0).tolist(),
        'left_right_completed_air_s': mean_air.mean(dim=0).tolist(),
        'left_sampled_vertical_load_share': float((load[:, 0] / load.sum(dim=1).clamp_min(1)).mean()),
        'mean_absolute_per_env_load_asymmetry': float(((load[:, 0] - load[:, 1]).abs() / load.sum(dim=1).clamp_min(1)).mean()),
        'flight_single_double_fraction': (support / support.sum().clamp_min(1)).tolist(),
        'env0': {'left_right_contact_fraction': fractions[0].tolist(), 'left_right_stance_s': mean_stance[0].tolist(), 'left_right_air_s': mean_air[0].tolist(), 'left_vertical_load_share': float(load[0, 0] / load[0].sum().clamp_min(1))},
    }
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('BALANCE_REPORT=' + json.dumps(report))
    env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        launcher.app.close()
