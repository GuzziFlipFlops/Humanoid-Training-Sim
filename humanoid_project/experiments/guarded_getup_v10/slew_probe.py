"""Compare recovery target slew rates in one exact-physics Isaac batch."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--replicas', type=int, default=16)
parser.add_argument('--seconds', type=float, default=15.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from candidate import make_cfg
from evaluation import audit

RATES = (2.5, 4., 6., 8., 10., 14.)


def main():
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = make_cfg(len(RATES) * args.replicas, 'discovery', 1, split=1,
                   start_mode='tilt5')
    cfg.log_dir = str(output)
    env = ManagerBasedRLEnv(cfg)
    try:
        term = env._recovery_reward
        action_term = env.action_manager.get_term('joint_pos')
        per_env_rate = torch.tensor(RATES, device=env.device).repeat_interleave(args.replicas)
        action_term._max_target_step = (per_env_rate * env.step_dt)[:, None]
        env.reset()
        actions = torch.zeros((env.num_envs, 22), device=env.device)
        term.capture = True
        records = {}
        steps = round(args.seconds / env.step_dt)
        for step in range(steps):
            env.step(actions)
            for item in fields(term.last_frame):
                value = getattr(term.last_frame, item.name).detach().cpu().numpy().copy()
                records.setdefault(item.name, []).append(value)
            records.setdefault('joint_pos', []).append(
                env.scene['robot'].data.joint_pos.detach().cpu().numpy().copy())
            records.setdefault('joint_vel', []).append(
                env.scene['robot'].data.joint_vel.detach().cpu().numpy().copy())
            records.setdefault('torque_estimate', []).append(
                env.scene['robot'].data.applied_torque.detach().cpu().numpy().copy())
            records.setdefault('penetration', []).append(
                term.captured['penetration'].copy())
            records.setdefault('terminated', []).append(
                env.reset_terminated.detach().cpu().numpy().copy())
            if (step + 1) % 250 == 0:
                heights = term.last_frame.height.reshape(len(RATES), args.replicas).mean(1)
                print('[SLEW] step', step + 1, dict(zip(RATES, heights.tolist())), flush=True)
        trace = {name: np.stack(values) for name, values in records.items()}
        reports = {}
        for index, rate in enumerate(RATES):
            ids = slice(index * args.replicas, (index + 1) * args.replicas)
            reports[str(rate)] = audit({name: value[:, ids] for name, value in trace.items()}, env.step_dt)
        (output / 'slew_probe.json').write_text(json.dumps(reports, indent=2))
        print('[SLEW] COMPLETE', json.dumps(reports), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    status = 0
    try:
        main()
    except BaseException:
        import traceback
        status = 1
        traceback.print_exc()
    finally:
        app.app.close()
    raise SystemExit(status)
