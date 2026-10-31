"""Replay retargeted recovery references through the exact Isaac physics asset.

This is a feasibility gate, not training.  A long PPO job is forbidden until at
least one unassisted variant reaches the same strict seven-second upright hold
used by the independent policy evaluator.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--group", type=int, choices=range(4), default=0)
parser.add_argument("--replicas", type=int, default=4)
parser.add_argument("--variant", type=int, default=-1)
parser.add_argument("--allow-unsafe", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from candidate import make_cfg
from reference_motion import retarget, time_warp


def main() -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Small grid: the variants alter only retargeting/time, never physics or
    # actuator ceilings.  Replicas expose sensitivity to the held-out pose-bank
    # jitter instead of accepting one lucky reset.
    variants = list(itertools.product(
        (0.55, 0.72, 0.88),       # roll scale
        (0.55, 0.72),             # yaw scale
        (0.88, 1.00),             # pitch scale
        (1.0, 1.35, 1.70),        # time warp
        (False, True),            # lateral mirror
    ))
    if args.variant >= 0:
        if args.variant >= len(variants):
            raise ValueError(f"variant must be below {len(variants)}")
        variants = [variants[args.variant]]
    descriptors = []
    for variant_id, variant in enumerate(variants):
        for replica in range(args.replicas):
            descriptors.append((variant_id, replica, variant))
    cfg = make_cfg(len(descriptors), stage="discovery", groups=4, split=1,
                   assist=0.0, forced_group=args.group, self_collisions=True)
    if args.allow_unsafe:
        cfg.terminations.unsafe = None
        cfg.terminations.terminal_fall = None
    cfg.seed = 4700 + args.group
    cfg.log_dir = str(output)
    env = ManagerBasedRLEnv(cfg)
    started = time.monotonic()
    try:
        env.reset()
        robot = env.scene["robot"]
        action_term = next(iter(env.action_manager._terms.values()))
        default = robot.data.default_joint_pos[0].detach().cpu()
        lower = robot.data.soft_joint_pos_limits[0, :, 0].detach().cpu()
        upper = robot.data.soft_joint_pos_limits[0, :, 1].detach().cpu()
        traces = []
        for _, _, (roll, yaw, pitch, warp, mirror) in descriptors:
            trace = retarget(robot.joint_names, default, lower, upper,
                             roll_scale=roll, yaw_scale=yaw,
                             pitch_scale=pitch, mirror=mirror)
            traces.append(time_warp(trace, warp).to(env.device))

        hold_steps = round(8.0 / env.step_dt)
        total_steps = max(len(trace) for trace in traces) + hold_steps
        ever_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        ever_handoff = torch.zeros_like(ever_done)
        handoff_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        safe_hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        max_safe_hold = torch.zeros_like(safe_hold)
        max_height = torch.zeros(env.num_envs, device=env.device)
        first_done = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        trace_height, trace_pelvis_tilt, trace_torso_tilt = [], [], []
        trace_penetration, trace_floor = [], []

        for step in range(total_steps):
            target = torch.stack([trace[min(step, len(trace) - 1)] for trace in traces])
            actions = action_term.normalized_for_target(target)
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            # Ignore the compulsory first post-reset motor-hold tick; any later
            # reset invalidates that rollout even if it subsequently restarts.
            if step > 0:
                first = done & (first_done < 0)
                first_done[first] = step
                ever_done |= done
            term = env._recovery_reward
            newly = term.handoff & ~ever_handoff
            handoff_step[newly] = step
            ever_handoff |= term.handoff
            frame = term.last_frame
            strict = (frame.safe & frame.contacts_fresh
                      & (frame.height >= .78) & (frame.height <= .94)
                      & (frame.pelvis_tilt < .20) & (frame.torso_tilt < .26)
                      & (frame.foot_load >= .12).all(1)
                      & (frame.foot_load.sum(1) >= .80)
                      & (frame.planar_speed < .15)
                      & (frame.vertical_speed.abs() < .10)
                      & (frame.angular_speed < .65)
                      & (frame.joint_speed_rms < 1.35)
                      & (frame.nonfoot_force < 8.0))
            safe_hold = torch.where(strict & ~done, safe_hold + 1, 0)
            max_safe_hold = torch.maximum(max_safe_hold, safe_hold)
            max_height = torch.maximum(max_height, frame.height)
            trace_height.append(frame.height.detach().cpu().numpy())
            trace_pelvis_tilt.append(frame.pelvis_tilt.detach().cpu().numpy())
            trace_torso_tilt.append(frame.torso_tilt.detach().cpu().numpy())
            trace_penetration.append(env._recovery_telemetry.last_penetration.detach().cpu().numpy())
            trace_floor.append(env._recovery_telemetry.last_floor.detach().cpu().numpy())
            if (step + 1) % 200 == 0:
                print(f"[REFERENCE] {step + 1}/{total_steps}; handoff={int(ever_handoff.sum())}; "
                      f"strict_7s={int((max_safe_hold * env.step_dt >= 7.).sum())}; "
                      f"max-height={float(max_height.max()):.3f}", flush=True)

        rows = []
        for index, (variant_id, replica, variant) in enumerate(descriptors):
            roll, yaw, pitch, warp, mirror = variant
            rows.append(dict(
                env=index, variant=variant_id, replica=replica, group=args.group,
                roll_scale=roll, yaw_scale=yaw, pitch_scale=pitch,
                time_warp=warp, mirror=mirror,
                handoff=bool(ever_handoff[index]),
                handoff_s=(float(handoff_step[index]) * env.step_dt if handoff_step[index] >= 0 else None),
                strict_hold_s=float(max_safe_hold[index]) * env.step_dt,
                max_height_m=float(max_height[index]),
                first_done_s=(float(first_done[index]) * env.step_dt if first_done[index] >= 0 else None),
                reset_or_unsafe=bool(ever_done[index]),
            ))
        # A reference is accepted only if every held-out replica of the same
        # variant completes an unassisted seven-second quiet hold without reset.
        accepted = []
        for variant_id in range(len(variants)):
            subset = [row for row in rows if row["variant"] == variant_id]
            if all(row["strict_hold_s"] >= 7.0 and not row["reset_or_unsafe"] for row in subset):
                accepted.append(dict(variant=variant_id, parameters=subset[0], replicas=len(subset)))
        report = dict(
            completed=True, accepted=accepted, accepted_count=len(accepted),
            tested_variants=len(variants), replicas=args.replicas, group=args.group,
            num_envs=env.num_envs, step_dt=env.step_dt,
            elapsed_s=time.monotonic() - started, rows=rows,
        )
        (output / "reference_probe.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(output / "reference_probe.npz",
                            max_height=max_height.cpu().numpy(),
                            max_safe_hold=max_safe_hold.cpu().numpy(),
                            handoff_step=handoff_step.cpu().numpy(),
                            height=np.stack(trace_height), pelvis_tilt=np.stack(trace_pelvis_tilt),
                            torso_tilt=np.stack(trace_torso_tilt),
                            penetration=np.stack(trace_penetration), floor=np.stack(trace_floor))
        print("[REFERENCE] COMPLETE " + json.dumps({k: report[k] for k in report if k != "rows"}), flush=True)
        if not accepted:
            raise RuntimeError("No retargeted reference passed the unassisted strict-hold gate")
    finally:
        env.close()


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        status = 1
        traceback.print_exc()
    finally:
        app.app.close()
    raise SystemExit(status)
