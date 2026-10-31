"""Deterministic behavioral gate for staged humanoid checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "source"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate a humanoid checkpoint against a behavioral promotion gate.")
parser.add_argument("--task", required=True)
parser.add_argument("--profile", choices=("stand", "walk", "run", "getup"), required=True)
parser.add_argument("--target_speed", type=float, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--eval_steps", type=int, default=500)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=47)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils import math as math_utils

import humanoid_tasks  # noqa: F401,E402


THRESHOLDS = {
    "stand": {
        "fall_fraction_max": 0.05,
        "upright_fraction_min": 0.95,
        "safe_height_fraction_min": 0.95,
        "speed_mae_max": 0.12,
        "lateral_speed_mean_max": 0.08,
        "contact_slip_mean_max": 0.10,
        "foot_crossing_fraction_max": 0.005,
        "knee_crossing_fraction_max": 0.005,
        "contact_foot_alignment_mean_max": 0.10,
    },
    "walk": {
        "fall_fraction_max": 0.12,
        "upright_fraction_min": 0.90,
        "safe_height_fraction_min": 0.90,
        "speed_mae_max": 0.30,
        "lateral_speed_mean_max": 0.16,
        "contact_slip_mean_max": 0.30,
        "foot_crossing_fraction_max": 0.02,
        "knee_crossing_fraction_max": 0.02,
        "contact_foot_alignment_mean_max": 0.18,
    },
    "run": {
        "fall_fraction_max": 0.20,
        "upright_fraction_min": 0.82,
        "safe_height_fraction_min": 0.82,
        "speed_mae_max": (
            0.40 if args_cli.target_speed <= 1.5 else
            (0.48 if args_cli.target_speed <= 2.0 else (0.58 if args_cli.target_speed <= 2.5 else 0.75))
        ),
        "lateral_speed_mean_max": 0.28,
        "contact_slip_mean_max": 0.50,
        "foot_crossing_fraction_max": 0.04,
        "knee_crossing_fraction_max": 0.04,
        "contact_foot_alignment_mean_max": 0.28,
    },
    # Discovery is allowed to promote once it supplies enough successful
    # trajectories for refinement. The refinement policy must pass the final
    # 85% randomized-pose requirement. Both tasks use the same strict dwell
    # termination, so a momentary upright pose cannot count as success.
    "getup": {
        "success_fraction_min": 0.65 if "GetUpDiscovery" in args_cli.task else 0.85,
    },
}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = True
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.rel_standing_envs = 1.0 if args_cli.profile == "stand" else 0.0
    command_cfg.ranges.lin_vel_x = (args_cli.target_speed, args_cli.target_speed)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.ranges.heading = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = handle_deprecated_rsl_rl_checkpoint(str(Path(args_cli.checkpoint).resolve()), installed_version)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    robot = raw.scene["robot"]
    sensor = raw.scene.sensors["contact_forces"]
    foot_body_ids = robot.find_bodies(["left_foot", "right_foot"])[0]
    shank_body_ids = robot.find_bodies(["left_shank", "right_shank"])[0]
    sensor_foot_ids = sensor.find_bodies(["left_foot", "right_foot"])[0]
    ever_fallen = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    ever_succeeded = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    sums = {
        "upright": 0.0,
        "height": 0.0,
        "speed_error": 0.0,
        "lateral": 0.0,
        "slip": 0.0,
        "foot_crossing": 0.0,
        "knee_crossing": 0.0,
        "foot_alignment": 0.0,
        "single_support": 0.0,
        "double_support": 0.0,
        "flight": 0.0,
        "swing_clearance": 0.0,
        "stride_separation": 0.0,
        "touchdown_air_time": 0.0,
    }
    sample_count = 0
    contact_count = 0.0
    swing_sample_count = 0.0
    single_support_count = 0.0
    touchdown_count = 0.0
    all_touchdown_count = 0.0
    premature_touchdown_count = 0.0
    obs = env.get_observations()

    with torch.inference_mode():
        for _ in range(args_cli.eval_steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            non_timeout_termination = dones.bool() & ~timeouts.bool()
            if args_cli.profile == "getup":
                ever_succeeded |= non_timeout_termination
            else:
                ever_fallen |= non_timeout_termination

            gravity_z = robot.data.projected_gravity_b[:, 2]
            height = robot.data.root_pos_w[:, 2]
            velocity = robot.data.root_lin_vel_b
            safe_height = 0.66 if args_cli.profile == "stand" else (0.60 if args_cli.profile == "walk" else 0.56)
            sums["upright"] += float((gravity_z < -0.82).sum())
            sums["height"] += float((height > safe_height).sum())
            sums["speed_error"] += float(torch.abs(velocity[:, 0] - args_cli.target_speed).sum())
            sums["lateral"] += float(torch.abs(velocity[:, 1]).sum())

            contacts = sensor.data.net_forces_w_history[:, :, sensor_foot_ids, :].norm(dim=-1).amax(dim=1) > 5.0
            support_count = contacts.sum(dim=1)
            single_support = support_count == 1
            sums["single_support"] += float(single_support.sum())
            sums["double_support"] += float((support_count == 2).sum())
            sums["flight"] += float((support_count == 0).sum())
            # Raw force-threshold transitions can count contact chatter as
            # multiple steps. ContactSensor supplies the completed swing time,
            # letting us separate real touchdown events from very short bounces.
            first_contact = sensor.compute_first_contact(raw.step_dt)[:, sensor_foot_ids]
            last_air_time = sensor.data.last_air_time[:, sensor_foot_ids]
            valid_touchdown = first_contact & (last_air_time >= 0.08)
            premature_touchdown = first_contact & (last_air_time < 0.12)
            touchdown_count += float(valid_touchdown.sum())
            all_touchdown_count += float(first_contact.sum())
            premature_touchdown_count += float(premature_touchdown.sum())
            sums["touchdown_air_time"] += float((last_air_time * first_contact).sum())
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, foot_body_ids, :2], dim=-1)
            sums["slip"] += float((foot_speed * contacts).sum())
            heading_quat_w = math_utils.yaw_quat(robot.data.root_quat_w)
            foot_pos_w = robot.data.body_pos_w[:, foot_body_ids, :]
            shank_pos_w = robot.data.body_pos_w[:, shank_body_ids, :]
            foot_relative_w = foot_pos_w - robot.data.root_pos_w[:, None, :]
            shank_relative_w = shank_pos_w - robot.data.root_pos_w[:, None, :]
            foot_heading_quat_w = heading_quat_w[:, None, :].expand(-1, foot_pos_w.shape[1], -1)
            shank_heading_quat_w = heading_quat_w[:, None, :].expand(-1, shank_pos_w.shape[1], -1)
            foot_pos_h = math_utils.quat_apply_inverse(
                foot_heading_quat_w.reshape(-1, 4), foot_relative_w.reshape(-1, 3)
            ).reshape_as(foot_relative_w)
            shank_pos_h = math_utils.quat_apply_inverse(
                shank_heading_quat_w.reshape(-1, 4), shank_relative_w.reshape(-1, 3)
            ).reshape_as(shank_relative_w)
            foot_lateral = foot_pos_h[..., 1]
            shank_lateral = shank_pos_h[..., 1]
            sums["foot_crossing"] += float((torch.abs(foot_lateral[:, 0] - foot_lateral[:, 1]) < 0.10).sum())
            sums["knee_crossing"] += float((torch.abs(shank_lateral[:, 0] - shank_lateral[:, 1]) < 0.09).sum())
            stride_separation = torch.abs(foot_pos_h[:, 0, 0] - foot_pos_h[:, 1, 0])
            sums["stride_separation"] += float((stride_separation * single_support).sum())
            single_support_count += float(single_support.sum())
            swing = ~contacts
            foot_height = foot_pos_w[..., 2] - raw.scene.env_origins[:, 2].unsqueeze(1)
            sums["swing_clearance"] += float((foot_height * swing).sum())
            swing_sample_count += float(swing.sum())

            foot_quat = robot.data.body_quat_w[:, foot_body_ids, :]
            local_up = torch.zeros((raw.num_envs, 2, 3), device=raw.device)
            local_up[..., 2] = 1.0
            local_forward = torch.zeros_like(local_up)
            local_forward[..., 0] = 1.0
            world_up = math_utils.quat_apply(foot_quat.reshape(-1, 4), local_up.reshape(-1, 3)).reshape_as(local_up)
            world_forward = math_utils.quat_apply(
                foot_quat.reshape(-1, 4), local_forward.reshape(-1, 3)
            ).reshape_as(local_forward)
            foot_heading_quat_w = heading_quat_w[:, None, :].expand(-1, foot_quat.shape[1], -1)
            foot_forward_h = math_utils.quat_apply_inverse(
                foot_heading_quat_w.reshape(-1, 4), world_forward.reshape(-1, 3)
            ).reshape_as(world_forward)
            alignment_error = torch.square(1.0 - world_up[..., 2]) + torch.square(foot_forward_h[..., 1])
            sums["foot_alignment"] += float((alignment_error * contacts).sum())
            contact_count += float(contacts.sum())
            sample_count += raw.num_envs

    metrics = {
        "profile": args_cli.profile,
        "target_speed_mps": args_cli.target_speed,
        "num_envs": raw.num_envs,
        "eval_steps": args_cli.eval_steps,
        "fall_fraction": float(ever_fallen.float().mean()),
        "upright_fraction": sums["upright"] / sample_count,
        "safe_height_fraction": sums["height"] / sample_count,
        "speed_mae": sums["speed_error"] / sample_count,
        "lateral_speed_mean": sums["lateral"] / sample_count,
        "contact_slip_mean": sums["slip"] / max(contact_count, 1.0),
        "foot_crossing_fraction": sums["foot_crossing"] / sample_count,
        "knee_crossing_fraction": sums["knee_crossing"] / sample_count,
        "contact_foot_alignment_mean": sums["foot_alignment"] / max(contact_count, 1.0),
        "single_support_fraction": sums["single_support"] / sample_count,
        "double_support_fraction": sums["double_support"] / sample_count,
        "flight_fraction": sums["flight"] / sample_count,
        "swing_clearance_mean": sums["swing_clearance"] / max(swing_sample_count, 1.0),
        "single_support_stride_mean": sums["stride_separation"] / max(single_support_count, 1.0),
        "step_rate_hz": touchdown_count / max(raw.num_envs * args_cli.eval_steps * raw.step_dt, 1.0e-6),
        "touchdown_air_time_mean": sums["touchdown_air_time"] / max(all_touchdown_count, 1.0),
        "premature_touchdown_fraction": premature_touchdown_count / max(all_touchdown_count, 1.0),
    }
    thresholds = THRESHOLDS[args_cli.profile]
    if args_cli.profile == "getup":
        metrics["success_fraction"] = float(ever_succeeded.float().mean())
        checks = {"success_fraction": metrics["success_fraction"] >= thresholds["success_fraction_min"]}
    else:
        checks = {
            "fall_fraction": metrics["fall_fraction"] <= thresholds["fall_fraction_max"],
            "upright_fraction": metrics["upright_fraction"] >= thresholds["upright_fraction_min"],
            "safe_height_fraction": metrics["safe_height_fraction"] >= thresholds["safe_height_fraction_min"],
            "speed_mae": metrics["speed_mae"] <= thresholds["speed_mae_max"],
            "lateral_speed_mean": metrics["lateral_speed_mean"] <= thresholds["lateral_speed_mean_max"],
            "contact_slip_mean": metrics["contact_slip_mean"] <= thresholds["contact_slip_mean_max"],
            "foot_crossing_fraction": metrics["foot_crossing_fraction"] <= thresholds["foot_crossing_fraction_max"],
            "knee_crossing_fraction": metrics["knee_crossing_fraction"] <= thresholds["knee_crossing_fraction_max"],
            "contact_foot_alignment_mean": metrics["contact_foot_alignment_mean"] <= thresholds["contact_foot_alignment_mean_max"],
        }
        # The new gait tasks must pass contact-pattern and geometry checks in
        # addition to speed. These prevent promotion of tiny-step, hopping, or
        # permanently double-supported solutions that can still track velocity.
        if "GaitWalk" in args_cli.task and args_cli.profile == "walk":
            gait_thresholds = {
                "contact_slip_mean_max": 0.18,
                "single_support_fraction_min": 0.22,
                "double_support_fraction_min": 0.025,
                "double_support_fraction_max": 0.35,
                "flight_fraction_max": 0.04,
                "swing_clearance_mean_min": 0.05,
                "single_support_stride_mean_min": 0.30,
                "step_rate_hz_min": 1.20,
                "step_rate_hz_max": 3.80,
                "touchdown_air_time_mean_min": 0.24,
                "touchdown_air_time_mean_max": 0.55,
                "premature_touchdown_fraction_max": 0.15,
            }
            thresholds.update(gait_thresholds)
            checks.update({
                "contact_slip_mean": metrics["contact_slip_mean"] <= gait_thresholds["contact_slip_mean_max"],
                "single_support_fraction": metrics["single_support_fraction"] >= gait_thresholds["single_support_fraction_min"],
                "double_support_fraction_min": metrics["double_support_fraction"] >= gait_thresholds["double_support_fraction_min"],
                "double_support_fraction_max": metrics["double_support_fraction"] <= gait_thresholds["double_support_fraction_max"],
                "flight_fraction": metrics["flight_fraction"] <= gait_thresholds["flight_fraction_max"],
                "swing_clearance_mean": metrics["swing_clearance_mean"] >= gait_thresholds["swing_clearance_mean_min"],
                "single_support_stride_mean": metrics["single_support_stride_mean"] >= gait_thresholds["single_support_stride_mean_min"],
                "step_rate_hz_min": metrics["step_rate_hz"] >= gait_thresholds["step_rate_hz_min"],
                "step_rate_hz_max": metrics["step_rate_hz"] <= gait_thresholds["step_rate_hz_max"],
                "touchdown_air_time_mean_min": metrics["touchdown_air_time_mean"] >= gait_thresholds["touchdown_air_time_mean_min"],
                "touchdown_air_time_mean_max": metrics["touchdown_air_time_mean"] <= gait_thresholds["touchdown_air_time_mean_max"],
                "premature_touchdown_fraction": metrics["premature_touchdown_fraction"] <= gait_thresholds["premature_touchdown_fraction_max"],
            })
        if "GaitSprint4" in args_cli.task and args_cli.profile == "run" and args_cli.target_speed >= 3.5:
            gait_thresholds = {
                "single_support_fraction_min": 0.18,
                "flight_fraction_min": 0.025,
                "double_support_fraction_max": 0.65,
                "swing_clearance_mean_min": 0.065,
                "single_support_stride_mean_min": 0.16,
                "step_rate_hz_min": 0.80,
                "step_rate_hz_max": 5.00,
            }
            thresholds.update(gait_thresholds)
            checks.update({
                "single_support_fraction": metrics["single_support_fraction"] >= gait_thresholds["single_support_fraction_min"],
                "flight_fraction": metrics["flight_fraction"] >= gait_thresholds["flight_fraction_min"],
                "double_support_fraction": metrics["double_support_fraction"] <= gait_thresholds["double_support_fraction_max"],
                "swing_clearance_mean": metrics["swing_clearance_mean"] >= gait_thresholds["swing_clearance_mean_min"],
                "single_support_stride_mean": metrics["single_support_stride_mean"] >= gait_thresholds["single_support_stride_mean_min"],
                "step_rate_hz_min": metrics["step_rate_hz"] >= gait_thresholds["step_rate_hz_min"],
                "step_rate_hz_max": metrics["step_rate_hz"] <= gait_thresholds["step_rate_hz_max"],
            })
    result = {"passed": all(checks.values()), "metrics": metrics, "thresholds": thresholds, "checks": checks}
    Path(args_cli.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PROMOTION_GATE=" + json.dumps(result, separators=(",", ":")))
    env.close()
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    # Hydra's decorated entry point does not reliably propagate the wrapped
    # function's return value.  Read the durable result so failed behavioral
    # gates produce a non-zero process exit code for chain orchestration.
    result_path = Path(args_cli.output)
    if not result_path.is_file():
        raise SystemExit(3)
    persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
    raise SystemExit(0 if persisted_result.get("passed", False) else 2)
