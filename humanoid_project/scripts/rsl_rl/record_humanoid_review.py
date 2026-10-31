"""Record one trained humanoid policy with synchronized actuator telemetry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--skill", required=True, choices=("stand", "walk", "sprint", "getup"))
parser.add_argument("--duration", required=True, type=float)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--target_speed", type=float, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import cv2
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401,E402
import humanoid_tasks  # noqa: F401,E402


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _prepare_cfg(env_cfg: ManagerBasedRLEnvCfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None
    if hasattr(env_cfg.commands, "base_velocity"):
        command = env_cfg.commands.base_velocity
        speed = args_cli.target_speed if args_cli.target_speed is not None else 0.0
        command.ranges.lin_vel_x = (speed, speed)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.rel_standing_envs = 1.0 if abs(speed) < 1.0e-6 else 0.0
        command.debug_vis = False
    # Keep the recovery task's randomized lying pose. Make locomotion captures repeatable.
    if args_cli.skill != "getup":
        if getattr(env_cfg.events, "reset_base", None) is not None:
            for values in env_cfg.events.reset_base.params.get("pose_range", {}).values():
                values = values
            env_cfg.events.reset_base.params["pose_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            }
            env_cfg.events.reset_base.params["velocity_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            }
        if getattr(env_cfg.events, "reset_joints", None) is not None:
            env_cfg.events.reset_joints.params["position_range"] = (1.0, 1.0)
            env_cfg.events.reset_joints.params["velocity_range"] = (0.0, 0.0)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    root = Path(args_cli.output_dir).resolve()
    video_dir = root / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args_cli.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    _prepare_cfg(env_cfg)
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg.seed = args_cli.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    dt = float(base_env.unwrapped.step_dt)
    frames = int(round(args_cli.duration / dt))
    video_path = video_dir / f"{args_cli.skill}.mp4"
    video_writer = None
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(handle_deprecated_rsl_rl_checkpoint(str(checkpoint), installed_version))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    joints = list(robot.joint_names)
    slugs = [_slug(name) for name in joints]
    limits = robot.data.joint_effort_limits[0].detach().cpu().tolist()
    velocity_limits = robot.data.soft_joint_vel_limits[0].detach().cpu().tolist()

    fields = ["time_s", "episode_time_s", "base_x_m", "base_y_m", "base_height_m",
              "base_vx_m_s", "base_vy_m_s", "base_vz_m_s", "base_wx_rad_s",
              "base_wy_rad_s", "base_wz_rad_s", "command_vx_m_s"]
    for slug in slugs:
        fields += [f"{slug}_position_rad", f"{slug}_velocity_rad_s",
                   f"{slug}_applied_torque_nm", f"{slug}_computed_torque_nm",
                   f"{slug}_mechanical_power_w"]

    obs = env.get_observations()
    csv_path = root / "telemetry.csv"
    falls = 0
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(frames):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
            falls += int(env.unwrapped.reset_terminated[0].item())
            frame = env.unwrapped.render()
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / dt,
                    (int(frame.shape[1]), int(frame.shape[0])),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Unable to create {video_path}")
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            data = robot.data
            pos = data.joint_pos[0].detach().cpu()
            vel = data.joint_vel[0].detach().cpu()
            applied = data.applied_torque[0].detach().cpu()
            computed = data.computed_torque[0].detach().cpu()
            root_pos = data.root_pos_w[0].detach().cpu()
            root_vel = data.root_lin_vel_b[0].detach().cpu()
            root_ang = data.root_ang_vel_b[0].detach().cpu()
            command_vx = 0.0
            if hasattr(env.unwrapped, "command_manager"):
                try:
                    command_vx = float(env.unwrapped.command_manager.get_command("base_velocity")[0, 0])
                except Exception:
                    pass
            row = {
                "time_s": (step + 1) * dt,
                "episode_time_s": float(env.unwrapped.episode_length_buf[0]) * dt,
                "base_x_m": float(root_pos[0]), "base_y_m": float(root_pos[1]),
                "base_height_m": float(root_pos[2]), "base_vx_m_s": float(root_vel[0]),
                "base_vy_m_s": float(root_vel[1]), "base_vz_m_s": float(root_vel[2]),
                "base_wx_rad_s": float(root_ang[0]), "base_wy_rad_s": float(root_ang[1]),
                "base_wz_rad_s": float(root_ang[2]), "command_vx_m_s": command_vx,
            }
            for i, slug in enumerate(slugs):
                tau, omega = float(applied[i]), float(vel[i])
                row.update({f"{slug}_position_rad": float(pos[i]),
                            f"{slug}_velocity_rad_s": omega,
                            f"{slug}_applied_torque_nm": tau,
                            f"{slug}_computed_torque_nm": float(computed[i]),
                            f"{slug}_mechanical_power_w": tau * omega})
            writer.writerow(row)
            env.unwrapped.sim.set_camera_view(
                [float(root_pos[0]) + 3.0, float(root_pos[1]) + 2.4, 1.55],
                [float(root_pos[0]), float(root_pos[1]), 0.78],
            )

    if video_writer is not None:
        video_writer.release()
    env.close()
    if not video_path.is_file():
        raise RuntimeError(f"No recording created at {video_path}")
    manifest = {
        "label": "Walking at 1 m/s" if args_cli.skill == "walk" else args_cli.skill,
        "skill": args_cli.skill, "task": args_cli.task, "duration_s": frames * dt,
        "dt_s": dt, "control_dt_s": dt, "control_steps": frames,
        "video_fps": 1.0 / dt, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "action_scale": float(env_cfg.actions.joint_pos.scale), "falls": falls,
        "video": str(video_path.relative_to(root)).replace("\\", "/"), "csv": csv_path.name,
        "joints": [{"name": n, "slug": s, "effort_limit_nm": float(l),
                    "velocity_limit_rad_s": float(v)}
                   for n, s, l, v in zip(joints, slugs, limits, velocity_limits)],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[REVIEW] {args_cli.skill}: {video_path}")
    print(f"[REVIEW] telemetry: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
