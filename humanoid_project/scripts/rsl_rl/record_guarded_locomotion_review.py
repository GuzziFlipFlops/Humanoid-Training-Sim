"""Record a guarded locomotion checkpoint for an exact number of simulation seconds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT_ROOT / "experiments" / "guarded_run_v8_posture6"
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(PROJECT_ROOT / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--speed", type=float, required=True)
parser.add_argument("--duration", type=float, default=15.0)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--seed", type=int, default=47)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)

import gymnasium as gym
import cv2
import torch
from packaging import version as packaging_version
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from humanoid_tasks.curriculum.agents.rsl_rl_cfg import HumanoidRunFastPPORunnerCfg
from candidate import make_cfg
import humanoid_tasks  # noqa: F401


def slug(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def main() -> None:
    checkpoint = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed capture: {output}")
    if args.duration <= 0.0 or args.speed <= 0.0:
        raise ValueError("Duration and speed must be positive")
    output.mkdir(parents=True, exist_ok=True)
    video_dir = output / "video"
    video_dir.mkdir()

    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    info = stored.get("infos") or {}
    scale = float(info.get("action_scale", 0.35))
    velocity_boost = float(info.get("velocity_boost", 1.0))
    boost_classes = tuple(info.get("boost_classes") or ())
    candidate_version = str(info.get("candidate_version", "unknown"))
    del stored

    cfg = make_cfg("speed", max(1.8, args.speed), scale, 1, velocity_boost, boost_classes)
    cfg.sim.device = args.device or "cuda:0"
    cfg.sim.render_interval = cfg.decimation
    cfg.seed = args.seed
    cfg.episode_length_s = max(60.0, args.duration + 5.0)
    cfg.events.push = None
    cfg.events.torso_mass = None
    cfg.events.torso_com = None
    cfg.events.actuator_gains = None
    cfg.events.physics_material.params.update(
        static_friction_range=(0.95, 0.95), dynamic_friction_range=(0.8, 0.8),
        restitution_range=(0.0, 0.0), num_buckets=1,
    )
    for field in ("pose_range", "velocity_range"):
        cfg.events.reset_base.params[field] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
    cfg.events.reset_joints.params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    cfg.observations.policy.enable_corruption = False
    command = cfg.commands.base_velocity
    command.class_type = UniformVelocityCommand
    command.ranges.lin_vel_x = (args.speed, args.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (1000.0, 1000.0)

    installed = metadata.version("rsl-rl-lib")
    agent = handle_deprecated_rsl_rl_cfg(HumanoidRunFastPPORunnerCfg(), installed)
    agent.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    base = gym.make("Isaac-Humanoid5ft-NaturalRun-v0", cfg=cfg, render_mode="rgb_array")
    dt = float(base.unwrapped.step_dt)
    frames = int(round(args.duration / dt))
    if abs(frames * dt - args.duration) > 1.0e-9:
        raise ValueError("Duration must be an exact multiple of the control step")
    video_path = video_dir / f"{slug(args.label)}.mp4"
    video_writer = None
    env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
    try:
        raw = env.unwrapped
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=raw.device)
        runner.load(
            handle_deprecated_rsl_rl_checkpoint(str(checkpoint), installed),
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False},
        )
        policy = runner.get_inference_policy(raw.device)
        robot = raw.scene["robot"]
        joints = list(robot.joint_names)
        joint_slugs = [slug(name) for name in joints]
        effort_limits = robot.data.joint_effort_limits[0].detach().cpu().tolist()
        velocity_limits = robot.data.soft_joint_vel_limits[0].detach().cpu().tolist()
        fields = [
            "time_s", "episode_time_s", "base_x_m", "base_y_m", "base_height_m",
            "base_vx_m_s", "base_vy_m_s", "base_vz_m_s", "base_wx_rad_s",
            "base_wy_rad_s", "base_wz_rad_s", "command_vx_m_s", "terminated",
        ]
        for name in joint_slugs:
            fields += [
                f"{name}_position_rad", f"{name}_velocity_rad_s",
                f"{name}_applied_torque_nm", f"{name}_computed_torque_nm",
                f"{name}_mechanical_power_w",
            ]
        obs = env.get_observations()
        csv_path = output / "telemetry.csv"
        falls = 0
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(frames):
                before = robot.data.root_pos_w[0].detach().cpu()
                raw.sim.set_camera_view(
                    [float(before[0]) + 3.0, float(before[1]) + 2.4, 1.55],
                    [float(before[0]), float(before[1]), 0.78],
                )
                with torch.inference_mode():
                    obs, _, done, _ = env.step(policy(obs))
                    if packaging_version.parse(installed) >= packaging_version.parse("4.0.0"):
                        policy.reset(done)
                falls += int(raw.reset_terminated[0].item())
                frame = raw.render()
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
                actual_command = float(raw.command_manager.get_command("base_velocity")[0, 0])
                row = {
                    "time_s": (step + 1) * dt,
                    "episode_time_s": float(raw.episode_length_buf[0]) * dt,
                    "base_x_m": float(root_pos[0]), "base_y_m": float(root_pos[1]),
                    "base_height_m": float(root_pos[2]), "base_vx_m_s": float(root_vel[0]),
                    "base_vy_m_s": float(root_vel[1]), "base_vz_m_s": float(root_vel[2]),
                    "base_wx_rad_s": float(root_ang[0]), "base_wy_rad_s": float(root_ang[1]),
                    "base_wz_rad_s": float(root_ang[2]), "command_vx_m_s": actual_command,
                    "terminated": int(raw.reset_terminated[0].item()),
                }
                for index, name in enumerate(joint_slugs):
                    tau, omega = float(applied[index]), float(vel[index])
                    row.update({
                        f"{name}_position_rad": float(pos[index]),
                        f"{name}_velocity_rad_s": omega,
                        f"{name}_applied_torque_nm": tau,
                        f"{name}_computed_torque_nm": float(computed[index]),
                        f"{name}_mechanical_power_w": tau * omega,
                    })
                writer.writerow(row)
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()

    if not video_path.is_file():
        raise RuntimeError("No MP4 was produced")
    manifest = {
        "label": args.label, "skill": slug(args.label), "speed_m_s": args.speed,
        "duration_s": frames * dt, "control_dt_s": dt, "control_steps": frames,
        "video_fps": 1.0 / dt, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "candidate_version": candidate_version, "action_scale": scale,
        "velocity_boost": velocity_boost, "boost_classes": list(boost_classes),
        "falls": falls, "video": str(video_path.relative_to(output)).replace("\\", "/"),
        "csv": csv_path.name,
        "joints": [
            {"name": name, "slug": name_slug, "effort_limit_nm": float(effort),
             "velocity_limit_rad_s": float(speed_limit)}
            for name, name_slug, effort, speed_limit in
            zip(joints, joint_slugs, effort_limits, velocity_limits)
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        launcher.app.close()
