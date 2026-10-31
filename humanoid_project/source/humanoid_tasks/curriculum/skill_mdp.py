"""Humanoid-specific observations, resets, rewards, and hard success gates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1).amax(dim=1) > threshold


def feet_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    return _contacts(env, sensor_cfg, threshold).to(torch.float32)


def both_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    return _contacts(env, sensor_cfg, threshold).all(dim=1).to(torch.float32)


def no_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    return (~_contacts(env, sensor_cfg, threshold).any(dim=1)).to(torch.float32)


def upright_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = torch.square(asset.data.projected_gravity_b[:, 0]) + torch.square(
        asset.data.projected_gravity_b[:, 1]
    )
    return torch.exp(-error / std**2)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.exp(-torch.square(asset.data.root_pos_w[:, 2] - target_height) / std**2)


def planar_still_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.exp(-torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) / std**2)


def joint_symmetry_l1(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset = env.scene[left_cfg.name]
    left = asset.data.joint_pos[:, left_cfg.joint_ids]
    right = asset.data.joint_pos[:, right_cfg.joint_ids]
    return torch.mean(torch.abs(left - right), dim=1)


def gated_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track planar velocity only while the robot is upright and above a safe height.

    This prevents the common reward exploit where a falling or sliding robot receives
    forward-velocity reward even though it is not performing valid locomotion.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    tracking = torch.exp(-error / std**2)
    upright_gate = torch.clamp((-asset.data.projected_gravity_b[:, 2] - 0.65) / 0.30, 0.0, 1.0)
    height_gate = torch.clamp((asset.data.root_pos_w[:, 2] - minimum_height) / 0.18, 0.0, 1.0)
    return tracking * upright_gate * height_gate


def lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)


def actuator_saturation_l2(
    env: ManagerBasedRLEnv,
    velocity_margin: float = 0.85,
    torque_margin: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize operation near peak joint limits before hard clipping occurs."""
    asset = env.scene[asset_cfg.name]
    velocity_limits = asset.data.soft_joint_vel_limits
    effort_limits = asset.data.joint_effort_limits
    velocity_ratio = torch.abs(asset.data.joint_vel) / torch.clamp(velocity_limits, min=1.0e-6)
    torque_ratio = torch.abs(asset.data.applied_torque) / torch.clamp(effort_limits, min=1.0e-6)
    velocity_excess = torch.relu((velocity_ratio - velocity_margin) / (1.0 - velocity_margin))
    torque_excess = torch.relu((torque_ratio - torque_margin) / (1.0 - torque_margin))
    return torch.mean(torch.square(velocity_excess) + torch.square(torque_excess), dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.10,
    std: float = 0.055,
) -> torch.Tensor:
    """Reward swing-foot clearance without rewarding a permanently high stance."""
    asset = env.scene[asset_cfg.name]
    contacts = _contacts(env, sensor_cfg)
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    clearance = torch.exp(-torch.square(foot_height - target_height) / std**2)
    return torch.sum(clearance * (~contacts).to(torch.float32), dim=1)


def bilateral_lateral_separation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_separation: float,
) -> torch.Tensor:
    """Penalize left/right limbs crossing the sagittal centerline.

    The asset configuration must resolve exactly two corresponding bodies in
    left/right order.  This remains useful while simplified neighboring
    collision shapes require global self-collision to stay disabled.
    """
    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    if body_pos_w.shape[1] != 2:
        raise ValueError("bilateral_lateral_separation_l2 requires exactly two bodies")
    # Measure separation in the robot's heading frame.  Reset yaw is randomized,
    # so world-Y separation incorrectly reports a normal stance as crossed when
    # the robot faces along world Y.
    relative_pos_w = body_pos_w - asset.data.root_pos_w[:, None, :]
    heading_quat_w = math_utils.yaw_quat(asset.data.root_quat_w)
    heading_quat_w = heading_quat_w[:, None, :].expand(-1, body_pos_w.shape[1], -1)
    body_pos_h = math_utils.quat_apply_inverse(
        heading_quat_w.reshape(-1, 4), relative_pos_w.reshape(-1, 3)
    ).reshape_as(relative_pos_w)
    lateral = body_pos_h[..., 1]
    separation = torch.abs(lateral[:, 0] - lateral[:, 1])
    return torch.square(torch.relu(minimum_separation - separation) / minimum_separation)


def step_width_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_width: float = 0.20,
) -> torch.Tensor:
    """Keep the feet near a humanlike width without prescribing cadence."""
    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    if body_pos_w.shape[1] != 2:
        raise ValueError("step_width_l2 requires exactly two feet")
    relative_pos_w = body_pos_w - asset.data.root_pos_w[:, None, :]
    heading_quat_w = math_utils.yaw_quat(asset.data.root_quat_w)
    heading_quat_w = heading_quat_w[:, None, :].expand(-1, body_pos_w.shape[1], -1)
    body_pos_h = math_utils.quat_apply_inverse(
        heading_quat_w.reshape(-1, 4), relative_pos_w.reshape(-1, 3)
    ).reshape_as(relative_pos_w)
    lateral = body_pos_h[..., 1]
    width = torch.abs(lateral[:, 0] - lateral[:, 1])
    return torch.square(width - target_width)


def contact_foot_alignment_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize contacted feet that are tilted or yawed across travel.

    Local +X is the forward direction and local +Z is up in the prepared Isaac
    asset.  Foot heading is compared with the robot's yaw frame rather than the
    world X axis because reset yaw is randomized.  Only stance feet are scored
    so swing-foot toe motion remains free.
    """
    asset = env.scene[asset_cfg.name]
    contacts = _contacts(env, sensor_cfg).to(torch.float32)
    quaternions = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    local_up = torch.zeros((*quaternions.shape[:-1], 3), device=asset.device)
    local_up[..., 2] = 1.0
    local_forward = torch.zeros_like(local_up)
    local_forward[..., 0] = 1.0
    world_up = math_utils.quat_apply(quaternions.reshape(-1, 4), local_up.reshape(-1, 3)).reshape_as(local_up)
    world_forward = math_utils.quat_apply(
        quaternions.reshape(-1, 4), local_forward.reshape(-1, 3)
    ).reshape_as(local_forward)
    heading_quat_w = math_utils.yaw_quat(asset.data.root_quat_w)
    heading_quat_w = heading_quat_w[:, None, :].expand(-1, quaternions.shape[1], -1)
    foot_forward_h = math_utils.quat_apply_inverse(
        heading_quat_w.reshape(-1, 4), world_forward.reshape(-1, 3)
    ).reshape_as(world_forward)
    tilt = torch.square(1.0 - world_up[..., 2])
    cross_heading = torch.square(foot_forward_h[..., 1])
    return torch.sum((tilt + cross_heading) * contacts, dim=1) / torch.clamp(contacts.sum(dim=1), min=1.0)


def single_support_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    minimum_command_speed: float = 0.20,
) -> torch.Tensor:
    """Encourage a biped support phase only when locomotion is commanded."""
    contacts = _contacts(env, sensor_cfg)
    command = env.command_manager.get_command(command_name)
    moving = torch.abs(command[:, 0]) > minimum_command_speed
    return ((contacts.sum(dim=1) == 1) & moving).to(torch.float32)


def fore_aft_foot_separation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    base_target: float = 0.14,
    speed_gain: float = 0.20,
    maximum_target: float = 0.40,
    std: float = 0.10,
    minimum_command_speed: float = 0.20,
) -> torch.Tensor:
    """Reward a bounded fore-aft stride during single support.

    Speed alone allowed the previous policy to use tiny, high-cadence steps.
    Measuring in the heading frame makes the term invariant to randomized yaw.
    It is active only during single support so a static split stance cannot farm it.
    """
    asset = env.scene[asset_cfg.name]
    contacts = _contacts(env, sensor_cfg)
    command = env.command_manager.get_command(command_name)
    speed = torch.abs(command[:, 0])
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    relative_pos_w = body_pos_w - asset.data.root_pos_w[:, None, :]
    heading_quat_w = math_utils.yaw_quat(asset.data.root_quat_w)
    heading_quat_w = heading_quat_w[:, None, :].expand(-1, body_pos_w.shape[1], -1)
    body_pos_h = math_utils.quat_apply_inverse(
        heading_quat_w.reshape(-1, 4), relative_pos_w.reshape(-1, 3)
    ).reshape_as(relative_pos_w)
    separation = torch.abs(body_pos_h[:, 0, 0] - body_pos_h[:, 1, 0])
    target = torch.clamp(base_target + speed_gain * speed, max=maximum_target)
    valid = (contacts.sum(dim=1) == 1) & (speed > minimum_command_speed)
    return torch.exp(-torch.square(separation - target) / std**2) * valid.to(torch.float32)


def swing_duration_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    base_target: float = 0.40,
    speed_gain: float = -0.10,
    minimum_target: float = 0.18,
    maximum_target: float = 0.42,
    std: float = 0.12,
    minimum_command_speed: float = 0.20,
) -> torch.Tensor:
    """Reward swing timing near a speed-dependent target, not unlimited air time."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contacts = _contacts(env, sensor_cfg)
    speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    target = torch.clamp(base_target + speed_gain * speed, min=minimum_target, max=maximum_target)
    swing = ~contacts
    score = torch.exp(-torch.square(air_time - target.unsqueeze(1)) / std**2) * swing.to(torch.float32)
    return score.sum(dim=1) / torch.clamp(swing.sum(dim=1), min=1)


def gait_mode_duration_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    base_target: float = 0.42,
    speed_gain: float = -0.08,
    minimum_target: float = 0.28,
    maximum_target: float = 0.40,
    std: float = 0.08,
    minimum_command_speed: float = 0.20,
) -> torch.Tensor:
    """Reward a bounded alternating gait phase instead of rapid contact switching.

    During single support the stance foot's contact timer and the swing foot's
    air timer advance together. Scoring both timers makes a complete phase near
    the requested duration more valuable than many short phases, while the
    Gaussian prevents the policy from farming arbitrarily long steps.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = _contacts(env, sensor_cfg)
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    mode_time = torch.where(contacts, contact_time, air_time)
    speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    target = torch.clamp(base_target + speed_gain * speed, min=minimum_target, max=maximum_target)
    score = torch.exp(-torch.square(mode_time - target.unsqueeze(1)) / std**2).mean(dim=1)
    valid = (contacts.sum(dim=1) == 1) & (speed > minimum_command_speed)
    return score * valid.to(torch.float32)


def touchdown_air_time_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    base_target: float = 0.40,
    speed_gain: float = -0.08,
    minimum_target: float = 0.26,
    maximum_target: float = 0.38,
    std: float = 0.07,
    minimum_command_speed: float = 0.20,
) -> torch.Tensor:
    """Reward touchdown after a speed-dependent swing duration.

    This closes the earlier loophole where the airborne foot received a small
    timing score but could land early without an event-level consequence.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    target = torch.clamp(base_target + speed_gain * speed, min=minimum_target, max=maximum_target)
    score = torch.exp(-torch.square(last_air_time - target.unsqueeze(1)) / std**2)
    moving = speed > minimum_command_speed
    return torch.sum(score * first_contact.to(torch.float32), dim=1) * moving.to(torch.float32)


def premature_touchdown_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    base_minimum: float = 0.34,
    speed_gain: float = -0.06,
    minimum_target: float = 0.22,
    maximum_target: float = 0.30,
    minimum_command_speed: float = 0.20,
) -> torch.Tensor:
    """Penalize early touchdown and short contact-chatter events."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    minimum = torch.clamp(base_minimum + speed_gain * speed, min=minimum_target, max=maximum_target)
    deficit = torch.clamp(minimum.unsqueeze(1) - last_air_time, min=0.0) / torch.clamp(
        minimum.unsqueeze(1), min=1.0e-6
    )
    moving = speed > minimum_command_speed
    return torch.sum(torch.square(deficit) * first_contact.to(torch.float32), dim=1) * moving.to(torch.float32)


def bounded_double_support_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_duration: float = 0.06,
    std: float = 0.03,
    minimum_command_speed: float = 0.20,
    maximum_command_speed: float = 1.40,
) -> torch.Tensor:
    """Reward a brief walking double-support phase without rewarding shuffling.

    The newer foot has the smaller contact timer. The Gaussian peaks at a short
    transfer duration and decays again, so keeping both feet planted cannot
    accumulate maximum reward indefinitely.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = _contacts(env, sensor_cfg)
    contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    newest_contact_time = torch.min(contact_time, dim=1).values
    speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    valid = contacts.all(dim=1) & (speed > minimum_command_speed) & (speed < maximum_command_speed)
    return torch.exp(-torch.square(newest_contact_time - target_duration) / std**2) * valid.to(torch.float32)


def swing_knee_flexion_exp(
    env: ManagerBasedRLEnv,
    knee_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    base_target: float = 0.58,
    speed_gain: float = 0.16,
    maximum_target: float = 1.10,
    std: float = 0.28,
    minimum_command_speed: float = 0.20,
) -> torch.Tensor:
    """Encourage anatomical knee flexion only on the airborne leg."""
    asset = env.scene[knee_cfg.name]
    contacts = _contacts(env, sensor_cfg)
    knee_pos = asset.data.joint_pos[:, knee_cfg.joint_ids]
    speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    target = torch.clamp(base_target + speed_gain * speed, max=maximum_target)
    swing = (~contacts) & (speed.unsqueeze(1) > minimum_command_speed)
    score = torch.exp(-torch.square(knee_pos - target.unsqueeze(1)) / std**2) * swing.to(torch.float32)
    return score.sum(dim=1) / torch.clamp(swing.sum(dim=1), min=1)


def contralateral_arm_swing_exp(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg,
    forearm_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    coupling: float = 0.55,
    std: float = 0.14,
    minimum_command_speed: float = 0.20,
) -> torch.Tensor:
    """Coordinate left/right forearms opposite to the legs in the heading frame."""
    asset = env.scene[foot_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, foot_cfg.body_ids, :]
    arm_pos_w = asset.data.body_pos_w[:, forearm_cfg.body_ids, :]
    heading_quat_w = math_utils.yaw_quat(asset.data.root_quat_w)

    def to_heading(body_pos_w: torch.Tensor) -> torch.Tensor:
        relative = body_pos_w - asset.data.root_pos_w[:, None, :]
        heading = heading_quat_w[:, None, :].expand(-1, body_pos_w.shape[1], -1)
        return math_utils.quat_apply_inverse(
            heading.reshape(-1, 4), relative.reshape(-1, 3)
        ).reshape_as(relative)

    foot_h = to_heading(foot_pos_w)
    arm_h = to_heading(arm_pos_w)
    foot_delta = foot_h[:, 0, 0] - foot_h[:, 1, 0]
    arm_delta = arm_h[:, 0, 0] - arm_h[:, 1, 0]
    error = arm_delta + coupling * foot_delta
    moving = torch.abs(env.command_manager.get_command(command_name)[:, 0]) > minimum_command_speed
    stride_present = torch.abs(foot_delta) > 0.06
    return torch.exp(-torch.square(error) / std**2) * (moving & stride_present).to(torch.float32)


def walking_flight_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    minimum_command_speed: float = 0.20,
) -> torch.Tensor:
    """Return one during an invalid walking flight phase (used with negative weight)."""
    contacts = _contacts(env, sensor_cfg)
    moving = torch.abs(env.command_manager.get_command(command_name)[:, 0]) > minimum_command_speed
    return ((contacts.sum(dim=1) == 0) & moving).to(torch.float32)


def running_flight_phase_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    minimum_command_speed: float = 2.20,
    minimum_height: float = 0.62,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward brief flight only for fast, upright running.

    Height and orientation gates prevent a fall or long jump from being treated
    as valid running flight.
    """
    asset = env.scene[asset_cfg.name]
    contacts = _contacts(env, sensor_cfg)
    speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    upright = asset.data.projected_gravity_b[:, 2] < -0.82
    safe_height = asset.data.root_pos_w[:, 2] > minimum_height
    return ((contacts.sum(dim=1) == 0) & (speed > minimum_command_speed) & upright & safe_height).to(torch.float32)


def fast_double_support_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    minimum_command_speed: float = 2.20,
) -> torch.Tensor:
    """Return one for double support at running speeds (used with negative weight)."""
    contacts = _contacts(env, sensor_cfg)
    speed = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    return ((contacts.sum(dim=1) == 2) & (speed > minimum_command_speed)).to(torch.float32)


def ankle_differential_violation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    motor_torque_limit: float = 150.88,
    motor_speed_limit: float = 9.56,
) -> torch.Tensor:
    """Penalize commands outside the two-motor ankle differential envelope.

    Joint order must be left pitch/roll followed by right pitch/roll.  Each
    physical motor sees the sum or difference of pitch and roll loading.
    """
    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids].reshape(-1, 2, 2)
    speed = asset.data.joint_vel[:, asset_cfg.joint_ids].reshape(-1, 2, 2)
    motor_torque = torch.stack((torque[:, :, 0] + torque[:, :, 1], torque[:, :, 0] - torque[:, :, 1]), dim=-1)
    motor_speed = torch.stack((speed[:, :, 0] + speed[:, :, 1], speed[:, :, 0] - speed[:, :, 1]), dim=-1)
    torque_excess = torch.relu(torch.abs(motor_torque) - motor_torque_limit) / motor_torque_limit
    speed_excess = torch.relu(torch.abs(motor_speed) - motor_speed_limit) / motor_speed_limit
    return torch.sum(torch.square(torque_excess) + torch.square(speed_excess), dim=(1, 2))


def run_speed_curriculum(
    env: ManagerBasedRLEnv,
    env_ids,
    command_name: str = "base_velocity",
    minimum_speed: float = 0.30,
    initial_max_speed: float = 1.20,
    final_max_speed: float = 2.60,
    ramp_steps: int = 48000,
) -> torch.Tensor:
    """Continuously expand the run command envelope while retaining walk speeds."""
    del env_ids
    progress = min(float(env.common_step_counter) / float(ramp_steps), 1.0)
    maximum_speed = initial_max_speed + progress * (final_max_speed - initial_max_speed)
    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.ranges.lin_vel_x = (minimum_speed, maximum_speed)
    return torch.tensor(maximum_speed, device=env.device)


def _allocate_state(env: ManagerBasedRLEnv, name: str, dtype=torch.float32) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None or value.shape[0] != env.num_envs:
        value = torch.zeros(env.num_envs, device=env.device, dtype=dtype)
        setattr(env, name, value)
    return value


def reset_skill_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    for name, dtype in (
        ("_getup_dwell", torch.float32),
        ("_getup_stage", torch.long),
        ("_getup_success", torch.bool),
        ("_jump_stage", torch.long),
        ("_jump_landing_dwell", torch.float32),
        ("_jump_success", torch.bool),
    ):
        _allocate_state(env, name, dtype)[env_ids] = 0


def reset_recovery_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_noise: float = 0.12,
    joint_noise: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset into discrete supine, prone, and left/right side poses."""
    asset = env.scene[asset_cfg.name]
    count = len(env_ids)
    root = asset.data.default_root_state[env_ids].clone()
    root[:, 0:2] = env.scene.env_origins[env_ids, 0:2]
    root[:, 0:2] += (2.0 * torch.rand((count, 2), device=asset.device) - 1.0) * position_noise
    root[:, 2] = env.scene.env_origins[env_ids, 2] + 0.22

    modes = torch.randint(0, 4, (count,), device=asset.device)
    # A small upright reset mixture teaches the terminal hold directly instead
    # of requiring every sample to rediscover it through a complete recovery.
    upright_reset = torch.rand(count, device=asset.device) < 0.10
    roll = torch.zeros(count, device=asset.device)
    pitch = torch.zeros(count, device=asset.device)
    roll = torch.where(modes == 2, torch.full_like(roll, torch.pi / 2.0), roll)
    roll = torch.where(modes == 3, torch.full_like(roll, -torch.pi / 2.0), roll)
    pitch = torch.where(modes == 0, torch.full_like(pitch, torch.pi / 2.0), pitch)
    pitch = torch.where(modes == 1, torch.full_like(pitch, -torch.pi / 2.0), pitch)
    roll = torch.where(upright_reset, torch.zeros_like(roll), roll)
    pitch = torch.where(upright_reset, torch.zeros_like(pitch), pitch)
    yaw = (2.0 * torch.rand(count, device=asset.device) - 1.0) * torch.pi
    root[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    root[:, 7:13] = 0.0
    root[:, 2] = torch.where(
        upright_reset,
        env.scene.env_origins[env_ids, 2] + asset.data.default_root_state[env_ids, 2],
        root[:, 2],
    )

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    noise_scale = torch.where(
        upright_reset.unsqueeze(1),
        torch.full_like(joint_pos, min(joint_noise, 0.08)),
        torch.full_like(joint_pos, joint_noise),
    )
    joint_pos += (2.0 * torch.rand_like(joint_pos) - 1.0) * noise_scale
    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, min=limits[:, :, 0], max=limits[:, :, 1])
    joint_vel = torch.zeros_like(joint_pos)

    asset.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    reset_skill_state(env, env_ids)


def sustained_getup_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    required_dwell_s: float = 2.0,
    max_planar_speed: float = 0.10,
    max_vertical_speed: float = 0.10,
    max_angular_speed: float = 0.25,
    max_joint_speed_rms: float = 0.90,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ordered recovery ending in a genuinely stationary two-foot hold.

    Success is deliberately delayed for several seconds. This makes the
    policy optimize the post-recovery stance instead of ending the episode as
    soon as it briefly passes through an upright pose.
    """
    asset = env.scene[asset_cfg.name]
    stage = _allocate_state(env, "_getup_stage", torch.long)
    old_stage = stage.clone()
    contacts = _contacts(env, sensor_cfg).all(dim=1)
    gravity_z = asset.data.projected_gravity_b[:, 2]
    height = asset.data.root_pos_w[:, 2]
    upright = gravity_z < -0.94
    tall = height > 0.78
    planar_speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    vertical_speed = torch.abs(asset.data.root_lin_vel_b[:, 2])
    angular_speed = torch.linalg.vector_norm(asset.data.root_ang_vel_b, dim=1)
    joint_speed_rms = torch.sqrt(torch.mean(torch.square(asset.data.joint_vel), dim=1))
    slow = (planar_speed < max_planar_speed) & (vertical_speed < max_vertical_speed)
    slow_rotation = angular_speed < max_angular_speed
    slow_joints = joint_speed_rms < max_joint_speed_rms

    stage[:] = torch.where((stage == 0) & (height > 0.40), 1, stage)
    stage[:] = torch.where((stage == 1) & (height > 0.66) & (gravity_z < -0.78), 2, stage)
    stable = (stage == 2) & contacts & upright & tall & slow & slow_rotation & slow_joints
    dwell = _allocate_state(env, "_getup_dwell")
    success = _allocate_state(env, "_getup_success", torch.bool)
    dwell[:] = torch.where(stable, dwell + env.step_dt, torch.zeros_like(dwell))
    first_success = (~success) & (dwell >= required_dwell_s)
    success[:] = success | first_success
    height_progress = torch.clamp((height - 0.22) / 0.54, 0.0, 1.0)
    upright_progress = torch.clamp((-gravity_z - 0.05) / 0.85, 0.0, 1.0)
    # Shaping is stage-limited.  The previous policy could repeatedly collect
    # height/upright reward without ever reaching a stationary two-foot dwell.
    reward = torch.where(stage == 0, 0.35 * height_progress, torch.zeros_like(height))
    reward += torch.where(stage == 1, 0.45 * upright_progress, torch.zeros_like(height))
    reward += 2.0 * stable.to(torch.float32)
    reward += 3.0 * (stage > old_stage).to(torch.float32)
    reward += 14.0 * first_success.to(torch.float32)
    return reward


def terminal_recovery_motion_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize drift and rotation once a recovery is nearly upright."""
    asset = env.scene[asset_cfg.name]
    gate = ((asset.data.root_pos_w[:, 2] > 0.66) & (asset.data.projected_gravity_b[:, 2] < -0.78)).to(torch.float32)
    planar = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    rotation = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    return gate * (planar + 0.35 * rotation)


def terminal_recovery_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Suppress joint chatter after the pelvis is upright and near stand height."""
    asset = env.scene[asset_cfg.name]
    gate = (
        (asset.data.root_pos_w[:, 2] > 0.70)
        & (asset.data.projected_gravity_b[:, 2] < -0.86)
    ).to(torch.float32)
    return gate * torch.mean(torch.square(asset.data.joint_vel), dim=1)


def terminal_recovery_posture_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bring the recovered robot back to its validated neutral standing pose."""
    asset = env.scene[asset_cfg.name]
    gate = (
        (asset.data.root_pos_w[:, 2] > 0.70)
        & (asset.data.projected_gravity_b[:, 2] < -0.86)
    ).to(torch.float32)
    error = torch.mean(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return gate * error


def upright_undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Require feet-only support after rising while allowing recovery contacts."""
    asset = env.scene[asset_cfg.name]
    gate = (
        (asset.data.root_pos_w[:, 2] > 0.70)
        & (asset.data.projected_gravity_b[:, 2] < -0.86)
    ).to(torch.float32)
    return gate * _contacts(env, sensor_cfg, threshold).to(torch.float32).sum(dim=1)


def getup_success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _allocate_state(env, "_getup_success", torch.bool)


def skill_clock(env: ManagerBasedRLEnv, cycle_time: float, mode: str) -> torch.Tensor:
    phase = torch.remainder(env.episode_length_buf.to(torch.float32) * env.step_dt, cycle_time) / cycle_time
    angle = 2.0 * torch.pi * phase
    if mode == "jump":
        progress = _allocate_state(env, "_jump_stage", torch.long).to(torch.float32) / 4.0
    else:
        progress = torch.zeros_like(phase)
    return torch.stack((torch.sin(angle), torch.cos(angle), progress), dim=-1)


def jump_sequence_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    stage = _allocate_state(env, "_jump_stage", torch.long)
    dwell = _allocate_state(env, "_jump_landing_dwell")
    success = _allocate_state(env, "_jump_success", torch.bool)
    old_stage = stage.clone()
    contacts = _contacts(env, sensor_cfg)
    both = contacts.all(dim=1)
    airborne = ~contacts.any(dim=1)
    height = asset.data.root_pos_w[:, 2]
    vz = asset.data.root_lin_vel_w[:, 2]
    upright = asset.data.projected_gravity_b[:, 2] < -0.86

    stage[:] = torch.where((stage == 0) & both & (height < 0.78), 1, stage)
    stage[:] = torch.where((stage == 1) & (vz > 0.65), 2, stage)
    stage[:] = torch.where((stage == 2) & airborne & (height > 0.90), 3, stage)
    landing = (stage == 3) & both & upright & (height > 0.72)
    dwell[:] = torch.where(landing, dwell + env.step_dt, torch.zeros_like(dwell))
    stage[:] = torch.where((stage == 3) & (dwell > 0.40), 4, stage)
    first_success = (~success) & (stage == 4)
    success[:] = success | first_success

    reward = 4.0 * (stage > old_stage).to(torch.float32)
    reward += torch.where(stage == 0, torch.exp(-torch.square(height - 0.70) / 0.08**2), 0.0)
    reward += torch.where(stage == 1, torch.clamp(vz, 0.0, 2.5), 0.0)
    reward += torch.where(stage == 2, airborne.to(torch.float32) + torch.clamp(height - 0.84, 0.0, 0.5), 0.0)
    reward += torch.where(stage == 3, 2.0 * landing.to(torch.float32), 0.0)
    reward += 12.0 * first_success.to(torch.float32)
    return reward


def jump_success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _allocate_state(env, "_jump_success", torch.bool)
