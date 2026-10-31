"""Heading commands and independent fixed-direction measurements.

The command generator emits a desired yaw rate, never a motor action. It follows
Isaac Lab's wrapped proportional heading-command formulation (gain 0.5), bounded
to Unitree G1's published angular-command limit of 0.2 rad/s. Stopped commands
remain exactly zero. The direction verifier uses raw poses in the reset heading
frame and does not use reward or the generator's error as its measurements.
"""
import math
import numpy as np
import torch

COMMAND_VERSION = 'fixed_reset_heading_yaw_rate_v1'
GAIN = .5
MAX_YAW_RATE = .2
HEADING_WALK_DEG = 20.
HEADING_RUN_DEG = 25.


def yaw_from_quat(quaternion):
    """Horizontal forward heading; quaternions are Isaac w,x,y,z."""
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2*(x*y+w*z), 1-2*(y*y+z*z))


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def heading_yaw_command(quaternion, target, moving):
    error = wrap_angle(target-yaw_from_quat(quaternion))
    return torch.where(moving, (GAIN*error).clamp(-MAX_YAW_RATE, MAX_YAW_RATE),
                       torch.zeros_like(error))


def direction_statistics(positions, quaternions, valid, reference_heading, dt, speed):
    """CPU verifier from T,N raw post-warmup samples, independent of reward.

    Paired velocities use only adjacent live samples; resets/falls cannot create
    artificial travel. Heading is compared with the *reset* heading, not a later
    orientation. Signed speed and absolute lateral speed use that same fixed
    frame. Per-trial endpoint quantities describe only surviving trajectories.
    """
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    reference_heading = np.broadcast_to(np.asarray(reference_heading), valid.shape[1:])
    assert positions.shape == valid.shape+(2,)
    assert quaternions.shape == valid.shape+(4,)
    assert dt > 0 and len(positions) >= 2
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    yaw = np.arctan2(2*(x*y+w*z), 1-2*(y*y+z*z))
    error = np.arctan2(np.sin(yaw-reference_heading), np.cos(yaw-reference_heading))
    forward = np.stack([np.cos(reference_heading), np.sin(reference_heading)], -1)
    sideways = np.stack([-np.sin(reference_heading), np.cos(reference_heading)], -1)
    delta = np.diff(positions, axis=0)
    paired = valid[1:] & valid[:-1]
    along = (delta*forward).sum(-1)/dt
    lateral = (delta*sideways).sum(-1)/dt
    surviving = valid.all(0)
    net = positions[-1]-positions[0]
    path = np.linalg.norm(delta, axis=-1).sum(0)
    efficiency = (net*forward).sum(-1)/np.maximum(path, 1e-8)
    threshold = math.radians(HEADING_WALK_DEG if speed <= 1. else HEADING_RUN_DEG)
    def measured_mean(values, mask):
        return float(values[mask].mean()) if mask.any() else 999.
    return dict(
        heading_abs_mean_deg=math.degrees(measured_mean(np.abs(error), valid)),
        heading_outside_fraction=measured_mean((np.abs(error)>threshold).astype(float), valid),
        heading_bound_deg=math.degrees(threshold),
        reset_frame_forward_speed_m_s=measured_mean(along, paired),
        reset_frame_lateral_speed_m_s=measured_mean(np.abs(lateral), paired),
        forward_path_efficiency=measured_mean(efficiency, surviving),
        final_heading_abs_mean_deg=math.degrees(measured_mean(np.abs(error[-1]), surviving)),
        direction_trials=int(valid.shape[1]), direction_surviving_trials=int(surviving.sum()))


def direction_checks(speed, stats):
    """Additional guards; existing survival/contact/step checks are retained."""
    if speed == 0:
        return {}
    return dict(
        fixed_heading=stats['heading_outside_fraction'] < .05,
        world_forward_speed=abs(stats['reset_frame_forward_speed_m_s']-speed) < max(.10, speed*.12),
        world_lateral_speed=stats['reset_frame_lateral_speed_m_s'] < .14,
        forward_net_travel=stats['forward_path_efficiency'] > .85)
