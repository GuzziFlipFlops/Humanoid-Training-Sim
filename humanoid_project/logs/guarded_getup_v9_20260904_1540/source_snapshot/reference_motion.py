"""Retarget the public HumanUP G1 recovery trace to the 22-DOF robot.

The downloaded files are motion priors only.  They are never treated as proof
that this robot can execute the motion: ``reference_probe.py`` must replay the
retargeted targets through this project's collision, inertia and actuator model.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "assets" / "humanup_reference"

HUMANUP_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee_pitch",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee_pitch",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow_pitch",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_pitch",
)


def _load_pickle_cpu(path: Path) -> torch.Tensor:
    """Load HumanUP's CUDA-saved pickle without requiring its original GPU."""
    old = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda data: torch.load(
            io.BytesIO(data), map_location="cpu", weights_only=False
        )
        with path.open("rb") as handle:
            value = pickle.load(handle)
    finally:
        torch.storage._load_from_bytes = old
    return torch.as_tensor(value, dtype=torch.float32, device="cpu")


def load_humanup() -> tuple[torch.Tensor, torch.Tensor]:
    q = _load_pickle_cpu(REFERENCE_DIR / "dof_pos_all.pkl")
    height = _load_pickle_cpu(REFERENCE_DIR / "head_height_all.pkl").flatten()
    if q.shape != (58, 23) or height.shape != (58,):
        raise RuntimeError(f"Unexpected HumanUP reference shapes: q={tuple(q.shape)}, h={tuple(height.shape)}")
    if not torch.isfinite(q).all() or not torch.isfinite(height).all():
        raise RuntimeError("HumanUP reference contains non-finite values")
    return q, height


def retarget(
    joint_names: list[str],
    default_q: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    roll_scale: float = 0.72,
    yaw_scale: float = 0.72,
    pitch_scale: float = 1.0,
    mirror: bool = False,
) -> torch.Tensor:
    """Return a limit-safe [T, 22] target trace in this robot's joint order.

    G1 has waist-roll and shoulder-yaw joints that this robot does not.  Those
    channels are omitted; head joints stay neutral.  Hip/ankle roll and hip yaw
    are scaled because this robot has narrower ranges than G1.  Pitch channels
    preserve the corrected Unitree-style sign convention.
    """
    source, _ = load_humanup()
    source_index = {name: index for index, name in enumerate(HUMANUP_NAMES)}
    out = default_q.detach().cpu().reshape(1, -1).repeat(len(source), 1)
    if out.shape[1] != len(joint_names):
        raise ValueError("Default pose and joint-name dimensions differ")

    def source_name(target: str) -> str:
        if not mirror:
            return target
        if target.startswith("left_"):
            return "right_" + target[5:]
        if target.startswith("right_"):
            return "left_" + target[6:]
        return target

    for target_index, target_name in enumerate(joint_names):
        name = source_name(target_name)
        if name not in source_index:
            continue
        values = source[:, source_index[name]].clone()
        if mirror and ("_roll" in target_name or "_yaw" in target_name):
            values.neg_()
        if "_roll" in target_name:
            values.mul_(roll_scale)
        elif "_yaw" in target_name:
            values.mul_(yaw_scale)
        elif "pitch" in target_name or "knee" in target_name:
            values.mul_(pitch_scale)
        out[:, target_index] = values

    # Leave a small hard-stop margin.  Unlike v8's +/-0.70-rad clamp, this
    # retains the deep knee flexion and hip excursion needed to put the feet
    # underneath the pelvis.
    lo = lower.detach().cpu() + 0.025
    hi = upper.detach().cpu() - 0.025
    out = torch.maximum(torch.minimum(out, hi), lo)

    # HumanUP records the discovered motion after its initial lying state has
    # already settled.  Our pose bank starts near the neutral joint pose, so a
    # direct first-frame jump caused collision-heavy torque impulses.  Move into
    # the first reference posture smoothly while the robot remains on the floor.
    prelude_frames = 50  # 1.0 s at 50 Hz
    prelude_u = torch.linspace(0.0, 1.0, prelude_frames + 1, dtype=torch.float32)[:-1]
    prelude_u = prelude_u * prelude_u * (3.0 - 2.0 * prelude_u)
    prelude = default_q.detach().cpu()[None] + prelude_u[:, None] * (
        out[:1] - default_q.detach().cpu()[None]
    )

    # Blend from the HumanUP terminal pose into this robot's independently
    # validated neutral standing pose before the stationary controller takes
    # over.  Smoothstep keeps target velocity continuous at both ends.
    blend_frames = 75  # 1.5 s at the 50 Hz policy rate
    u = torch.linspace(0.0, 1.0, blend_frames + 1, dtype=torch.float32)[1:]
    u = u * u * (3.0 - 2.0 * u)
    tail = out[-1:] + u[:, None] * (default_q.detach().cpu()[None] - out[-1:])
    hold = default_q.detach().cpu().reshape(1, -1).repeat(100, 1)
    return torch.cat((prelude, out, tail, hold), dim=0)


def time_warp(trace: torch.Tensor, factor: float) -> torch.Tensor:
    """Resample a target trace; factor > 1 executes it more slowly."""
    if factor <= 0:
        raise ValueError("Time-warp factor must be positive")
    count = max(2, round(len(trace) * factor))
    where = torch.linspace(0, len(trace) - 1, count)
    lo = where.floor().long()
    hi = where.ceil().long().clamp_max(len(trace) - 1)
    alpha = (where - lo.float())[:, None]
    return trace[lo] * (1.0 - alpha) + trace[hi] * alpha
