"""Fast GUI validation of every joint on the five-foot humanoid."""

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Sweep all 22 humanoid joints with a live terminal status display.")
parser.add_argument("--seconds_per_joint", type=float, default=0.55)
parser.add_argument("--cycles", type=float, default=1.0)
parser.add_argument("--hold_seconds", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from humanoid_cfg import HUMANOID_5FT_CFG  # noqa: E402


def actuator_spec(joint_name: str) -> tuple[float, float]:
    if "hip_yaw" in joint_name:
        return 50.0, 8.0
    if "hip_roll" in joint_name:
        return 132.25, 6.0
    if "hip_pitch" in joint_name:
        return 226.8, 14.0
    if "knee_pitch" in joint_name:
        return 243.0, 14.0
    if "ankle_pitch" in joint_name:
        return 115.0, 8.0
    if "ankle_roll" in joint_name:
        return 65.0, 6.0
    if joint_name == "waist_yaw":
        return 50.0, 8.0
    if joint_name == "waist_pitch":
        return 60.0, 8.0
    if "shoulder_pitch" in joint_name:
        return 32.0, 15.0
    if "shoulder_roll" in joint_name:
        return 30.0, 15.0
    if "elbow_pitch" in joint_name:
        return 26.0, 18.0
    return 7.0, 5.0


SEMANTIC_ORDER = [
    "waist_yaw", "waist_pitch",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow_pitch",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow_pitch",
    "head_yaw", "head_pitch",
]


def set_joint_camera(sim, joint_name: str) -> str:
    if "roll" in joint_name:
        sim.set_camera_view(eye=[2.8, 0.0, 1.35], target=[0.0, 0.0, 0.78])
        return "+X forward-axis view (roll moves left/right)"
    if "pitch" in joint_name or "elbow" in joint_name:
        sim.set_camera_view(eye=[0.0, 2.8, 1.35], target=[0.0, 0.0, 0.78])
        return "+Y side view (pitch moves forward/back)"
    sim.set_camera_view(eye=[2.3, 2.3, 2.25], target=[0.0, 0.0, 0.80])
    return "elevated view (yaw rotates around vertical Z)"


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=[2.7, 2.5, 1.65], target=[0.0, 0.0, 0.78])
    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/Light", light_cfg)

    test_cfg = HUMANOID_5FT_CFG.copy()
    test_cfg.init_state.pos = (0.0, 0.0, 0.84)
    test_cfg.spawn.rigid_props.disable_gravity = True
    test_cfg.spawn.articulation_props.fix_root_link = True
    robot = Articulation(test_cfg)
    sim.reset()

    nominal = robot.data.default_joint_pos.clone()
    zeros = torch.zeros_like(nominal)
    robot.write_joint_state_to_sim(nominal, zeros)
    robot.reset()
    limits = robot.data.soft_joint_pos_limits[0]
    dt = sim.get_physics_dt()
    steps_per_joint = max(1, round(args_cli.seconds_per_joint / dt))
    joint_lookup = {name: index for index, name in enumerate(robot.joint_names)}
    missing = [name for name in SEMANTIC_ORDER if name not in joint_lookup]
    if missing:
        raise RuntimeError(f"Imported articulation is missing expected joints: {missing}")
    total = len(SEMANTIC_ORDER)

    print("\n" + "=" * 88)
    print(f" FAST HUMANOID JOINT VALIDATION | {total} joints | {args_cli.seconds_per_joint:.2f} s each")
    print(" Root fixed and gravity disabled: motion shown is the isolated joint axis.")
    print("=" * 88, flush=True)

    for display_index, joint_name in enumerate(SEMANTIC_ORDER):
        joint_index = joint_lookup[joint_name]
        view_label = set_joint_camera(sim, joint_name)
        # Remove all residual position/velocity from the previous joint.  Without
        # this, a fast sequential test measures several settling joints at once.
        robot.write_joint_state_to_sim(nominal, zeros)
        robot.set_joint_position_target(nominal)
        robot.reset()
        robot.write_data_to_sim()
        for settle_step in range(8):
            sim.step(render=(settle_step % 2 == 0))
            robot.update(dt)
        lower = float(limits[joint_index, 0])
        upper = float(limits[joint_index, 1])
        center = float(nominal[0, joint_index])
        effort, velocity = actuator_spec(joint_name)
        available = max(0.02, min(center - lower, upper - center))
        # One sinusoid begins and ends at nominal.  Bound its peak velocity to
        # 65% of the approved actuator speed for a readable but nonviolent test.
        amplitude = min(0.55, 0.65 * available, 0.65 * velocity * args_cli.seconds_per_joint / (2.0 * math.pi))
        safe_low = center - amplitude
        safe_high = center + amplitude
        print(
            f"\n[{display_index + 1:02d}/{total:02d}] {joint_name:<24} "
            f"axis_view={view_label}\n"
            f"    sweep=[{math.degrees(safe_low):+6.1f}°, {math.degrees(safe_high):+6.1f}°]  "
            f"limits=[{math.degrees(lower):6.1f}°, {math.degrees(upper):6.1f}°]  "
            f"peak={effort:6.2f} Nm  vmax={velocity:4.1f} rad/s",
            flush=True,
        )
        for step in range(steps_per_joint):
            phase = 2.0 * math.pi * args_cli.cycles * step / steps_per_joint
            command = center + amplitude * math.sin(phase)
            targets = nominal.clone()
            targets[:, joint_index] = command
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step(render=(step % 2 == 0))
            robot.update(dt)
            if step % max(1, steps_per_joint // 4) == 0:
                measured = float(robot.data.joint_pos[0, joint_index])
                error = math.degrees(command - measured)
                print(
                    f"    command={math.degrees(command):+6.1f}°  measured={math.degrees(measured):+6.1f}°  error={error:+5.1f}°",
                    flush=True,
                )
            if not simulation_app.is_running():
                return

    robot.set_joint_position_target(nominal)
    print(f"\n[PASS] All {total} joints completed. Holding nominal pose for {args_cli.hold_seconds:.1f} s.", flush=True)
    for _ in range(max(1, round(args_cli.hold_seconds / dt))):
        if not simulation_app.is_running():
            return
        robot.set_joint_position_target(nominal)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
