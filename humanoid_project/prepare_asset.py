"""Create the Isaac-coordinate URDF and copy traceable dynamics data.

Fusion uses X=left, Y=forward, Z=up. Isaac locomotion conventions use
X=forward, Y=left, Z=up. This tool rotates the URDF data without changing the
native Fusion assembly, then copies the actuator/dynamics manifests beside it.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path(r"C:\Users\rshan\Documents\GitHub\Humanoid-Robot\CAD\Simulation\Humanoid5ft")
SOURCE_URDF = SOURCE_ROOT / "humanoid_5ft_ragdoll.urdf"
OUTPUT_URDF = PROJECT_ROOT / "assets" / "urdf" / "humanoid_5ft_isaac.urdf"

LEG_PITCH_JOINTS = {
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
}


def _numbers(text: str) -> list[float]:
    return [float(value) for value in text.split()]


def _format(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _rotate_xyz(values: list[float]) -> list[float]:
    """Map CAD (left, forward, up) to Isaac (forward, left, up)."""
    x_right, y_forward, z_up = values
    return [y_forward, -x_right, z_up]


def main() -> None:
    if not SOURCE_URDF.is_file():
        raise FileNotFoundError(f"Fusion URDF does not exist: {SOURCE_URDF}")

    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    root.set("name", "humanoid_5ft_isaac")

    for origin in root.findall(".//origin"):
        if "xyz" in origin.attrib:
            origin.set("xyz", _format(_rotate_xyz(_numbers(origin.attrib["xyz"]))))
    for axis in root.findall(".//axis"):
        axis.set("xyz", _format(_rotate_xyz(_numbers(axis.attrib["xyz"]))))
    for box in root.findall(".//box"):
        x_left, y_forward, z_up = _numbers(box.attrib["size"])
        box.set("size", _format([y_forward, x_left, z_up]))

    # Policies and default poses follow Isaac/Unitree's anatomical convention:
    # pitch rotates around +Y, roll around +X, and yaw around +Z.  Reject a
    # regenerated Fusion export immediately if a leg pitch sign regresses.
    axis_audit = {}
    for joint in root.findall("./joint"):
        name = joint.attrib["name"]
        axis = _numbers(joint.find("axis").attrib["xyz"])
        axis_audit[name] = axis
        if name in LEG_PITCH_JOINTS and axis != [0.0, 1.0, 0.0]:
            raise ValueError(f"{name} must export on Isaac +Y, received {axis}")

    OUTPUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)

    manifest_targets = {
        "humanoid_5ft_ragdoll_dynamics.json": "humanoid_5ft_dynamics.json",
        "humanoid_5ft_ragdoll_isaac_actuators.json": "humanoid_5ft_actuators.json",
        "fusion_export_status.json": "fusion_export_status.json",
    }
    for source_name, target_name in manifest_targets.items():
        shutil.copy2(SOURCE_ROOT / source_name, PROJECT_ROOT / "assets" / "urdf" / target_name)

    dynamics = json.loads((PROJECT_ROOT / "assets" / "urdf" / "humanoid_5ft_dynamics.json").read_text())
    summary = {
        "source": str(SOURCE_URDF),
        "output": str(OUTPUT_URDF),
        "coordinate_mapping": "isaac_x=cad_y, isaac_y=-cad_x, isaac_z=cad_z",
        "height_m": dynamics["height_m"],
        "modeled_mass_kg": dynamics["modeled_mass_kg"],
        "ankle_linkage_ratio": dynamics["ankle_transmission"]["torque_reduction_ratio"],
        "ankle_effective_motor_to_joint_ratio": dynamics["ankle_transmission"][
            "effective_motor_to_joint_ratio"
        ],
        "leg_pitch_axis_isaac": [0.0, 1.0, 0.0],
        "axis_audit": axis_audit,
    }
    (PROJECT_ROOT / "assets" / "urdf" / "asset_prepare_status.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
