"""Isaac Lab articulation config for the 1.524 m, 38 kg humanoid."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parent
USD_PATH = PROJECT_ROOT / "assets" / "usd" / "humanoid_5ft.usd"


HUMANOID_5FT_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=3.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # The simplified box collision bodies overlap at several adjacent
            # hip/waist joints.  As with Isaac Lab's official G1 minimal asset,
            # self-collision must remain disabled until explicit collision
            # filtering or non-overlapping collision meshes are authored.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.84),
        joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": -0.18,
            ".*_knee_pitch": 0.36,
            ".*_ankle_pitch": -0.18,
            ".*_ankle_roll": 0.0,
            "waist_.*": 0.0,
            ".*_shoulder_pitch": 0.10,
            ".*_shoulder_roll": 0.0,
            ".*_elbow_pitch": 0.25,
            "head_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.92,
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch"],
            effort_limit_sim={
                ".*_hip_yaw": 50.0,
                ".*_hip_roll": 132.25,
                ".*_hip_pitch": 226.8,
            },
            velocity_limit_sim={
                ".*_hip_yaw": 8.0,
                ".*_hip_roll": 6.0,
                ".*_hip_pitch": 14.0,
            },
            stiffness={
                ".*_hip_yaw": 120.0,
                ".*_hip_roll": 180.0,
                ".*_hip_pitch": 220.0,
            },
            damping={
                ".*_hip_yaw": 4.0,
                ".*_hip_roll": 6.0,
                ".*_hip_pitch": 7.0,
            },
            armature={
                ".*_hip_yaw": 0.02025,
                ".*_hip_roll": 0.15552,
                ".*_hip_pitch": 0.08960,
            },
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_pitch"],
            effort_limit_sim=243.0,
            velocity_limit_sim=14.0,
            stiffness=240.0,
            damping=7.0,
            armature=0.10240,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch", ".*_ankle_roll"],
            effort_limit_sim={".*_ankle_pitch": 115.0, ".*_ankle_roll": 65.0},
            velocity_limit_sim={".*_ankle_pitch": 8.0, ".*_ankle_roll": 6.0},
            stiffness={".*_ankle_pitch": 140.0, ".*_ankle_roll": 120.0},
            damping={".*_ankle_pitch": 5.0, ".*_ankle_roll": 4.0},
            armature=0.171396,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw", "waist_pitch"],
            effort_limit_sim={"waist_yaw": 50.0, "waist_pitch": 60.0},
            velocity_limit_sim=8.0,
            stiffness={"waist_yaw": 100.0, "waist_pitch": 140.0},
            damping={"waist_yaw": 3.0, "waist_pitch": 5.0},
            armature=0.02025,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_elbow_pitch"],
            effort_limit_sim={
                ".*_shoulder_pitch": 32.0,
                ".*_shoulder_roll": 30.0,
                ".*_elbow_pitch": 26.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch": 15.0,
                ".*_shoulder_roll": 15.0,
                ".*_elbow_pitch": 18.0,
            },
            stiffness={".*_shoulder_.*": 50.0, ".*_elbow_pitch": 40.0},
            damping={".*_shoulder_.*": 2.0, ".*_elbow_pitch": 1.5},
            armature={".*_shoulder_.*": 0.00972, ".*_elbow_pitch": 0.00768},
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw", "head_pitch"],
            effort_limit_sim=7.0,
            velocity_limit_sim=5.0,
            stiffness=15.0,
            damping=0.5,
            armature=0.00150,
        ),
    },
)
