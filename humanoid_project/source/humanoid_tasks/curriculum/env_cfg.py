"""Manager-based curriculum environments for the custom five-foot humanoid."""

import math
import sys
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as locomotion_mdp

from . import skill_mdp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from humanoid_cfg import HUMANOID_5FT_CFG  # noqa: E402


FEET = SceneEntityCfg("contact_forces", body_names=["left_foot", "right_foot"])
FOOT_BODIES = SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])
SHANK_BODIES = SceneEntityCfg("robot", body_names=["left_shank", "right_shank"])
FOREARM_BODIES = SceneEntityCfg("robot", body_names=["left_forearm", "right_forearm"])
LEFT_LEG = SceneEntityCfg(
    "robot",
    joint_names=[
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee_pitch",
        "left_ankle_pitch", "left_ankle_roll",
    ],
)
RIGHT_LEG = SceneEntityCfg(
    "robot",
    joint_names=[
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee_pitch",
        "right_ankle_pitch", "right_ankle_roll",
    ],
)
ANKLE_DIFFERENTIALS = SceneEntityCfg(
    "robot",
    joint_names=["left_ankle_pitch", "left_ankle_roll", "right_ankle_pitch", "right_ankle_roll"],
)
KNEES = SceneEntityCfg("robot", joint_names=["left_knee_pitch", "right_knee_pitch"])
UPPER_BODY = SceneEntityCfg(
    "robot",
    joint_names=["waist_.*", ".*_shoulder_roll", "head_.*"],
)
ARMS = SceneEntityCfg(
    "robot",
    joint_names=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_elbow_pitch"],
)
LATERAL_LEG_JOINTS = SceneEntityCfg(
    "robot",
    joint_names=[".*_hip_yaw", ".*_hip_roll", ".*_ankle_roll"],
)
NON_FOOT_CONTACTS = SceneEntityCfg(
    "contact_forces",
    body_names=["torso", "head", ".*_thigh", ".*_shank", ".*_upper_arm", ".*_forearm"],
)


@configclass
class HumanoidSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(160.0, 160.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.95,
                dynamic_friction=0.80,
                restitution=0.0,
            ),
        ),
    )
    robot: ArticulationCfg = HUMANOID_5FT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2200.0),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.35,
        use_default_offset=True,
    )


def _velocity_command(x_range, yaw_range, standing_fraction=0.1):
    return locomotion_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=standing_fraction,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=locomotion_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=x_range,
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=yaw_range,
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ZeroCommandsCfg:
    base_velocity = _velocity_command((0.0, 0.0), (0.0, 0.0), 1.0)


@configclass
class TurnStandingCommandsCfg:
    base_velocity = _velocity_command((0.0, 0.0), (-0.75, 0.75), 0.15)


@configclass
class WalkCommandsCfg:
    # Train over a useful range while making 1.0 m/s the upper walking target.
    base_velocity = _velocity_command((0.20, 1.00), (0.0, 0.0), 0.15)


@configclass
class FixedWalk1CommandsCfg:
    # Deterministic playback command: exactly 1.0 m/s, no standing samples.
    base_velocity = _velocity_command((1.00, 1.00), (0.0, 0.0), 0.0)


@configclass
class GaitWalkCommandsCfg:
    # Concentrate training on a useful human walking range while retaining a
    # small stationary fraction so the policy can stop without falling.
    base_velocity = _velocity_command((0.55, 1.05), (0.0, 0.0), 0.06)


@configclass
class GaitSprint4CommandsCfg:
    # The curriculum expands this upper limit to 4.2 m/s. The lower end keeps
    # the walk-to-run transition represented in the same command-conditioned policy.
    base_velocity = _velocity_command((0.80, 1.50), (0.0, 0.0), 0.05)


@configclass
class WalkTurnCommandsCfg:
    base_velocity = _velocity_command((0.30, 1.00), (-0.80, 0.80), 0.08)


@configclass
class RunFastCommandsCfg:
    # Warm-started from walking; this range exposes the policy to the transition
    # from a fast walk into the requested 2.5 m/s running gait.
    base_velocity = _velocity_command((0.30, 1.20), (0.0, 0.0), 0.12)


@configclass
class Sprint4CommandsCfg:
    # Preserve lower-speed transitions while progressively exposing 4 m/s.
    base_velocity = _velocity_command((0.50, 2.80), (0.0, 0.0), 0.08)


@configclass
class RunTurnCommandsCfg:
    base_velocity = _velocity_command((1.20, 2.60), (-1.00, 1.00), 0.03)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.10, n_max=0.10))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.025, n_max=0.025))
        commands = ObsTerm(
            func=locomotion_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.20, n_max=0.20), scale=0.1)
        feet_contacts = ObsTerm(func=skill_mdp.feet_contact_state, params={"sensor_cfg": FEET})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class JumpObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        commands = ObsTerm(func=skill_mdp.skill_clock, params={"cycle_time": 4.5, "mode": "jump"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class BaseEventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.70, 1.15),
            "dynamic_friction_range": (0.55, 0.95),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 64,
        },
    )
    torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (-0.40, 0.40),
            "operation": "add",
        },
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005)},
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.02),
                "roll": (-0.04, 0.04), "pitch": (-0.04, 0.04), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.03, 0.03),
                "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.92, 1.08), "velocity_range": (0.0, 0.0)},
    )
    reset_state = EventTerm(func=skill_mdp.reset_skill_state, mode="reset")


@configclass
class LocomotionEventsCfg(BaseEventsCfg):
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 14.0),
        params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.20, 0.20), "yaw": (-0.20, 0.20)}},
    )


@configclass
class GaitWalkEventsCfg(BaseEventsCfg):
    # Smaller disturbances let gait quality converge while still training
    # recovery. The final evaluator remains deterministic.
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(7.0, 12.0),
        params={"velocity_range": {"x": (-0.16, 0.16), "y": (-0.12, 0.12), "yaw": (-0.12, 0.12)}},
    )


@configclass
class GaitRunEventsCfg(BaseEventsCfg):
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(7.0, 12.0),
        params={"velocity_range": {"x": (-0.22, 0.22), "y": (-0.16, 0.16), "yaw": (-0.14, 0.14)}},
    )


@configclass
class RecoveryEventsCfg(BaseEventsCfg):
    reset_base = None
    reset_joints = None
    reset_state = None
    reset_recovery = EventTerm(func=skill_mdp.reset_recovery_pose, mode="reset")


@configclass
class StandRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-200.0)
    upright = RewTerm(func=skill_mdp.upright_exp, weight=3.5, params={"std": 0.22})
    height = RewTerm(func=skill_mdp.base_height_exp, weight=2.5, params={"target_height": 0.84, "std": 0.10})
    feet_support = RewTerm(func=skill_mdp.both_feet_contact, weight=1.5, params={"sensor_cfg": FEET})
    planar_still = RewTerm(func=skill_mdp.planar_still_exp, weight=1.0, params={"std": 0.25})
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    roll_pitch_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)
    upper_body_posture = RewTerm(func=mdp.joint_deviation_l1, weight=-0.10, params={"asset_cfg": UPPER_BODY})
    lateral_posture = RewTerm(func=mdp.joint_deviation_l1, weight=-0.10, params={"asset_cfg": LATERAL_LEG_JOINTS})
    symmetry = RewTerm(
        func=skill_mdp.joint_symmetry_l1,
        weight=-0.12,
        params={"left_cfg": LEFT_LEG, "right_cfg": RIGHT_LEG},
    )
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    feet_slide = RewTerm(
        func=locomotion_mdp.feet_slide,
        weight=-0.30,
        params={"sensor_cfg": FEET, "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])},
    )
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": NON_FOOT_CONTACTS, "threshold": 5.0})
    ankle_differential = RewTerm(
        func=skill_mdp.ankle_differential_violation_l2,
        weight=-2.0,
        params={"asset_cfg": ANKLE_DIFFERENTIALS},
    )


@configclass
class TurnStandingRewardsCfg(StandRewardsCfg):
    yaw_tracking = RewTerm(
        func=locomotion_mdp.track_ang_vel_z_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )
    zero_linear = RewTerm(
        func=locomotion_mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.12)},
    )


@configclass
class RecoveryRewardsCfg:
    recovery = RewTerm(
        func=skill_mdp.sustained_getup_reward,
        weight=3.0,
        params={
            "sensor_cfg": FEET,
            "required_dwell_s": 1.25,
            "max_planar_speed": 0.20,
            "max_vertical_speed": 0.14,
            "max_angular_speed": 0.40,
            "max_joint_speed_rms": 1.50,
        },
    )
    # Discovery gets weak dense shaping, while completion still depends on the
    # ordered stationary hold in sustained_getup_reward.
    height = RewTerm(func=skill_mdp.base_height_exp, weight=0.5, params={"target_height": 0.84, "std": 0.28})
    upright = RewTerm(func=skill_mdp.upright_exp, weight=0.5, params={"std": 0.50})
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-8)
    joint_power = RewTerm(func=skill_mdp.joint_power_l1, weight=-2.0e-6)
    head_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["head"]), "threshold": 8.0},
    )
    ankle_differential = RewTerm(
        func=skill_mdp.ankle_differential_violation_l2,
        weight=-1.0,
        params={"asset_cfg": ANKLE_DIFFERENTIALS},
    )
    terminal_motion = RewTerm(func=skill_mdp.terminal_recovery_motion_l2, weight=-2.0)
    terminal_joint_motion = RewTerm(func=skill_mdp.terminal_recovery_joint_velocity_l2, weight=-0.04)
    terminal_posture = RewTerm(func=skill_mdp.terminal_recovery_posture_l1, weight=-0.75)
    terminal_contacts = RewTerm(
        func=skill_mdp.upright_undesired_contacts,
        weight=-1.5,
        params={"sensor_cfg": NON_FOOT_CONTACTS, "threshold": 5.0},
    )
    foot_crossing = RewTerm(
        func=skill_mdp.bilateral_lateral_separation_l2,
        weight=-6.0,
        params={"asset_cfg": FOOT_BODIES, "minimum_separation": 0.09},
    )
    knee_crossing = RewTerm(
        func=skill_mdp.bilateral_lateral_separation_l2,
        weight=-6.0,
        params={"asset_cfg": SHANK_BODIES, "minimum_separation": 0.08},
    )


@configclass
class RecoveryRefineRewardsCfg(RecoveryRewardsCfg):
    recovery = RewTerm(
        func=skill_mdp.sustained_getup_reward,
        weight=4.0,
        params={
            "sensor_cfg": FEET,
            "required_dwell_s": 2.5,
            "max_planar_speed": 0.08,
            "max_vertical_speed": 0.08,
            "max_angular_speed": 0.20,
            "max_joint_speed_rms": 0.80,
        },
    )
    # Refinement must not farm independent height/upright shaping. It receives
    # task reward only from ordered recovery and the terminal hold.
    height = None
    upright = None
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-3.0e-7)
    joint_power = RewTerm(func=skill_mdp.joint_power_l1, weight=-1.0e-5)
    terminal_motion = RewTerm(func=skill_mdp.terminal_recovery_motion_l2, weight=-5.0)
    terminal_joint_motion = RewTerm(func=skill_mdp.terminal_recovery_joint_velocity_l2, weight=-0.12)
    terminal_posture = RewTerm(func=skill_mdp.terminal_recovery_posture_l1, weight=-1.5)
    terminal_contacts = RewTerm(
        func=skill_mdp.upright_undesired_contacts,
        weight=-3.0,
        params={"sensor_cfg": NON_FOOT_CONTACTS, "threshold": 5.0},
    )


@configclass
class LocomotionRewardsCfg(StandRewardsCfg):
    velocity_tracking = RewTerm(
        func=skill_mdp.gated_velocity_tracking_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.20), "minimum_height": 0.62},
    )
    yaw_tracking = RewTerm(
        func=locomotion_mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.20)},
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.35},
    )
    feet_slide = RewTerm(
        func=locomotion_mdp.feet_slide,
        weight=-0.25,
        params={"sensor_cfg": FEET, "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])},
    )
    planar_still = None
    symmetry = None
    feet_support = None
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.50)
    lateral_velocity = RewTerm(func=skill_mdp.lateral_velocity_l2, weight=-0.75)
    joint_power = RewTerm(func=skill_mdp.joint_power_l1, weight=-1.0e-5)


@configclass
class RunRewardsCfg(LocomotionRewardsCfg):
    velocity_tracking = RewTerm(
        func=skill_mdp.gated_velocity_tracking_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.30), "minimum_height": 0.58},
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=0.8,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.28},
    )
    height = RewTerm(func=skill_mdp.base_height_exp, weight=1.0, params={"target_height": 0.81, "std": 0.16})
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-7.5e-7)
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.006)


@configclass
class Sprint4RewardsCfg(RunRewardsCfg):
    velocity_tracking = RewTerm(
        func=skill_mdp.gated_velocity_tracking_exp,
        weight=6.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.38), "minimum_height": 0.54},
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.22},
    )
    foot_clearance = RewTerm(
        func=skill_mdp.foot_clearance_reward,
        weight=0.35,
        params={"sensor_cfg": FEET, "asset_cfg": FOOT_BODIES, "target_height": 0.105, "std": 0.06},
    )
    height = RewTerm(func=skill_mdp.base_height_exp, weight=0.8, params={"target_height": 0.78, "std": 0.20})
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-8.0e-7)
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.12)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.004)
    joint_power = RewTerm(func=skill_mdp.joint_power_l1, weight=-1.25e-5)
    actuator_saturation = RewTerm(
        func=skill_mdp.actuator_saturation_l2,
        weight=-1.5,
        params={"velocity_margin": 0.85, "torque_margin": 0.80},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": NON_FOOT_CONTACTS, "threshold": 5.0},
    )


@configclass
class NaturalStandRewardsCfg(StandRewardsCfg):
    """Standing rewards with explicit posture and limb-clearance constraints."""

    arm_posture = RewTerm(func=mdp.joint_deviation_l1, weight=-0.12, params={"asset_cfg": ARMS})
    foot_width = RewTerm(
        func=skill_mdp.step_width_l2,
        weight=-8.0,
        params={"asset_cfg": FOOT_BODIES, "target_width": 0.21},
    )
    knee_crossing = RewTerm(
        func=skill_mdp.bilateral_lateral_separation_l2,
        weight=-8.0,
        params={"asset_cfg": SHANK_BODIES, "minimum_separation": 0.105},
    )
    arm_torso_clearance = RewTerm(
        func=skill_mdp.bilateral_lateral_separation_l2,
        weight=-2.0,
        params={"asset_cfg": FOREARM_BODIES, "minimum_separation": 0.34},
    )
    contact_foot_alignment = RewTerm(
        func=skill_mdp.contact_foot_alignment_l2,
        weight=-2.0,
        params={"sensor_cfg": FEET, "asset_cfg": FOOT_BODIES},
    )


@configclass
class NaturalLocomotionRewardsCfg(LocomotionRewardsCfg):
    """G1-like gait shaping without prescribing a fixed motion trajectory."""

    arm_posture = RewTerm(func=mdp.joint_deviation_l1, weight=-0.08, params={"asset_cfg": ARMS})
    foot_width = RewTerm(
        func=skill_mdp.step_width_l2,
        weight=-5.0,
        params={"asset_cfg": FOOT_BODIES, "target_width": 0.19},
    )
    foot_crossing = RewTerm(
        func=skill_mdp.bilateral_lateral_separation_l2,
        weight=-12.0,
        params={"asset_cfg": FOOT_BODIES, "minimum_separation": 0.10},
    )
    knee_crossing = RewTerm(
        func=skill_mdp.bilateral_lateral_separation_l2,
        weight=-10.0,
        params={"asset_cfg": SHANK_BODIES, "minimum_separation": 0.09},
    )
    arm_torso_clearance = RewTerm(
        func=skill_mdp.bilateral_lateral_separation_l2,
        weight=-1.5,
        params={"asset_cfg": FOREARM_BODIES, "minimum_separation": 0.32},
    )
    contact_foot_alignment = RewTerm(
        func=skill_mdp.contact_foot_alignment_l2,
        weight=-1.5,
        params={"sensor_cfg": FEET, "asset_cfg": FOOT_BODIES},
    )
    single_support = RewTerm(
        func=skill_mdp.single_support_reward,
        weight=0.30,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )


@configclass
class GaitWalkRewardsCfg(NaturalLocomotionRewardsCfg):
    """Walking rewards that directly reject the short-step/stiff-arm solution."""

    velocity_tracking = RewTerm(
        func=skill_mdp.gated_velocity_tracking_exp,
        weight=3.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16), "minimum_height": 0.64},
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.34},
    )
    feet_slide = RewTerm(
        func=locomotion_mdp.feet_slide,
        weight=-0.80,
        params={"sensor_cfg": FEET, "asset_cfg": FOOT_BODIES},
    )
    stride_length = RewTerm(
        func=skill_mdp.fore_aft_foot_separation_exp,
        weight=1.75,
        params={
            "asset_cfg": FOOT_BODIES, "sensor_cfg": FEET, "command_name": "base_velocity",
            "base_target": 0.16, "speed_gain": 0.24, "maximum_target": 0.44, "std": 0.08,
        },
    )
    swing_clearance = RewTerm(
        func=skill_mdp.foot_clearance_reward,
        weight=0.75,
        params={"sensor_cfg": FEET, "asset_cfg": FOOT_BODIES, "target_height": 0.085, "std": 0.045},
    )
    swing_timing = RewTerm(
        func=skill_mdp.gait_mode_duration_exp,
        weight=1.40,
        params={
            "sensor_cfg": FEET, "command_name": "base_velocity", "base_target": 0.42,
            "speed_gain": -0.08, "minimum_target": 0.28, "maximum_target": 0.40, "std": 0.08,
        },
    )
    touchdown_timing = RewTerm(
        func=skill_mdp.touchdown_air_time_exp,
        weight=1.20,
        params={
            "sensor_cfg": FEET, "command_name": "base_velocity", "base_target": 0.40,
            "speed_gain": -0.08, "minimum_target": 0.26, "maximum_target": 0.38, "std": 0.07,
        },
    )
    premature_touchdown = RewTerm(
        func=skill_mdp.premature_touchdown_l2,
        weight=-4.0,
        params={
            "sensor_cfg": FEET, "command_name": "base_velocity", "base_minimum": 0.34,
            "speed_gain": -0.06, "minimum_target": 0.22, "maximum_target": 0.30,
        },
    )
    walking_double_support = RewTerm(
        func=skill_mdp.bounded_double_support_exp,
        weight=0.35,
        params={
            "sensor_cfg": FEET, "command_name": "base_velocity",
            "target_duration": 0.06, "std": 0.03,
        },
    )
    swing_knee = RewTerm(
        func=skill_mdp.swing_knee_flexion_exp,
        weight=0.50,
        params={
            "knee_cfg": KNEES, "sensor_cfg": FEET, "command_name": "base_velocity",
            "base_target": 0.56, "speed_gain": 0.16, "maximum_target": 0.78, "std": 0.26,
        },
    )
    arm_swing = RewTerm(
        func=skill_mdp.contralateral_arm_swing_exp,
        weight=0.35,
        params={
            "foot_cfg": FOOT_BODIES, "forearm_cfg": FOREARM_BODIES,
            "command_name": "base_velocity", "coupling": 0.55, "std": 0.14,
        },
    )
    walking_flight = RewTerm(
        func=skill_mdp.walking_flight_phase,
        weight=-1.5,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )
    single_support = RewTerm(
        func=skill_mdp.single_support_reward,
        weight=0.25,
        params={"sensor_cfg": FEET, "command_name": "base_velocity"},
    )
    # Neutral-arm deviation previously suppressed the arm swing we now reward.
    arm_posture = RewTerm(func=mdp.joint_deviation_l1, weight=-0.02, params={"asset_cfg": ARMS})
    roll_pitch_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.16)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.016)
    acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-4.0e-7)
    joint_power = RewTerm(func=skill_mdp.joint_power_l1, weight=-1.2e-5)
    contact_foot_alignment = RewTerm(
        func=skill_mdp.contact_foot_alignment_l2,
        weight=-2.0,
        params={"sensor_cfg": FEET, "asset_cfg": FOOT_BODIES},
    )


@configclass
class NaturalRunRewardsCfg(NaturalLocomotionRewardsCfg):
    velocity_tracking = RewTerm(
        func=skill_mdp.gated_velocity_tracking_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.30), "minimum_height": 0.58},
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=0.8,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.28},
    )
    height = RewTerm(func=skill_mdp.base_height_exp, weight=1.0, params={"target_height": 0.81, "std": 0.16})
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-7.5e-7)
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.008)
    joint_power = RewTerm(func=skill_mdp.joint_power_l1, weight=-1.25e-5)


@configclass
class NaturalSprint4RewardsCfg(NaturalRunRewardsCfg):
    velocity_tracking = RewTerm(
        func=skill_mdp.gated_velocity_tracking_exp,
        weight=6.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.38), "minimum_height": 0.54},
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.22},
    )
    foot_clearance = RewTerm(
        func=skill_mdp.foot_clearance_reward,
        weight=0.35,
        params={"sensor_cfg": FEET, "asset_cfg": FOOT_BODIES, "target_height": 0.105, "std": 0.06},
    )
    height = RewTerm(func=skill_mdp.base_height_exp, weight=0.8, params={"target_height": 0.78, "std": 0.20})
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-8.0e-7)
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.12)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.006)
    joint_power = RewTerm(func=skill_mdp.joint_power_l1, weight=-1.25e-5)
    actuator_saturation = RewTerm(
        func=skill_mdp.actuator_saturation_l2,
        weight=-1.5,
        params={"velocity_margin": 0.85, "torque_margin": 0.80},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": NON_FOOT_CONTACTS, "threshold": 5.0},
    )
    foot_crossing = RewTerm(
        func=skill_mdp.bilateral_lateral_separation_l2,
        weight=-16.0,
        params={"asset_cfg": FOOT_BODIES, "minimum_separation": 0.095},
    )
    knee_crossing = RewTerm(
        func=skill_mdp.bilateral_lateral_separation_l2,
        weight=-14.0,
        params={"asset_cfg": SHANK_BODIES, "minimum_separation": 0.085},
    )


@configclass
class GaitSprint4RewardsCfg(GaitWalkRewardsCfg):
    """Fast-running rewards with explicit flight and reduced double support."""

    velocity_tracking = RewTerm(
        func=skill_mdp.gated_velocity_tracking_exp,
        weight=6.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.30), "minimum_height": 0.56},
    )
    height = RewTerm(func=skill_mdp.base_height_exp, weight=0.70, params={"target_height": 0.78, "std": 0.18})
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=0.50,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "threshold": 0.22},
    )
    stride_length = RewTerm(
        func=skill_mdp.fore_aft_foot_separation_exp,
        weight=1.0,
        params={
            "asset_cfg": FOOT_BODIES, "sensor_cfg": FEET, "command_name": "base_velocity",
            "base_target": 0.18, "speed_gain": 0.12, "maximum_target": 0.66, "std": 0.15,
        },
    )
    swing_clearance = RewTerm(
        func=skill_mdp.foot_clearance_reward,
        weight=0.85,
        params={"sensor_cfg": FEET, "asset_cfg": FOOT_BODIES, "target_height": 0.13, "std": 0.065},
    )
    swing_timing = RewTerm(
        func=skill_mdp.gait_mode_duration_exp,
        weight=0.60,
        params={
            "sensor_cfg": FEET, "command_name": "base_velocity", "base_target": 0.26,
            "speed_gain": -0.025, "minimum_target": 0.14, "maximum_target": 0.24, "std": 0.06,
        },
    )
    touchdown_timing = RewTerm(
        func=skill_mdp.touchdown_air_time_exp,
        weight=0.50,
        params={
            "sensor_cfg": FEET, "command_name": "base_velocity", "base_target": 0.25,
            "speed_gain": -0.022, "minimum_target": 0.14, "maximum_target": 0.23, "std": 0.055,
        },
    )
    premature_touchdown = RewTerm(
        func=skill_mdp.premature_touchdown_l2,
        weight=-2.0,
        params={
            "sensor_cfg": FEET, "command_name": "base_velocity", "base_minimum": 0.22,
            "speed_gain": -0.020, "minimum_target": 0.11, "maximum_target": 0.20,
        },
    )
    walking_double_support = None
    swing_knee = RewTerm(
        func=skill_mdp.swing_knee_flexion_exp,
        weight=0.60,
        params={
            "knee_cfg": KNEES, "sensor_cfg": FEET, "command_name": "base_velocity",
            "base_target": 0.62, "speed_gain": 0.12, "maximum_target": 1.10, "std": 0.30,
        },
    )
    arm_swing = RewTerm(
        func=skill_mdp.contralateral_arm_swing_exp,
        weight=0.45,
        params={
            "foot_cfg": FOOT_BODIES, "forearm_cfg": FOREARM_BODIES,
            "command_name": "base_velocity", "coupling": 0.65, "std": 0.17,
        },
    )
    walking_flight = None
    running_flight = RewTerm(
        func=skill_mdp.running_flight_phase_reward,
        weight=0.55,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "minimum_command_speed": 2.2},
    )
    fast_double_support = RewTerm(
        func=skill_mdp.fast_double_support_phase,
        weight=-0.70,
        params={"sensor_cfg": FEET, "command_name": "base_velocity", "minimum_command_speed": 2.2},
    )
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.06)
    roll_pitch_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.09)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.006)
    acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    joint_power = RewTerm(func=skill_mdp.joint_power_l1, weight=-1.25e-5)
    actuator_saturation = RewTerm(
        func=skill_mdp.actuator_saturation_l2,
        weight=-1.5,
        params={"velocity_margin": 0.85, "torque_margin": 0.80},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": NON_FOOT_CONTACTS, "threshold": 5.0},
    )


@configclass
class RunCurriculumCfg:
    speed = CurrTerm(
        func=skill_mdp.run_speed_curriculum,
        params={
            "command_name": "base_velocity",
            "minimum_speed": 0.30,
            "initial_max_speed": 1.20,
            "final_max_speed": 2.60,
            "ramp_steps": 48000,
        },
    )


@configclass
class Sprint4CurriculumCfg:
    speed = CurrTerm(
        func=skill_mdp.run_speed_curriculum,
        params={
            "command_name": "base_velocity",
            "minimum_speed": 0.50,
            "initial_max_speed": 2.80,
            "final_max_speed": 4.20,
            "ramp_steps": 60000,
        },
    )


@configclass
class GaitSprint4CurriculumCfg:
    speed = CurrTerm(
        func=skill_mdp.run_speed_curriculum,
        params={
            "command_name": "base_velocity",
            "minimum_speed": 0.80,
            "initial_max_speed": 1.50,
            "final_max_speed": 4.20,
            "ramp_steps": 55000,
        },
    )


@configclass
class JumpRewardsCfg:
    sequence = RewTerm(func=skill_mdp.jump_sequence_reward, weight=4.0, params={"sensor_cfg": FEET})
    horizontal_stability = RewTerm(func=skill_mdp.planar_still_exp, weight=1.0, params={"std": 0.55})
    limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.75)
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-8.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.004)


@configclass
class StandingTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    low_base = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.55})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.90})
    torso_or_head_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso", "head"]), "threshold": 5.0},
    )


@configclass
class RecoveryTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=skill_mdp.getup_success_termination)


@configclass
class JumpTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=skill_mdp.jump_success_termination)
    head_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["head"]), "threshold": 8.0},
    )


@configclass
class HumanoidStandEnvCfg(ManagerBasedRLEnvCfg):
    scene: HumanoidSceneCfg = HumanoidSceneCfg(num_envs=4096, env_spacing=2.2)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: ZeroCommandsCfg = ZeroCommandsCfg()
    events: BaseEventsCfg = BaseEventsCfg()
    rewards: StandRewardsCfg = StandRewardsCfg()
    terminations: StandingTerminationsCfg = StandingTerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (3.0, 3.0, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.8)


@configclass
class HumanoidTurnStandingEnvCfg(HumanoidStandEnvCfg):
    commands: TurnStandingCommandsCfg = TurnStandingCommandsCfg()
    rewards: TurnStandingRewardsCfg = TurnStandingRewardsCfg()


@configclass
class HumanoidGetUpDiscoveryEnvCfg(HumanoidStandEnvCfg):
    events: RecoveryEventsCfg = RecoveryEventsCfg()
    rewards: RecoveryRewardsCfg = RecoveryRewardsCfg()
    terminations: RecoveryTerminationsCfg = RecoveryTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 8.0
        self.actions.joint_pos.scale = 0.70


@configclass
class HumanoidGetUpRefineEnvCfg(HumanoidGetUpDiscoveryEnvCfg):
    rewards: RecoveryRefineRewardsCfg = RecoveryRefineRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.actions.joint_pos.scale = 0.35


@configclass
class HumanoidWalkEnvCfg(HumanoidStandEnvCfg):
    commands: WalkCommandsCfg = WalkCommandsCfg()
    events: LocomotionEventsCfg = LocomotionEventsCfg()
    rewards: LocomotionRewardsCfg = LocomotionRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.actions.joint_pos.scale = 0.45


@configclass
class HumanoidWalkTurnEnvCfg(HumanoidWalkEnvCfg):
    commands: WalkTurnCommandsCfg = WalkTurnCommandsCfg()


@configclass
class HumanoidRunFastEnvCfg(HumanoidWalkEnvCfg):
    commands: RunFastCommandsCfg = RunFastCommandsCfg()
    rewards: RunRewardsCfg = RunRewardsCfg()
    curriculum: RunCurriculumCfg = RunCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = 0.55


@configclass
class HumanoidSprint4EnvCfg(HumanoidRunFastEnvCfg):
    commands: Sprint4CommandsCfg = Sprint4CommandsCfg()
    rewards: Sprint4RewardsCfg = Sprint4RewardsCfg()
    curriculum: Sprint4CurriculumCfg = Sprint4CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 16.0
        self.actions.joint_pos.scale = 0.65


@configclass
class HumanoidNaturalStandEnvCfg(HumanoidStandEnvCfg):
    rewards: NaturalStandRewardsCfg = NaturalStandRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = 0.25


@configclass
class HumanoidNaturalWalkEnvCfg(HumanoidWalkEnvCfg):
    rewards: NaturalLocomotionRewardsCfg = NaturalLocomotionRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = 0.35


@configclass
class HumanoidNaturalWalkFixed1EnvCfg(HumanoidNaturalWalkEnvCfg):
    commands: FixedWalk1CommandsCfg = FixedWalk1CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep evaluation repeatable: no random velocity kicks during playback.
        self.events.push = None


@configclass
class HumanoidGaitWalkEnvCfg(HumanoidNaturalWalkEnvCfg):
    commands: GaitWalkCommandsCfg = GaitWalkCommandsCfg()
    events: GaitWalkEventsCfg = GaitWalkEventsCfg()
    rewards: GaitWalkRewardsCfg = GaitWalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 22.0
        self.actions.joint_pos.scale = 0.38


@configclass
class HumanoidGaitSprint4EnvCfg(HumanoidSprint4EnvCfg):
    commands: GaitSprint4CommandsCfg = GaitSprint4CommandsCfg()
    events: GaitRunEventsCfg = GaitRunEventsCfg()
    rewards: GaitSprint4RewardsCfg = GaitSprint4RewardsCfg()
    curriculum: GaitSprint4CurriculumCfg = GaitSprint4CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 18.0
        self.actions.joint_pos.scale = 0.58


@configclass
class HumanoidNaturalRunEnvCfg(HumanoidRunFastEnvCfg):
    rewards: NaturalRunRewardsCfg = NaturalRunRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = 0.45


@configclass
class HumanoidNaturalSprint4EnvCfg(HumanoidSprint4EnvCfg):
    rewards: NaturalSprint4RewardsCfg = NaturalSprint4RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = 0.55

@configclass
class HumanoidRunTurnEnvCfg(HumanoidRunFastEnvCfg):
    commands: RunTurnCommandsCfg = RunTurnCommandsCfg()


@configclass
class HumanoidJumpEnvCfg(HumanoidStandEnvCfg):
    observations: JumpObservationsCfg = JumpObservationsCfg()
    rewards: JumpRewardsCfg = JumpRewardsCfg()
    terminations: JumpTerminationsCfg = JumpTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5.0
        self.actions.joint_pos.scale = 0.70


def _make_play(base_cls):
    @configclass
    class PlayCfg(base_cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 1
            self.observations.policy.enable_corruption = False
            if hasattr(self.commands, "base_velocity"):
                self.commands.base_velocity.debug_vis = True

    return PlayCfg


HumanoidStandEnvCfg_PLAY = _make_play(HumanoidStandEnvCfg)
HumanoidTurnStandingEnvCfg_PLAY = _make_play(HumanoidTurnStandingEnvCfg)
HumanoidGetUpDiscoveryEnvCfg_PLAY = _make_play(HumanoidGetUpDiscoveryEnvCfg)
HumanoidGetUpRefineEnvCfg_PLAY = _make_play(HumanoidGetUpRefineEnvCfg)
HumanoidWalkEnvCfg_PLAY = _make_play(HumanoidWalkEnvCfg)
HumanoidWalkTurnEnvCfg_PLAY = _make_play(HumanoidWalkTurnEnvCfg)
HumanoidRunFastEnvCfg_PLAY = _make_play(HumanoidRunFastEnvCfg)
HumanoidSprint4EnvCfg_PLAY = _make_play(HumanoidSprint4EnvCfg)
HumanoidRunTurnEnvCfg_PLAY = _make_play(HumanoidRunTurnEnvCfg)
HumanoidJumpEnvCfg_PLAY = _make_play(HumanoidJumpEnvCfg)
HumanoidNaturalStandEnvCfg_PLAY = _make_play(HumanoidNaturalStandEnvCfg)
HumanoidNaturalWalkEnvCfg_PLAY = _make_play(HumanoidNaturalWalkEnvCfg)
HumanoidNaturalWalkFixed1EnvCfg_PLAY = _make_play(HumanoidNaturalWalkFixed1EnvCfg)
HumanoidGaitWalkEnvCfg_PLAY = _make_play(HumanoidGaitWalkEnvCfg)
HumanoidGaitSprint4EnvCfg_PLAY = _make_play(HumanoidGaitSprint4EnvCfg)
HumanoidNaturalRunEnvCfg_PLAY = _make_play(HumanoidNaturalRunEnvCfg)
HumanoidNaturalSprint4EnvCfg_PLAY = _make_play(HumanoidNaturalSprint4EnvCfg)
