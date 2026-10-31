"""Isolated recovery task; never imports the old get-up rewards or mutates locomotion."""
from pathlib import Path
import json
import importlib.util
import sys
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, ObservationTermCfg, EventTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul
from humanoid_tasks.curriculum.env_cfg import HumanoidStandEnvCfg, ObservationsCfg
from geometry import Geometry
from reward_core import RecoveryReward
from telemetry import Telemetry

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('recovery_ankle_envelope', HERE.parent / 'guarded_run_v4/contact_state.py')
_ankle_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ankle_module)
differential_cost = _ankle_module.differential_cost


def zero(env): return torch.zeros(env.num_envs, device=env.device)


def get_state(env):
    if not hasattr(env, '_recovery_telemetry'): env._recovery_telemetry = Telemetry(env)
    return env._recovery_telemetry


def critic_geometry(env):
    # Privileged critic only; the actor does not depend on absolute pose/COM.
    d = env.scene['robot'].data
    return torch.cat((d.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3],
                      d.root_lin_vel_b, d.body_quat_w[:, [env.scene['robot'].body_names.index('torso')]].flatten(1)), dim=1)


def critic_memory(env):
    if hasattr(env, '_recovery_reward') and env._recovery_reward.last_frame is not None:
        term = env._recovery_reward
        memory = term.core.state_observation(term.last_frame)
        return torch.where(term.pending[:, None], 0., memory)
    return torch.zeros((env.num_envs, 13), device=env.device)


@configclass
class RecoveryObservations(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        base_lin_vel = None  # not directly available from an IMU on hardware
        commands = None
        feet_contacts = None  # actor uses IMU/encoder history, not privileged contacts
        def __post_init__(self):
            self.history_length = 4
            self.flatten_history_dim = True
            self.enable_corruption = False
    @configclass
    class CriticCfg(ObservationsCfg.PolicyCfg):
        geometry = ObservationTermCfg(func=critic_geometry)
        memory = ObservationTermCfg(func=critic_memory)
        def __post_init__(self): self.enable_corruption = False
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


def reset_bank(env, env_ids):
    if env_ids is None: env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, '_recovery_bank'):
        data = torch.load(HERE / 'assets/pose_bank.pt', map_location=env.device, weights_only=False)
        if not data.get('validated'): raise RuntimeError('Pose bank not validated')
        robot_names = env.scene['robot'].joint_names
        order = [data['joint_names'].index(n) for n in robot_names]
        data['q'] = data['q'][:, order]
        env._recovery_bank = data
        env._recovery_groups = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._recovery_pose_ids = torch.zeros_like(env._recovery_groups)
    bank = env._recovery_bank
    n = len(env_ids)
    # 15% stationary/easy rehearsal improves the tail hold but NEVER enters
    # the measured lying-recovery denominator.
    if hasattr(env, '_forced_group'):
        groups = torch.full((n,), env._forced_group, device=env.device, dtype=torch.long)
    else:
        groups = torch.randint(env.cfg.recovery_groups, (n,), device=env.device)
        groups[torch.rand(n, device=env.device) < .15] = 4
    choices = torch.empty(n, device=env.device, dtype=torch.long)
    for group in range(5):
        mask = groups == group
        pool = torch.where((bank['group'] == group) & (bank['split'] == env.cfg.recovery_split))[0]
        if pool.numel() == 0: raise RuntimeError(f'Empty audited pose group {group}, split {env.cfg.recovery_split}')
        choices[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), device=env.device)]
    root = bank['root'][choices].clone()
    # Free heading variation; ground geometry and collision checks are yaw invariant.
    yaw = (torch.rand(n, device=env.device) - .5) * 2 * torch.pi
    yaw_q = torch.zeros((n, 4), device=env.device)
    yaw_q[:, 0], yaw_q[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    root[:, :3] = quat_apply(yaw_q, root[:, :3]) + env.scene.env_origins[env_ids]
    root[:, 3:7] = quat_mul(yaw_q, root[:, 3:7])
    root[:, 7:] = 0.
    robot = env.scene['robot']
    robot.write_root_state_to_sim(root, env_ids)
    robot.write_joint_state_to_sim(bank['q'][choices], torch.zeros_like(bank['q'][choices]), env_ids=env_ids)
    robot.set_joint_position_target(bank['q'][choices], env_ids=env_ids)
    env._recovery_groups[env_ids] = groups
    env._recovery_pose_ids[env_ids] = choices


class RecoveryTerm(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.core = RecoveryReward(env.num_envs, env.device, env.step_dt)
        self.pending = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.last_frame = None
        self.last_out = None
        self.counts = torch.zeros((5, 3), device=env.device)  # ended / strict success / post-success loss
        self.best_height = torch.zeros(env.num_envs, device=env.device)
        self.totals = torch.zeros(5, device=env.device)
        self.last_tick = -1
        self.capture = False
        self.captured = None
        env._recovery_reward = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.pending[ids] = True
        self.best_height[ids] = 0.
        # Observation is evaluated after reset; no prior episode anchor memory.
        self.core.hold_steps[ids] = 0
        self.core.ever_success[ids] = False
        self.core.invalidated[ids] = False
        return {}

    def __call__(self, env):
        if env.common_step_counter == self.last_tick: return self.last_out.rate
        sensor = get_state(env)
        f = sensor.measure(full_geometry=True)
        self.last_frame = f
        if self.pending.any():
            ids = self.pending.nonzero().flatten()
            valid = (sensor.last_penetration < .015) & (sensor.last_floor > -.025)
            # Core initializes from measured post-reset state. All poses were
            # independently settled/audited; invalid runtime resets fail loudly.
            if not valid[ids].all(): raise RuntimeError('Audited reset caused penetration; stop the training')
            self.core.reset(ids, f, env._recovery_groups, valid)
            easy = ids[env._recovery_groups[ids] == 4]
            self.core.eligible[easy] = True  # reward terminal-standing rehearsal; report separately
            self.pending[ids] = False
        out = self.core.update(f, env.common_step_counter, env.reset_buf)
        self.last_tick = env.common_step_counter
        self.last_out = out
        if self.capture:
            from dataclasses import fields
            self.captured = {k.name: getattr(f, k.name).detach().cpu().numpy().copy() for k in fields(f)}
            self.captured.update(joint_pos=env.scene['robot'].data.joint_pos.detach().cpu().numpy().copy(),
                                 joint_vel=env.scene['robot'].data.joint_vel.detach().cpu().numpy().copy(),
                                 torque_estimate=env.scene['robot'].data.applied_torque.detach().cpu().numpy().copy(),
                                 penetration=sensor.last_penetration.detach().cpu().numpy().copy(),
                                 terminated=env.reset_terminated.detach().cpu().numpy().copy())
        self.best_height = torch.maximum(self.best_height, f.height)
        self.totals += torch.stack((out.progress.sum(), out.completion.sum(), out.hold_motion_cost.sum(),
                                   out.failure.sum(), out.stable.float().sum()))
        for group in range(5):
            ended = env.reset_buf & (env._recovery_groups == group)
            self.counts[group] += torch.stack((ended.float().sum(), (ended & out.currently_valid).float().sum(),
                                               (ended & out.invalidated_after_success).float().sum()))
        return out.rate


def unsafe_state(env):
    d = env.scene['robot'].data
    if not torch.isfinite(d.joint_pos).all() or not torch.isfinite(d.root_state_w).all():
        raise RuntimeError('Nonfinite recovery physics; abort rather than train through NaNs')
    head = env.scene.sensors['ground_head'].data.force_matrix_w
    if head is None: raise RuntimeError('Head-ground contact signal unavailable')
    # Recovery may use hands/knees, but not a violent head impact or leaving
    # the isolated floor. Fixed robot mass is 38 kg in this candidate.
    return ((d.root_lin_vel_w.norm(dim=1) > 8.) | (d.root_ang_vel_w.norm(dim=1) > 20.)
            | (head[:, 0, 0].norm(dim=1) > 2. * 38. * 9.81)
            | ((d.root_pos_w[:, :2] - env.scene.env_origins[:, :2]).norm(dim=1) > 1.25))


def unreachable(env):
    d = env.scene['robot'].data
    return (((d.soft_joint_pos_limits[..., 0] - d.joint_pos_target).clamp_min(0.)
            + (d.joint_pos_target - d.soft_joint_pos_limits[..., 1]).clamp_min(0.)) / .5).square().clamp_max(4.).mean(1)


class AnkleCost(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        names = ['left_ankle_pitch', 'left_ankle_roll', 'right_ankle_pitch', 'right_ankle_roll']
        self.ids = [self.robot.joint_names.index(n) for n in names]
        assert [self.robot.joint_names[i] for i in self.ids] == names
    def __call__(self, env):
        d = self.robot.data
        return differential_cost(d.applied_torque[:, self.ids].reshape(-1, 2, 2),
                                 d.joint_vel[:, self.ids].reshape(-1, 2, 2))


def power(env):
    d = env.scene['robot'].data
    return (d.applied_torque * d.joint_vel).abs().sum(1)


@configclass
class Rewards:
    recovery = RewardTermCfg(func=RecoveryTerm, weight=1.)
    limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-.5)
    target_limits = RewardTermCfg(func=unreachable, weight=-.25)
    ankle = RewardTermCfg(func=AnkleCost, weight=-1.)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-.001)
    power = RewardTermCfg(func=power, weight=-2e-6)


@configclass
class ProbeRewards:
    zero = RewardTermCfg(func=zero, weight=1.)


@configclass
class Endings:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    unsafe = TerminationTermCfg(func=unsafe_state)


@configclass
class RecoveryCfg(HumanoidStandEnvCfg):
    recovery_groups: int = 1
    recovery_split: int = 0


def make_cfg(count=64, stage='discovery', groups=1, split=0, probe=False):
    cfg = RecoveryCfg()
    cfg.seed = 47
    cfg.scene.num_envs = count
    cfg.scene.env_spacing = 3.5
    # PhysX filtered contact views require one matching partner per environment.
    # Use a local static floor; a global plane cannot provide this filter mapping.
    cfg.scene.ground = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Ground',
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., -.05)),
        spawn=sim_utils.CuboidCfg(size=(3.4, 3.4, .1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.95, dynamic_friction=.8, restitution=0.)))
    cfg.scene.robot.spawn.usd_path = str(HERE / 'assets/recovery.usda')
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 12
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    cfg.recovery_groups, cfg.recovery_split = groups, split
    cfg.curriculum = None
    cfg.episode_length_s = 25.
    cfg.actions.joint_pos.scale = .7  # separate policy, fixed through discovery/refinement
    cfg.observations = RecoveryObservations()
    cfg.terminations = Endings()
    cfg.rewards = ProbeRewards() if probe else Rewards()
    if not probe and stage == 'refine':
        cfg.rewards.action_rate.weight = -.005
        cfg.rewards.power.weight = -1e-5
    cfg.events.torso_mass = None
    cfg.events.torso_com = None
    cfg.events.actuator_gains = None
    cfg.events.reset_state = None
    cfg.events.physics_material.params.update(static_friction_range=(.9, 1.), dynamic_friction_range=(.75, .85),
                                               restitution_range=(0., 0.), num_buckets=8)
    cfg.events.reset_base.params['pose_range'] = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.events.reset_base.params['velocity_range'] = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.events.reset_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    if not probe:
        cfg.events.reset_base = EventTermCfg(func=reset_bank, mode='reset')
        cfg.events.reset_joints = None
    for name in Geometry().links:
        setattr(cfg.scene, 'ground_' + name, ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Robot/' + name, filter_prim_paths_expr=['{ENV_REGEX_NS}/Ground'],
            history_length=1, update_period=cfg.sim.dt, max_contact_data_count_per_prim=8))
    return cfg
