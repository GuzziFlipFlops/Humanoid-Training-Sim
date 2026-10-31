"""Isolated recovery task; never imports the old get-up rewards or mutates locomotion."""
from pathlib import Path
import json
import importlib.util
import sys
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ManagerTermBase, RewardTermCfg, ObservationTermCfg, EventTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul
from humanoid_tasks.curriculum.env_cfg import HumanoidStandEnvCfg, ObservationsCfg
from geometry import Geometry
from reward_core import RecoveryReward, Limits, terminal_rehearsal_score, terminal_fall_mask
from telemetry import Telemetry
from assistance import vertical_assist_force

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('recovery_ankle_envelope', HERE.parent / 'guarded_run_v4/contact_state.py')
_ankle_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ankle_module)
differential_cost = _ankle_module.differential_cost


def zero(env): return torch.zeros(env.num_envs, device=env.device)


def terminal_rehearsal(env):
    term = env._recovery_reward
    if term.last_frame is None:
        return zero(env)
    return terminal_rehearsal_score(term.last_frame) * (env._recovery_groups == 4)


def terminal_motion_cost(env):
    """Strongly suppress drift/thrashing only while rehearsing a high stand."""
    term = env._recovery_reward
    if term.last_frame is None:
        return zero(env)
    f = term.last_frame
    normalized = ((f.planar_speed / .08).square() + (f.vertical_speed / .08).square()
                  + (f.angular_speed / .25).square() + (f.joint_speed_rms / .8).square())
    active = (env._recovery_groups == 4) & (f.height > .70)
    # log1p bounds magnitude without a flat saturated region.  The v4-era
    # standing policy spins near 3 rad/s; a hard clamp had zero gradient there.
    return torch.log1p(normalized) * active * f.contacts_fresh * f.safe


def terminal_fall(env):
    """Immediate negative outcome for falling from the terminal-hold reset."""
    if not hasattr(env, '_recovery_groups'):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    frame = get_state(env).measure(full_geometry=False)
    return terminal_fall_mask(frame, env._recovery_groups, env.episode_length_buf)


def terminal_failure_cost(env):
    """One event penalty for an actual terminal-rehearsal fall.

    A strict-hold timeout is deliberately *not* a failure event.  The previous
    version made every upright-but-still-moving environment receive the full
    recovery failure penalty together at the first episode boundary.  That
    synchronized return discontinuity erased the standing bootstrap.
    """
    if not hasattr(env, '_recovery_groups'):
        return zero(env)
    terminal_group = env._recovery_groups == 4
    # Reward terms are integrated by step_dt, so divide here to make the
    # configured weight an event cost rather than a per-second cost.
    return (env.reset_terminated & terminal_group).float() / env.step_dt


class ResetAwarePositionAction(JointPositionAction):
    """One 20 ms motor-hold tick after reset, before the first policy action.

    This produces fresh contact data for start validation without evaluating a
    random PPO action as if it were the reset pose. No gravity/body assistance.
    """
    def process_actions(self, actions):
        super().process_actions(actions)
        if hasattr(self._env, '_recovery_reward'):
            term = self._env._recovery_reward
            pending = term.pending
            self._processed_actions[pending] = self._asset.data.joint_pos[pending][:, self._joint_ids]

            # Upright rehearsal samples do not need to rediscover balance.
            # Route them straight to the separately audited controller instead
            # of allowing one out-of-distribution recovery action to kick them.
            term.handoff |= self._env._recovery_groups == 4

            # Once recovery reaches a conservative two-foot upright basin,
            # hand control to the separately audited stationary controller.
            # This latch prevents PPO from collecting a rise bonus and then
            # walking, thrashing, or falling after the nominal get-up.
            if term.last_frame is not None:
                f = term.last_frame
                ready = (f.contacts_fresh & f.safe & (f.height >= .78) & (f.height <= .94)
                         & (f.pelvis_tilt < .26) & (f.torso_tilt < .32)
                         & (f.foot_load >= .10).all(1) & (f.foot_load.sum(1) >= .75)
                         & (f.planar_speed < .25) & (f.vertical_speed.abs() < .18)
                         & (f.angular_speed < 1.5) & (f.nonfoot_force < 10.))
                term.handoff |= ready
                active = term.handoff & ~pending
                if active.any():
                    target = self._asset.data.default_joint_pos[active][:, self._joint_ids].clone()
                    gravity = self._asset.data.projected_gravity_b[active]
                    omega = self._asset.data.root_ang_vel_b[active]
                    pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
                    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
                    names = self._asset.joint_names[self._joint_ids]
                    for side in ('left', 'right'):
                        p = names.index(side + '_ankle_pitch')
                        r = names.index(side + '_ankle_roll')
                        target[:, p] += (.65 * pitch + .16 * omega[:, 1]).clamp(-.18, .18)
                        target[:, r] += (.65 * roll + .16 * omega[:, 0]).clamp(-.12, .12)
                    self._processed_actions[active] = target

        # HoST-inspired discovery aid: reduce effective weight near the floor,
        # taper to zero before the strict standing region. Global +Z force is
        # applied at the pelvis COM and is never enabled in promotion evals.
        ratio = float(self._env.cfg.recovery_assist)
        body_id = self._asset.body_names.index('pelvis')
        height = self._asset.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        force = torch.zeros((self.num_envs, 1, 3), device=self.device)
        force[:, 0, 2] = vertical_assist_force(height, ratio)
        torque = torch.zeros_like(force)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=force, torques=torque, body_ids=[body_id], is_global=True)
        self._env._assist_force_z = force[:, 0, 2]


def get_state(env):
    if not hasattr(env, '_recovery_telemetry'): env._recovery_telemetry = Telemetry(env)
    return env._recovery_telemetry


def critic_geometry(env):
    # Privileged critic only; the actor does not depend on absolute pose/COM.
    d = env.scene['robot'].data
    assist = torch.full((env.num_envs, 1), float(env.cfg.recovery_assist), device=env.device)
    return torch.cat((d.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3],
                      d.root_lin_vel_b, d.body_quat_w[:, [env.scene['robot'].body_names.index('torso')]].flatten(1),
                      assist), dim=1)


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
        def __post_init__(self):
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
    if env.cfg.recovery_forced_group >= 0:
        groups = torch.full((n,), env.cfg.recovery_forced_group, device=env.device, dtype=torch.long)
    elif hasattr(env, '_forced_group'):
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
    q = bank['q'][choices].clone()
    # Terminal rehearsal must use the same canonical standing manifold as the
    # validated standing bootstrap.  v4 sampled "standing" snapshots from an
    # older recovery controller; those snapshots differed across hips, knees,
    # ankles, waist and arms and made the bootstrap thrash immediately.
    standing = groups == 4
    if standing.any():
        root[standing] = 0.
        root[standing, 2] = .84
        root[standing, 3] = 1.
        q[standing] = env.scene['robot'].data.default_joint_pos[env_ids[standing]]
    # Free heading variation; ground geometry and collision checks are yaw invariant.
    yaw = (torch.rand(n, device=env.device) - .5) * 2 * torch.pi
    yaw_q = torch.zeros((n, 4), device=env.device)
    yaw_q[:, 0], yaw_q[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    root[:, :3] = quat_apply(yaw_q, root[:, :3]) + env.scene.env_origins[env_ids]
    root[:, 3:7] = quat_mul(yaw_q, root[:, 3:7])
    root[:, 7:] = 0.
    robot = env.scene['robot']
    robot.write_root_state_to_sim(root, env_ids)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    robot.set_joint_position_target(q, env_ids=env_ids)
    env._recovery_groups[env_ids] = groups
    env._recovery_pose_ids[env_ids] = choices


class RecoveryTerm(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # v5's 15-point cost occurred at every unsuccessful episode ending.
        # Moving and failing early therefore incurred it more often than lying
        # still until timeout. Promotion is already decided by strict external
        # evaluation, so discovery uses dense floor/rise shaping instead.
        failure_penalty = 0. if env.cfg.recovery_stage == 'discovery' else 3.
        self.core = RecoveryReward(env.num_envs, env.device, env.step_dt,
                                   Limits(progress_budget=4., failure_penalty=failure_penalty))
        self.pending = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.last_frame = None
        self.last_out = None
        self.counts = torch.zeros((5, 3), device=env.device)  # ended / strict success / post-success loss
        self.best_height = torch.zeros(env.num_envs, device=env.device)
        self.totals = torch.zeros(5, device=env.device)
        self.last_tick = -1
        self.capture = False
        self.captured = None
        self.handoff = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._recovery_reward = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.pending[ids] = True
        self.best_height[ids] = 0.
        # Observation is evaluated after reset; no prior episode anchor memory.
        self.core.hold_steps[ids] = 0
        self.core.ever_success[ids] = False
        self.core.invalidated[ids] = False
        self.handoff[ids] = False
        return {}

    def __call__(self, env):
        if env.common_step_counter == self.last_tick: return self.last_out.rate
        sensor = get_state(env)
        # unsafe_state has already checked this exact pre-reset physics state.
        f = sensor.measure(full_geometry=False)
        self.last_frame = f
        if self.pending.any():
            ids = self.pending.nonzero().flatten()
            valid = (sensor.last_penetration < .015) & (sensor.last_floor > -.025)
            # Core initializes from measured post-reset state. All poses were
            # independently settled/audited; invalid runtime resets fail loudly.
            if not valid[ids].all():
                bad = ids[~valid[ids]]
                detail = dict(env_ids=bad.tolist(), groups=env._recovery_groups[bad].tolist(),
                    pose_ids=env._recovery_pose_ids[bad].tolist(), floor=sensor.last_floor[bad].tolist(),
                    penetration=sensor.last_penetration[bad].tolist(),
                    root=env.scene['robot'].data.root_state_w[bad].tolist(),
                    joints=env.scene['robot'].data.joint_pos[bad].tolist())
                (Path(env.cfg.log_dir)/'reset_failure.json').write_text(json.dumps(detail, indent=2))
                raise RuntimeError(f'Audited reset caused penetration: {len(bad)} envs; floor={detail["floor"][:8]}, depth={detail["penetration"][:8]}')
            self.core.reset(ids, f, env._recovery_groups, valid)
            # Group 4 is a terminal-hold rehearsal, not a lying recovery.  It
            # has its own dense score and actual-fall event below.  Feeding it
            # through RecoveryReward would label a safe strict-gate timeout as
            # the same failure as never rising from the floor.
            easy = ids[env._recovery_groups[ids] == 4]
            self.core.eligible[easy] = False
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
    penetration, floor = get_state(env).geometry_check()
    return ((penetration >= .015) | (floor <= -.025)
            | (d.root_lin_vel_w.norm(dim=1) > 8.) | (d.root_ang_vel_w.norm(dim=1) > 20.)
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


def collision_margin_cost(env):
    depth = get_state(env).last_penetration
    return ((depth - .003).clamp_min(0.) / .012).square().clamp_max(4.)


def rise_speed_cost(env):
    d = env.scene['robot'].data
    return (d.root_lin_vel_w[:, 2].abs() - 1.).clamp_min(0.).square() + .1 * (
        d.root_ang_vel_b.norm(dim=1) - 4.).clamp_min(0.).square()


def discovery_rise_reward(env):
    """Dense encouragement to leave the floor and become upright."""
    term = env._recovery_reward
    if term.last_frame is None: return zero(env)
    f = term.last_frame
    active = (env._recovery_groups < 4) & ~term.handoff & f.contacts_fresh & f.safe
    height = ((f.height - .12) / (.78 - .12)).clamp(0., 1.)
    upright = .5 * (1. + torch.cos(f.torso_tilt.clamp(0., torch.pi)))
    # Height gates the orientation credit: rolling upright on the floor cannot
    # farm this term. It stops when the audited standing controller takes over.
    return active * height * (.25 + .75 * upright)


def discovery_upward_velocity_reward(env):
    """Reward useful upward motion only before the terminal standing basin."""
    term = env._recovery_reward
    if term.last_frame is None: return zero(env)
    f = term.last_frame
    active = ((env._recovery_groups < 4) & ~term.handoff & (f.height < .76)
              & f.contacts_fresh & f.safe)
    return active * f.vertical_speed.clamp(0., 1.)


def floor_stall_cost(env):
    """Small per-second cost for remaining at the reset floor height."""
    term = env._recovery_reward
    if term.last_frame is None: return zero(env)
    f = term.last_frame
    active = (env._recovery_groups < 4) & ~term.handoff & f.contacts_fresh & f.safe
    # Unlike an episode-end penalty, this decreases immediately as the pelvis
    # rises and cannot make attempting recovery less attractive than waiting.
    return active * ((.22 - f.height) / .12).clamp(0., 1.)


@configclass
class Rewards:
    recovery = RewardTermCfg(func=RecoveryTerm, weight=1.)
    discovery_rise = RewardTermCfg(func=discovery_rise_reward, weight=8.)
    discovery_upward_velocity = RewardTermCfg(func=discovery_upward_velocity_reward, weight=2.)
    floor_stall = RewardTermCfg(func=floor_stall_cost, weight=-.25)
    terminal_rehearsal = RewardTermCfg(func=terminal_rehearsal, weight=4.0)
    terminal_motion = RewardTermCfg(func=terminal_motion_cost, weight=-.60)
    terminal_failure = RewardTermCfg(func=terminal_failure_cost, weight=-15.)
    limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-.5)
    target_limits = RewardTermCfg(func=unreachable, weight=-.25)
    ankle = RewardTermCfg(func=AnkleCost, weight=-1.)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-.001)
    power = RewardTermCfg(func=power, weight=-2e-6)
    collision_margin = RewardTermCfg(func=collision_margin_cost, weight=-2.)
    rise_speed = RewardTermCfg(func=rise_speed_cost, weight=-.5)


@configclass
class ProbeRewards:
    zero = RewardTermCfg(func=zero, weight=1.)


@configclass
class Endings:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    unsafe = TerminationTermCfg(func=unsafe_state)
    terminal_fall = TerminationTermCfg(func=terminal_fall)


@configclass
class RecoveryCfg(HumanoidStandEnvCfg):
    recovery_groups: int = 1
    recovery_split: int = 0
    recovery_assist: float = 0.
    recovery_forced_group: int = -1
    recovery_stage: str = 'discovery'


def make_cfg(count=64, stage='discovery', groups=1, split=0, probe=False, assist=0., forced_group=-1,
             self_collisions=True):
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
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = self_collisions
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 12
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    if not 0. <= assist <= .25: raise ValueError('Training assistance must be within [0,.25] body weight')
    if forced_group not in (-1, 0, 1, 2, 3, 4): raise ValueError('Invalid forced pose group')
    cfg.recovery_groups, cfg.recovery_split = groups, split
    cfg.recovery_assist, cfg.recovery_forced_group = assist, forced_group
    cfg.recovery_stage = stage
    cfg.curriculum = None
    # Hold rehearsal only needs the seven-second audited dwell plus margin.
    # Recovery retains enough time to rise and complete the same dwell.
    cfg.episode_length_s = 10. if stage == 'hold' else 20.
    # Recovery needs large coordinated hip/knee excursions. The failed v5
    # candidate reduced this to .25 rad and never left the floor.
    cfg.actions.joint_pos.scale = .70
    if not probe: cfg.actions.joint_pos.class_type = ResetAwarePositionAction
    cfg.observations = RecoveryObservations()
    cfg.terminations = Endings()
    cfg.rewards = ProbeRewards() if probe else Rewards()
    if not probe and stage == 'discovery':
        # Style costs must not suppress the first useful recovery motion.
        # Penetration and severe impact remain hard terminations.
        cfg.rewards.limits.weight = -.10
        cfg.rewards.target_limits.weight = -.05
        cfg.rewards.ankle.weight = -.05
        cfg.rewards.action_rate.weight = -.0001
        cfg.rewards.power.weight = -2e-7
        cfg.rewards.collision_margin.weight = -.20
        cfg.rewards.rise_speed.weight = -.05
    elif not probe and stage == 'refine':
        cfg.rewards.discovery_upward_velocity.weight = .5
        cfg.rewards.floor_stall.weight = -.15
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
