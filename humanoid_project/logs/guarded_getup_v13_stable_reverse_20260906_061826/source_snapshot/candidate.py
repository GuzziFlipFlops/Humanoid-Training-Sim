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


def unsafe_recovery_cost(env):
    """Event cost only for an objectively unsafe lying-recovery termination.

    A normal timeout is deliberately free, so safe attempts are not suppressed.
    This closes the v7 shortcut where crashing early avoided many seconds of
    floor-stall cost.
    """
    if not hasattr(env, '_recovery_groups'): return zero(env)
    lying_group = env._recovery_groups < 4
    return (env.reset_terminated & lying_group).float() / env.step_dt


class ResetAwarePositionAction(JointPositionAction):
    """One 20 ms motor-hold tick after reset, before the first policy action.

    This produces fresh contact data for start validation without evaluating a
    random PPO action as if it were the reset pose. No gravity/body assistance.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # A uniform +/-0.70-rad offset made deep flexion mechanically
        # unreachable: knee targets stopped near 1.06 rad although the joint
        # and the HumanUP reference require roughly 2.1-2.2 rad.  Use an
        # asymmetric normalized mapping: zero remains the safe default pose,
        # while +/-1 reaches the audited URDF limit (with a hard-stop margin).
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._default = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._lower = limits[..., 0] + .025
        self._upper = limits[..., 1] - .025
        if not ((self._lower < self._default) & (self._default < self._upper)).all():
            raise RuntimeError('Default recovery pose must lie strictly inside every joint limit')

    def normalized_for_target(self, target):
        """Convert physical joint targets to the bounded policy coordinate."""
        target = torch.maximum(torch.minimum(target, self._upper), self._lower)
        positive = (target - self._default) / (self._upper - self._default)
        negative = (target - self._default) / (self._default - self._lower)
        return torch.where(target >= self._default, positive, negative).clamp(-1., 1.)

    def process_actions(self, actions):
        bounded = actions.clamp(-1., 1.)
        self._raw_actions[:] = bounded
        positive = self._default + bounded * (self._upper - self._default)
        negative = self._default + bounded * (self._default - self._lower)
        self._processed_actions = torch.where(bounded >= 0., positive, negative)
        # No reset hold, upright handoff, posture controller, action masking,
        # or success latch: PPO owns all 22 targets from the first control tick.

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
            self.history_length = 4
            self.flatten_history_dim = True
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
    mode = env.cfg.recovery_start_mode
    # Five-degree reverse-curriculum states were checked with the exact URDF
    # box geometry; the Z values leave about 6 mm of ground clearance.  Small
    # increments avoid asking PPO to discover an entirely different maneuver
    # at the old 15-degree discontinuities.
    tilted = {
        'tilt1': (-1., .549), 'tilt2': (-2., .551),
        'tilt3': (-3., .552), 'tilt4': (-4., .553),
        'tilt5': (-5., .554), 'tilt10': (-10., .556),
        'tilt15': (-15., .553), 'tilt20': (-20., .547),
        'tilt25': (-25., .536), 'tilt30': (-30., .521),
        'tilt35': (-35., .503), 'tilt40': (-40., .481),
        'tilt45': (-45., .455), 'tilt50': (-50., .425),
        'tilt55': (-55., .393), 'tilt60': (-60., .357),
        'tilt65': (-65., .319), 'tilt70': (-70., .278),
        'tilt75': (-75., .236), 'tilt80': (-80., .191),
        'tilt85': (-85., .145), 'tucked_supine': (-90., .099),
    }
    if mode in ('crouch', 'deep_crouch', *tilted):
        groups = torch.zeros(n, device=env.device, dtype=torch.long)
        choices = torch.full((n,), -1, device=env.device, dtype=torch.long)
        q = env.scene['robot'].data.default_joint_pos[env_ids].clone()
        names = env.scene['robot'].joint_names
        if mode == 'crouch':
            hip, knee, ankle, root_z = -1.00, 2.00, -1.00, .550
        elif mode == 'deep_crouch':
            hip, knee, ankle = -1.20, 2.35, -1.10
            root_z = .447
        else:
            # Change one physical variable at a time from the exact crouch
            # that passed 16/16.  The earlier tilt states also deepened the
            # knee to 2.35 rad; commanding neutral from that compressed pose
            # launched every deterministic trial and made the curriculum
            # invalid before PPO ever acted.
            hip, knee, ankle = -1.00, 2.00, -1.00
            if env.cfg.recovery_split == 0:
                # Training at an endpoint includes every already-learned
                # five-degree state.  This is the reverse-curriculum analogue
                # of replay and prevents a later tilt from erasing easier
                # crouch recoveries.  Evaluation (split 1) remains fixed at
                # the named endpoint, so this cannot make a gate easier.
                maximum = 90 if mode == 'tucked_supine' else int(mode[4:])
                levels = [(0, .550)] + sorted(
                    (round(abs(value[0])), value[1]) for value in tilted.values()
                    if round(abs(value[0])) <= maximum)
                angle_table = torch.tensor([-item[0] for item in levels], device=env.device)
                height_table = torch.tensor([item[1] for item in levels], device=env.device)
                indices = torch.randint(len(levels), (n,), device=env.device)
                angle_degrees = angle_table[indices]
                root_z = height_table[indices]
            else:
                angle_degrees = torch.full((n,), tilted[mode][0], device=env.device)
                root_z = tilted[mode][1]
        for side in ('left', 'right'):
            q[:, names.index(side + '_hip_pitch')] = hip
            q[:, names.index(side + '_knee_pitch')] = knee
            q[:, names.index(side + '_ankle_pitch')] = ankle
        root = torch.zeros((n, 13), device=env.device)
        root[:, 2] = root_z
        if mode in tilted:
            if env.cfg.recovery_split != 0:
                angle_degrees = torch.full((n,), tilted[mode][0], device=env.device)
            angle = torch.deg2rad(angle_degrees)
            root[:, 3] = torch.cos(angle / 2.)
            root[:, 5] = torch.sin(angle / 2.)
        else:
            root[:, 3] = 1.
    else:
        if mode != 'lying':
            raise RuntimeError(f'Unsupported recovery start mode: {mode}')
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
        # Discovery timeouts are not event-penalized.  The dense floor cost
        # already makes waiting unattractive, while a timeout penalty is paid
        # more often by policies that attempt and fall early.  That ordering
        # was the central "stay down" shortcut in the rejected v5-v8 runs.
        # Refinement may penalize a failed episode once a recovery already
        # exists, but it stays small relative to the completion event.
        failure_penalty = 0. if env.cfg.recovery_stage == 'discovery' else 5.
        self.core = RecoveryReward(env.num_envs, env.device, env.step_dt,
                                   Limits(progress_budget=24., completion_bonus=40.,
                                          failure_penalty=failure_penalty))
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
    """Deprecated v8 absolute reward; retained as a zero regression guard."""
    return zero(env)


def discovery_upward_velocity_reward(env):
    """Deprecated v8 one-way velocity reward; rising and falling must balance."""
    return zero(env)


def floor_stall_cost(env):
    """Small per-second cost for remaining at the reset floor height."""
    term = env._recovery_reward
    if term.last_frame is None: return zero(env)
    f = term.last_frame
    active = (env._recovery_groups < 4) & ~term.handoff & f.contacts_fresh & f.safe
    # Unlike an episode-end penalty, this decreases immediately as the pelvis
    # rises and cannot make attempting recovery less attractive than waiting.
    # Maintain an incentive throughout the low/mid-height dead zone.  Unsafe
    # early termination has a larger event cost, so crashing cannot avoid it.
    return active * ((.72 - f.height) / .64).clamp(0., 1.)


def rise_height_reward(env):
    """Discovery-only objective: continuously reward raising the pelvis."""
    height = env.scene['robot'].data.root_pos_w[:, 2]
    return ((height - .06) / (.78 - .06)).clamp(0., 1.35)


def rise_upward_velocity_reward(env):
    """Reward positive vertical pelvis motion; downward motion earns nothing."""
    return env.scene['robot'].data.root_lin_vel_w[:, 2].clamp(0., 3.)


def stand_height_band_reward(env):
    """Positive standing-height target that does not reward jumping above it."""
    height = env.scene['robot'].data.root_pos_w[:, 2]
    escaped_floor = ((height - .35) / .35).clamp(0., 1.)
    return escaped_floor * torch.exp(-torch.square((height - .85) / .12))


def approach_upward_velocity_reward(env):
    """Help the rise below standing height, then remove the incentive to launch."""
    height = env.scene['robot'].data.root_pos_w[:, 2]
    fade = ((.72 - height) / .32).clamp(0., 1.)
    return fade * env.scene['robot'].data.root_lin_vel_w[:, 2].clamp(0., 3.)


def stand_upright_reward(env):
    """Positive-only upright shaping, activated after meaningful floor escape."""
    term = env._recovery_reward
    if term.last_frame is None:
        return zero(env)
    f = term.last_frame
    height_gate = ((f.height - .25) / .45).clamp(0., 1.)
    pelvis = .5 * (1. + torch.cos(f.pelvis_tilt.clamp(0., torch.pi)))
    torso = .5 * (1. + torch.cos(f.torso_tilt.clamp(0., torch.pi)))
    return height_gate * .5 * (pelvis + torso)


def stand_two_foot_reward(env):
    """Reward loading both feet only after the pelvis is already off the floor."""
    term = env._recovery_reward
    if term.last_frame is None:
        return zero(env)
    f = term.last_frame
    height_gate = ((f.height - .55) / .17).clamp(0., 1.)
    each = (f.foot_load / .10).clamp(0., 1.).amin(1)
    total = (f.foot_load.sum(1) / .75).clamp(0., 1.)
    return height_gate * each * total * f.contacts_fresh


def stand_quiet_reward(env):
    """Reward staying upright and quiet; never punish motion needed to rise."""
    term = env._recovery_reward
    if term.last_frame is None:
        return zero(env)
    f = term.last_frame
    height_gate = ((f.height - .68) / .08).clamp(0., 1.)
    upright_gate = (f.pelvis_tilt < torch.deg2rad(torch.tensor(35., device=env.device)))
    upright_gate &= (f.torso_tilt < torch.deg2rad(torch.tensor(35., device=env.device)))
    quiet = torch.exp(-1.0 * f.planar_speed.square()
                      - .20 * f.angular_speed.square()
                      - .08 * f.joint_speed_rms.square())
    return height_gate * upright_gate * quiet


@configclass
class Rewards:
    recovery = RewardTermCfg(func=RecoveryTerm, weight=1.)
    discovery_rise = RewardTermCfg(func=discovery_rise_reward, weight=0.)
    discovery_upward_velocity = RewardTermCfg(func=discovery_upward_velocity_reward, weight=0.)
    floor_stall = RewardTermCfg(func=floor_stall_cost, weight=-.45)
    terminal_rehearsal = RewardTermCfg(func=terminal_rehearsal, weight=4.0)
    terminal_motion = RewardTermCfg(func=terminal_motion_cost, weight=-.60)
    terminal_failure = RewardTermCfg(func=terminal_failure_cost, weight=-15.)
    # Larger than the complete 24-point potential rise plus the roughly
    # 9-point floor-cost that an early reset can avoid.  A violent launch can
    # therefore never beat either a safe partial rise or simply timing out.
    unsafe_recovery = RewardTermCfg(func=unsafe_recovery_cost, weight=-40.)
    limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-.5)
    target_limits = RewardTermCfg(func=unreachable, weight=-.25)
    ankle = RewardTermCfg(func=AnkleCost, weight=-1.)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-.001)
    power = RewardTermCfg(func=power, weight=-2e-6)
    collision_margin = RewardTermCfg(func=collision_margin_cost, weight=-2.)
    rise_speed = RewardTermCfg(func=rise_speed_cost, weight=-.5)


@configclass
class RiseOnlyRewards:
    # Keep RecoveryTerm at zero weight only so evaluation can collect its
    # synchronized telemetry. It contributes nothing to the PPO return.
    recovery_telemetry = RewardTermCfg(func=RecoveryTerm, weight=0.)
    height = RewardTermCfg(func=rise_height_reward, weight=12.)
    upward_velocity = RewardTermCfg(func=rise_upward_velocity_reward, weight=1.5)


@configclass
class StandOnlyRewards:
    # V12 proved that height-only discovery learned a transient rise followed
    # by a fall.  V13 remains positive-only but rewards the standing-height
    # band, upright support, and quiet dwell from the first easy reverse stage.
    recovery_telemetry = RewardTermCfg(func=RecoveryTerm, weight=0.)
    height_band = RewardTermCfg(func=stand_height_band_reward, weight=10.)
    upward_velocity_below_stand = RewardTermCfg(func=approach_upward_velocity_reward, weight=.75)
    upright = RewardTermCfg(func=stand_upright_reward, weight=6.)
    two_foot_support = RewardTermCfg(func=stand_two_foot_reward, weight=6.)
    quiet_stand = RewardTermCfg(func=stand_quiet_reward, weight=6.)


@configclass
class ProbeRewards:
    zero = RewardTermCfg(func=zero, weight=1.)


@configclass
class Endings:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    unsafe = TerminationTermCfg(func=unsafe_state)
    terminal_fall = TerminationTermCfg(func=terminal_fall)


@configclass
class RiseOnlyEndings:
    # Discovery is intentionally permissive. Joint limits and the episode
    # timeout remain; nonfinite simulation state is still rejected by worker.py.
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class RecoveryCfg(HumanoidStandEnvCfg):
    recovery_groups: int = 1
    recovery_split: int = 0
    recovery_assist: float = 0.
    recovery_forced_group: int = -1
    recovery_stage: str = 'discovery'
    recovery_start_mode: str = 'lying'


def make_cfg(count=64, stage='discovery', groups=1, split=0, probe=False, assist=0., forced_group=-1,
             self_collisions=True, start_mode='lying'):
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
    # Simplified box self-collisions previously stopped useful discovery motion.
    # Geometry violations remain visible in diagnostic telemetry.
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 12
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    # V11 stopped below its second unassisted height gate.  V12 permits a
    # stronger *training-only* scaffold while retaining the same spatial fade
    # to exactly zero by 0.70 m and keeping every evaluation unassisted.
    if not 0. <= assist <= .40: raise ValueError('Training assistance must be within [0,.40] body weight')
    if forced_group not in (-1, 0, 1, 2, 3, 4): raise ValueError('Invalid forced pose group')
    valid_starts = ('lying', 'crouch', 'deep_crouch',
                    *(f'tilt{angle}' for angle in (1, 2, 3, 4, *range(5, 90, 5))),
                    'tucked_supine')
    if start_mode not in valid_starts: raise ValueError('Invalid recovery start mode')
    cfg.recovery_groups, cfg.recovery_split = groups, split
    cfg.recovery_assist, cfg.recovery_forced_group = assist, forced_group
    cfg.recovery_stage = stage
    cfg.recovery_start_mode = start_mode
    cfg.curriculum = None
    # Hold rehearsal only needs the seven-second audited dwell plus margin.
    # Recovery retains enough time to rise and complete the same dwell.
    cfg.episode_length_s = 10. if stage == 'hold' else 20.
    # Recovery needs large coordinated hip/knee excursions. The failed v5
    # candidate reduced this to .25 rad and never left the floor.
    # ResetAwarePositionAction supplies a limit-aware nonlinear mapping.  Keep
    # these identity values so serialized configs do not imply the old v8
    # +/-0.70-rad contract.
    cfg.actions.joint_pos.scale = 1.0
    cfg.actions.joint_pos.use_default_offset = False
    if not probe: cfg.actions.joint_pos.class_type = ResetAwarePositionAction
    cfg.observations = RecoveryObservations()
    cfg.terminations = RiseOnlyEndings()
    cfg.rewards = ProbeRewards() if probe else (StandOnlyRewards() if stage == 'refine' else RiseOnlyRewards())
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
