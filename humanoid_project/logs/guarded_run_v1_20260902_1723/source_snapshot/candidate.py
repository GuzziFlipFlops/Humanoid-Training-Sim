"""Isolated experimental task. Does not modify the archived NaturalWalk/Run tasks."""
import math
import torch
from isaaclab.envs import mdp
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from humanoid_tasks.curriculum.env_cfg import HumanoidNaturalRunEnvCfg
from contact_state import CompletedStances, differential_cost

ANKLES = ['left_ankle_pitch', 'left_ankle_roll', 'right_ankle_pitch', 'right_ankle_roll']


class MixedSpeeds(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        n = len(env_ids)
        low = torch.rand(n, device=self.device) < .40
        walking = .3 + .7 * torch.rand(n, device=self.device)
        high = 1.0 + max(0., self.cfg.ranges.lin_vel_x[1] - 1.) * torch.rand(n, device=self.device)
        self.vel_command_b[env_ids, 0] = torch.where(low, walking, high)


class AnkleEnvelope(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        self.ids = [self.robot.joint_names.index(n) for n in ANKLES]
        assert [self.robot.joint_names[i] for i in self.ids] == ANKLES

    def __call__(self, env):
        d = self.robot.data
        return differential_cost(d.applied_torque[:, self.ids].reshape(-1, 2, 2),
                                 d.joint_vel[:, self.ids].reshape(-1, 2, 2))


class StanceBalance(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        self.sensor = env.scene.sensors['contact_forces']
        self.ids = [self.sensor.body_names.index(n) for n in ('left_foot', 'right_foot')]
        self.tracker = CompletedStances(env.num_envs, env.device, env.step_dt)
        self.previous_command = torch.zeros((env.num_envs, 3), device=env.device)
        self.cooldown = torch.ones(env.num_envs, device=env.device)
        env._guarded_stance = self

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.cooldown[ids] = 1.
        self.previous_command[ids] = 0.

    def __call__(self, env):
        cmd = env.command_manager.get_command('base_velocity')
        changed = (cmd - self.previous_command).abs().amax(1) > .05
        self.tracker.reset(changed)
        self.cooldown[changed] = 1.
        self.previous_command.copy_(cmd)
        self.cooldown.sub_(env.step_dt).clamp_min_(0.)
        self.tracker.update(self.sensor.data.net_forces_w[:, self.ids, 2] > 5.)
        d = self.robot.data
        valid = ((cmd[:, 0] > .4) & (cmd[:, 1].abs() < .1) & (cmd[:, 2].abs() < .1)
                 & (self.cooldown <= 0.) & ((d.root_lin_vel_b[:, 0] - cmd[:, 0]).abs() < .5)
                 & (d.root_lin_vel_b[:, 1].abs() < .25) & (d.root_ang_vel_b[:, 2].abs() < .4)
                 & (d.projected_gravity_b[:, 2] < -.85))
        return self.tracker.penalty() * valid.float()


def guarded_push(env, env_ids, velocity_range, asset_cfg=SceneEntityCfg('robot')):
    mdp.push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    if hasattr(env, '_guarded_stance'):
        env._guarded_stance.reset(env_ids)


def unreachable_targets(env):
    d = env.scene['robot'].data
    bounds = d.joint_pos_limits
    excess = ((bounds[..., 0] - d.joint_pos_target).clamp_min(0.)
              + (d.joint_pos_target - bounds[..., 1]).clamp_min(0.))
    # Penalize the request, not just the resulting joint angle. No hard action clipping.
    fast = (env.command_manager.get_command('base_velocity')[:, 0] - 1.).clamp(0., 1.)
    return (excess / .35).square().clamp_max(4.).mean(1) * fast


class TorsoLean(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        self.idx = self.robot.body_names.index('torso')

    def __call__(self, env):
        q = self.robot.data.body_quat_w[:, self.idx]
        up = torch.zeros((env.num_envs, 3), device=env.device)
        up[:, 2] = 1.
        local_up = quat_apply_inverse(q, up)
        tilt = torch.acos(local_up[:, 2].clamp(-1., 1.))
        speed = env.command_manager.get_command('base_velocity')[:, 0].abs()
        # Allow a speed-dependent lean cone (23 to 29 degrees), not a fixed vertical torso.
        allowance = .35 + .04 * speed.clamp_max(4.)
        return ((tilt - allowance).clamp_min(0.) / .35).square().clamp_max(4.) * (speed - 1.).clamp(0., 1.)


def make_cfg(stage='index', ceiling=1.8, scale=.45, count=8192):
    cfg = HumanoidNaturalRunEnvCfg()
    cfg.scene.num_envs = count
    cfg.actions.joint_pos.scale = scale
    cfg.curriculum = None  # Only the external measured gates can increase speed.
    cfg.commands.base_velocity.class_type = MixedSpeeds
    cfg.commands.base_velocity.ranges.lin_vel_x = (.3, ceiling)
    cfg.commands.base_velocity.rel_standing_envs = .05
    cfg.rewards.ankle_differential = RewardTermCfg(func=AnkleEnvelope, weight=-2.)
    cfg.events.push.func = guarded_push
    if stage in ('balance', 'targets', 'torso', 'speed'):
        cfg.rewards.completed_stance_balance = RewardTermCfg(func=StanceBalance, weight=-.20)
    if stage in ('targets', 'torso', 'speed'):
        cfg.rewards.unreachable_target = RewardTermCfg(func=unreachable_targets, weight=-.25)
    if stage in ('torso', 'speed'):
        cfg.rewards.actual_torso_lean = RewardTermCfg(func=TorsoLean, weight=-.40)
    return cfg
