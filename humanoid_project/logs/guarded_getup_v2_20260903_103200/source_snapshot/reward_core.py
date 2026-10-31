"""Get-up reward core, independent of Isaac; adapter lives in candidate.py.

Call once per control step with measured pre-reset telemetry. Isaac RewardManager
multiplies a term by dt: feed RewardOutput.rate, NOT RewardOutput.step_reward.
See README.md for the telemetry contract and prerequisites for an Isaac adapter.
"""
from dataclasses import dataclass, fields
import math
import torch


@dataclass(frozen=True)
class Limits:
    # Provisional for the current 0.84 m pelvis reference, NOT human height.
    min_height: float = .78
    max_height: float = .94
    max_pelvis_tilt: float = math.radians(12)
    max_torso_tilt: float = math.radians(12)
    max_foot_tilt: float = math.radians(15)
    max_planar_speed: float = .08
    max_vertical_speed: float = .08
    max_angular_speed: float = .25
    max_joint_speed_rms: float = .8
    min_each_foot_weight: float = .10
    min_total_foot_weight: float = .75
    max_total_foot_weight: float = 1.25
    max_nonfoot_force: float = 5.
    max_slip: float = .05
    min_support_margin: float = .01
    max_hold_drift: float = .05
    max_hold_foot_drift: float = .02
    max_hold_yaw: float = math.radians(10)
    dwell_s: float = 2.
    post_hold_s: float = 5.
    progress_budget: float = 1.
    completion_bonus: float = 20.
    failure_penalty: float = 3.
    hold_motion_weight: float = .10


@dataclass
class Frame:
    # All fields are batch tensors; scalar quantities have shape [N].
    height: torch.Tensor             # pelvis height above the local ground, meters
    pelvis_tilt: torch.Tensor        # unsigned angle from vertical, radians
    torso_tilt: torch.Tensor         # actual torso BODY, not pelvis quaternion
    xy: torch.Tensor                 # [N,2], world pelvis position
    yaw: torch.Tensor                # world pelvis yaw, radians
    planar_speed: torch.Tensor       # norm, not signed forward speed
    vertical_speed: torch.Tensor    # signed vertical velocity
    angular_speed: torch.Tensor     # norm of full 3D angular velocity
    joint_speed_rms: torch.Tensor
    foot_load: torch.Tensor          # [N,2], upward force / actual total weight
    foot_tilt: torch.Tensor          # [N,2], sole-normal tilt from ground normal
    foot_slip: torch.Tensor          # [N,2], sole contact-point planar speed
    foot_xy: torch.Tensor            # [N,2,2], world sole positions
    nonfoot_force: torch.Tensor      # maximum current non-foot ground force (N)
    support_margin: torch.Tensor     # signed COM margin inside loaded-sole hull (m)
    contacts_fresh: torch.Tensor     # bool: physics samples current and not reset-stale
    safe: torch.Tensor               # bool: finite dynamics, no severe impact/penetration

    def validate(self, n, device):
        vector_shapes = {'xy': (n, 2), 'foot_load': (n, 2), 'foot_tilt': (n, 2),
                         'foot_slip': (n, 2), 'foot_xy': (n, 2, 2)}
        for item in fields(self):
            value = getattr(self, item.name)
            if value.shape != vector_shapes.get(item.name, (n,)) or value.device != device:
                raise ValueError(f'Bad shape/device for {item.name}')
            if item.name in ('contacts_fresh', 'safe'):
                if value.dtype != torch.bool:
                    raise ValueError(f'{item.name} must be boolean')
            elif not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite {item.name}: abort, do not hide invalid physics')


@dataclass
class RewardOutput:
    step_reward: torch.Tensor
    rate: torch.Tensor
    progress: torch.Tensor
    completion: torch.Tensor
    hold_motion_cost: torch.Tensor
    failure: torch.Tensor
    stable: torch.Tensor
    success: torch.Tensor            # event: first uninterrupted 2+5 seconds only
    currently_valid: torch.Tensor    # live, can turn false AFTER a success
    invalidated_after_success: torch.Tensor  # latches, invalidates entire eval trial


class RecoveryReward:
    def __init__(self, count, device='cpu', dt=.02, limits=None):
        if count < 1 or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Positive count and finite dt required')
        self.n, self.device, self.dt = count, torch.device(device), dt
        self.limits = limits or Limits()
        self.dwell_steps = math.ceil(self.limits.dwell_s / dt)
        self.total_steps = math.ceil((self.limits.dwell_s + self.limits.post_hold_s) / dt)
        self.best = torch.zeros(count, device=self.device)
        self.hold_steps = torch.zeros(count, dtype=torch.long, device=self.device)
        self.anchor_xy = torch.zeros((count, 2), device=self.device)
        self.anchor_yaw = torch.zeros(count, device=self.device)
        self.anchor_feet = torch.zeros((count, 2, 2), device=self.device)
        self.eligible = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.ever_success = self.eligible.clone()
        self.invalidated = self.eligible.clone()
        self.finished = self.eligible.clone()
        self.initialized = self.eligible.clone()
        self.pose_group = torch.full((count,), -1, dtype=torch.long, device=self.device)
        self.last_tick = -1

    def quality(self, f):
        # A capped exploration signal, not a standing/completion classifier.
        # Audited lying pelvis heights are about .08-.10 m. The old .25 m
        # floor erased credit for the entire initial rise. Keep a smooth
        # orientation signal beyond 90 degrees, allowing early rolling.
        height = ((f.height - .06) / (self.limits.min_height - .06)).clamp(0., 1.)
        upright = .5 * (1. + torch.cos(f.torso_tilt.clamp(0., math.pi)))
        return .5 * height + .5 * upright

    def reset(self, ids, frame, pose_group, validated_pose):
        """Initialize AFTER pose settling. Labels 0..3: supine/prone/left/right.

        Non-lying training starts may be kept, but are never recovery successes.
        validated_pose must come from the collision/pose audit, not a height test.
        frame/labels are full batch tensors; only ids are reset.
        """
        frame.validate(self.n, self.device)
        if pose_group.shape != (self.n,) or validated_pose.shape != (self.n,):
            raise ValueError('Reset labels/mask must cover full batch')
        self.best[ids] = self.quality(frame)[ids]
        self.hold_steps[ids] = 0
        self.ever_success[ids] = False
        self.invalidated[ids] = False
        self.finished[ids] = False
        self.initialized[ids] = True
        self.anchor_xy[ids] = frame.xy[ids]
        self.anchor_yaw[ids] = frame.yaw[ids]
        self.anchor_feet[ids] = frame.foot_xy[ids]
        self.pose_group[ids] = pose_group[ids]
        lying = (frame.height < .45) & (frame.torso_tilt > math.radians(60))
        valid_group = (pose_group >= 0) & (pose_group <= 3)
        self.eligible[ids] = (validated_pose & frame.safe & frame.contacts_fresh & lying & valid_group)[ids]

    def stable_mask(self, f):
        c = self.limits
        load = f.foot_load.sum(1)
        return ((f.height >= c.min_height) & (f.height <= c.max_height)
                & (f.pelvis_tilt <= c.max_pelvis_tilt) & (f.torso_tilt <= c.max_torso_tilt)
                & (f.planar_speed <= c.max_planar_speed) & (f.vertical_speed.abs() <= c.max_vertical_speed)
                & (f.angular_speed <= c.max_angular_speed) & (f.joint_speed_rms <= c.max_joint_speed_rms)
                & (f.foot_load >= c.min_each_foot_weight).all(1)
                & (load >= c.min_total_foot_weight) & (load <= c.max_total_foot_weight)
                & (f.foot_tilt <= c.max_foot_tilt).all(1) & (f.foot_slip <= c.max_slip).all(1)
                & (f.nonfoot_force <= c.max_nonfoot_force) & (f.support_margin >= c.min_support_margin)
                & f.contacts_fresh & f.safe)

    def update(self, f, tick, episode_end=None):
        """One call per tick, before auto-reset. episode_end includes real timeouts.

        An adapter must cache output if rewards and termination both request it.
        The endpoint success event does NOT request an automatic reset.
        """
        f.validate(self.n, self.device)
        if not self.initialized.all() or tick <= self.last_tick:
            raise ValueError('Initialize all envs; update once per strictly increasing tick')
        self.last_tick = tick
        c = self.limits
        active = self.eligible & ~self.finished
        ended = torch.zeros_like(active) if episode_end is None else episode_end
        if ended.shape != (self.n,) or ended.dtype != torch.bool:
            raise ValueError('episode_end must be a full boolean vector')
        quality = self.quality(f)
        progress = (quality - self.best).clamp_min(0.) * active * f.safe * c.progress_budget
        self.best.copy_(torch.maximum(self.best, quality))
        stable = self.stable_mask(f) & active
        starting = stable & (self.hold_steps == 0) & ~self.ever_success
        self.anchor_xy[starting] = f.xy[starting]
        self.anchor_yaw[starting] = f.yaw[starting]
        self.anchor_feet[starting] = f.foot_xy[starting]
        drift = (f.xy - self.anchor_xy).norm(dim=1)
        yaw = torch.atan2(torch.sin(f.yaw - self.anchor_yaw), torch.cos(f.yaw - self.anchor_yaw)).abs()
        foot_drift = (f.foot_xy - self.anchor_feet).norm(dim=2).amax(1)
        # Anchors are fixed over the whole hold; they cannot follow a skating robot.
        anchored = stable & (drift <= c.max_hold_drift) & (yaw <= c.max_hold_yaw) & (foot_drift <= c.max_hold_foot_drift)
        newly_invalid = self.ever_success & active & ~anchored & ~self.invalidated
        self.invalidated |= newly_invalid
        self.hold_steps = torch.where(anchored, self.hold_steps + 1, 0)
        success = (self.hold_steps >= self.total_steps) & ~self.ever_success & active
        self.ever_success |= success
        valid = self.ever_success & anchored & ~self.invalidated
        # Target motion only once posture/support make stationary standing feasible.
        upright_supported = active & (f.height >= c.min_height) & (f.torso_tilt <= c.max_torso_tilt)
        upright_supported &= (f.foot_load >= c.min_each_foot_weight).all(1)
        motion = (f.planar_speed.square() / c.max_planar_speed**2
                  + f.angular_speed.square() / c.max_angular_speed**2
                  + f.joint_speed_rms.square() / c.max_joint_speed_rms**2).clamp_max(10.)
        motion_cost = motion * upright_supported * c.hold_motion_weight * self.dt
        completion = success.float() * c.completion_bonus
        # Revoke the completion credit once if the robot subsequently walks/falls
        # out of the hold, rather than leaving that shortcut profitable in reward.
        failure = ((ended & active & ~valid).float() * c.failure_penalty
                   + newly_invalid.float() * c.completion_bonus)
        self.finished |= ended
        reward = progress + completion - motion_cost - failure
        return RewardOutput(reward, reward / self.dt, progress, completion, motion_cost, failure,
                            stable, success, valid, self.invalidated.clone())

    def state_observation(self, frame):
        """Reward memory for a NEW recovery critic, never locomotion observations.

        Deployment actor should use proprioceptive history; this privileged critic
        state includes the anchors which make hold/drift rewards history-dependent.
        """
        memory = torch.stack((self.best, (self.hold_steps.float() / self.total_steps).clamp_max(1.),
                              self.eligible.float(), self.ever_success.float(), self.invalidated.float()), dim=1)
        yaw = frame.yaw - self.anchor_yaw
        return torch.cat((memory, frame.xy - self.anchor_xy,
                          torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=1),
                          (frame.foot_xy - self.anchor_feet).flatten(1)), dim=1)
