"""Measured walking credit, separate from the independent promotion evaluator.

Pure torch: CPU exploit tests require neither Isaac nor a second GPU process.
Rates are integrated once by control dt. Completed-step credit is an event.
"""
import torch

REWARD_VERSION = 'net_travel_v3'
STOP_SPEED = .08


class ForwardProgress:
    """New territory, not repeated forward halves of a rocking cycle.

    Progress uses a fixed world heading per moving episode. Positive command
    resampling cannot reset the frontier or renew free swing guidance. All
    fields are cleared on a physical episode reset; stopping starts a new bout.
    """
    def __init__(self, count, device):
        self.origin = torch.zeros((count, 2), device=device)
        self.heading = torch.zeros_like(self.origin)
        self.frontier = torch.zeros(count, device=device)
        self.elapsed = torch.zeros_like(self.frontier)
        self.since_step = torch.full_like(self.frontier, 100.)
        self.moving = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.moving[ids] = False
        self.frontier[ids] = 0.
        self.elapsed[ids] = 0.
        self.since_step[ids] = 100.

    def update(self, root_xy, heading, command, events, dt):
        moving = command > STOP_SPEED
        start = moving & ~self.moving
        self.origin = torch.where(start[:, None], root_xy, self.origin)
        self.heading = torch.where(start[:, None], heading, self.heading)
        self.frontier = torch.where(start, 0., self.frontier)
        self.elapsed = torch.where(start, 0., self.elapsed) + dt
        self.since_step = torch.where(start, 100., self.since_step) + dt
        self.since_step = torch.where(events.sum(1) > 0, 0., self.since_step)
        distance = ((root_xy-self.origin)*self.heading).sum(1)
        advance = (distance-self.frontier).clamp_min(0.)
        self.frontier = torch.maximum(self.frontier, distance)
        self.moving.copy_(moving)
        fraction = (advance/(command.clamp_min(.10)*dt)).clamp(0., 1.)*moving
        recent = self.since_step < 1.5/(.90+.50*command)
        # A finite discovery window, then renew only after a verified landing.
        budget = (self.elapsed < 2.5/(.90+.50*command)) | recent
        return fraction, budget.float()*moving, recent & moving


class StepTracker:
    """Loaded stance -> uninterrupted air -> loaded landing with forward travel.

    Reject contact chatter, in-place kicks, planted-foot sliding, same-foot
    repetition, walking hops and events carried across resets/command changes.
    This state only shapes training; it cannot qualify a policy for promotion.
    """
    def __init__(self, count, device):
        self.contact = torch.zeros((count, 2), dtype=torch.bool, device=device)
        self.ground = torch.zeros((count, 2), device=device)
        self.air = torch.zeros_like(self.ground)
        self.active = self.contact.clone()
        self.armed = self.contact.clone()
        self.clearance = torch.zeros_like(self.ground)
        self.anchor = torch.zeros((count, 2, 2), device=device)
        self.last_ground = torch.zeros_like(self.anchor)
        self.root_anchor = torch.zeros_like(self.anchor)
        self.heading = torch.zeros_like(self.anchor)
        self.last_foot = torch.full((count,), -1, dtype=torch.long, device=device)
        self.age = torch.full_like(self.ground, 10.)
        self.counts = torch.zeros_like(self.ground)

    def reset(self, ids):
        for value in [self.contact, self.ground, self.air, self.active, self.armed,
                      self.clearance, self.anchor, self.last_ground,
                      self.root_anchor, self.heading, self.counts]:
            value[ids] = 0
        self.last_foot[ids] = -1
        self.age[ids] = 10.

    def update(self, contact, foot_xy, foot_clearance, root_xy, heading,
               command, eligible, dt):
        moving = command > STOP_SPEED
        live = moving & eligible
        self.age += dt
        lift = self.contact & ~contact & (self.ground >= .06 - 1e-5) & live[:, None]
        self.anchor = torch.where(lift[..., None], self.last_ground, self.anchor)
        self.root_anchor = torch.where(lift[..., None], root_xy[:, None], self.root_anchor)
        self.heading = torch.where(lift[..., None], heading[:, None], self.heading)
        self.active = (self.active | lift) & live[:, None]
        self.armed = torch.where(lift, False, self.armed) & self.active
        self.clearance = torch.where(lift, 0., self.clearance)
        self.ground = torch.where(contact, self.ground + dt, 0.)
        self.air = torch.where(contact, 0., self.air + dt)
        self.armed |= (self.air >= .08 - 1e-5) & self.active
        self.clearance = torch.maximum(self.clearance, foot_clearance * (~contact))
        settled = self.active & self.armed & contact & (self.ground >= .06 - 1e-5)
        foot_advance = ((foot_xy - self.anchor) * self.heading).sum(-1)
        body_advance = ((root_xy[:, None] - self.root_anchor) * self.heading).sum(-1)
        frequency = .90 + .50 * command
        minimum_advance = torch.maximum(torch.full_like(command, .020), .10 * command / frequency)
        candidate = (settled & (self.clearance >= .012)
                     & (foot_advance >= minimum_advance[:, None])
                     & (body_advance >= .003)
                     & (self.age >= (.35 / frequency)[:, None]))
        # A simultaneous landing in walking is a hop, not two alternating steps.
        candidate &= ~((candidate.sum(1) > 1) & (command <= 1.1))[:, None]
        events = torch.zeros_like(self.ground)
        for side in (0, 1):
            credit = candidate[:, side] & (self.last_foot != side)
            events[:, side] = credit.float()
            self.last_foot = torch.where(credit, side, self.last_foot)
            self.age[:, side] = torch.where(credit, 0., self.age[:, side])
        # Consume the landing even if rejected, before the foot can slide forward.
        self.active &= ~settled
        self.armed &= self.active
        self.last_ground = torch.where(contact[..., None], foot_xy, self.last_ground)
        self.contact.copy_(contact)
        self.counts += events
        return events


def walking_terms(command, velocity, healthy, slip, contact, expected_contact,
                  clearance, relfeet, target_feet, reference_error, events, dt,
                  *, advance_fraction=None, swing_budget=None, recent_step=None):
    """No locomotion credit for zero progress with two planted feet.

    Dense swing guidance is temporary until a real step succeeds. Positive
    tracking requires fresh territory plus an unloaded foot/recent valid step.
    A small survival floor avoids making falling the shortest profitable exit.
    """
    moving = command > STOP_SPEED
    progress = (velocity[:, 0] / command.clamp_min(.10)).clamp(-1., 1.)
    sigma = torch.maximum(torch.full_like(command, .10), .40 * command)
    accuracy = torch.exp(-((velocity[:, 0] - command).square()
                          + velocity[:, 1].square()) / sigma.square())
    traction = torch.exp(-(slip / .08).square())
    if advance_fraction is None:
        advance_fraction = progress.clamp_min(0.)  # stateless diagnostic callers
    if swing_budget is None:
        swing_budget = torch.ones_like(command)
    if recent_step is None:
        recent_step = events.sum(1) > 0
    locomoting = ((~contact).any(1) | recent_step).float()
    fresh_progress = advance_fraction * traction * locomoting
    swing = ~expected_contact
    actual_swing = swing & ~contact
    desired_clearance = .025 + .012 * command.clamp_max(2.5)
    lift_quality = (clearance / desired_clearance[:, None]).clamp(0., 1.)
    scales = torch.tensor([.08, .05, .035], device=relfeet.device)
    foot_quality = torch.exp(-((relfeet - target_feet) / scales).square().sum(-1))
    swing_quality = (actual_swing * lift_quality * foot_quality).sum(1) * swing_budget
    activity = torch.maximum(fresh_progress,
                             swing_quality.clamp_max(1.))
    live = healthy * moving
    return dict(
        track=2.5 * fresh_progress * accuracy * live,
        progress=.75 * (fresh_progress + progress.clamp_max(0.)) * live,
        reference=.35 * torch.exp(-reference_error / .06) * activity * live,
        gait=.80 * swing_quality * live,
        missing_swing=-.35 * (swing & contact).float().mean(1) * live,
        completed_step=.30 * events.sum(1) * live / dt,
        # This multiplies existing posture terms, not the locomotion terms above.
        posture_scale=torch.where(moving, .25 + .75 * activity, 1.),
        progress_fraction=progress,
        swing_quality=swing_quality,
    )
