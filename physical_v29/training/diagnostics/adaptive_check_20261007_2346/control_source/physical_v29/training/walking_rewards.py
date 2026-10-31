"""Measured walking credit, separate from the independent promotion evaluator.

Pure torch: CPU exploit tests require neither Isaac nor a second GPU process.
Rates are integrated once by control dt. Completed-step credit is an event.
"""
import torch

REWARD_VERSION = 'gait_integrity_v6'
STOP_SPEED = .08
RUN_SPEED = 1.1


def fall_penalty(terminated, dt, event_cost=8.):
    """Return a rate whose integrated cost is eight per fall at every dt.

    Formerly -30 was integrated as a rate: only -0.6 at 50 Hz. A reset also
    renews discovery shaping, so that weak exit cost could subsidize falling.
    """
    return -event_cost*terminated.float()/dt


def load_transfer(command, force, expected_contact, healthy, budget, weight):
    """Finite guidance for shifting load before a foot can lift.

    Equal double support pays zero. Stationary load rocking loses this credit
    when the discovery window expires, unless a real alternating step renews it.
    """
    fractions = force.clamp_min(0.)/force.clamp_min(0.).sum(1, keepdim=True).clamp_min(1.)
    desired = expected_contact.float()*2.-1.
    contrast = (fractions*desired).sum(1).clamp_min(0.)
    supported = force.clamp_min(0.).sum(1) > .65*weight
    single = expected_contact.sum(1) == 1
    return .4*contrast*single*supported*(command > STOP_SPEED)*healthy*budget


def running_flight(command, contact, expected_contact, clearance, healthy,
                   advance_fraction, swing_budget, recent_step):
    """Reward a brief commanded flight phase with real forward activity.

    Airborne time by itself is not success. Two-foot hops receive no landing
    events, and motion-free flight cannot receive this term.
    """
    expected_flight = ~expected_contact.any(1)
    actual_flight = ~contact.any(1)
    lift = (clearance.amin(1)/.025).clamp(0., 1.)
    activity = (advance_fraction > 0.) | recent_step
    return .25*(command > RUN_SPEED)*expected_flight*actual_flight*lift*healthy*activity*swing_budget


def standing_support(command, contact, relcom, relfeet, drift):
    """Smooth costs for the failed independent standing support/anchor checks.

    Use the same conservative sole rectangle, but a continuous distance cost.
    These costs cannot disappear by losing reward eligibility and never apply
    to commanded walking or running.
    """
    stopped = command <= STOP_SPEED
    lower = relfeet[:, :, :2].amin(1)-relfeet.new_tensor([.055, .035])
    upper = relfeet[:, :, :2].amax(1)+relfeet.new_tensor([.055, .035])
    outside = ((lower-relcom[:, :2]).clamp_min(0.)
               + (relcom[:, :2]-upper).clamp_min(0.)).norm(dim=1)
    return dict(stand_support=-.80*((~contact.all(1)).float()
                                  + (outside/.04).clamp(0., 1.))*stopped,
                stand_anchor=-.40*(drift/.25).square().clamp(0., 1.)*stopped)


class StandingAnchor:
    """Hold the location where a stop begins; do not undo prior forward travel."""
    def __init__(self, count, device):
        self.origin = torch.zeros((count, 2), device=device)
        self.stopped = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.stopped[ids] = False

    def update(self, root_xy, command):
        stopped = command <= STOP_SPEED
        start = stopped & ~self.stopped
        self.origin = torch.where(start[:, None], root_xy, self.origin)
        self.stopped.copy_(stopped)
        return (root_xy-self.origin).norm(dim=1)*stopped


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
        self.event_age = torch.full((count,), 10., device=device)
        self.age = torch.full_like(self.ground, 10.)
        self.touch_age = torch.full_like(self.ground, 100.)
        self.paired = self.contact.clone()
        self.impact_advance = torch.zeros_like(self.ground)
        self.impact_body_advance = torch.zeros_like(self.ground)
        self.counts = torch.zeros_like(self.ground)

    def reset(self, ids):
        for value in [self.contact, self.ground, self.air, self.active, self.armed,
                      self.clearance, self.anchor, self.last_ground,
                      self.root_anchor, self.heading, self.counts, self.paired,
                      self.impact_advance, self.impact_body_advance]:
            value[ids] = 0
        self.last_foot[ids] = -1
        self.event_age[ids] = 10.
        self.age[ids] = 10.
        self.touch_age[ids] = 100.

    def update(self, contact, foot_xy, foot_clearance, root_xy, heading,
               command, eligible, dt):
        moving = command > STOP_SPEED
        live = moving & eligible
        self.age += dt
        self.event_age += dt
        self.touch_age += dt
        lift = self.contact & ~contact & (self.ground >= .06 - 1e-5) & live[:, None]
        self.anchor = torch.where(lift[..., None], self.last_ground, self.anchor)
        self.root_anchor = torch.where(lift[..., None], root_xy[:, None], self.root_anchor)
        self.heading = torch.where(lift[..., None], heading[:, None], self.heading)
        self.active = (self.active | lift) & live[:, None]
        self.paired = torch.where(lift, False, self.paired)
        # A walking swing that turns into bilateral flight is a hop. Running
        # can include flight and is checked by its landing timing instead.
        walking_flight = (command <= RUN_SPEED) & ~contact.any(1)
        self.active &= ~walking_flight[:, None]
        self.armed = torch.where(lift, False, self.armed) & self.active
        self.clearance = torch.where(lift, 0., self.clearance)
        self.ground = torch.where(contact, self.ground + dt, 0.)
        self.air = torch.where(contact, 0., self.air + dt)
        self.armed |= (self.air >= .08 - 1e-5) & self.active
        touchdown = contact & ~self.contact & self.armed
        self.impact_advance = torch.where(touchdown,
            ((foot_xy-self.anchor)*self.heading).sum(-1), self.impact_advance)
        self.impact_body_advance = torch.where(touchdown,
            ((root_xy[:, None]-self.root_anchor)*self.heading).sum(-1), self.impact_body_advance)
        self.touch_age = torch.where(touchdown, 0., self.touch_age)
        pair = ((self.touch_age[:, 0]-self.touch_age[:, 1]).abs() <= .08+1e-5)
        pair &= self.touch_age.amax(1) <= .08+1e-5
        self.paired |= pair[:, None] & self.active
        self.clearance = torch.maximum(self.clearance, foot_clearance * (~contact))
        # Wait through the complete 80ms paired-impact window before payout.
        settled = self.active & self.armed & contact & (self.ground >= .10 - 1e-5)
        frequency = .90 + .50 * command
        minimum_advance = torch.maximum(torch.full_like(command, .020), .10 * command / frequency)
        candidate = (settled & (self.clearance >= .012)
                     & (self.impact_advance >= minimum_advance[:, None])
                     & (self.impact_body_advance >= .003)
                     & (self.age >= (.70 / frequency)[:, None]) & ~self.paired)
        # Biped running includes flight, but simultaneous bilateral landings
        # are still hops. The old running exception paid two steps in one tick.
        candidate &= ~(candidate.sum(1) > 1)[:, None]
        # Contacts staggered by one noisy control tick are still a hop. Human
        # running landings alternate about half a cycle apart; require a modest
        # fraction of that interval without prescribing an exact cadence.
        # Broad upper cadence bound: no more than 1.43 times target cycle rate.
        # The independent evaluator still measures ALL physical landings.
        candidate &= (self.event_age >= .35/frequency)[:, None]
        events = torch.zeros_like(self.ground)
        for side in (0, 1):
            credit = candidate[:, side] & (self.last_foot != side)
            events[:, side] = credit.float()
            self.last_foot = torch.where(credit, side, self.last_foot)
            self.event_age = torch.where(credit, 0., self.event_age)
            self.age[:, side] = torch.where(credit, 0., self.age[:, side])
        # Consume the landing even if rejected, before the foot can slide forward.
        self.active &= ~settled
        self.armed &= self.active
        self.last_ground = torch.where(contact[..., None], foot_xy, self.last_ground)
        self.contact.copy_(contact)
        self.counts += events
        return events


class StepDiscovery:
    """Finite measured progress toward the first landing, without a time limit.

    Each foot earns at most one unit in a physical episode. Only a physical
    reset renews the bank; accepted steps, command changes, rocking and kicks
    cannot replenish it. Continuing gait earns separate locomotion credit.
    Credit
    is paid on a new high-water mark, never for holding an unloaded pose.
    """
    def __init__(self, count, device):
        self.best = torch.zeros((count, 2), device=device)
        self.ground = torch.zeros_like(self.best)
        self.ready = torch.zeros_like(self.best, dtype=torch.bool)
        self.started = torch.zeros(count, dtype=torch.bool, device=device)
        self.anchor = torch.zeros((count, 2, 2), device=device)
        self.root_anchor = torch.zeros_like(self.anchor)
        self.heading = torch.zeros((count, 2), device=device)

    def reset(self, ids):
        for value in (self.best, self.ground, self.ready, self.started,
                      self.anchor, self.root_anchor, self.heading):
            value[ids] = 0

    def update(self, force, foot_xy, clearance, root_xy, heading, command,
               eligible, healthy, events, weight, dt):
        moving = command > STOP_SPEED
        start = moving & ~self.started
        self.anchor = torch.where(start[:, None, None], foot_xy, self.anchor)
        self.root_anchor = torch.where(start[:, None, None], root_xy[:, None], self.root_anchor)
        self.heading = torch.where(start[:, None], heading, self.heading)
        self.started |= start
        force = force.clamp_min(0.)
        contact = force > .05*weight
        self.ground = torch.where(contact, self.ground+dt, 0.)
        self.ready |= (self.ground >= .06-1e-5) & moving[:, None]
        supported = force.sum(1) > .65*weight
        live = moving & eligible & supported
        fraction = force/force.sum(1, keepdim=True).clamp_min(1.)
        transfer = (1.-2.*fraction).clamp(0., 1.)
        unloaded = ~contact & self.ready
        lift = (clearance/(.025+.012*command.clamp_max(2.5))[:, None]).clamp(0., 1.)*unloaded
        advance = ((foot_xy-self.anchor)*self.heading[:, None]).sum(-1)
        body_advance = ((root_xy[:, None]-self.root_anchor)*self.heading[:, None]).sum(-1)
        stride = torch.maximum(torch.full_like(command, .020), .10*command/(.90+.50*command))
        placement = (advance/stride[:, None]).clamp(0., 1.)*unloaded*(body_advance >= .003)
        score = (.20*transfer+.40*lift+.40*placement)*self.ready*live[:, None]
        increase = (score-self.best).clamp_min(0.)
        self.best = torch.maximum(self.best, score)
        credit = (increase*healthy[:, None]).sum(1)/dt
        return credit


def walking_terms(command, velocity, healthy, slip, contact, expected_contact,
                  clearance, relfeet, target_feet, reference_error, events, dt,
                  *, advance_fraction=None, swing_budget=None, recent_step=None):
    """No locomotion credit for zero progress with two planted feet.

    Dense swing guidance is temporary until a real step succeeds. Positive
    tracking requires fresh territory plus an unloaded foot/recent valid step.
    Full balance gradients keep the transferred controller valuable. A bounded
    motion-deficit cost exceeds the maximum posture-only income at rest, so
    avoiding all steps is no longer an indefinitely profitable moving policy.
    """
    moving = command > STOP_SPEED
    progress = (velocity[:, 0] / command.clamp_min(.10)).clamp(-1., 1.)
    sigma = torch.maximum(torch.full_like(command, .10), .40 * command)
    accuracy = torch.exp(-((velocity[:, 0] - command).square()
                          + velocity[:, 1].square()) / sigma.square())
    # Running impacts tolerate more slip; independent promotion limits remain
    # stricter evidence than this smooth shaping term.
    traction_scale = torch.where(command > RUN_SPEED, .14, .08)
    traction = torch.exp(-(slip / traction_scale).square())
    if advance_fraction is None:
        advance_fraction = progress.clamp_min(0.)  # stateless diagnostic callers
    if swing_budget is None:
        swing_budget = torch.ones_like(command)
    if recent_step is None:
        recent_step = events.sum(1) > 0
    legal_support = contact.any(1) | (command > RUN_SPEED)
    locomoting = ((~contact).any(1) | recent_step).float()*legal_support
    fresh_progress = advance_fraction * traction * locomoting
    swing = ~expected_contact
    actual_swing = swing & ~contact
    desired_clearance = .025 + .012 * command.clamp_max(2.5)
    lift_quality = (clearance / desired_clearance[:, None]).clamp(0., 1.)
    # The former 5cm lateral band fought the COM shift needed for single support.
    scales = torch.tensor([.08, .10, .035], device=relfeet.device)
    foot_quality = torch.exp(-((relfeet - target_feet) / scales).square().sum(-1))
    swing_quality = (actual_swing * lift_quality * foot_quality).sum(1) * swing_budget * legal_support
    activity = torch.maximum(fresh_progress,
                             swing_quality.clamp_max(1.))
    live = healthy * moving
    progress_activity = (fresh_progress*accuracy*healthy).clamp(0., 1.)
    return dict(
        track=2.5 * fresh_progress * accuracy * live,
        progress=.75 * (fresh_progress + progress.clamp_max(0.)) * live,
        reference=.35 * torch.exp(-reference_error / .06) * activity * live,
        gait=.80 * swing_quality * live,
        missing_swing=-.35 * (swing & contact).float().mean(1) * live,
        completed_step=.30 * events.sum(1) * live / dt,
        flight=running_flight(command, contact, expected_contact, clearance, live,
                             advance_fraction, swing_budget, recent_step),
        walking_flight=-1.0*moving*(command <= RUN_SPEED)*(~contact.any(1)),
        # Do NOT multiply this cost by eligibility: crossing/slipping/falling
        # must not turn off the cost while posture income remains active.
        motion_deficit=-1.80*moving*(1.-progress_activity),
        # Standing balance should not lose 75% of its incentive when a walking
        # command begins. Promotion still requires bilateral steps and net speed.
        posture_scale=torch.ones_like(command),
        progress_fraction=progress,
        swing_quality=swing_quality,
        progress_activity=progress_activity,
    )
