"""Independent physical landings, never training credit or its rate filters.

All qualifying landings remain counted, including repeated same-foot steps.
Order is reported separately so deleting unwanted steps cannot fake cadence.
Hop fractions count each physical touchdown at most once, even for overlapping
pairs. A landing waits through the paired-impact window before classification.
"""
import torch


class IndependentLandings:
    def __init__(self, count, device):
        self.air = torch.zeros((count, 2), device=device)
        self.ground = torch.zeros_like(self.air)
        self.active = torch.zeros_like(self.air, dtype=torch.bool)
        self.armed = self.active.clone()
        self.paired = self.active.clone()
        self.maximum_clearance = torch.zeros_like(self.air)
        self.previous_contact = torch.ones_like(self.armed)
        self.previous_position = torch.zeros((count, 2, 2), device=device)
        self.takeoff_position = self.previous_position.clone()
        self.takeoff_root = self.previous_position.clone()
        self.heading = self.previous_position.clone()
        self.impact_travel = torch.zeros_like(self.air)
        self.impact_root_travel = torch.zeros_like(self.air)
        self.touch_time = torch.full_like(self.air, -100.)
        self.hop_marked = self.active.clone()
        self.last_foot = torch.full((count,), -1, dtype=torch.long, device=device)
        self.same_foot_events = torch.zeros(count, device=device)
        self.time = 0.
        self.initialized = False

    def update(self, contact, position, clearance, root_xy, forward, command, alive, dt):
        self.time += dt
        if not self.initialized:
            self.previous_position.copy_(position)
            self.previous_contact.copy_(contact)
            self.initialized = True
        live = alive & (command > .08)
        takeoff = self.previous_contact & ~contact & (self.ground >= .06-1e-5)
        self.takeoff_position = torch.where(takeoff[:, :, None], self.previous_position, self.takeoff_position)
        self.takeoff_root = torch.where(takeoff[:, :, None], root_xy[:, None], self.takeoff_root)
        self.heading = torch.where(takeoff[:, :, None], forward[:, None], self.heading)
        self.active = (self.active | takeoff) & live[:, None]
        self.armed = torch.where(takeoff, False, self.armed) & self.active
        self.paired = torch.where(takeoff, False, self.paired)
        self.maximum_clearance = torch.where(takeoff, 0., self.maximum_clearance)
        self.maximum_clearance = torch.maximum(self.maximum_clearance, clearance*(~contact))
        # Count raw air-qualified impacts independently of valid stance/steps.
        touchdown = contact & ~self.previous_contact & (self.air >= .08-1e-5)
        self.touch_time = torch.where(touchdown, self.time, self.touch_time)
        self.hop_marked = torch.where(touchdown, False, self.hop_marked)
        paired = ((self.touch_time[:, 0]-self.touch_time[:, 1]).abs() <= .08+1e-5)
        paired &= touchdown.any(1) & live
        new_hop_touches = paired[:, None] & ~self.hop_marked
        hop_touches = new_hop_touches.sum(1).float()
        self.hop_marked |= new_hop_touches
        self.paired |= paired[:, None] & self.active
        touches = (touchdown*live[:, None]).sum(1).float()
        # Freeze travel at impact, before any sliding during settling.
        self.impact_travel = torch.where(touchdown,
            ((position-self.takeoff_position)*self.heading).sum(-1), self.impact_travel)
        self.impact_root_travel = torch.where(touchdown,
            ((root_xy[:, None]-self.takeoff_root)*self.heading).sum(-1), self.impact_root_travel)
        self.air = torch.where(contact, 0., self.air+dt)
        self.ground = torch.where(contact, self.ground+dt, 0.)
        self.armed |= (self.air >= .08-1e-5) & self.active
        landing = self.armed & contact & (self.ground >= .10-1e-5)
        minimum_clearance = torch.where(command > 1.1, .020, .010)
        accepted = (landing & (self.maximum_clearance >= minimum_clearance[:, None])
            & (self.impact_travel >= .020) & (self.impact_root_travel >= .003)
            & ~self.paired & live[:, None])
        self.same_foot_events.zero_()
        for side in (0, 1):
            self.same_foot_events += (accepted[:, side] & (self.last_foot == side)).float()
            self.last_foot = torch.where(accepted[:, side], side, self.last_foot)
        event_clearance = self.maximum_clearance*accepted
        self.active &= ~landing
        self.armed &= self.active
        self.previous_contact.copy_(contact)
        self.previous_position.copy_(position)
        return accepted.float(), event_clearance, hop_touches, touches
