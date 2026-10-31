"""Independent landing measurements from physics, never training rewards.

Measure continuous air/load dwell, clearance, foot/body travel and impact timing.
Flight is legal; paired hops are not alternating gait.
"""
import torch


class IndependentLandings:
    def __init__(self, count, device):
        self.air = torch.zeros((count,2),device=device)
        self.ground = torch.zeros_like(self.air)
        self.armed = torch.zeros_like(self.air,dtype=torch.bool)
        self.maximum_clearance = torch.zeros_like(self.air)
        self.previous_contact = torch.ones_like(self.armed)
        self.previous_position = torch.zeros((count,2,2),device=device)
        self.takeoff_position = self.previous_position.clone()
        self.takeoff_root = self.previous_position.clone()
        self.heading = self.previous_position.clone()
        self.touch_time = torch.full_like(self.air,-100.)
        self.time = 0.
        self.initialized = False

    def update(self, contact, position, clearance, root_xy, forward, command, alive, dt):
        self.time += dt
        if not self.initialized:
            self.previous_position.copy_(position)
            self.previous_contact.copy_(contact)
            self.initialized = True
        takeoff = self.previous_contact & ~contact
        self.takeoff_position = torch.where(takeoff[:,:,None],self.previous_position,self.takeoff_position)
        self.takeoff_root = torch.where(takeoff[:,:,None],root_xy[:,None],self.takeoff_root)
        self.heading = torch.where(takeoff[:,:,None],forward[:,None],self.heading)
        self.maximum_clearance = torch.where(takeoff,0.,self.maximum_clearance)
        self.maximum_clearance = torch.maximum(self.maximum_clearance,clearance*(~contact))
        self.armed |= (self.air >= .08-1e-5)
        touchdown = contact & ~self.previous_contact & self.armed
        self.touch_time = torch.where(touchdown,self.time,self.touch_time)
        paired = (self.touch_time[:,0]-self.touch_time[:,1]).abs() <= .08+1e-5
        hop = paired & touchdown.any(1) & alive & (command > .08)
        touches = touchdown.sum(1)*alive*(command > .08)
        self.air = torch.where(contact,0.,self.air+dt)
        self.ground = torch.where(contact,self.ground+dt,0.)
        landing = self.armed & contact & (self.ground >= .06-1e-5)
        foot_travel = ((position-self.takeoff_position)*self.heading).sum(-1)
        root_travel = ((root_xy[:,None]-self.takeoff_root)*self.heading).sum(-1)
        minimum_clearance = torch.where(command > 1.1,.020,.010)
        accepted = (landing & (self.maximum_clearance >= minimum_clearance[:,None])
            & (foot_travel >= .020) & (root_travel >= .003) & ~paired[:,None]
            & alive[:,None] & (command > .08)[:,None])
        event_clearance = self.maximum_clearance*accepted
        self.armed &= ~landing
        self.previous_contact.copy_(contact)
        self.previous_position.copy_(position)
        return accepted.float(),event_clearance,hop.float(),touches.float()
