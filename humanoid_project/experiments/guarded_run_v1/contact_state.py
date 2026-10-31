"""Causal, reset-safe completed-stance statistics; independent of Isaac imports."""
import torch


class CompletedStances:
    def __init__(self, count, device, dt=0.02, debounce=0.03):
        self.dt, self.debounce = dt, debounce
        shape = (count, 2)
        self.stable = torch.zeros(shape, dtype=torch.bool, device=device)
        self.initialized = torch.zeros(count, dtype=torch.bool, device=device)
        self.pending = torch.zeros(shape, device=device)
        self.started = torch.zeros(shape, device=device)
        self.full = torch.zeros(shape, dtype=torch.bool, device=device)
        self.last = torch.zeros(shape, device=device)
        self.age = torch.full(shape, 100., device=device)
        self.count = torch.zeros(shape, dtype=torch.long, device=device)
        self.time = torch.zeros(count, device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for a in (self.stable, self.initialized, self.pending, self.started,
                  self.full, self.last, self.count, self.time):
            a[ids] = 0
        self.age[ids] = 100.

    def update(self, contact):
        self.time += self.dt
        self.age += self.dt
        first = ~self.initialized
        self.stable[first] = contact[first]
        self.initialized[:] = True
        differs = contact != self.stable
        self.pending = torch.where(differs, self.pending + self.dt, 0.)
        change = differs & (self.pending >= self.debounce)
        transition = self.time[:, None] - self.pending + self.dt
        touchdown = change & contact
        liftoff = change & ~contact
        duration = transition - self.started
        completed = liftoff & self.full & (duration >= .06) & (duration <= 1.5)
        self.last = torch.where(completed, duration, self.last)
        self.age = torch.where(completed, 0., self.age)
        self.count += completed.long()
        self.started = torch.where(touchdown, transition, self.started)
        self.full = torch.where(touchdown, True, torch.where(liftoff, False, self.full))
        self.stable = torch.where(change, contact, self.stable)
        self.pending = torch.where(change, 0., self.pending)
        return completed, torch.where(completed, duration, 0.)

    def penalty(self):
        valid = (self.count > 0).all(1) & (self.age < 1.5).all(1)
        relative = (self.last[:, 0] - self.last[:, 1]).abs() / self.last.sum(1).clamp_min(.12)
        # Deadband: tolerate modest natural variability; never a prescribed step length or angle.
        cost = ((relative - .10).clamp_min(0.) / .90).square().clamp_max(1.)
        return torch.where(valid, cost, 0.)


def differential_cost(torque, speed, torque_limit=150.88, speed_limit=9.56):
    """Input is [environment, leg, (pitch,roll)], never articulation-order reshaping."""
    tm = torch.stack((torque[..., 0] + torque[..., 1], torque[..., 0] - torque[..., 1]), -1)
    vm = torch.stack((speed[..., 0] + speed[..., 1], speed[..., 0] - speed[..., 1]), -1)
    return (((tm.abs() / torque_limit - 1.).clamp_min(0.) ** 2)
            + ((vm.abs() / speed_limit - 1.).clamp_min(0.) ** 2)).sum((1, 2))
