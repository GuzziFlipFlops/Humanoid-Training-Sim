"""Diagnostic command-local tolerance scaling; no weight or gate changes.

The e-fold velocity-error width is the EXISTING speed tolerance for moving
commands <=1 m/s. Stopped and faster commands retain the pinned 0.5 m/s width.
This tests objective precision, not a claim that a narrow kernel is optimal.
"""
import torch

VERSION = 'diagnostic_slow_tracking_existing_tolerance_v1'

def tracking_width(commands, original_std=.5):
    speed = commands[:, :2].norm(dim=1)
    tolerance = torch.maximum(torch.full_like(speed, .10), speed * .12)
    eligible = (speed > .1) & (speed <= 1.)
    return torch.where(eligible, tolerance, torch.full_like(speed, original_std))

def score(commands, yaw_velocity, original_std=.5):
    error = (commands[:, :2] - yaw_velocity[:, :2]).square().sum(1)
    return torch.exp(-error / tracking_width(commands, original_std).square())
