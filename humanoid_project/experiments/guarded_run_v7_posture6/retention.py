"""Pure-torch slow-walk torso protection, calibrated to recorded passing gait."""
import math
import torch


def slow_torso_cost(tilt, command_speed):
    speed = command_speed.abs()
    # 8 degrees at .5 m/s, 6 at 1 m/s. The approved walker measured 6.63/4.52.
    # A cone, not an exact torso pose; full high-speed lean is handled separately.
    allowance = (10. - 4. * speed.clamp(0., 1.)) * math.pi / 180.
    excess = ((tilt - allowance).clamp_min(0.) / (5.*math.pi/180.)).clamp_max(2.)
    cost = torch.where(excess < .2, 2.5*excess.square(), excess-.1)
    walking = ((1.8-speed)/.8).clamp(0.,1.)
    return cost * walking


def high_speed_torso_cost(tilt, command_speed):
    """V5/V6 torso cone, exactly inactive through 1 m/s."""
    speed = command_speed.abs()
    allowance = (8. + speed.clamp(0., 6.)) * math.pi / 180.
    normalized = (tilt - allowance).clamp_min(0.) / .25
    cost = torch.where(normalized < 1., .5 * normalized.square(), normalized - .5)
    return cost.clamp_max(3.) * (speed - 1.).clamp(0., 1.)
