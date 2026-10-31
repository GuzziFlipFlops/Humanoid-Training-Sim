"""Read-only reconstruction of the frozen physical action-to-target map."""
import torch

def clone_observation_mapping(observation):
    """Own teacher tensor storage for raw dict and wrapped TensorDict alike."""
    return {key: value.clone() for key, value in observation.items()}

def project_targets(action, reference, scales, low, high, previous, slew, step_dt, pairs=()):
    effective = action.clamp(-3., 3.)
    desired = reference + scales * effective
    joint_clipped = (desired < low) | (desired > high)
    desired = torch.maximum(torch.minimum(desired, high), low)
    slew_limited = (desired - previous).abs() > slew * step_dt
    desired = previous + (desired - previous).clamp(-slew * step_dt, slew * step_dt)
    invalid = torch.zeros_like(desired, dtype=torch.bool)
    for _, ids, model in pairs:
        _, _, valid, _ = model.state(desired[:, ids])
        invalid[:, ids] = ~valid[:, None]
        desired[:, ids] = torch.where(valid[:, None], desired[:, ids], previous[:, ids])
    return desired, joint_clipped, slew_limited, invalid

def ramp(t, beginning=5., seconds=.5):
    x = max(0., min(1., (t - beginning) / seconds))
    return x*x*(3.-2.*x)
