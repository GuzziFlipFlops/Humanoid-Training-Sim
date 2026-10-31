"""Training-only rehearsal at the exact commands required by independent gates."""
from curriculum_provenance import EVALUATION_SPEEDS

TRAINING_COMMAND_VERSION = 'gait_consistent_running_frontier_v2'


def sample_training_speeds(upper, ids, r, v):
    """Keep the old 32% standing / 24% prior / 44% frontier mixture.

    Prior rows now rehearse discrete evaluated speeds, rather than a widening
    uniform interval. Frontier rows include its exact command as well as the
    original 0.8-to-1.0 range. This changes training exposure, never motor actions.
    Draw count and the permanently stopped one-fifth cohort remain unchanged.
    """
    import torch  # Keep supervisor/worker pre-AppLauncher imports CPU-only.
    assert 0 <= upper <= 5 and r.shape == v.shape == ids.shape
    if upper == 0:
        return torch.zeros_like(r)
    prior = [s for s in EVALUATION_SPEEDS if 0 < s < upper-1e-5]
    anchors = torch.tensor(prior or [upper], device=r.device, dtype=r.dtype)
    index = (v * len(anchors)).long().clamp_max(len(anchors)-1)
    # Keep frontier exposure inside the current running gait. At the first
    # running stage, the old 1.2-1.5 interval predominantly trained walking.
    # Draw count, stopped/prior cohorts and all <=1.0 commands stay identical.
    frontier_lower = max(1.5, upper*.80) if upper >= 1.5 else upper*.80
    frontier = (upper*(.80+.20*v) if frontier_lower == upper*.80
                else frontier_lower+(upper-frontier_lower)*v)
    speed = torch.where(r < .15, 0., torch.where(r < .45, anchors[index],
            torch.where(r < .75, upper, frontier)))
    return torch.where(ids % 5 == 0, 0., speed)


def command_coverage(commands, upper):
    values = commands[:, 0]
    return {
        'version': TRAINING_COMMAND_VERSION,
        'stopped_fraction': float((values <= .1).float().mean()),
        'exact_command_fractions': {str(s): float(((values-s).abs()<.005).float().mean())
                                    for s in EVALUATION_SPEEDS if 0 < s <= upper+1e-5},
        'minimum_m_s': float(values.min()), 'maximum_m_s': float(values.max()),
        'scope': 'Current full-population training commands; not a behavioral gate.',
    }
