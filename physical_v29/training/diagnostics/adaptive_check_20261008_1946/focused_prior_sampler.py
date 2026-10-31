"""Bounded diagnostic prior-command resampling; no actor or physics override."""
from command_curriculum import sample_training_speeds as baseline
from curriculum_provenance import EVALUATION_SPEEDS

VERSION = 'focused_failed_prior_rehearsal_v1'
FOCUS = (.15, 1.5)


def sample_focused(upper, ids, r, v):
    import torch
    original = baseline(upper, ids, r, v)
    if abs(upper - 3.5) > 1e-5:
        return original
    prior = [s for s in EVALUATION_SPEEDS if 0 < s < upper - 1e-5]
    # One additional copy of each repeatedly failed exact prior command.
    # The existing standing/prior/frontier mixture and draw count are retained.
    anchors = [s for s in prior for _ in range(2 if s in FOCUS else 1)]
    table = torch.tensor(anchors, device=r.device, dtype=r.dtype)
    index = (v * len(anchors)).long().clamp_max(len(anchors) - 1)
    selected = (r >= .15) & (r < .45) & (ids % 5 != 0)
    return torch.where(selected, table[index], original)
