"""Pure training-assistance schedule shared by Isaac code and CPU tests."""
import torch


ROBOT_MASS_KG = 38.0
GRAVITY_M_S2 = 9.81
FULL_FORCE_N = ROBOT_MASS_KG * GRAVITY_M_S2
FADE_START_HEIGHT_M = 0.20
ZERO_HEIGHT_M = 0.70


def vertical_assist_force(height_m: torch.Tensor, body_weight_fraction: float) -> torch.Tensor:
    """Return a bounded +Z force that vanishes before standing can pass.

    The force is a discovery scaffold only.  Strict success begins at a 0.78 m
    pelvis height, while this function is identically zero from 0.70 m upward.
    """
    if not 0.0 <= body_weight_fraction <= 0.60:
        raise ValueError("body_weight_fraction must be in [0, 0.60]")
    fade = ((ZERO_HEIGHT_M - height_m) /
            (ZERO_HEIGHT_M - FADE_START_HEIGHT_M)).clamp(0.0, 1.0)
    return fade * (FULL_FORCE_N * body_weight_fraction)
