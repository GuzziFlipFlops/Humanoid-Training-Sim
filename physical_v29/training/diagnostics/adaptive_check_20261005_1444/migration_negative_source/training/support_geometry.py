"""Gravity-projected planar support geometry, usable without Isaac imports."""
import torch
from unitree_reward_port import quat_apply_inverse, yaw_quat


def horizontal_support(root_quat, root_pos, feet, com, contact):
    """Retain the existing conservative sole rectangle, in a yaw-only frame.

    Full root roll/pitch would mix COM/foot height into the support plane.
    This is a standing diagnostic, not a dynamic ZMP/whole-body guarantee.
    """
    yaw = yaw_quat(root_quat)
    rel_com = quat_apply_inverse(yaw, com-root_pos)
    rel_feet = quat_apply_inverse(yaw[:,None,:].expand(-1,2,-1).reshape(-1,4),
        (feet-root_pos[:,None,:]).reshape(-1,3)).reshape(-1,2,3)
    low = rel_feet[:,:,:2].amin(1)-torch.tensor([.055,.035],device=feet.device,dtype=feet.dtype)
    high = rel_feet[:,:,:2].amax(1)+torch.tensor([.055,.035],device=feet.device,dtype=feet.dtype)
    excess = torch.maximum(low-rel_com[:,:2], rel_com[:,:2]-high).clamp_min(0.)
    outside = (excess>0.).any(1)|(~contact.all(1))
    return outside, excess.norm(dim=1)
