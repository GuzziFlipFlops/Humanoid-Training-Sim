"""Foot positions from the unchanged CAD IK joint clips, in pelvis coordinates."""
import numpy as np


def target_limits(metadata, names=None):
    """The exact controller envelope, without expanding any CAD joint limit."""
    by_name = {j['name']: j for j in metadata['joints']}
    names = names or list(by_name)
    lower, upper = [], []
    for name in names:
        lo, hi = by_name[name]['limits_rad']
        center, radius = .5*(lo+hi), .475*(hi-lo)
        lo, hi = center-radius, center+radius
        if 'ankle_roll' in name: lo, hi = max(lo, -.13), min(hi, .13)
        if name in ('waist_pitch', 'waist_roll'): lo, hi = -.20, .20
        lower.append(lo); upper.append(hi)
    return np.array(lower), np.array(upper)


def bounded_motion(metadata, motion):
    """Keep joint and foot guidance consistent with realizable controller targets.

    This projection is a feasibility guard, not a dynamically balanced gait.
    The original reference archive remains intact and its clipping is audited.
    """
    result = {k: motion[k].copy() for k in motion.files} if hasattr(motion, 'files') else {k: np.array(v).copy() for k,v in motion.items()}
    lo, hi = target_limits(metadata, motion['names'].tolist())
    assert np.all(result['stand'] >= lo) and np.all(result['stand'] <= hi), 'Standing prior exceeds controller envelope'
    result['q'] = np.clip(result['q'], lo, hi).astype(np.float32)
    return result


def foot_reference(metadata, motion):
    names = motion['names'].tolist()
    by_child = {j['child']: j for j in metadata['joints']}
    q = motion['q'].reshape(-1, len(names))
    all_feet = []
    for side in ('left', 'right'):
        chain = []
        link = side + '_foot'
        while link != 'pelvis':
            joint = by_child[link]
            chain.append(joint)
            link = joint['parent']
        pos = np.zeros((len(q), 3))
        rot = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
        for joint in reversed(chain):
            pos += np.einsum('nij,j->ni', rot, joint['origin_xyz_m'])
            axis = np.asarray(joint['axis'], dtype=float)
            x, y, z = axis
            skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            angle = q[:, names.index(joint['name'])]
            delta = (np.eye(3)[None] + np.sin(angle)[:, None, None] * skew
                     + (1 - np.cos(angle))[:, None, None] * (skew @ skew))
            rot = rot @ delta
        all_feet.append(pos)
    return np.stack(all_feet, axis=1).reshape(*motion['q'].shape[:2], 2, 3).astype(np.float32)
