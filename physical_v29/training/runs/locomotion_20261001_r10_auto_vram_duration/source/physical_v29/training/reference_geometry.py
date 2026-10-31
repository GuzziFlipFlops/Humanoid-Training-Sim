"""Foot positions from the unchanged CAD IK joint clips, in pelvis coordinates."""
import numpy as np


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
