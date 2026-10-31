"""Exact box-URDF kinematics and separating-axis checks for recovery preflight."""
from pathlib import Path
import itertools
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'assets/urdf/humanoid_5ft_isaac.urdf'


def numbers(value): return np.fromstring(value, sep=' ')


class Geometry:
    def __init__(self):
        root = ET.parse(URDF).getroot()
        self.links = {}
        self.joints = []
        for link in root.findall('link'):
            c = link.find('collision')
            if c.find('geometry/box') is None:
                raise ValueError('Recovery audit currently requires box colliders')
            self.links[link.get('name')] = dict(center=numbers(c.find('origin').get('xyz')),
                rotation=Rotation.from_euler('xyz', numbers(c.find('origin').get('rpy', '0 0 0'))).as_matrix(),
                half=numbers(c.find('geometry/box').get('size')) / 2,
                mass=float(link.find('inertial/mass').get('value')))
        for joint in root.findall('joint'):
            self.joints.append(dict(name=joint.get('name'), parent=joint.find('parent').get('link'),
                child=joint.find('child').get('link'), xyz=numbers(joint.find('origin').get('xyz')),
                rotation=Rotation.from_euler('xyz', numbers(joint.find('origin').get('rpy', '0 0 0'))).as_matrix(),
                axis=numbers(joint.find('axis').get('xyz')),
                lower=float(joint.find('limit').get('lower')), upper=float(joint.find('limit').get('upper'))))
        self.names = [j['name'] for j in self.joints]
        self.nominal = {n: (-.18 if 'hip_pitch' in n or 'ankle_pitch' in n else
                            .36 if 'knee_pitch' in n else .10 if 'shoulder_pitch' in n else
                            .25 if 'elbow' in n else 0.) for n in self.names}
        # Only direct neighbors or serial gimbal frames at the same mechanical
        # joint are filtered. No left/right leg or forearm/torso exclusion.
        self.filtered = {frozenset((j['parent'], j['child'])) for j in self.joints}
        clusters = [['pelvis', 'waist_yaw_link', 'torso'], ['torso', 'neck_yaw_link', 'head']]
        for side in ('left', 'right'):
            clusters += [['pelvis', f'{side}_hip_yaw_link', f'{side}_hip_roll_link', f'{side}_thigh'],
                         [f'{side}_shank', f'{side}_ankle_pitch_link', f'{side}_foot'],
                         ['torso', f'{side}_shoulder_pitch_link', f'{side}_upper_arm']]
        for cluster in clusters:
            self.filtered.update(frozenset(pair) for pair in itertools.combinations(cluster, 2))
        self.pairs = [(a, b) for a, b in itertools.combinations(self.links, 2)
                      if frozenset((a, b)) not in self.filtered]

    def fk(self, angles=None, root_rotation=None, root_pos=None):
        q = self.nominal if angles is None else angles
        poses = {'pelvis': (np.eye(3) if root_rotation is None else root_rotation,
                            np.zeros(3) if root_pos is None else np.array(root_pos))}
        for j in self.joints:
            r, p = poses[j['parent']]
            poses[j['child']] = (r @ j['rotation'] @ Rotation.from_rotvec(j['axis'] * q[j['name']]).as_matrix(),
                                 p + r @ j['xyz'])
        return poses

    def boxes(self, poses):
        return {n: (p + r @ self.links[n]['center'], r @ self.links[n]['rotation'], self.links[n]['half'])
                for n, (r, p) in poses.items()}

    @staticmethod
    def overlap(a, b):
        ca, ra, ha = a; cb, rb, hb = b
        axes = [*ra.T, *rb.T, *(np.cross(x, y) for x in ra.T for y in rb.T)]
        min_depth = float('inf')
        for axis in axes:
            length = np.linalg.norm(axis)
            if length < 1e-8: continue
            axis = axis / length
            depth = np.abs(axis @ ra) @ ha + np.abs(axis @ rb) @ hb - abs(axis @ (cb - ca))
            if depth <= 0: return 0.
            min_depth = min(min_depth, depth)
        return min_depth

    def penetrations(self, poses, tolerance=.001):
        boxes = self.boxes(poses)
        return [(a, b, depth) for a, b in self.pairs
                if (depth := self.overlap(boxes[a], boxes[b])) > tolerance]

    def floor_z(self, poses):
        return min(c[2] - np.abs(r[2]) @ h for c, r, h in self.boxes(poses).values())

    def pose(self, group, rng, jitter=.04):
        q = {j['name']: float(np.clip(self.nominal[j['name']] + rng.uniform(-jitter, jitter),
                                      j['lower'] + .02, j['upper'] - .02)) for j in self.joints}
        rotations = [(0., -np.pi / 2, 0.), (0., np.pi / 2, 0.), (np.pi / 2, 0., 0.), (-np.pi / 2, 0., 0.)]
        rotation = Rotation.from_euler('xyz', rotations[group]).as_matrix()
        poses = self.fk(q, rotation)
        return q, rotation, -self.floor_z(poses) + .006


if __name__ == '__main__':
    g = Geometry()
    print('Bodies', len(g.links), 'joints', len(g.joints), 'mass', sum(x['mass'] for x in g.links.values()))
    print('Filtered mechanical-neighbor pairs', len(g.filtered))
    print('Neutral non-neighbor penetrations', g.penetrations(g.fk()))
    rng = np.random.default_rng(47)
    for group in range(4):
        q, rotation, z = g.pose(group, rng)
        print('pose', group, 'pelvis_height', z, 'penetrations', g.penetrations(g.fk(q, rotation, [0, 0, z])))
