"""CPU-only comparison of saved COM with frozen URDF sole geometry.

Never changes a policy, reward, gate or active run. Footprint bounds are not
contact-pressure measurements; near-ground corners are diagnostic only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from build_unitree_review import fk, pose


def hull(points):
    points = sorted(set(map(tuple, points)))
    if len(points) < 3:
        return np.asarray(points)
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in points:
        while len(lower) > 1 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) > 1 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1]+upper[:-1])


def inside(p, polygon):
    if len(polygon) < 3:
        return False
    edge = np.roll(polygon, -1, axis=0)-polygon
    rel = p-polygon
    return bool(np.all(edge[:, 0]*rel[:, 1]-edge[:, 1]*rel[:, 0] >= -1e-7))


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--evaluation', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
urdf = args.run/'source/physical_v29/training/assets/robot_training.urdf'
tree = ET.parse(urdf).getroot()
evaluation = args.run/args.evaluation
with np.load(evaluation/'gait_diagnostics_50hz.npz') as data:
    trace = {key:data[key] for key in ('joint_names', 'time', 'q', 'root_state', 'com', 'foot_position', 'contact')}
with np.load(evaluation/'telemetry.npz') as data:
    telemetry = {key:data[key] for key in ('step_dt', 'metrics')}
names = trace['joint_names'].tolist()
dt = float(telemetry['step_dt'])
start = int(2/dt)
rows, fk_errors = [], []
geometry = {}
for side in ('left', 'right'):
    collision = tree.find(f"link[@name='{side}_foot']/collision")
    geometry[side] = (pose(collision.find('origin')),
                      np.fromstring(collision.find('geometry/box').get('size'), sep=' '))
for tick in range(start, len(trace['time']), 10):
    for robot in range(trace['q'].shape[1]):
        state = trace['root_state'][tick, robot]
        com = trace['com'][tick, robot]
        feet = trace['foot_position'][tick, robot]
        contact = trace['contact'][tick, robot].astype(bool)
        w, x, y, z = state[3:7]
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        rel_feet = (feet[:, :2]-state[:2])@rotation.T
        rel_com = (com[:2]-state[:2])@rotation.T
        old_low = rel_feet.min(0)-[.055, .035]
        old_high = rel_feet.max(0)+[.055, .035]
        old_outside = bool(not contact.all() or np.any(rel_com < old_low) or np.any(rel_com > old_high))
        recorded = bool(telemetry['metrics'][tick-start, robot, 15])
        assert old_outside == recorded, 'Existing support formula was not reproduced'
        transforms, _ = fk(tree, dict(zip(names, trace['q'][tick, robot])), state)
        centers, full, inset, z_spans = [], [], [], []
        for j, side in enumerate(('left', 'right')):
            foot = transforms[side+'_foot']
            fk_errors.append(float(np.max(np.abs(foot[:3, 3]-feet[j]))))
            offset, size = geometry[side]
            T = foot@offset
            centers.append(T[:2, 3])
            corners = np.array([[sx*size[0]/2, sy*size[1]/2, -size[2]/2, 1]
                                for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])@T.T
            z_spans.append(float(np.ptp(corners[:, 2])))
            if contact[j]:
                full.extend(corners[:, :3])
                local = np.array([[sx*(size[0]/2-.01), sy*(size[1]/2-.005), -size[2]/2, 1]
                                  for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])@T.T
                inset.extend(local[:, :3])
        centers = (np.array(centers)-state[:2])@rotation.T
        corrected_low = centers.min(0)-[.055, .035]
        corrected_high = centers.max(0)+[.055, .035]
        shifted_outside = bool(not contact.all() or np.any(rel_com < corrected_low) or np.any(rel_com > corrected_high))
        full = np.asarray(full).reshape(-1, 3)
        inset = np.asarray(inset).reshape(-1, 3)
        full_outside = not inside(com[:2], hull(full[:, :2]))
        inset_outside = not inside(com[:2], hull(inset[:, :2]))
        near = full[full[:, 2] <= full[:, 2].min()+.002] if len(full) else full
        near_outside = not inside(com[:2], hull(near[:, :2]))
        rows.append(dict(t=float(trace['time'][tick]), robot=robot, old_outside=old_outside,
                         shifted_rectangle_outside=shifted_outside, full_sole_outside=full_outside,
                         inset_sole_outside=inset_outside, near_ground_corner_outside=near_outside,
                         both_loaded=bool(contact.all()), sole_vertical_spans_m=z_spans))
assert max(fk_errors) < .001, 'Frozen URDF FK does not match simulated feet'
report = dict(
    source=str(evaluation), urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
    actual_simulator_com=True, sample_ticks=len(rows)//trace['q'].shape[1],
    sampled_robots=trace['q'].shape[1], samples=len(rows), max_fk_error_m=max(fk_errors),
    sole_geometry={side:dict(origin_xyz=T[:3, 3].tolist(), box_size_m=size.tolist())
                   for side, (T, size) in geometry.items()},
    fractions={key:float(np.mean([row[key] for row in rows])) for key in
               ('old_outside', 'shifted_rectangle_outside', 'full_sole_outside',
                'inset_sole_outside', 'near_ground_corner_outside', 'both_loaded')},
    mean_sole_vertical_span_m=float(np.mean([row['sole_vertical_spans_m'] for row in rows])),
    scope='Every tenth post-warmup tick of the first 16 saved robots. Shifted rectangle '
          'uses unchanged .055/.035 half sizes but actual collision centers. Inset footprint '
          'uses 10mm longitudinal and 5mm lateral margins. Near-ground uses corners within '
          '2mm of lowest loaded sole point. None is measured contact pressure or a gate pass. '
          'No source, policy, active run or qualification was modified.',
    rows=rows,
)
args.output.mkdir(parents=True, exist_ok=True)
(args.output/'sole_support_audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key:value for key,value in report.items() if key != 'rows'}, indent=2))
