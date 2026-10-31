"""Measured-joint FK native-forward comparison; never an optical gaze claim."""
import json, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
review = read(OUT / 'matched_coupled_teacher_review.json')
run = Path(read(OUT / 'coupled_training_trial/diagnostic_pause.json')['run'])
root = ET.parse(run / 'source/physical_v29/training/assets/robot_training.urdf').getroot()
joints = {j.find('child').attrib['link']: j for j in root.findall('joint')}
head = next(j for j in root.findall('joint') if j.attrib['name'] == 'head_pitch').find('child').attrib['link']
chain = []
link = head
while link in joints:
    j = joints[link]
    chain.insert(0, j)
    link = j.find('parent').attrib['link']
assert link == 'pelvis'

def origin(j):
    a = j.find('origin')
    return np.eye(3) if a is None else Rotation.from_euler('xyz', np.fromstring(a.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()

zero = np.eye(3)
for j in chain:
    zero = zero @ origin(j)
native_forward = zero.T @ np.array([1., 0., 0.])
assert np.allclose(zero @ native_forward, [1., 0., 0.])
records = []
for seed in [229, 241]:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    plotted = False
    for row in review['records']:
        if row['updates'] != 128 or row['seed'] != seed:
            continue
        for label in ['nominal', 'push']:
            if label not in row['tests']:
                continue
            with np.load(Path(row['tests'][label]['folder']) / 'gait_diagnostics_50hz.npz') as d:
                q, names = d['q'], d['joint_names'].tolist()
                shape = q.shape[:2]
                rot = np.tile(np.eye(3), shape + (1, 1))
                for j in chain:
                    rot = rot @ origin(j)
                    if j.attrib['type'] == 'revolute':
                        axis = np.fromstring(j.find('axis').attrib['xyz'], sep=' ')
                        axis /= np.linalg.norm(axis)
                        angle = q[:, :, names.index(j.attrib['name'])]
                        rot = rot @ Rotation.from_rotvec((angle[:, :, None] * axis).reshape(-1, 3)).as_matrix().reshape(shape + (3, 3))
                quat = d['root_state'][:, :, 3:7]
                world = Rotation.from_quat(quat[:, :, [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(shape + (3, 3)) @ rot
                forward = (world @ native_forward[..., None])[..., 0]
                yaw = np.arctan2(forward[:, :, 1], forward[:, :, 0])
                error = np.arctan2(np.sin(yaw - d['heading_target']), np.cos(yaw - d['heading_target']))
                pitch = np.arctan2(forward[:, :, 2], np.linalg.norm(forward[:, :, :2], axis=-1))
                for speed in np.unique(d['commands']):
                    ids = np.flatnonzero(np.isclose(d['commands'], speed))
                    e, p = [], []
                    for robot in ids:
                        mask = d['time'] >= 2.02
                        terminal = np.flatnonzero(d['done'][:, robot])
                        if len(terminal):
                            mask &= np.arange(len(mask)) < terminal[0]
                        e.extend(np.rad2deg(error[mask, robot]).tolist())
                        p.extend(np.rad2deg(pitch[mask, robot]).tolist())
                    if not e:
                        continue
                    records.append(dict(seed=seed, branch=row['branch'], checkpoint=row['checkpoint'], sha256=row['checkpoint_sha256'], test=label,
                        command=float(speed), selected_representatives=len(ids), samples=len(e), first_episode_only=True,
                        world_native_heading_RMS_deg=float(np.sqrt(np.mean(np.square(e)))), world_native_heading_p95_deg=float(np.quantile(np.abs(e), .95)),
                        world_native_elevation_RMS_deg=float(np.sqrt(np.mean(np.square(p)))), world_native_elevation_mean_deg=float(np.mean(p))))
                    if label == 'nominal' and any(np.isclose(speed, x) for x in [.15, 1.5, 2.5, 3.5]):
                        col = next(i for i, x in enumerate([.15, 1.5, 2.5, 3.5]) if np.isclose(speed, x))
                        robot = ids[0]
                        mask = (d['time'] >= 2.02) & (d['time'] <= 8)
                        terminal = np.flatnonzero(d['done'][:, robot])
                        if len(terminal):
                            mask &= np.arange(len(mask)) < terminal[0]
                        axes[0, col].plot(d['time'][mask], np.rad2deg(error[mask, robot]), label=row['branch'])
                        axes[1, col].plot(d['time'][mask], np.rad2deg(pitch[mask, robot]), label=row['branch'])
                        axes[0, col].set_title(f'{speed:g} m/s, representative {robot}')
                        plotted = True
    for i in range(2):
        for ax in axes[i]:
            ax.grid(alpha=.2)
            ax.set_xlabel('Simulation seconds')
            ax.set_ylabel('Native heading error (deg)' if i == 0 else 'Native elevation (deg)')
    if plotted:
        axes[0, 0].legend()
        fig.suptitle(f'Seed {seed}: measured-joint FK native-forward head axis, 128-update actors\nIncludes base, waist and neck; CAD face/camera optical axis remains uncalibrated')
        fig.tight_layout(rect=[0, 0, 1, .92])
        fig.savefig(OUT / f'seed{seed}_native_head_axis.png', dpi=160)
    plt.close(fig)
result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), records=records,
    chain=[j.attrib['name'] for j in chain], native_forward_axis_in_head_link=native_forward.tolist(),
    optical_gaze_calibrated=False, natural_motion_certified=False,
    scope='16 selected representatives per gate at 50 Hz, first episodes only after 2.02 s. Native-zero forward maps to pelvis +X at zero modeled joints. Prior same-runtime Isaac/FK parity is known, but this is CPU FK and not a new optical or locomotion gate.')
(OUT / 'native_head_axis_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps([r for r in records if r['test'] == 'nominal' and r['command'] in [.15, 1.5, 3.5]], indent=2))
