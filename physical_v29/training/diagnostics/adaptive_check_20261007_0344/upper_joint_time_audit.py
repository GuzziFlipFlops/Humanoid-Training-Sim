"""CPU-only joint time-series audit; never loads or changes a policy/simulator.

Run with --gate-review from review_gates.py and --output-dir for a new check.
Native joint angles are diagnostic, not a human-likeness qualification gate.
"""
import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def correlation(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    den = np.sqrt((a*a).sum(0) * (b*b).sum(0))
    out = np.divide((a*b).sum(0), den, out=np.full(den.shape, np.nan), where=den > 1e-8)
    return float(np.nanmedian(out)) if np.isfinite(out).any() else None


def coordination(q, ids, dt):
    hips = q[:, :, [ids['left_hip_pitch'], ids['right_hip_pitch']]]
    y = hips - hips.mean(0)
    freqs = np.fft.rfftfreq(len(q), dt)
    spectrum = np.abs(np.fft.rfft(y, axis=0))**2
    band = (freqs >= .5) & (freqs <= min(4.3, .45/dt))
    if spectrum[band].sum() < 1e-8:
        return {'hip_cycle_hz': None, 'arms': []}
    peak = np.flatnonzero(band)[np.argmax(spectrum[band].mean((1, 2)))]
    arms = []
    for side in ['left', 'right']:
        arm, hip = q[:, :, ids[side+'_shoulder_pitch']], q[:, :, ids[side+'_hip_pitch']]
        a, h = np.fft.rfft(arm-arm.mean(0), axis=0), np.fft.rfft(hip-hip.mean(0), axis=0)
        phase = np.angle(a[peak]*np.conj(h[peak]))
        amplitude = np.quantile(arm, .95, axis=0)-np.quantile(arm, .05, axis=0)
        weight = np.abs(a[peak])*np.abs(h[peak])
        phase_mean = np.angle(np.sum(weight*np.exp(1j*phase)))
        arms.append({'side': side, 'shoulder_hip_correlation': correlation(arm, hip),
                     'phase_at_hip_cycle_deg': float(np.rad2deg(phase_mean)),
                     'shoulder_pitch_p95_p05_range_deg': float(np.median(amplitude)),
                     'phase_is_weak_motion_diagnostic': bool(np.median(amplitude) < 5)})
    return {'hip_cycle_hz': float(freqs[peak]), 'arms': arms,
            'left_right_shoulder_pitch_correlation': correlation(q[:, :, ids['left_shoulder_pitch']], q[:, :, ids['right_shoulder_pitch']]),
            'left_right_elbow_correlation': correlation(q[:, :, ids['left_elbow_pitch']], q[:, :, ids['right_elbow_pitch']])}


def validate_metrics():
    t = np.arange(1000)*.02
    a = np.sin(2*np.pi*1.5*t)[:, None]
    assert abs(correlation(a, -a)+1) < 1e-12
    assert abs(correlation(a, a)-1) < 1e-12
    assert correlation(np.zeros_like(a), a) is None
    assert abs(np.rad2deg(np.angle(np.fft.rfft(-a[:, 0])[30]*np.conj(np.fft.rfft(a[:, 0])[30])))) == 180


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate-review', type=Path, default=Path(__file__).with_name('initial_gate_review.json'))
    p.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    p.add_argument('--reference-stage', default='run_300')
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_metrics()
    review = json.loads(args.gate_review.read_text(encoding='utf-8-sig'))
    selected = [('latest', review['latest'])]
    promoted = [x for x in review['promotions'] if x.get('stage') == args.reference_stage]
    if promoted and promoted[0]['checkpoint'] != review['latest']['checkpoint']:
        selected.append(('qualified_'+args.reference_stage, promoted[0]['matching_pair']))
    urdf = Path(review['run'])/'source/physical_v29/training/assets/robot_training.urdf'
    root = ET.parse(urdf).getroot()
    axes = {j.attrib['name']: {'axis': j.find('axis').attrib['xyz'], 'origin_rpy': j.find('origin').attrib.get('rpy', '0 0 0')}
            for j in root.findall('joint') if j.find('axis') is not None and any(k in j.attrib['name'] for k in ['shoulder', 'hip_pitch', 'elbow', 'head'])}
    report = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'gate_review': str(args.gate_review.resolve()),
              'urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(), 'native_joint_axes': axes,
              'scope': 'Truthful actual joint angles/velocities from completed independent gates. Full-population 10 Hz angle statistics and representative 50 Hz traces; no policy, training, or physics changes.',
              'interpretation': 'Native shoulder/hip pitch axes have the same dominant positive Y sign, but hips are canted. Negative ipsilateral correlation and near-180-degree phase are counter-swing indicators, not exact human anatomical angles. Low-amplitude phase is unreliable. Head zero is torso-forward, not absolute world gaze. 10 Hz data cannot diagnose higher-frequency jitter; 50 Hz diagnostics cover only selected robots.',
              'reference_comparison': {'human': 'Speed-dependent swing amplitude, bilateral coordination and smoothness compared as features with Mirelman et al. 2015; their wrist IMUs are not these shoulder/elbow encoders. No numerical human pass threshold is imported.',
                                       'human_source': 'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043',
                                       'unitree_source': 'https://github.com/unitreerobotics/unitree_rl_mjlab',
                                       'unitree_limit': 'Official repository provides policy playback/motion workflows, but no verified speed-matched G1/H1 walking encoder trajectory was obtained in this check. Exact angle-over-time Unitree similarity remains unmeasured; configuration/rest angles or dance motion are not substitutes.'},
              'records': [], 'metric_fixtures_passed': 4}
    joint_keys = ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']
    for tag, pair in selected:
        for label in ['nominal', 'pushed']:
            folder = Path(pair[label])
            with np.load(folder/'telemetry.npz') as archive:z={k:archive[k] for k in archive.files}
            assert np.isfinite(z['q']).all() and np.isfinite(z['qd']).all()
            names = z['joint_names'].tolist(); ids = {k:names.index(k) for k in names}
            q, qd = np.rad2deg(z['q']).astype(float), np.rad2deg(z['qd']).astype(float)
            with np.load(folder/'gait_diagnostics_50hz.npz') as archive:d={k:archive[k] for k in archive.files}
            # Compare first episodes only, excluding the terminal/reset tick.
            # Population angles are sampled at the same ticks as metrics[::5].
            alive = z['metrics'][::5, :, 0] > .5
            assert alive.shape == q.shape[:2]
            for case, command in enumerate(z['case_speeds']):
                population = z['case_labels'] == case
                valid_rows = alive[:, population].all(1)
                if valid_rows.sum() < 10: continue
                a = q[valid_rows][:, population]; v = qd[valid_rows][:, population]
                stats = {}
                for joint in joint_keys:
                    j = ids[joint]
                    stats[joint] = {'mean_deg':float(a[:,:,j].mean()),
                                    'median_per_robot_p95_p05_range_deg':float(np.median(np.quantile(a[:,:,j],.95,axis=0)-np.quantile(a[:,:,j],.05,axis=0))),
                                    'velocity_rms_deg_s':float(np.sqrt(np.mean(v[:,:,j]**2)))}
                head = np.linalg.norm(a[:, :, [ids['head_yaw'], ids['head_pitch']]], axis=-1)
                row = {'actor_tag':tag, 'checkpoint':pair['checkpoint'], 'sha256':pair['sha256'], 'test':label,
                       'folder':str(folder), 'command_m_s':float(command), 'population':int(population.sum()),
                       'first_episode_population_samples':int(valid_rows.sum()),
                       'reset_or_terminal_population_rows_excluded':int((~valid_rows).sum()),
                       'head_rms_deg':float(np.sqrt(np.mean(head**2))), 'head_p95_deg':float(np.quantile(head,.95)),
                       'head_within_5deg_fraction':float(np.mean(head<=5)), 'joints':stats,
                       'coordination':coordination(a, ids, .1)}
                sample = np.isclose(d['commands'], command)
                if sample.any():
                    terminal = np.flatnonzero(d['done'][:, sample].any(1))
                    mask = d['time'] >= 2
                    if len(terminal):mask &= np.arange(len(mask)) < terminal[0]
                    if mask.sum()<10:report['records'].append(row);continue
                    u = np.rad2deg(d['q'][mask][:, sample]); vel = np.rad2deg(d['qd'][mask][:, sample])
                    target = np.rad2deg(d['target'][mask][:, sample]); upper = [ids[k] for k in joint_keys]
                    spec = np.abs(np.fft.rfft(u[:,:,upper]-u[:,:,upper].mean(0), axis=0))**2
                    f = np.fft.rfftfreq(len(u), float(d['step_dt']))
                    row['representative_50hz'] = {'robots':int(sample.sum()),
                        'first_episode_samples':int(mask.sum()),'reset_ticks_total':int(d['done'][:,sample].sum()),
                        'target_joint_rms_error_deg':float(np.sqrt(np.mean((target[:,:,upper]-u[:,:,upper])**2))),
                        'above_8hz_angle_power_fraction':float(spec[f>8].sum()/max(spec[1:].sum(),1e-12)),
                        'upper_acceleration_rms_deg_s2':float(np.sqrt(np.mean((np.diff(vel[:,:,upper],axis=0)/float(d['step_dt']))**2))),
                        'raw_upper_action_outside_clamp_fraction':float(np.mean(np.abs(d['raw_action'][mask][:,sample][:,:,upper])>3))}
                report['records'].append(row)
            if tag == 'latest' and label == 'nominal':
                plot_trace(d, ids, args.output_dir, Path(pair['checkpoint']).parent.name)
    (args.output_dir/'upper_joint_time_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ['actor_tag','test','command_m_s','head_rms_deg','coordination']} for r in report['records'] if r['test']=='nominal'], indent=2))


def plot_trace(d, ids, out, actor):
    speeds = [1., 2.5, 3., 3.5]
    fig, axes = plt.subplots(5,4,figsize=(18,14),sharex=True)
    velocity_fig, velocity_axes = plt.subplots(5,4,figsize=(18,14),sharex=True)
    rows = [('head_yaw','head_pitch'),('left_shoulder_pitch','right_shoulder_pitch'),('left_shoulder_roll','right_shoulder_roll'),('left_elbow_pitch','right_elbow_pitch'),('left_hip_pitch','right_hip_pitch')]
    cols = ['#1768ac','#c94c4c']
    csv_rows = []
    for col,speed in enumerate(speeds):
        candidates = np.flatnonzero(np.isclose(d['commands'],speed))
        if not len(candidates): continue
        robot = candidates[0]; time = d['time']; mask=(time>=2)&(time<=8)
        terminal=np.flatnonzero(d['done'][:,robot])
        if len(terminal):mask &= np.arange(len(mask)) < terminal[0]
        for row,joints in enumerate(rows):
            ax=axes[row,col]
            for color,joint in zip(cols,joints):
                q=np.rad2deg(d['q'][:,robot,ids[joint]])
                ax.plot(time[mask],q[mask],color=color,label=joint.replace('_',' '),lw=1.2)
                velocity_axes[row,col].plot(time[mask],np.rad2deg(d['qd'][mask,robot,ids[joint]]),color=color,label=joint.replace('_',' '),lw=1.2)
                if row < 4:
                    target=np.rad2deg(d['target'][:,robot,ids[joint]])
                    ax.plot(time[mask],target[mask],color=color,alpha=.3,ls='--',lw=.8)
                csv_rows.extend((speed,int(robot),float(t),joint,float(v),float(np.rad2deg(d['target'][i,robot,ids[joint]])),float(np.rad2deg(d['qd'][i,robot,ids[joint]]))) for i,t,v in zip(np.flatnonzero(mask),time[mask],q[mask]))
            ax.grid(alpha=.2); ax.legend(fontsize=7,loc='upper right')
            if col==0:ax.set_ylabel('Native angle (degrees)')
            if row==0:ax.set_title(f'{speed:g} m/s command | robot {robot}')
            if row==4:ax.set_xlabel('Simulation time (seconds)')
            va=velocity_axes[row,col];va.grid(alpha=.2);va.legend(fontsize=7,loc='upper right')
            if col==0:va.set_ylabel('Joint velocity (degrees/s)')
            if row==0:va.set_title(f'{speed:g} m/s command | robot {robot}')
            if row==4:va.set_xlabel('Simulation time (seconds)')
    fig.suptitle(f'Actual head, arm and leg angles over time — {actor}, nominal test\nSolid: measured joints; faint dashed: physical targets. Representative robots; population statistics saved separately.',fontsize=13)
    fig.tight_layout(rect=[0,0,.999,.94]);fig.savefig(out/'upper_joint_angles.png',dpi=170);fig.savefig(out/'upper_joint_angles.svg');plt.close(fig)
    velocity_fig.suptitle(f'Actual joint velocities — {actor}, nominal test\n50 Hz representative traces, first episode; population motion statistics saved separately.',fontsize=13)
    velocity_fig.tight_layout(rect=[0,0,.999,.94]);velocity_fig.savefig(out/'upper_joint_velocities.png',dpi=170);velocity_fig.savefig(out/'upper_joint_velocities.svg');plt.close(velocity_fig)
    with (out/'upper_joint_angles.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['command_m_s','robot_index','sim_time_s','joint','actual_deg','target_deg','actual_velocity_deg_s']);w.writerows(csv_rows)


if __name__=='__main__':main()
