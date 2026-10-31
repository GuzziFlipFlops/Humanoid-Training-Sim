"""CPU audit of saved CAD frames, reference limits and failed policy telemetry.

No simulator is created. Outputs distinguish kinematic agreement from dynamic
balance, and use actual native saved joint origins rather than geometry.origin.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent


def fk(metadata, q, link):
    lookup = {j['name']: i for i, j in enumerate(metadata['joints'])}
    by_child = {j['child']: j for j in metadata['joints']}
    chain = []
    while link != 'pelvis':
        j = by_child[link]; chain.append(j); link = j['parent']
    p, r = np.zeros(3), np.eye(3)
    for j in reversed(chain):
        p += r @ np.array(j['origin_xyz_m'])
        r = r @ Rotation.from_rotvec(np.array(j['axis'])*q[lookup[j['name']]]).as_matrix()
    return p, r


def main():
    import xml.etree.ElementTree as ET
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((HERE.parent/'assets/export_metadata.json').read_text())
    native = json.loads((HERE.parents[1]/'outputs/shin_cad_20260912/v29/native_joint_map_saved.json').read_text())
    native = {j['name']: j for j in native if j.get('anatomical')}
    motion = np.load(HERE/'assets/motion_reference.npz')
    names = [j['name'] for j in meta['joints']]
    order = [motion['names'].tolist().index(n) for n in names]
    stand, clips = motion['stand'][order], motion['q'][:, :, order]
    limits = json.loads((HERE.parent/'assets/actuator_limits.json').read_text())
    xml = ET.parse(HERE/'assets/robot_training.urdf').getroot()
    urdf = {j.attrib['name']: j for j in xml.findall('joint')}
    coordinate = np.array(meta['coordinate_rotation_cad_to_isaac'])
    base = np.array(meta['base_native_mm'])
    rows, issues = [], []
    for i, j in enumerate(meta['joints']):
        n = native[j['native_name']]; x = urdf[j['name']]
        p, _ = fk(meta, np.zeros(len(names)), j['child'])
        native_p = coordinate @ (np.array(n['origin_mm'])-base)/1000
        expected_axis = coordinate @ np.array(n['axis'])
        origin = np.fromstring(x.find('origin').attrib['xyz'], sep=' ')
        rpy = np.fromstring(x.find('origin').attrib['rpy'], sep=' ')
        xml_axis = np.fromstring(x.find('axis').attrib['xyz'], sep=' ')
        xml_lims = np.array([float(x.find('limit').attrib[k]) for k in ('lower', 'upper')])
        lo, hi = j['limits_rad']; soft_lo, soft_hi = .5*(lo+hi) + np.array([-1, 1])*.475*(hi-lo)
        if 'ankle_roll' in j['name']: soft_lo, soft_hi = max(soft_lo, -.13), min(soft_hi, .13)
        if j['name'] in ('waist_pitch', 'waist_roll'): soft_lo, soft_hi = -.20, .20
        errors = dict(native_position_mm=float(np.linalg.norm(p-native_p)*1000),
            native_axis=float(np.linalg.norm(expected_axis-j['axis'])),
            urdf_origin=float(np.linalg.norm(origin-j['origin_xyz_m'])),
            urdf_axis=float(np.linalg.norm(xml_axis-j['axis'])),
            urdf_rpy=float(np.linalg.norm(rpy)),
            urdf_limits=float(np.linalg.norm(xml_lims-j['limits_rad'])),
            native_limits=float(np.linalg.norm(np.array(n['limits_rad'])-j['limits_rad'])))
        if any(v > (0.002 if k == 'native_position_mm' else 1e-6) for k, v in errors.items()):
            issues.append(dict(joint=j['name'], errors=errors))
        row = dict(joint=j['name'], parent=j['parent'], child=j['child'],
            axis_x=j['axis'][0], axis_y=j['axis'][1], axis_z=j['axis'][2],
            origin_x_m=float(p[0]), origin_y_m=float(p[1]), origin_z_m=float(p[2]),
            lower_deg=math.degrees(lo), upper_deg=math.degrees(hi),
            target_lower_deg=math.degrees(soft_lo), target_upper_deg=math.degrees(soft_hi),
            stand_deg=math.degrees(float(stand[i])), native_origin_error_mm=errors['native_position_mm'],
            native_axis_error=errors['native_axis'])
        for ci, speed in enumerate(motion['speeds']):
            q = clips[ci, :, i]; freq = .9+.5*float(speed)
            qd = (np.roll(q, -1)-np.roll(q, 1))*len(q)*freq/2
            qd_jump = np.diff(np.r_[q, q[0]])*len(q)*freq
            key = str(float(speed)).replace('.', 'p')
            row[key+'_min_deg'], row[key+'_max_deg'] = math.degrees(float(q.min())), math.degrees(float(q.max()))
            row[key+'_target_clip_fraction'] = float(((q < soft_lo) | (q > soft_hi)).mean())
            row[key+'_peak_rad_s'] = float(np.max(np.abs(qd_jump)))
            row[key+'_speed_fraction'] = float(np.max(np.abs(qd))/limits[j['name']]['velocity_rad_s'])
        rows.append(row)
    with (out/'joint_axes_and_limits.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    jacobians = {}
    for side in ('left', 'right'):
        ids = [i for i, n in enumerate(names) if n.startswith(side) and any(s in n for s in ('hip_', 'knee_', 'ankle_'))]
        p, r = fk(meta, stand, side+'_foot'); jac = []
        for i in ids:
            q = stand.copy(); q[i] += .0001
            p2, r2 = fk(meta, q, side+'_foot')
            jac.append(np.r_[(p2-p)/.0001, Rotation.from_matrix(r2@r.T).as_rotvec()/.0001])
        jac = np.array(jac).T
        sv = np.linalg.svd(jac, compute_uv=False)
        jacobians[side] = dict(joints=[names[i] for i in ids], spatial_jacobian=jac.tolist(),
            rank=int(np.linalg.matrix_rank(jac, tol=1e-5)), singular_values=sv.tolist(),
            condition_number=float(sv[0]/sv[-1]), foot_position_m=p.tolist())
    gates = []
    for p in sorted(run.glob('*_gate_*_nominal/evaluation.json')):
        r = json.loads(p.read_text()); gates.append(dict(source=str(p), passed=r['passed'],
            cases=[dict(c['statistics'], failures=c['failures']) for c in r['cases']]))
    telemetry = sorted(run.glob('*_gate_*_nominal/telemetry.npz'))[-1]
    rec = np.load(telemetry); q = rec['q']; qd = rec['qd']; tau = rec['tau']
    alive=rec['metrics'][::5,:,0]>.5
    assert alive.shape==q.shape[:2],'Joint sampling and survival trace differ'
    kinematic_rollout = []
    stand_order = [names.index(n) for n in rec['joint_names']]
    # Normalize against the actor that produced this rollout, not the older
    # standing prior: its evolving physical-input statistics can differ.
    command = json.loads((telemetry.parent/'command.json').read_text())['argv']
    prior = Path(command[command.index('--checkpoint')+1])
    model = torch.load(prior, weights_only=False, map_location='cpu', mmap=True)['actor_state_dict']
    mean = model['obs_normalizer._mean'].numpy().reshape(4, 80)[-1]
    std = model['obs_normalizer._std'].numpy().reshape(4, 80)[-1]+.01
    for i, name in enumerate(rec['joint_names']):
        normalized_q = ((q[:, :, i]-stand[stand_order[i]])-mean[9+i])/std[9+i]
        normalized_qd = (qd[:, :, i]*.08-mean[32+i])/std[32+i]
        row=rows[names.index(name)];lo,hi=np.deg2rad([row['lower_deg'],row['upper_deg']])
        softlo,softhi=np.deg2rad([row['target_lower_deg'],row['target_upper_deg']])
        valid_q=q[:,:,i][alive];valid_qd=qd[:,:,i][alive]
        kinematic_rollout.append(dict(joint=str(name),hard_lower_deg=row['lower_deg'],hard_upper_deg=row['upper_deg'],
            min_deg=math.degrees(float(q[:, :, i].min())), max_deg=math.degrees(float(q[:, :, i].max())),
            prefall_min_deg=math.degrees(float(valid_q.min())) if valid_q.size else None,
            prefall_max_deg=math.degrees(float(valid_q.max())) if valid_q.size else None,
            prefall_hard_limit_exceeded_fraction=float(((valid_q<lo-.005)|(valid_q>hi+.005)).mean()) if valid_q.size else None,
            prefall_controller_band_exceeded_fraction=float(((valid_q<softlo)|(valid_q>softhi)).mean()) if valid_q.size else None,
            prefall_speed_limit_exceeded_fraction=float((abs(valid_qd)>limits[name]['velocity_rad_s']).mean()) if valid_q.size else None,
            abs_q_normalized_p99=float(np.quantile(abs(normalized_q), .99)),
            abs_qd_normalized_p99=float(np.quantile(abs(normalized_qd), .99)),
            peak_tau_nm=float(abs(tau[:, :, i]).max()),
            continuous_exceeded_fraction=float((abs(tau[:, :, i]) > limits[name]['continuous_Nm']).mean())))
    with (out/'measured_joint_ranges.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(kinematic_rollout[0]));writer.writeheader();writer.writerows(kinematic_rollout)
    startup_summary=None
    startup_path=telemetry.parent/'startup_diagnostics.npz'
    if startup_path.exists():
        start=np.load(startup_path)
        startup_summary=dict(source=str(startup_path),scope='Sampled first2s of16robots; not all falls retained',
            failure_counts={str(n):int(start['failure_causes'][:,:,i].sum()) for i,n in enumerate(start['failure_names'])},
            action_abs_max=float(abs(start['action']).max()),
            mean_target_error_rad=float(abs(start['q']-start['target']).mean()))
    report = dict(source_run=str(run), telemetry_source=str(telemetry), controlled_dof=len(names),
        cad_urdf_frame_mismatches=issues, maximum_native_origin_error_mm=max(r['native_origin_error_mm'] for r in rows),
        maximum_native_axis_error=max(r['native_axis_error'] for r in rows),
        leg_jacobians=jacobians, nominal_gate_history=gates,
        recent_joint_telemetry=kinematic_rollout,
        observation_normalizer_checkpoint=str(prior),
        observation_normalizer_scope='Saved producing actor, last history frame; q/qd only, absolute p99 includes all saved samples.',
        startup_diagnostics=startup_summary,
        reference_clipping=[{k:v for k,v in r.items() if k in ('joint',) or k.endswith('target_clip_fraction') or k.endswith('peak_rad_s')}
                            for r in rows if any(r[k] > 0 for k in r if k.endswith('target_clip_fraction'))],
        incentives=dict(control_dt=.02, old_fall_event_penalty=-30*.02,
            note='Termination was a rate integrated with dt; falling renews the temporary initial swing discovery window.'),
        limitations=['Axes matching saved CAD and full-rank leg Jacobians do not prove contact balance.',
            'Reference clips have no explicit lateral COM weight-shift plan.',
            'Dynamics use provisional masses/inertias and simplified collision envelopes.',
            'Saved joint test checked bounded motions near neutral, not every limit or dynamic gait.'])
    (out/'diagnosis.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ('controlled_dof','cad_urdf_frame_mismatches','maximum_native_origin_error_mm',
        'maximum_native_axis_error','reference_clipping')}, indent=2))
    print('Leg Jacobian ranks:', {k:v['rank'] for k,v in jacobians.items()})
    print('Largest normalized physical observation p99:', max(r['abs_q_normalized_p99'] for r in kinematic_rollout),
        max(r['abs_qd_normalized_p99'] for r in kinematic_rollout))


if __name__ == '__main__': main()
