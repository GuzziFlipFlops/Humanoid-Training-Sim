"""CPU conversion from frozen pivot/state arrays to installed motor shafts.

No Isaac import or production mutation. Differential states must come from the
same actuator computation as joint_pos/vel and requested/applied pivot effort.
Non-differential shaft angles are relative; encoder zero calibration is absent.
"""
from __future__ import annotations
import numpy as np


def installed_motor_arrays(joint_names, records, pairs, joint_pos, joint_vel,
                           requested_joint_torque, applied_joint_torque, pair_states):
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (joint_pos, joint_vel, requested_joint_torque, applied_joint_torque)]
    q, qd, requested, applied = arrays
    if not all(x.shape == q.shape for x in arrays) or q.shape[-1] != len(joint_names):
        raise ValueError('Synchronized arrays must have identical shapes and named joint axis')
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('Nonfinite joint or effort samples')
    names, refs, positions, velocities, requests, torques, validity = [], [], [], [], [], [], []
    paired = set()
    for key, ids in pairs:
        if paired.intersection(ids):
            raise ValueError('Overlapping differential joint mappings')
        paired.update(ids)
        state = pair_states[key]
        for field, dest in [('theta', positions), ('motor_velocity', velocities),
                            ('motor_requested', requests), ('motor_applied', torques)]:
            value = np.asarray(state[field], dtype=np.float64)
            if value.shape != q.shape[:-1] + (2,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid {key} {field} shape/data')
            dest.append(value)
        valid = np.asarray(state['valid'], dtype=bool)
        if valid.shape != q.shape[:-1]:
            raise ValueError('Differential validity mask is not aligned')
        validity.append(np.repeat(valid[..., None], 2, axis=-1))
        for j in (1, 2):
            names.append(f'{key}_motor_{j}')
            refs.append(dict(name=names[-1], shaft='installed factory-geared output driving CAD crank',
                             position_zero='CAD theta reference', joint_names=[joint_names[i] for i in ids],
                             external_mapping='pose-dependent CAD Jacobian; exact actuator state'))
    for i, name in enumerate(joint_names):
        if i in paired:
            continue
        record = records[name]
        reduction = float(record.get('physical_external_reduction', record.get('reduction', 1.)))
        efficiency = float(record.get('transmission_efficiency', 1.)) if reduction != 1. else 1.
        if reduction <= 0 or not 0 < efficiency <= 1:
            raise ValueError('Invalid reduction/efficiency')
        shaft = ('bare DJI motor shaft before total reduction' if str(record.get('motor', '')).startswith('DJI')
                 else 'installed factory-geared or servo output before external transmission')
        names.append(name + '_motor')
        refs.append(dict(name=names[-1], shaft=shaft, position_zero='relative uncalibrated pivot zero times reduction',
                         joint_names=[name], reduction=reduction, modeled_external_efficiency=efficiency,
                         internal_rotor_torque='unavailable for factory-geared/servo outputs'))
        positions.append((q[..., i] * reduction)[..., None])
        velocities.append((qd[..., i] * reduction)[..., None])
        requests.append((requested[..., i] / (reduction * efficiency))[..., None])
        torques.append((applied[..., i] / (reduction * efficiency))[..., None])
        validity.append(np.ones(q.shape[:-1] + (1,), dtype=bool))
    angle, velocity, request, torque = [np.concatenate(x, axis=-1) for x in
                                      (positions, velocities, requests, torques)]
    return dict(motor_names=names, shaft_references=refs, position_rad=angle,
                velocity_rad_s=velocity, velocity_RPM=velocity * 60 / (2 * np.pi),
                requested_torque_Nm=request, applied_torque_Nm=torque,
                signed_mechanical_power_W=torque * velocity,
                valid=np.concatenate(validity, axis=-1),
                power_scope='modeled mechanical shaft power; not electrical/thermal measurement')


def summarize_motor_arrays(data, alive_mask):
    torque = np.asarray(data['applied_torque_Nm'])
    rpm = np.asarray(data['velocity_RPM'])
    power = np.asarray(data['signed_mechanical_power_W'])
    mask = np.asarray(alive_mask, dtype=bool)
    if mask.shape != torque.shape[:-1]:
        raise ValueError('Alive/reset/settling selection must align with leading sample axes')
    rows = []
    for i, name in enumerate(data['motor_names']):
        valid = mask & data['valid'][..., i]
        t, r, p = [x[..., i][valid] for x in (torque, rpm, power)]
        if not len(t):
            rows.append(dict(motor=name, samples=0, unavailable='no valid alive samples'))
            continue
        k = int(np.argmax(np.abs(p)))
        rows.append(dict(motor=name, samples=len(t), shaft=data['shaft_references'][i],
                         torque_RMS_Nm=float(np.sqrt(np.mean(t * t))),
                         torque_p95_abs_Nm=float(np.quantile(np.abs(t), .95)),
                         torque_peak_abs_Nm=float(np.abs(t).max()),
                         speed_p95_abs_RPM=float(np.quantile(np.abs(r), .95)),
                         speed_peak_abs_RPM=float(np.abs(r).max()),
                         signed_mean_mechanical_power_W=float(p.mean()),
                         mean_positive_mechanical_power_W=float(np.maximum(p, 0).mean()),
                         mean_braking_mechanical_power_W=float(np.maximum(-p, 0).mean()),
                         peak_abs_mechanical_power_W=float(np.abs(p).max()),
                         simultaneous_peak_power_torque_Nm=float(t[k]),
                         simultaneous_peak_power_speed_RPM=float(r[k])))
    return rows


def self_test():
    # Real ratios, signs and coincident power, not a mirrored implementation test.
    records = dict(hip=dict(motor='DJI10018', reduction=15, transmission_efficiency=.95),
                   shoulder=dict(motor='6010-8', physical_external_reduction=4, transmission_efficiency=.95),
                   direct=dict(motor='7010-8'))
    q = np.array([[.1, .2, .3], [.2, .4, .6]])
    qd = np.array([[2., -3., 4.], [-1., 5., -6.]])
    torque = np.array([[142.5, 38., 7.], [71.25, -19., 8.]])
    d = installed_motor_arrays(list(records), records, [], q, qd, torque, torque, {})
    np.testing.assert_allclose(d['velocity_rad_s'], [[30, -12, 4], [-15, 20, -6]])
    np.testing.assert_allclose(d['applied_torque_Nm'], [[10, 10, 7], [5, -5, 8]])
    np.testing.assert_allclose(d['signed_mechanical_power_W'], [[300, -120, 28], [-75, -100, -48]])
    rows = summarize_motor_arrays(d, np.array([True, False]))
    assert rows[0]['torque_RMS_Nm'] == 10 and rows[0]['samples'] == 1
    assert rows[1]['mean_positive_mechanical_power_W'] == 0 and rows[1]['mean_braking_mechanical_power_W'] == 120
    # A coupled mechanism's individual motors must retain its measured state.
    pair_records = dict(pitch={}, roll={})
    pair = dict(theta=np.array([[.4, -.7]]), motor_velocity=np.array([[11., -13.]]),
                motor_requested=np.array([[4., 6.]]), motor_applied=np.array([[3., -5.]]), valid=np.array([True]))
    z = installed_motor_arrays(['pitch', 'roll'], pair_records, [('ankle', [0, 1])],
                               np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), {'ankle': pair})
    np.testing.assert_allclose(z['signed_mechanical_power_W'], [[33., 65.]])
    np.testing.assert_allclose(z['position_rad'], [[.4, -.7]])
    pair['valid'][:] = False
    z = installed_motor_arrays(['pitch', 'roll'], pair_records, [('ankle', [0, 1])],
                               np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), {'ankle': pair})
    assert summarize_motor_arrays(z, np.array([True]))[0]['samples'] == 0
    return dict(passed=True, checks=['external shaft reductions', 'transmission efficiency once',
                                    'signed mechanical power', 'alive/settling exclusion',
                                    'coupled individual motor states', 'invalid linkage exclusion'])


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), indent=2))
