"""Final output torque, after all reductions, from preserved native capture."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

TRAIN = Path('E:/Aditya/robot_sim/physical_v29/training')
SOURCE = TRAIN / 'diagnostics/final_check_20261010_0905/selected_diverse_nominal'
RUN = TRAIN / 'runs/locomotion_20261007_r40_goal_budget_full_state'
OUT = Path(__file__).resolve().parent
mapping = json.loads((SOURCE / 'motor_mapping.json').read_text())
records = json.loads((RUN / 'source/physical_v29/assets/actuator_limits.json').read_text())
geometry = json.loads((RUN / 'source/physical_v29/assets/differential_geometry.json').read_text())
previous = json.loads((TRAIN / 'diagnostics/running_torque_p99_20261010/torque_p99.json').read_text())
evaluation = json.loads((SOURCE / 'evaluation.json').read_text())
case = next(c for c in evaluation['cases'] if c['statistics']['command_m_s'] == 3.5)
diversity = json.loads((SOURCE / 'initial_diversity.json').read_text())
assert case['passed'] and case['statistics']['trials'] == 32
assert diversity['unique_initial_states_by_command']['3.5'] == 32
rows = []
validation = []
with np.load(SOURCE / 'native_motor_200hz.npz', allow_pickle=False) as z:
    torque = z['applied_pivot_torque_Nm'].astype(np.float64)
    requested = z['requested_pivot_torque_Nm'].astype(np.float64)
    times = z['interval_start_s']
    commands = z['command_m_s']
    alive = z['alive'].astype(bool)
    assert torque.shape == requested.shape == (4000, 64, 23)
    assert alive.all() and z['valid'].all() and not z['invalid_transmission'].any()
    assert not z['terminated'].any() and not z['truncated'].any()
    assert np.isfinite(torque).all() and np.isfinite(requested).all()
    assert np.allclose(np.diff(times), .005, rtol=0, atol=2e-6)
    selected = np.isclose(commands, 3.5)
    assert selected.sum() == 32
    mask = (times[:, None] >= 2) & selected[None, :] & alive
    assert mask.sum() == 115200
    for pair, spec in geometry.items():
        ids = [mapping['joint_names'].index(n) for n in spec['joint_names']]
        J = z[f'cad_{pair}_J'].astype(np.float64)
        m = z[f'cad_{pair}_motor_applied'].astype(np.float64)
        eta = records[spec['joint_names'][0]]['transmission_efficiency']
        reconstructed = eta * np.einsum('...ij,...i->...j', J, m)
        np.testing.assert_allclose(reconstructed, torque[..., ids], rtol=5e-7, atol=3e-5)
        validation.append(dict(pair=pair, joint_names=spec['joint_names'],
                               equation='output_torque = efficiency * J_transpose * individual_motor_torque',
                               max_output_reconstruction_error_Nm=float(np.abs(reconstructed-torque[..., ids]).max())))
    for j, name in enumerate(mapping['joint_names']):
        x = torque[..., j][mask]
        req = requested[..., j][mask]
        row = dict(output_joint=name, samples=int(x.size),
                   torque_p99_abs_Nm=float(np.quantile(np.abs(x), .99, method='linear')),
                   torque_max_abs_Nm=float(np.abs(x).max()), torque_RMS_Nm=float(np.sqrt(np.mean(x*x))),
                   signed_min_Nm=float(x.min()), signed_max_Nm=float(x.max()),
                   requested_torque_p99_abs_Nm=float(np.quantile(np.abs(req), .99)),
                   requested_torque_max_abs_Nm=float(np.abs(req).max()),
                   reference='final anatomical pivot output after all reductions/linkage and modeled efficiency')
        earlier = next((r for r in previous['population']['rows']
                        if r['command_m_s']==3.5 and r.get('pivot_joint')==name), None)
        if earlier:
            assert row['torque_p99_abs_Nm'] == earlier['pivot_torque_p99_abs_Nm']
            assert row['torque_max_abs_Nm'] == earlier['pivot_torque_max_abs_Nm']
        rows.append(row)
result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
              actor=previous['actor'], actor_sha256=previous['actor_sha256'],
              measured_signed_fixed_frame_speed_m_s=case['statistics']['reset_frame_forward_speed_m_s'],
              scope='32 independently initialized nominal-physics trials; applied output torques at 200 Hz, 2–20 seconds',
              statistic='99th percentile and maximum of absolute applied torque, including driving and braking',
              rows=rows, cad_virtual_work_validation=validation,
              source_capture=str(SOURCE / 'native_motor_200hz.npz'),
              limitation='Simulated applied actuator effort, not measured hardware torque; a configured limit is a cap, not a required load.',
              training_action='none; CPU-only analysis, no restart')
(OUT / 'output_torque_p99.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
with (OUT / 'output_torque_p99.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines = ['# Final output torque after all reductions', '',
         f"Measured signed fixed-frame speed **{result['measured_signed_fixed_frame_speed_m_s']:.6f} m/s** at command3.5.", '',
         'The previous motor-shaft table was before added transmissions. This table instead reads the saved applied_pivot_torque_Nm array directly in its named runtime joint order. No reduction is applied again. All factory gearing, added gearing/chain and the modeled transmission efficiency are already represented in these final output efforts. Ankle and waist rows are the combined differential pitch/roll outputs, not either individual motor.', '',
         '32 independently initialized nominal-physics trials; native200Hz; 2–20s after settling,115200 valid samples/output. No falls,resets or invalid transmissions. P99/max use absolute applied torque including braking. These are simulation applied actuator efforts,not measured hardware values. Configured torque limits cap available torque; they do not force the controller to use that torque. Some yaw/elbow outputs reach their caps,while hip/knee loads here are lower. This alone neither establishes a hardware ceiling nor validates simulation realism.', '',
         '| Final output joint | P99 absolute N·m | Maximum absolute N·m |', '|---|---:|---:|']
for r in rows:
    lines.append(f"| {r['output_joint']} | {r['torque_p99_abs_Nm']:.3f} | {r['torque_max_abs_Nm']:.3f} |")
lines += ['', 'Validation: all three differential outputs reconstruct from efficiency × CAD J-transpose × individual motor torque within recorded floating-point error; all17 non-differential p99/max outputs exactly match the prior independently converted pivot statistics. Original native capture and frozen sources are preserved. Actor1064/current running case passes; complete lower-speed retention remains unqualified.', '',
          '[Full-precision CSV](output_torque_p99.csv), [statistics and validation](output_torque_p99.json), [CPU analysis](analyze.py).', '',
          'Training and automation remain stopped; no new simulator worker,learning or production change.']
(OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
files = [SOURCE/'native_motor_200hz.npz',SOURCE/'motor_mapping.json',SOURCE/'evaluation.json',SOURCE/'initial_diversity.json',
         RUN/'source/physical_v29/assets/actuator_limits.json',RUN/'source/physical_v29/assets/differential_geometry.json',
         RUN/'source/physical_v29/cad_differential.py', OUT/'analyze.py', OUT/'REPORT.md', OUT/'output_torque_p99.json', OUT/'output_torque_p99.csv']
(OUT/'artifact_hashes.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2),encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'REPORT.md'),rows=rows,cad_validation=validation),indent=2))
