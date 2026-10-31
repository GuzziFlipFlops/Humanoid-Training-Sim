"""CPU-only observed RPM percentages with matched shaft references and limits."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import hashlib
import numpy as np

TRAIN=Path('E:/Aditya/robot_sim/physical_v29/training')
RUN=TRAIN/'runs/locomotion_20261007_r40_goal_budget_full_state'
SOURCE=TRAIN/'diagnostics/final_check_20261010_0905/selected_diverse_nominal'
OUT=Path(__file__).resolve().parent
mapping=json.loads((SOURCE/'motor_mapping.json').read_text())
records=json.loads((RUN/'source/physical_v29/assets/actuator_limits.json').read_text())
geometry=json.loads((RUN/'source/physical_v29/assets/differential_geometry.json').read_text())
evaluation=json.loads((SOURCE/'evaluation.json').read_text())
case=next(c for c in evaluation['cases'] if c['statistics']['command_m_s']==3.5)
assert case['passed'] and case['statistics']['trials']==32
diversity=json.loads((SOURCE/'initial_diversity.json').read_text())
assert diversity['unique_initial_states_by_command']['3.5']==32
factor=60/(2*np.pi)
rows=[]; motors=[]; checks=[]
with np.load(SOURCE/'native_motor_200hz.npz',allow_pickle=False) as z:
    qd=z['joint_velocity_rad_s'].astype(float)
    motor_velocity=z['velocity_rad_s'].astype(float)
    motor_rpm=z['velocity_RPM'].astype(float)
    motor_torque=z['applied_torque_Nm'].astype(float)
    valid=z['valid'].astype(bool); alive=z['alive'].astype(bool)
    selected=np.isclose(z['command_m_s'],3.5)
    mask=(z['interval_start_s'][:,None]>=2)&selected[None,:]&alive
    assert selected.sum()==32 and mask.sum()==115200
    assert alive.all() and valid.all() and not z['terminated'].any() and not z['truncated'].any()
    assert not z['invalid_transmission'].any() and np.isfinite(qd).all()
    np.testing.assert_allclose(motor_rpm,motor_velocity*factor,rtol=2e-7,atol=1e-4)
    for pair,spec in geometry.items():
        ids=[mapping['joint_names'].index(n) for n in spec['joint_names']]
        m=np.einsum('...ij,...j->...i',z[f'cad_{pair}_J'].astype(float),qd[...,ids])
        np.testing.assert_allclose(m,z[f'cad_{pair}_motor_velocity'],rtol=5e-7,atol=3e-5)
        checks.append(dict(pair=pair,max_J_velocity_error_rad_s=float(np.abs(m-z[f'cad_{pair}_motor_velocity']).max())))
    for j,name in enumerate(mapping['joint_names']):
        rec=records[name]
        bound=rec.get('URDF_bounding_velocity_rad_s',rec['velocity_rad_s'])*factor
        x=np.abs(qd[...,j][mask])*factor
        p99=float(np.quantile(x,.99,method='linear')); peak=float(x.max())
        rows.append(dict(output_joint=name,reference='final pivot output after all reduction',
                         denominator_type=('configured PhysX per-axis bounding speed; coupled physical cap is at motors' if 'differential' in rec else 'configured actuator output speed cap'),
                         configured_max_RPM=float(bound),observed_p99_RPM=p99,observed_max_RPM=peak,
                         p99_percent_of_configured_max=100*p99/bound,peak_percent_of_configured_max=100*peak/bound,
                         samples=int(x.size),coupled_differential=rec.get('differential','')))
    for i,ref in enumerate(mapping['shaft_references']):
        if 'reduction' in ref:
            j=mapping['joint_names'].index(ref['joint_names'][0]); rec=records[ref['joint_names'][0]]
            reduction=ref['reduction']
            np.testing.assert_allclose(motor_velocity[...,i],qd[...,j]*reduction,rtol=2e-7,atol=2e-5)
            # The executable non-paired actuator cap is output speed × reduction.
            # Shoulder-pitch metadata also declares500RPM, but actual output120RPM×4=480RPM.
            cap=rec['velocity_rad_s']*factor*reduction
            reason='executable output speed cap times physical external reduction'
        else:
            rec=records[ref['joint_names'][0]]; cap=float(rec['motor_speed_cap_RPM'])
            reason='native CAD differential individual installed-motor speed cap'
        v=mask&valid[...,i]
        x=np.abs(motor_rpm[...,i][v]); vel=motor_velocity[...,i][v]; tau=motor_torque[...,i][v]
        p99=float(np.quantile(x,.99)); peak=float(x.max())
        above=x>cap*(1+1e-6)
        driving=above&(tau*vel>1e-6); braking=above&(tau*vel<-1e-6)
        motors.append(dict(motor=ref['name'],shaft=ref['shaft'],configured_max_RPM=float(cap),denominator_type=reason,
                           observed_p99_RPM=p99,observed_max_RPM=peak,p99_percent_of_configured_max=100*p99/cap,
                           peak_percent_of_configured_max=100*peak/cap,samples=int(x.size),
                           above_cap_sample_count=int(above.sum()),above_cap_fraction_percent=100*float(above.mean()),
                           above_cap_driving_sample_count=int(driving.sum()),above_cap_braking_sample_count=int(braking.sum()),
                           above_cap_zero_torque_sample_count=int((above&~driving&~braking).sum())))
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),
            measured_signed_fixed_frame_speed_m_s=case['statistics']['reset_frame_forward_speed_m_s'],
            actor=str(RUN/'1064_run_350_train_352/candidate.pt'),
            actor_sha256='0ad6a1cc1c018951b82d05c67d02ca7a2663eb93845030e255a04503c60a61d0',
            scope='32 independently initialized nominal-physics trials;200Hz;2–20seconds;115200 samples per axis',
            percentage_definition='100 * percentile_or_peak_absolute_observed_RPM / matched_reference_configured_max_RPM',
            output_rows=rows,motor_rows=motors,CAD_velocity_validation=checks,
            limits='Configured simulation operating caps/bounds, not a hardware loaded-speed measurement. Observed passive/braking speed can exceed a driving cap. Differential pitch/roll bounds are not independent actuator capabilities.',
            training_action='none;CPU-only preserved capture analysis')
(OUT/'rpm_utilization.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for title,rr in [('output_rpm_utilization',rows),('motor_rpm_utilization',motors)]:
    with (OUT/f'{title}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
lines=['# Running RPM as percentage of configured maximum','',
       f"Measured signed fixed-frame speed **{result['measured_signed_fixed_frame_speed_m_s']:.6f}m/s**.32 independently initialized nominal-physics trials,native200Hz,2–20s after settling,115200 valid samples per axis.No falls/resets/invalid transmissions.",'',
       'Percentage=100×absolute observed RPM/configured maximum RPM at the same reference. P99 means99th percentile across all trial/time samples; peak is the observed maximum. These are modeled operating caps,not measured hardware loaded capability. No transmission-efficiency factor is applied to speed. Fixed reductions divide both observed motor RPM and its cap by the same ratio,so percentage is unchanged. The shoulder executable cap is120 outputRPM(480RPM before external4:1);metadata500RPM is not the executable cap.','',
       '## Final outputs after all reductions','',
       'Ankle/waist pitch+roll rows marked coupled compare against the configured per-axis PhysX bounding speed,NOT an independent attainable motor cap. Their true motor-cap percentages follow in the second table.','',
       '| Output | Configured RPM | P99 RPM | Peak RPM | P99 % | Peak % |','|---|---:|---:|---:|---:|---:|']
for r in rows:
    label=r['output_joint']+(' (coupled bound)' if r['coupled_differential'] else '')
    lines.append(f"| {label} | {r['configured_max_RPM']:.2f} | {r['observed_p99_RPM']:.2f} | {r['observed_max_RPM']:.2f} | {r['p99_percent_of_configured_max']:.2f}% | {r['peak_percent_of_configured_max']:.2f}% |")
lines+=['','## Actual differential motor-cap use','','Each shaft below is the factory-geared installed output driving its CAD crank. Pose-dependent J,not a fixed ratio,maps these to final pitch/roll.','',
        '| Motor | Configured RPM | P99 RPM | Peak RPM | P99 % | Peak % | Above-cap fraction | Driving above cap samples |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in motors[:6]:
    lines.append(f"| {r['motor']} | {r['configured_max_RPM']:.0f} | {r['observed_p99_RPM']:.2f} | {r['observed_max_RPM']:.2f} | {r['p99_percent_of_configured_max']:.2f}% | {r['peak_percent_of_configured_max']:.2f}% | {r['above_cap_fraction_percent']:.4f}% | {r['above_cap_driving_sample_count']} |")
lines+=['','Above100% means actual observed overspeed,not additional authorized driving speed. Simulated mechanisms can move faster through momentum/external forces while their actuator brakes or applies zero torque. Native synchronized torque×velocity classification above records this;isolated overspeed does not establish a locomotion ceiling.','',
        '[Output CSV](output_rpm_utilization.csv),[motor CSV](motor_rpm_utilization.csv),[full statistics](rpm_utilization.json),[reproducible CPU analysis](analyze.py). Training and automation remain stopped;no simulator or learner restarted.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
files=[SOURCE/'native_motor_200hz.npz',SOURCE/'motor_mapping.json',SOURCE/'evaluation.json',SOURCE/'initial_diversity.json',
       RUN/'source/physical_v29/assets/actuator_limits.json',RUN/'source/physical_v29/assets/differential_geometry.json',
       RUN/'source/physical_v29/hardware_actuators.py',RUN/'source/physical_v29/cad_differential.py',
       OUT/'analyze.py',OUT/'REPORT.md',OUT/'rpm_utilization.json',OUT/'output_rpm_utilization.csv',OUT/'motor_rpm_utilization.csv']
(OUT/'artifact_hashes.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2),encoding='utf-8')
print((OUT/'REPORT.md').read_text(encoding='utf-8'))
