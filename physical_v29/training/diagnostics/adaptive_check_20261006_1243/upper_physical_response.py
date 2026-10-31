"""Actual 50Hz upper response and exact native 200Hz cap audit."""
import json,numpy as np,importlib.util
from pathlib import Path
OUT=Path(__file__).resolve().parent;folder=OUT/'steering_response'
d=np.load(folder/'telemetry.npz');names=d['joint_names'].tolist();window=d['time']>=5
spec=importlib.util.spec_from_file_location('upper_audit',OUT/'upper_joint_time_audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
arrays={k:d[k] for k in ['q','qd','target','raw_action','target_clipped','target_slew','torque_saturation']}
mot=json.loads((folder/'substep_motor_audit.json').read_text())
upper=['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch'];rows=[]
for i,label in enumerate(d['groups']):
 if label not in ['1.0_control','3.5_control']:continue
 ids=slice(i*32,(i+1)*32);record=dict(cohort=str(label),robots=32,rate_Hz=50,joints={})
 qi=arrays['q'][window,ids];ti=arrays['target'][window,ids];lookup={n:j for j,n in enumerate(names)}
 record['actual_coordination_50Hz']=audit.coordination(qi,lookup,.02)
 target_with_actual_hips=qi.copy()
 for n in upper:target_with_actual_hips[:,:,lookup[n]]=ti[:,:,lookup[n]]
 record['target_vs_actual_hips_coordination_50Hz']=audit.coordination(target_with_actual_hips,lookup,.02)
 motors={x['name']:x for x in mot['commands'][i]['motors']}
 for name in upper:
  j=names.index(name);q=arrays['q'][window,ids,j];target=arrays['target'][window,ids,j];qd=arrays['qd'][window,ids,j]
  record['joints'][name]=dict(actual_mean_deg=float(np.rad2deg(q).mean()),actual_population_median_range_deg=float(np.median(np.rad2deg(np.quantile(q,.95,axis=0)-np.quantile(q,.05,axis=0)))),target_joint_RMS_deg=float(np.rad2deg(np.sqrt(((target-q)**2).mean()))),physical_velocity_RMS_deg_s=float(np.rad2deg(np.sqrt((qd**2).mean()))),raw_action_outside_clamp_fraction=float((abs(arrays['raw_action'][window,ids,j])>3).mean()),target_limit_fraction=float(arrays['target_clipped'][window,ids,j].mean()),target_slew_fraction=float(arrays['target_slew'][window,ids,j].mean()),requested_applied_torque_difference_fraction=float(arrays['torque_saturation'][window,ids,j].mean()),native_200Hz_speed_audit=motors[name])
  freq=np.fft.rfftfreq(len(q),.02);hz=record['actual_coordination_50Hz']['hip_cycle_hz'];k=int(np.argmin(abs(freq-hz)))
  qf=np.fft.rfft(q-q.mean(0),axis=0)[k];tf=np.fft.rfft(target-target.mean(0),axis=0)[k]
  cross=(qf*np.conj(tf)).sum();auto_t=(abs(tf)**2).sum();auto_q=(abs(qf)**2).sum()
  record['joints'][name]['descriptive_gait_frequency_response']=dict(frequency_Hz=float(freq[k]),actual_minus_target_phase_deg=float(np.rad2deg(np.angle(cross))),pooled_gain=float(abs(cross)/max(float(auto_t),1e-12)),pooled_cross_coherence=float(abs(cross)**2/max(float(auto_t*auto_q),1e-12)),actual_gait_component_peak_amplitude_deg=float(np.rad2deg(2.*abs(qf)/len(q)).mean()),target_gait_component_peak_amplitude_deg=float(np.rad2deg(2.*abs(tf)/len(q)).mean()),causal_identification=False)
 rows.append(record)
result=dict(completed=True,checkpoint=json.loads((folder/'result.json').read_text())['checkpoint'],cohorts=rows,scope='32 physical robots per straight-command cohort at1/3.5m/s. Truthful 50Hz target/joint/velocity/torque difference; exact native200Hz driving speed audit. Torque-difference fractions combine torque/speed/transmission constraints and are not alone torque-ceiling attribution. Diagnostic only; no human-like certification.')
(OUT/'upper_physical_response.json').write_text(json.dumps(result,indent=2));print(json.dumps({r['cohort']:{n:{k:v for k,v in q.items() if k in ['target_joint_RMS_deg','target_limit_fraction','target_slew_fraction','requested_applied_torque_difference_fraction']} for n,q in r['joints'].items()} for r in rows},indent=2))
