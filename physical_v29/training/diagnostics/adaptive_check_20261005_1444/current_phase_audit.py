"""Measured current-target delay and phase alignment; no simulation changes."""
import json,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
d=np.load(OUT/'original_offline_nominal/gait_diagnostics_50hz.npz');names=list(d['joint_names'])
time=d['time'];use=(time>=5.);commands=d['commands'];q=d['q'];target=d['target'];forces=d['foot_force'];clearance=d['clearance']
rows=[]
for command in [.15,.35,.65,1.,1.5]:
 ids=np.flatnonzero(np.isclose(commands,command));assert len(ids)
 frequency=1.25 if command<1.5 else .9+.5*command
 phase=np.remainder(time[:,None]*frequency,1.)
 # No resets occurred: action observes tick i clock; reward uses tick i+1.
 assert not d['done'][:,ids].any()
 phase_pre=np.remainder(phase-.02*frequency,1.)
 duty=.55 if command<1.5 else .43
 foot_phase=np.remainder(phase[:,:,None]+np.array([0.,.5]),1.)
 swing=np.repeat(foot_phase>=duty,len(ids),axis=1)
 force=forces[:,ids];cl=clearance[:,ids];intended_flight=swing.all(-1)
 actual=((cl>.001).all(-1)&(np.abs(force)<1).all(-1))
 delays=[]
 for name in ['left_hip_pitch','right_hip_pitch','left_knee_pitch','right_knee_pitch','left_ankle_pitch','right_ankle_pitch']:
  j=names.index(name);a=target[use][:,ids,j];b=q[use][:,ids,j];a=a-a.mean(0);b=b-b.mean(0)
  scores=[]
  for lag in range(11):
   x=a if lag==0 else a[:-lag];y=b if lag==0 else b[lag:]
   scores.append(float((x*y).sum()/max(1e-12,np.sqrt((x*x).sum()*(y*y).sum()))))
  lag=int(np.argmax(scores));delays.append(dict(joint=name,lag_ms=20*lag,correlation=scores[lag]))
 row=dict(command=command,traced_robots=len(ids),frequency_Hz=frequency,duty=duty,
  pre_action_to_reward_phase_delta_cycles=.02*frequency,
  intended_two_foot_swing_window_ms=(.5-duty)/frequency*1000 if duty<.5 else 0.,
  geometric_flight_fraction=float(actual[use].mean()),target_joint_delay=delays)
 if command>=1.5:
  bw=34.463*9.81
  row.update(intended_flight_loaded_foot_fraction=float((np.max(force[use],-1)>1)[intended_flight[use]].mean()),
   intended_flight_mean_total_force_N=float(force[use].sum(-1)[intended_flight[use]].mean()),
   intended_flight_max_minimum_clearance_m=float(cl[use].min(-1)[intended_flight[use]].max()))
 rows.append(row)
record=dict(completed=True,rows=rows,scope='50Hz bounded traces, correlation over periodic trajectories; lag is descriptive, not identified causality or a motor ceiling. No phase rewrite or action pulse deployed.')
(OUT/'current_phase_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
