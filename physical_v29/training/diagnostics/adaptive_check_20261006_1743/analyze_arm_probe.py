import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent;folder=out/'arm_response';result=json.loads((folder/'result.json').read_text());assert result['completed']
with np.load(folder/'telemetry.npz') as a:z={k:a[k] for k in a.files}
names=z['joint_names'].tolist();ids={n:names.index(n) for n in names};per=int(z['robots_per_group']);window=z['time']>=5
rows=[];fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
for i,summary in enumerate(result['groups']):
 group=slice(i*per,(i+1)*per);q=z['q'][window,group];target=z['target'][window,group];a=[]
 frequency=summary['diagnostic_sine_frequency_Hz']
 if frequency is None:
  hip=q[:,:,ids['left_hip_pitch']];f=np.fft.rfftfreq(len(q),.02);y=abs(np.fft.rfft(hip-hip.mean(0),axis=0)).mean(1)
  band=(f>.5)&(f<4.);frequency=float(f[np.flatnonzero(band)[np.argmax(y[band])]])
 for side in ['left','right']:
  j=ids[side+'_shoulder_pitch'];time=z['time'][window]
  phasor=np.exp(-2j*np.pi*frequency*time)[:,None]
  actual=((q[:,:,j]-q[:,:,j].mean(0))*phasor).sum(0)
  requested=((target[:,:,j]-target[:,:,j].mean(0))*phasor).sum(0)
  phase=np.angle((actual*np.conj(requested)).sum());gain=np.median(abs(actual)/np.maximum(abs(requested),1e-8))
  a.append(dict(side=side,actual_minus_target_phase_deg=float(np.rad2deg(phase)),gain=float(gain),
   target_joint_RMS_deg=float(np.rad2deg(np.sqrt(np.mean((q[:,:,j]-target[:,:,j])**2)))),
   actual_range_deg=float(np.rad2deg(np.median(np.quantile(q[:,:,j],.95,axis=0)-np.quantile(q[:,:,j],.05,axis=0)))),
   damping_Nms_rad=float(z['damping'][window,group,j].mean()),stiffness_Nm_rad=float(z['stiffness'][window,group,j].mean())))
 row=dict(summary,frequency_Hz=frequency,shoulder_response=a);rows.append(row)
 if i<6:
  ax=axes[i%3,i//3];mask=(z['time']>=3)&(z['time']<=8);robot=i*per
  for color,side in [('blue','left'),('red','right')]:
   j=ids[side+'_shoulder_pitch'];ax.plot(z['time'][mask],np.rad2deg(z['q'][mask,robot,j]),color=color,label=side)
   ax.plot(z['time'][mask],np.rad2deg(z['target'][mask,robot,j]),color=color,alpha=.3,ls='--')
  ax.set_title(summary['name']);ax.grid(alpha=.2);ax.legend();ax.set_ylabel('Shoulder pitch degrees')
fig.tight_layout();fig.savefig(out/'arm_response.png',dpi=160);plt.close(fig)
record=dict(completed=True,checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],rows=rows,
 corrected_prior_diagnostic_assumption='Earlier locked-subtree damping4 analogy did not describe actual training damping1. Actual arm tensors measured here; stiffness35 and damping1 in control.',
 limitations='Closed-loop free-base tests with truthful student feedback. Sine offsets applied only diagnostically through preserved target limits/slew; do not establish open-loop transfer functions or identify hardware inertia. Damping variants are not deployed or qualified; no convergence or human-motion proof.')
(out/'arm_response_analysis.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['name','survival','mean_forward_world_speed','mean_absolute_world_lateral_velocity','frequency_Hz','shoulder_response']} for r in rows],indent=2))
