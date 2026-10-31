"""Separate world angular-z from heading derivative on truthful saved states."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
with np.load(OUT/'heading_response/telemetry.npz') as z:
    root=z['root'].astype(np.float64);command=z['action_command'].astype(np.float64)
    time=z['time'];groups=z['groups'].tolist();per=int(z['robots_per_group'])
w,x,y,z=[root[:,:,3+i] for i in range(4)]
yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
unwrapped=np.unwrap(yaw,axis=0);finite=np.diff(unwrapped,axis=0)/.02
corrected=root[:,:,12]+np.tan(pitch)*(root[:,:,10]*np.cos(yaw)+root[:,:,11]*np.sin(yaw))
rows=[]
for i,name in enumerate(groups):
    if '3.5' not in name:continue
    ids=slice(i*per,(i+1)*per);window=time>=5.;paired=time[1:]>=5.
    signal=command[window,ids,2].mean(1);freq=np.fft.rfftfreq(len(signal),.02)
    power=abs(np.fft.rfft(signal-signal.mean()))**2;moving=freq>.3
    peak=int(np.argmax(np.where(moving,power,0)))
    rows.append(dict(group=name,mean_world_angular_z_rad_s=float(root[window,ids,12].mean()),
      mean_heading_finite_difference_rad_s=float(finite[paired,ids].mean()),
      mean_heading_kinematic_derivative_rad_s=float(corrected[window,ids].mean()),
      mean_actual_yaw_command_rad_s=float(command[window,ids,2].mean()),
      actual_command_variation_RMS_rad_s=float(np.std(command[window,ids,2],axis=0).mean()),
      dominant_command_variation_Hz=float(freq[peak]),
      command_variation_power_below_half_Hz_fraction=float(power[(freq>0)&(freq<.5)].sum()/max(power[1:].sum(),1e-20)),
      kinematic_vs_forward_finite_difference_RMS_rad_s=float(np.sqrt(((corrected[1:,ids][paired]-finite[paired,ids])**2).mean()))))
result=dict(completed=True,groups=rows,
    interpretation='Quaternion heading derivative differs from world angular-z when the base tilts. The analytic ZYX kinematics and50Hz forward difference have different sampling alignment; their RMS disagreement is reported. Command spectra are descriptive, not a causal delay ID or proof smoothing helps. All signals come from actual saved physical states; no production change.',
    yaw_rate_field_in_response_analysis='mean_yaw_rate denotes world angular-z, not quaternion heading derivative',
    formula='heading_dot = omega_world_z + tan(pitch)*(omega_world_x*cos(yaw)+omega_world_y*sin(yaw)',
    native_limits_and_gates_unchanged=True)
(OUT/'angular_heading_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
