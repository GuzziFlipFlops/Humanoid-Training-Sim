import json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent;z=np.load(out/'arm_response/telemetry.npz');motor=np.load(out/'arm_response/substep_motor_200hz.npz');names=z['joint_names'].tolist();mn=motor['motor_names'].tolist();per=int(z['robots_per_group'])
limits=json.loads((out/'control_source/physical_v29/assets/actuator_limits.json').read_text());r=json.loads((out/'arm_response/result.json').read_text());rows=[]
for group,summary in enumerate(r['groups']):
 selection=slice(group*per,(group+1)*per);mask=z['time']>=5;items=[]
 for side in ['left','right']:
  name=side+'_shoulder_pitch';j=names.index(name);mi=mn.index(name);tau=z['tau'][mask,selection,j];v=z['qd'][mask,selection,j];cap=limits[name]['effort_Nm']
  items.append(dict(joint=name,torque_p99_Nm=float(np.quantile(abs(tau),.99)),torque_peak_Nm=float(abs(tau).max()),peak_cap_Nm=cap,declared_continuous_Nm=limits[name]['continuous_Nm'],
    velocity_p99_rad_s=float(np.quantile(abs(v),.99)),velocity_cap_rad_s=limits[name]['velocity_rad_s'],
    cap_ratio_p99_at_200hz=float(np.quantile(motor['cap_ratio'][1000:,selection,mi],.99)),
    driving_speed_clip_fraction_at_200hz=float(motor['speed_clip'][1000:,selection,mi].mean()),
    computed_applied_difference_fraction_50hz=float(z['torque_saturation'][mask,selection,j].mean())))
 rows.append(dict(name=summary['name'],joints=items))
(out/'arm_load_audit.json').write_text(json.dumps(dict(completed=True,rows=rows,ceiling_proven=False,scope='Actual50Hz effort and200Hz native driving-speed clips; declared continuous assumptions are not thermally validated. Computed/applied differences are not standalone torque-limit attribution.'),indent=2)+'\n')
print(json.dumps(rows[3:6],indent=2))
