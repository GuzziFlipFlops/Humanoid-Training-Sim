import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
result=dict(scope='Diagnostic joint telemetry, not natural-arm certification or posture gates.',groups={})
for name in ['control','all_prior']:
 result['groups'][name]={}
 for suffix in ['nominal','push']:
  folder=OUT/('matched_'+name+'_'+suffix);z=np.load(folder/'telemetry.npz');gate=json.loads((folder/'evaluation.json').read_text())
  names=list(z['joint_names']);headids=[names.index(n) for n in ['head_yaw','head_pitch']]
  armids=[i for i,n in enumerate(names) if 'shoulder' in n or 'elbow' in n]
  rows=[]
  for c,case in enumerate(gate['cases']):
   mask=z['case_labels']==c;head=np.rad2deg(np.linalg.norm(z['q'][:,mask][:,:,headids],axis=-1))
   q=z['q'][:,mask][:,:,armids];qd=z['qd'][:,mask][:,:,armids]
   rows.append(dict(command=case['statistics']['command_m_s'],head_RMS_deg=float(np.sqrt((head**2).mean())),
    head_p95_deg=float(np.percentile(head,95)),fraction_head_within_5deg=float((head<=5.).mean()),
    arm_joint_mean_rad={names[i]:float(q[:,:,j].mean()) for j,i in enumerate(armids)},
    arm_joint_std_rad={names[i]:float(q[:,:,j].std()) for j,i in enumerate(armids)},
    arm_velocity_abs_p95_rad_s={names[i]:float(np.percentile(np.abs(qd[:,:,j]),95)) for j,i in enumerate(armids)}))
  result['groups'][name][suffix]=rows
(OUT/'matched_upper_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:[{s:r[s] for s in ['command','head_RMS_deg','head_p95_deg','fraction_head_within_5deg']} for r in v['nominal']] for k,v in result['groups'].items()},indent=2))
