"""CPU-only independent upper-body diagnostics from full-population gate telemetry.

Run after every matched pair; not a posture certificate or gate relaxation.
"""
import argparse,json,numpy as np
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args()
run=Path(args.run);rows=[]
for folder in sorted(run.glob('*_gate_*')):
 if not (folder/'result.json').exists() or not (folder/'telemetry.npz').exists():continue
 result=json.loads((folder/'result.json').read_text());cfg=json.loads((folder/'run_config.json').read_text())
 x=np.load(folder/'telemetry.npz');names=list(x['joint_names']);q=x['q'];qd=x['qd']
 for label,speed in enumerate(x['case_speeds']):
  sel=x['case_labels']==label;pos=q[:,sel];vel=qd[:,sel];head=pos[:,:,[names.index('head_yaw'),names.index('head_pitch')]]
  entry=dict(gate=str(folder),checkpoint=cfg['arguments']['checkpoint'],perturb=result['perturb'],
   command_m_s=float(speed),trials=int(sel.sum()),locomotion_case_passed=result['cases'][label]['passed'],
   locomotion_failures=result['cases'][label]['failures'],
   head_combined_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(head**2,-1))))),
   head_both_within_5deg_fraction=float(np.mean(np.max(np.abs(head),-1)<=np.deg2rad(5))),joints={})
  for name in ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']:
   i=names.index(name);z=np.rad2deg(pos[:,:,i])
   entry['joints'][name]=dict(mean_deg=float(z.mean()),p95_abs_deg=float(np.quantile(np.abs(z),.95)),
    std_deg=float(z.std()),rms_velocity_rad_s=float(np.sqrt(np.mean(vel[:,:,i]**2))))
  rows.append(entry)
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),cases=rows,
 scope='Diagnostic measurements on saved 10Hz full-population positions/velocities. A five-degree band is a reporting indicator, not a locomotion gate or natural-motion certification. Frame QA and matching locomotion retention remain required.'),indent=2)+'\n')
print(json.dumps(dict(gates=len({r['gate'] for r in rows}),cases=len(rows),output=str(out))))
