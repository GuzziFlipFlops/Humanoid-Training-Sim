"""Synchronous contact/objective attribution on current and qualified teacher."""
import json,numpy as np
from pathlib import Path
OUT=Path(__file__).resolve().parent;F=OUT/'motor_retention_response'
r=json.loads((F/'result.json').read_text())
with np.load(F/'telemetry.npz') as z:
 t=z['time'];window=t>=5.;force=z['force'][window];clearance=z['clearance'][window];phase=z['reward_phase'][window];alive=z['alive'][window]
 contact=z['contact'][window];reward=z['reward_rate'][window];actions=z['action_command'][window]
 rows=[]
 for i,g in enumerate(r['groups']):
  ids=slice(32*i,32*(i+1));f=force[:,ids];cl=clearance[:,ids];c=contact[:,ids];valid=alive[:,ids]
  geometric=(cl>.001).all(-1);unloaded=(f<1).all(-1);true=geometric&unloaded&valid
  landing=(c[1:]&~c[:-1])
  row=dict(name=g['name'],actor=g['actor'],command=g['command'],force_threshold_gap_fraction=float((~c.any(-1)&valid).mean()),
   unloaded_force_fraction=float((unloaded&valid).mean()),sole_clearance_gap_fraction=float((geometric&valid).mean()),
   true_unloaded_geometric_flight_fraction=float(true.mean()),grounded_force_gap_fraction=float((~c.any(-1)&~geometric&valid).mean()),
   loaded_both_fraction=float(c.all(-1).mean()),left_right_mean_force_N=f.mean((0,1)).tolist(),
   left_right_landings_total=landing.sum((0,1)).tolist(),simultaneous_contact_arrivals_total=int(landing.all(-1).sum()),
   minimum_sole_clearance_quantiles_m=np.quantile(cl.min(-1),[0,.5,.95,1]).tolist(),
   true_flight_reward_phase_quantiles=np.quantile(phase[:,ids][true],[0,.25,.5,.75,1]).tolist() if true.any() else None,
   complete_reward_rate=float(reward[:,ids][valid].mean()),reward_terms=g['reward_terms'],
   no_reset=True,actual_command_x_mean=float(actions[:,ids,0].mean()),diagnostic_not_gate_classification=True)
  rows.append(row)
result=dict(completed=True,records=rows,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],
 scope='Actual simultaneous force/sole/contact/reward/commands from paired normal-limit20s physics,after5s. Contact arrivals are descriptive counts without the unchanged qualifying-step/startup landing classifier; do not substitute them for no-hopping or flight gates.',
 no_gate_change=True,no_inference_override=True)
(OUT/'contact_objective_analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:x[k] for k in ['name','force_threshold_gap_fraction','sole_clearance_gap_fraction','true_unloaded_geometric_flight_fraction','complete_reward_rate']} for x in rows]))
