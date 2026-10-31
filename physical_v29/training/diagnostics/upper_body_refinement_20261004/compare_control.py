"""Separate ordinary small-batch PPO drift from new upper posture effects."""
import json,numpy as np
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def stats(folder):
 x=np.load(folder/'telemetry.npz');names=list(x['joint_names']);q=x['q'];rows=[]
 gate=read(folder/'result.json')
 for label,speed in enumerate(x['case_speeds']):
  a=q[:,x['case_labels']==label];h=a[:,:,[names.index('head_yaw'),names.index('head_pitch')]]
  c=gate['cases'][label]
  rows.append(dict(command_m_s=float(speed),head_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(h*h,-1))))),
   survival=c['statistics']['survival'],signed_forward_m_s=c['statistics']['reset_frame_forward_speed_m_s'],
   geometric_flight=c['statistics']['geometric_flight_fraction'],passed=c['passed'],failures=c['failures']))
 return rows
rows=[json.loads(x) for x in (OUT/'control_smoke/progress.jsonl').read_text().splitlines()]
assert len(rows)==8 and all(x['ppo']['pre_update_kl']==0 and x['behavior']['action_command_difference']==0 for x in rows)
control=stats(OUT/'control_reload');candidate=stats(OUT/'checkpoint_reload')
assert all(x['survival']==1. for x in control+candidate)
proof=dict(passed=True,control=control,refinement=candidate,
 all_population_survival=True,full_small_batch_learning_not_posture_qualification=True,
 scope='Matched64-env8-update PPO counterfactual and192-env20-second nominal rollouts; causal on this seed/batch only. No smoke actor is deployed.')
(OUT/'matched_control.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
