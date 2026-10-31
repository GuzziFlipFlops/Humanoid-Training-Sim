import json,sys,math
from pathlib import Path
import numpy as np,torch
out=Path(__file__).resolve().parent;sys.path.insert(0,str(out/'control_source/physical_v29/training'))
import unitree_reward_port as port
from direction_kernel import score
review=json.loads((out/'initial_gate_review.json').read_text());pair=review['latest'];rows=[]
std=port.profile()['track_lin_vel_xy']['params']['std']
for label in ['nominal','pushed']:
 with np.load(Path(pair[label])/'gait_diagnostics_50hz.npz') as d:
  mask=d['time']>=2;root=torch.from_numpy(d['root_state'][mask].copy());commands=torch.from_numpy(d['observed_command'][mask].copy())
  heading=torch.from_numpy(d['heading_target'][mask].copy());q=root[:,:,3:7];v=root[:,:,7:10]
  yaw=torch.atan2(2*(q[:,:,0]*q[:,:,3]+q[:,:,1]*q[:,:,2]),1-2*(q[:,:,2]**2+q[:,:,3]**2))
  h=torch.atan2(torch.sin(heading-yaw),torch.cos(heading-yaw))
  vyaw=port.quat_apply_inverse(port.yaw_quat(q),v)
  shape=commands.shape[:2]
  old=torch.exp(-((commands[:,:,:2]-vyaw[:,:,:2])**2).sum(-1)/std**2)
  new=score(commands.reshape(-1,3),vyaw.reshape(-1,3),h.reshape(-1),std).reshape(shape)
  assert torch.equal(new[commands[:,:,0]<=3.+1e-6],old[commands[:,:,0]<=3.+1e-6])
  for speed in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5]:
   ids=np.isclose(d['commands'],speed)
   if not ids.any():continue
   a,b=old[:,ids],new[:,ids];yv=vyaw[:,ids]
   fixed=torch.stack([torch.cos(h[:,ids])*yv[:,:,0]+torch.sin(h[:,ids])*yv[:,:,1],-torch.sin(h[:,ids])*yv[:,:,0]+torch.cos(h[:,ids])*yv[:,:,1]],-1)
   rows.append(dict(test=label,command=speed,representative_robots=int(ids.sum()),samples=int(mask.sum()),
    old_raw_tracking_mean=float(a.mean()),candidate_raw_tracking_mean=float(b.mean()),
    heading_abs_mean_deg=float(torch.rad2deg(h[:,ids].abs()).mean()),
    body_lateral_abs_mean=float(yv[:,:,1].abs().mean()),fixed_lateral_abs_mean=float(fixed[:,:,1].abs().mean()),
    candidate_weight4_rate_delta_before_healthy_gating=float((4*(b-a)).mean())))
record=dict(completed=True,checkpoint=pair['checkpoint'],std=std,weight=4.,rows=rows,
 scope='Counterfactual raw tracking rates on truthful saved states; not physical return, actual advantages, policy gradients or convergence proof. Prior <=3 bitwise unchanged. No deployed change.')
(out/'direction_state_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(rows[-2:],indent=2))
