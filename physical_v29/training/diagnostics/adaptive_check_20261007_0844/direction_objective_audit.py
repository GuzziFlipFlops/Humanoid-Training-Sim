"""Pinned-kernel counterexamples and exact current representative rollouts."""
import sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path.insert(0,str(TRAIN));torch.set_num_threads(4)
from unitree_reward_port import FUNCTIONS,profile,yaw_quat,quat_apply_inverse
p=profile('g1')['track_lin_vel_xy'];assert p['function']=='track_lin_vel_xy_yaw_frame_exp'
std=p['params']['std'];function=FUNCTIONS[p['function']]
def score(q,v,c):
 env=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(root_quat_w=q,root_lin_vel_w=v))},command_manager=SimpleNamespace(get_command=lambda name:c))
 return function(env,std=std,command_name='base_velocity')
synthetic=[]
for speed in [.15,.35,.65,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
 for angle in [0.,1.,2.,3.,4.,6.,8.,12.,20.]:
  phi=np.deg2rad(angle);q=torch.tensor([[np.cos(phi/2),0.,0.,np.sin(phi/2)]],dtype=torch.float64)
  v=torch.tensor([[speed*np.cos(phi),speed*np.sin(phi),0.]],dtype=torch.float64);c=torch.tensor([[speed,0.,-.5*phi]],dtype=torch.float64)
  tracking=float(score(q,v,c));assert abs(tracking-1.)<1e-12
  heading=np.exp(-(phi/.35)**2);quality=4*tracking+heading
  synthetic.append(dict(command_m_s=speed,path_heading_deg=angle,fixed_frame_lateral_m_s=float(v[0,1]),
   fails_existing_world_lateral_gate=float(v[0,1])>=.14,tracking_rate=4*tracking,heading_rate=heading,
   partial_direction_quality_rate=quality,partial_rate_gap_from_straight=5-quality,
   scope='Hold health and other objective terms fixed. Not complete return, policy response, or physical running proof.'))
assert len(synthetic)==108
# Gravity-aligned yaw kernel must not accidentally become full-body tracking.
phi=.13;roll=.20;cy,sy=np.cos(phi/2),np.sin(phi/2);cr,sr=np.cos(roll/2),np.sin(roll/2)
q0=torch.tensor([[cy,0.,0.,sy]],dtype=torch.float64);q1=torch.tensor([[cy*cr,cy*sr,sy*sr,sy*cr]],dtype=torch.float64)
v=torch.tensor([[3.5,0.,1.]],dtype=torch.float64);c=torch.tensor([[3.5,0.,0.]],dtype=torch.float64)
assert torch.allclose(score(q0,v,c),score(q1,v,c),atol=1e-12,rtol=0.)
gate=json.loads((OUT/'boundary_gate_review.json').read_text())['latest'];rows=[]
for label in ['nominal','pushed']:
 f=Path(gate[label])/'gait_diagnostics_50hz.npz';z=np.load(f)
 roots=z['root_state'];commands=z['action_command'];targets=z['heading_target'];time=z['time']
 done=np.cumsum(z['done'].astype(np.int64),axis=0)>0;valid=(time[:,None]>=2)&(~done)
 q=torch.from_numpy(roots[:,:,3:7].reshape(-1,4));vel=torch.from_numpy(roots[:,:,7:10].reshape(-1,3));cmd=torch.from_numpy(commands.reshape(-1,3))
 actual=score(q,vel,cmd).numpy().reshape(valid.shape)
 w,x,y,zz=np.moveaxis(roots[:,:,3:7],-1,0);yaw=np.arctan2(2*(x*y+w*zz),1-2*(y*y+zz*zz))
 error=np.arctan2(np.sin(yaw-targets),np.cos(yaw-targets))
 ref=np.stack([np.cos(targets),np.sin(targets)],-1);side=np.stack([-np.sin(targets),np.cos(targets)],-1)
 fixed_lateral=np.abs((roots[:,:,7:9]*side).sum(-1));heading=np.exp(-(error/.35)**2)
 for command in sorted(set(commands[0,:,0].tolist())):
  mask=valid & np.isclose(commands[:,:,0],command)
  if not mask.any():continue
  rows.append(dict(test=label,command_m_s=command,representative_samples=int(mask.sum()),
   tracking_kernel_mean=float(actual[mask].mean()),heading_kernel_mean=float(heading[mask].mean()),
   partial_tracking_plus_heading_rate_mean=float((4*actual+heading)[mask].mean()),
   fixed_lateral_velocity_mean_m_s=float(fixed_lateral[mask].mean()),
   heading_abs_mean_deg=float(np.rad2deg(np.abs(error[mask])).mean()),
   scope='Only first16 representative robots at50Hz, first episodes after2s; full population qualification remains separate. Raw instantaneous velocity, not independent10Hz finite-difference speed.'))
result=dict(passed=True,synthetic_fixture_count=108,gravity_aligned_kernel_fixture=True,
 pinned_tracker_function=p['function'],pinned_std=std,local_tracking_weight=4.,local_heading_width_rad=.35,
 source_hashes={name:hashlib.sha256((TRAIN/name).read_bytes()).hexdigest() for name in ['unitree_reward_port.py','stage_rewards.py','unitree_vendor/velocity_rewards.py']},
 actual_checkpoint=gate['checkpoint'],synthetic=synthetic,representative_rollout=rows,
 conclusion='Yaw-aligned tracking is implemented correctly. Failed fixed-frame lateral paths can retain high partial direction quality. This is objective tolerance evidence, not proof of a porting bug, causal repair, or infeasible motor limit. No new reward is deployed.')
(OUT/'direction_objective_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(passed=True,synthetic_fixtures=108,current_3_5=[r for r in rows if abs(r['command_m_s']-3.5)<1e-6],example=[r for r in synthetic if r['command_m_s']==3.5 and r['path_heading_deg']==8.]),indent=2))
