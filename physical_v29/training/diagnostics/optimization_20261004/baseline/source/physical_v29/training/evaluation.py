"""Independent rollout gates from raw physics telemetry, never reward averages.

No success resets, helper controller or training reward used for promotion.
Nominal and perturbed batches are separate processes; every prior command is
reevaluated. Articulation/contact simplifications remain explicit limitations.
"""
import json,math
import numpy as np
import torch
from isaaclab.utils.math import quat_apply_inverse,quat_apply
from evaluation_steps import IndependentLandings
from support_geometry import horizontal_support
from heading_control import yaw_from_quat,direction_statistics,direction_checks,COMMAND_VERSION
GATE_VERSION='physical_landing_v5_fixed_direction'

def gate_case(speed,stats,perturb=False):
 failures=[]
 checks={'survival':stats['survival']>=(.90 if perturb else .97),
  'height':stats['height_error_m']<.070,
  'pelvis_tilt':stats['tilt_deg']<(14. if speed<=1. else 22.),
  'torso_tilt':stats['torso_tilt_deg']<(14. if speed<=1. else 22.),
  'nonfoot_contact':stats['nonfoot_fraction']<.015,
  'no_invalid_differential':stats['invalid_samples']==0,
  'no_crossing':stats['crossing_fraction']<.015,
  'foot_slip':stats['slip_m_s']<(.12 if speed==0 else .18 if speed<=1 else .25)}
 if speed==0:
  checks.update(stationary=stats['mean_speed_m_s']<.12,
                anchor=stats['mean_final_drift_m']<.25,
                both_loaded_feet=stats['two_feet_fraction']>.85,
                com_over_support=stats['support_outside_fraction']<.03)
 else:
  checks.update(speed=abs(stats['forward_speed_m_s']-speed)<max(.10,speed*.12),
                lateral_speed=stats['lateral_speed_m_s']<.14,
                completed_left_steps=stats['steps_left']>=5.,completed_right_steps=stats['steps_right']>=5.,
                bilateral_steps=stats['step_asymmetry']<.18,
                bilateral_stance=stats['stance_asymmetry']<.18,
                no_hopping=stats.get('hop_landing_fraction',1.)<.02,
                gait_cadence=abs(stats['cycle_hz']-(.9+.5*speed))<max(.45,.4*(.9+.5*speed)),
                swing_clearance=stats['swing_clearance_m']>(.010 if speed<=1 else .020))
  if speed>=1.5:checks['flight']=.015<stats['flight_fraction']<.45
  checks['alternating_landings']=stats.get('same_foot_fraction',1.)<.02
  if speed<=1.:checks['walking_support']=stats['flight_fraction']<.02
  checks.update(direction_checks(speed,stats))
 failures=[k for k,v in checks.items() if not v]
 return {'passed':not failures,'checks':checks,'failures':failures}

def evaluate(env,policy,speed,duration,seed,perturb,out,play=False):
 raw=env.unwrapped;device=raw.device
 speeds=sorted(set([0.,speed]+[v for v in [.15,.35,.65,1.,1.5,2.,2.5] if v<=speed+1e-5]))
 count=raw.num_envs;labels=np.arange(count)%len(speeds);cmd=torch.tensor(np.array(speeds)[labels],device=device,dtype=torch.float32)
 reference_heading=yaw_from_quat(raw.robot.data.default_root_state[:,3:7]).clone()
 raw.configure_fixed_commands(cmd,reference_heading)
 env.reset()
 alive=torch.ones(count,dtype=torch.bool,device=device);previous_contact=None
 strikes=torch.zeros((count,2),device=device);contacts=torch.zeros((count,2),device=device)
 verifier=IndependentLandings(count,device)
 clearance_sum=torch.zeros((count,2),device=device)
 hop_touches=torch.zeros(count,device=device);touch_count=torch.zeros_like(hop_touches)
 repeated_landings=torch.zeros_like(hop_touches)
 samples=[];joint_q=[];joint_qd=[];joint_tau=[];foot_positions=[];foot_contacts=[];foot_clearance=[];root_poses=[]
 startup={k:[] for k in ['time','q','target','action','root_state','foot_force','done','failure_causes']}
 trace={k:[] for k in ['time','q','qd','tau','target','raw_action','applied_action','observed_command','action_command','heading_target','root_state','foot_force','foot_position','com','contact','clearance','landings','hop_touches','same_foot_events','done','failure_causes']}
 reason_names=['low_height','pelvis_tilt','torso_tilt','invalid_differential','nonfinite','excessive_drift','nonfoot_impact']
 rng=np.random.default_rng(seed)
 push=torch.tensor(rng.uniform(-.25,.25,(count,2)),device=device,dtype=torch.float32)
 steps=round(duration/raw.step_dt)
 if play:raw.sim.set_camera_view((2.7,-2.7,1.7),(0.,0.,.8))
 with torch.inference_mode():
  for i in range(steps):
   obs=env.get_observations();observed_command=raw.history[:16,-1,6:9].clone();action=policy(obs);_,_,done,_=env.step(action)
   trace['observed_command'].append(observed_command.cpu().numpy().copy())
   trace['action_command'].append(raw.action_commands[:16].cpu().numpy().copy())
   trace['heading_target'].append(reference_heading[:16].cpu().numpy().copy())
   alive&=~done.bool() # reset hook preserves fixed commands and reset heading
   if perturb and i in [int(5/raw.step_dt),int(12/raw.step_dt)]:
    vel=raw.robot.data.root_vel_w.clone();vel[:,:2]+=push;raw.robot.write_root_velocity_to_sim(vel)
   m=raw.telemetry()
   forward=quat_apply(raw.robot.data.root_quat_w,torch.tensor([1.,0.,0.],device=device).expand(count,-1))[:,:2]
   forward=forward/forward.norm(dim=-1,keepdim=True).clamp_min(.1)
   landed,landed_clearance,hops,touches=verifier.update(m['contact'],m['foot_position'][:,:,:2],m['clearance'],
    raw.robot.data.root_pos_w[:,:2],forward,cmd,alive,raw.step_dt)
   # Count impacts over the complete rollout, including pairs spanning 2s.
   # Otherwise the warmup boundary could count a pair but only one denominator.
   hop_touches+=hops;touch_count+=touches
   trace['time'].append((i+1)*raw.step_dt)
   for key,value in [('q',raw.robot.data.joint_pos),('qd',raw.robot.data.joint_vel),('tau',raw.robot.data.applied_torque),
                     ('target',raw.targets),('raw_action',action),('applied_action',raw.actions),
                     ('root_state',raw.robot.data.root_state_w),('foot_force',m['foot_force']),
                     ('foot_position',m['foot_position']),('com',m['com']),('contact',m['contact']),('clearance',m['clearance']),
                     ('landings',landed),('hop_touches',hops),('same_foot_events',verifier.same_foot_events),('done',done)]:
    trace[key].append(value[:16].cpu().numpy().copy())
   trace['failure_causes'].append(torch.stack([raw.termination_reasons[k] for k in reason_names],1)[:16].cpu().numpy().copy())
   # Preserve startup failures that the normal average excludes. Sixteen
   # bounded traces are enough to inspect targets/actions and why balance fails.
   if i<=int(2/raw.step_dt) and i%5==0:
    startup['time'].append((i+1)*raw.step_dt)
    for key,value in [('q',raw.robot.data.joint_pos),('target',raw.targets),('action',action),
                      ('root_state',raw.robot.data.root_state_w),('foot_force',m['foot_force']),('done',done)]:
     startup[key].append(value[:16].cpu().numpy().copy())
    startup['failure_causes'].append(torch.stack([raw.termination_reasons[k] for k in reason_names],1)[:16].cpu().numpy().copy())
   if i>=int(2/raw.step_dt):
    c=m['contact'];contacts+=c*alive[:,None]
    clearance=m['clearance']
    strikes+=landed;clearance_sum+=landed_clearance
    repeated_landings+=verifier.same_foot_events
    previous_contact=c.clone()
    # Conservative inscribed support rectangle, expressed in the robot's yaw frame.
    # Smaller than either sole and the double-foot convex hull; used only at rest.
    outside,_=horizontal_support(raw.robot.data.root_quat_w,raw.robot.data.root_pos_w,
      m['foot_position'],m['com'],c)
    entry=torch.stack([alive.float(),m['height'],raw.reference_height,m['velocity'][:,0],m['velocity'][:,1].abs(),m['velocity'][:,:2].norm(dim=-1),m['tilt'],m['torso_tilt'],m['slip'],m['nonfoot']>25.,m['invalid'],m['crossing'],c.all(1),~c.any(1),m['drift'],outside],1)
    samples.append(entry.cpu().numpy())
    if i%5==0:
     joint_q.append(raw.robot.data.joint_pos.cpu().numpy().copy());joint_qd.append(raw.robot.data.joint_vel.cpu().numpy().copy());joint_tau.append(raw.robot.data.applied_torque.cpu().numpy().copy())
     foot_positions.append(m['foot_position'].cpu().numpy().copy());foot_contacts.append(c.cpu().numpy().copy());foot_clearance.append(clearance.cpu().numpy().copy());root_poses.append(raw.robot.data.root_state_w[:,:7].cpu().numpy().copy())
   if play and i%10==0:
    p=raw.robot.data.root_pos_w[0].cpu().numpy();raw.sim.set_camera_view(p+np.array([2.7,-2.7,.7]),p+np.array([0,0,-.15]))
 a=np.stack(samples);final_alive=alive.cpu().numpy();st=strikes.cpu().numpy();ct=contacts.cpu().numpy();cl=clearance_sum.cpu().numpy();cases=[]
 hp=hop_touches.cpu().numpy();tc=touch_count.cpu().numpy();repeated=repeated_landings.cpu().numpy()
 for ci,s in enumerate(speeds):
  mask=labels==ci;am=a[:,mask];valid=am[:,:,0]>.5
  def mean_col(k):return float(am[:,:,k][valid].mean()) if valid.any() else 999.
  strides=st[mask].mean(0);stance=ct[mask].mean(0)
  stat=dict(command_m_s=s,trials=int(mask.sum()),survival=float(final_alive[mask].mean()),
   height_error_m=float(np.abs(am[:,:,1]-am[:,:,2])[valid].mean()) if valid.any() else 999.,
   forward_speed_m_s=mean_col(3),lateral_speed_m_s=mean_col(4),mean_speed_m_s=mean_col(5),
   tilt_deg=math.degrees(mean_col(6)),torso_tilt_deg=math.degrees(mean_col(7)),slip_m_s=mean_col(8),
   nonfoot_fraction=mean_col(9),invalid_samples=int(am[:,:,10].sum()),crossing_fraction=mean_col(11),
   two_feet_fraction=mean_col(12),flight_fraction=mean_col(13),mean_final_drift_m=float(am[-1,:,14].mean()),
   support_outside_fraction=mean_col(15),
   cycle_hz=float(strides.mean()/max(duration-2.,1.)),
   swing_clearance_m=float(cl[mask].sum()/max(st[mask].sum(),1.)),
   steps_left=float(strides[0]),steps_right=float(strides[1]),step_asymmetry=float(abs(strides[0]-strides[1])/max(strides.mean(),1.)),
   stance_asymmetry=float(abs(stance[0]-stance[1])/max(stance.mean(),1.)))
  stat['hop_landing_fraction']=float(hp[mask].sum()/max(tc[mask].sum(),1.))
  stat['same_foot_fraction']=float(repeated[mask].sum()/max(st[mask].sum(),1.))
  poses=np.stack(root_poses)
  # Full population, same live mask and post-warmup window as the gait averages.
  stat.update(direction_statistics(poses[:,mask,:2],poses[:,mask,3:7],
    a[::5,mask,0]>.5,reference_heading[mask].cpu().numpy(),raw.step_dt*5,s))
  gate=gate_case(s,stat,perturb);cases.append({'statistics':stat,**gate})
 report={'passed':all(c['passed'] for c in cases),'cases':cases,'seed':seed,'perturb':perturb,'duration_s':duration,'gate_version':GATE_VERSION,
  'command_version':COMMAND_VERSION,'direction_scope':'Reset-heading target held for the full rollout; raw 10Hz full-population world travel and heading guards; desired yaw-rate feedback observed by the actor, no motor-action helper',
  'metric_scope':'20-second free-base contact rollouts; first 2s excluded from averages but all startup falls retained',
  'landing_scope':'Raw qualifying steps and repeated-foot order after 2s; unique hop impacts over all 20s. No training cadence filter.',
  'limitations':'Provisional mass/inertia and simplified collision envelopes; natural appearance still needs video review. No recovery/backflip claim.'}
 (out/'evaluation.json').write_text(json.dumps(report,indent=2))
 np.savez_compressed(out/'startup_diagnostics.npz',**{k:np.array(v) for k,v in startup.items()},
  failure_names=np.array(reason_names),joint_names=np.array(raw.robot.joint_names),commands=cmd[:16].cpu().numpy(),
  scope='State after env.step; automatic resets can replace poses on a done tick. Failure flags refer to the pre-reset state.')
 np.savez_compressed(out/'gait_diagnostics_50hz.npz',**{k:np.array(v) for k,v in trace.items()},
  failure_names=np.array(reason_names),joint_names=np.array(raw.robot.joint_names),commands=cmd[:16].cpu().numpy(),
  step_dt=raw.step_dt,scope='First 16 robots, every control tick. Post-step poses may reset on done; raw action is before clamping.')
 np.savez_compressed(out/'telemetry.npz',metrics=a,case_labels=labels,case_speeds=np.array(speeds),reference_heading=reference_heading.cpu().numpy(),joint_names=np.array(raw.robot.joint_names),q=np.stack(joint_q),qd=np.stack(joint_qd),tau=np.stack(joint_tau),foot_position=np.stack(foot_positions),foot_contact=np.stack(foot_contacts),sole_clearance=np.stack(foot_clearance),root_pose=np.stack(root_poses),step_dt=raw.step_dt)
 print('[V29] GATE '+json.dumps(report),flush=True)
 return report
