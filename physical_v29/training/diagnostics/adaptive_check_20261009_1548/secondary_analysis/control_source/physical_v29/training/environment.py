"""23-axis v29 proprioceptive locomotion environment; no cameras or old policies.

Lightweight provisional dynamics, exact measured joint frames and CAD motor-space
limits. Four-frame actor history, privileged critic, residual task-space gait
reference, mild dynamics randomization and independent raw-telemetry gates.
"""
from pathlib import Path
import sys,json,math
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv,DirectRLEnvCfg
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply,quat_apply_inverse,quat_from_euler_xyz
import isaaclab.sim as sim_utils
from hardware_actuators import make_actuators
from walking_rewards import StepTracker,StepDiscovery,ForwardProgress,StandingAnchor,walking_terms,STOP_SPEED,RUN_SPEED,fall_penalty,load_transfer,standing_support
from reference_geometry import foot_reference,bounded_motion

HERE=Path(__file__).parent
META=json.loads((HERE.parent/'assets/export_metadata.json').read_text())
MASS=json.loads((HERE/'mass_estimate.json').read_text())
MOTION=np.load(HERE/'assets/motion_reference.npz')
BOUNDED_MOTION=bounded_motion(META,MOTION)
FRAMES=80;HISTORY=4

@configclass
class LocomotionCfg(DirectRLEnvCfg):
 decimation=4
 episode_length_s=20.
 action_space=23
 observation_space=FRAMES*HISTORY
 state_space=0
 sim=sim_utils.SimulationCfg(dt=.005,render_interval=4,physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.85,dynamic_friction=.8,restitution=0.,friction_combine_mode='average'))
 scene=InteractiveSceneCfg(num_envs=64,env_spacing=3.,replicate_physics=True,clone_in_fabric=False)
 stage_speed=0.
 evaluation=False
 perturb=False
 robot_usd=''
 seed=71
 ui_window_class_type=None

def robot_config(usd):
 gains={};damping={};armature={}
 for j in META['joints']:
  n=j['name']
  if '_hip_' in n or '_knee_' in n:gains[n]=150.;damping[n]=3.;armature[n]=.008
  elif 'ankle' in n:gains[n]=80.;damping[n]=1.5;armature[n]=.004
  elif 'waist' in n:gains[n]=90.;damping[n]=2.;armature[n]=.004
  elif 'head' in n:gains[n]=10.;damping[n]=.15;armature[n]=.0004
  else:gains[n]=35.;damping[n]=1.;armature[n]=.001
 default=dict(zip(MOTION['names'].tolist(),MOTION['stand'].tolist()))
 return ArticulationCfg(prim_path='/World/envs/env_.*/Robot',
  spawn=sim_utils.UsdFileCfg(usd_path=str(usd),activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,max_linear_velocity=30.,max_angular_velocity=60.,max_depenetration_velocity=2.),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,solver_position_iteration_count=8,solver_velocity_iteration_count=2)),
  init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,float(MOTION['root_height'])+.004),joint_pos=default,joint_vel={'.*':0.}),
  soft_joint_pos_limit_factor=.95,actuators=make_actuators(gains,damping,armature))

class LocomotionEnv(DirectRLEnv):
 cfg:LocomotionCfg
 def __init__(self,cfg,**kw):
  super().__init__(cfg,**kw)
  names=self.robot.joint_names;self.lookup={n:i for i,n in enumerate(names)}
  assert set(names)=={j['name'] for j in META['joints']} and len(names)==23
  self.body_lookup={n:i for i,n in enumerate(self.robot.body_names)}
  self.feet=[self.body_lookup[s+'_foot'] for s in ['left','right']]
  self.torso=self.body_lookup['torso'];self.pelvis=self.body_lookup['pelvis']
  self.contact_feet=[self.contacts.body_names.index(s+'_foot') for s in ['left','right']]
  self.nonfeet=[i for i,n in enumerate(self.contacts.body_names) if n not in ['left_foot','right_foot']]
  ids=[MOTION['names'].tolist().index(n) for n in names]
  self.clips=torch.as_tensor(BOUNDED_MOTION['q'][:,:,ids],device=self.device)
  self.foot_clips=torch.as_tensor(foot_reference(META,BOUNDED_MOTION),device=self.device)
  self.reference_feet=self.foot_clips[0,0].expand(self.num_envs,-1,-1).clone()
  self.leg_ids=[self.lookup[s+'_'+j] for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_pitch','ankle_roll']]
  self.speeds=torch.as_tensor(MOTION['speeds'],device=self.device,dtype=torch.float32)
  self.heights=torch.as_tensor(MOTION['heights'],device=self.device,dtype=torch.float32)
  self.stand=torch.as_tensor(MOTION['stand'][ids],device=self.device)
  self.actions=torch.zeros((self.num_envs,23),device=self.device);self.previous_actions=self.actions.clone()
  self.targets=self.stand.expand(self.num_envs,-1).clone();self.previous_targets=self.targets.clone()
  self.reference=self.targets.clone();self.reference_height=torch.full((self.num_envs,),float(MOTION['root_height']),device=self.device)
  self.commands=torch.zeros((self.num_envs,3),device=self.device);self.phase=torch.zeros(self.num_envs,device=self.device)
  self.command_travel_budget=torch.zeros(self.num_envs,device=self.device);self._command_travel_tick=-1
  self.history=torch.zeros((self.num_envs,HISTORY,FRAMES),device=self.device)
  self._last_history_tick=-1
  self.start_xy=torch.zeros((self.num_envs,2),device=self.device)
  self.domain=torch.ones((self.num_envs,5),device=self.device)
  self.previous_vel=torch.zeros_like(self.actions)
  self.scales=torch.tensor([.18 if 'waist' in n else .18 if 'ankle_roll' in n else .25 if 'head' in n else .35 for n in names],device=self.device)
  self.target_lo=self.robot.data.soft_joint_pos_limits[0,:,0].clone();self.target_hi=self.robot.data.soft_joint_pos_limits[0,:,1].clone()
  for s in ['left','right']:
   i=self.lookup[s+'_ankle_roll'];self.target_lo[i]=max(float(self.target_lo[i]),-.13);self.target_hi[i]=min(float(self.target_hi[i]),.13)
  for n in ['waist_pitch','waist_roll']:
   i=self.lookup[n];self.target_lo[i]=-.20;self.target_hi[i]=.20
  self.actuator=self.robot.actuators['cad_hardware']
  self.counters={};self.metrics={};self.invalid=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
  self.slew=torch.tensor([4.5 if 'elbow' in n else 4.0 if 'head' in n else 12. for n in names],device=self.device)
  self.records=json.loads((HERE.parent/'assets/actuator_limits.json').read_text())
  self.continuous=torch.tensor([self.records[n].get('continuous_Nm',self.records[n]['effort_Nm']) for n in names],device=self.device)
  self.command_ticks=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
  self.total_weight=MASS['total_mass_kg']*9.81
  self.link_masses=self.robot.data.default_mass[0].to(self.device)
  corners=[]
  for side in ['left','right']:
   box=MASS['links'][side+'_foot']['collision_box'];center=np.array(box['center']);size=np.array(box['size'])
   corners.append([center+.5*size*np.array([x,y,z]) for x in [-1,1] for y in [-1,1] for z in [-1,1]])
  self.sole_corners=torch.as_tensor(np.array(corners),dtype=torch.float32,device=self.device)
  self.step_tracker=StepTracker(self.num_envs,self.device)
  self.forward_progress=ForwardProgress(self.num_envs,self.device)
  self.step_discovery=StepDiscovery(self.num_envs,self.device)
  self.standing_anchor=StandingAnchor(self.num_envs,self.device)
  self.reward_audit={};self.reward_audit_ticks=0
  self.joint_audit={};self.joint_audit_ticks=0
  self.target_clipped=torch.zeros_like(self.actions,dtype=torch.bool)
  self.target_slew_limited=self.target_clipped.clone()
  self._telemetry_tick=-1

 def _setup_scene(self):
  self.robot=Articulation(robot_config(self.cfg.robot_usd))
  self.contacts=ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*',history_length=3,track_air_time=True,update_period=.005))
  sim_utils.spawn_ground_plane('/World/ground',sim_utils.GroundPlaneCfg(size=(20000.,20000.),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.85,dynamic_friction=.8,restitution=0.)))
  self.scene.clone_environments(copy_from_source=False)
  self.scene.articulations['robot']=self.robot;self.scene.sensors['contacts']=self.contacts
  self.scene.filter_collisions(global_prim_paths=['/World/ground'])
  if self.sim.has_gui():
   light=sim_utils.DomeLightCfg(intensity=700.,visible_in_primary_ray=False);light.func('/World/Light',light)

 def _sample_commands(self,ids):
  upper=self.cfg.stage_speed
  self.commands[ids]=0.
  if upper>0:
   r=torch.rand(len(ids),device=self.device);v=torch.rand(len(ids),device=self.device)
   # Retain stopping and lower speeds while exposing the current frontier often.
   self.commands[ids,0]=torch.where(r<.15,0.,torch.where(r<.45,.12+v*(min(upper,1.)-.12),upper*(.80+.20*v)))
   # One fifth rehearse full standing episodes; command switches previously
   # limited almost all standing practice to only a few seconds at a time.
   self.commands[ids[ids%5==0],0]=0.
  if hasattr(self,'step_tracker'):self.step_tracker.reset(ids)
  self.command_ticks[ids]=torch.randint(200,400,(len(ids),),device=self.device)

 def _reference(self):
  speed=self.commands[:,0].abs().clamp(0,2.5)
  hi=torch.searchsorted(self.speeds,speed).clamp(1,len(self.speeds)-1);lo=hi-1
  w=((speed-self.speeds[lo])/(self.speeds[hi]-self.speeds[lo])).clamp(0,1)
  # Genuine stopping has no clock or gait target.
  ph=self.phase*120;i=ph.long()%120;j=(i+1)%120;t=(ph-i.float())[:,None]
  qa=(1-t)*self.clips[lo,i]+t*self.clips[lo,j];qb=(1-t)*self.clips[hi,i]+t*self.clips[hi,j]
  q=(1-w[:,None])*qa+w[:,None]*qb
  fa=(1-t[:,:,None])*self.foot_clips[lo,i]+t[:,:,None]*self.foot_clips[lo,j]
  fb=(1-t[:,:,None])*self.foot_clips[hi,i]+t[:,:,None]*self.foot_clips[hi,j]
  feet=(1-w[:,None,None])*fa+w[:,None,None]*fb
  ramp=(self.episode_length_buf.float()*self.step_dt/1.).clamp(0,1)
  moving=speed>STOP_SPEED
  self.reference=torch.where(moving[:,None],self.stand+(q-self.stand)*ramp[:,None],self.stand)
  self.reference_feet=torch.where(moving[:,None,None],self.foot_clips[0,0]+(feet-self.foot_clips[0,0])*ramp[:,None,None],self.foot_clips[0,0])
  duties=torch.tensor([.62,.62,.62,.62,.48,.43,.43],device=self.device)
  self.reference_duty=(1-w)*duties[lo]+w*duties[hi]
  self.reference_height=(1-w)*self.heights[lo]+w*self.heights[hi]
  self.reference_height=torch.where(speed<.08,self.heights[0],self.reference_height)

 def _pre_physics_step(self,actions):
  self.previous_actions.copy_(self.actions);self.actions=actions.clamp(-3.,3.).clone()
  if not self.cfg.evaluation and not getattr(self,'defer_command_resampling',False):
   self.command_ticks-=1;ids=(self.command_ticks<=0).nonzero(as_tuple=False).flatten()
   if ids.numel():self._sample_commands(ids)
  speed=self.commands[:,0].abs();freq=.90+.50*speed
  self.phase=torch.remainder(self.phase+self.step_dt*freq*(speed>.08),1.)
  self._reference()
  desired=self.reference+self.scales*self.actions
  self.target_clipped=(desired<self.target_lo)|(desired>self.target_hi)
  desired=torch.maximum(torch.minimum(desired,self.target_hi),self.target_lo)
  self.target_slew_limited=(desired-self.targets).abs()>self.slew*self.step_dt
  desired=self.targets+(desired-self.targets).clamp(-self.slew*self.step_dt,self.slew*self.step_dt)
  # Invalid target combinations are projected to the previous valid target.
  for _,ids,model in self.actuator.pairs:
   _,_,valid,_=model.state(desired[:,ids])
   desired[:,ids]=torch.where(valid[:,None],desired[:,ids],self.targets[:,ids])
  self.previous_targets.copy_(self.targets);self.targets=desired
  # One control tick of delay for a subset; policy observes four frames.
  self.delayed_targets=torch.where(self.domain[:,2:3]>.5,self.previous_targets,self.targets)
  if not self.cfg.evaluation and self.cfg.stage_speed>=.35:
   ids=((self.episode_length_buf==350)&(torch.rand(self.num_envs,device=self.device)<.3)).nonzero(as_tuple=False).flatten()
   if ids.numel():
    vel=self.robot.data.root_vel_w[ids].clone();vel[:,:2]+=(torch.rand((len(ids),2),device=self.device)-.5)*.40
    self.robot.write_root_velocity_to_sim(vel,env_ids=ids)

 def _apply_action(self):
  self.robot.set_joint_position_target(self.delayed_targets)

 def telemetry(self):
  stamp=self.robot.data._sim_timestamp
  if getattr(self,'_telemetry_tick',None)==stamp:return self._telemetry_cache
  d=self.robot.data
  footv=d.body_lin_vel_w[:,self.feet,:];footp=d.body_link_pos_w[:,self.feet,:]
  force=self.contacts.data.net_forces_w[:,self.contact_feet,:]
  contact=force[:,:,2]>self.total_weight*.05
  footquat=d.body_link_quat_w[:,self.feet,:]
  vertices=quat_apply(footquat[:,:,None,:].expand(-1,-1,8,-1).reshape(-1,4),self.sole_corners[None].expand(self.num_envs,-1,-1,-1).reshape(-1,3)).reshape(self.num_envs,2,8,3)
  clearance=(footp[:,:,2]+vertices[:,:,:,2].amin(2)).clamp_min(0.)
  torso_g=quat_apply_inverse(d.body_link_quat_w[:,self.torso],torch.tensor([0,0,-1.],device=self.device).expand(self.num_envs,3))
  tilt=torch.acos((-d.projected_gravity_b[:,2]).clamp(-1,1));torso_tilt=torch.acos((-torso_g[:,2]).clamp(-1,1))
  slip=(footv[:,:,:2].norm(dim=-1)*contact).sum(1)/contact.sum(1).clamp_min(1)
  nonfoot=self.contacts.data.net_forces_w[:,self.nonfeet,:].norm(dim=-1).amax(1)
  invalid=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
  for _,ids,model in self.actuator.pairs:
   _,_,valid,_=model.state(d.joint_pos[:,ids]);invalid|=~valid
  # Source left is negative Y; yaw-free root coordinates handle turns correctly.
  rel=quat_apply_inverse(d.root_quat_w[:,None,:].expand(-1,2,-1).reshape(-1,4),(footp-d.root_pos_w[:,None,:]).reshape(-1,3)).reshape(-1,2,3)
  com=(d.body_com_pos_w*self.link_masses[None,:,None]).sum(1)/self.link_masses.sum()
  self._telemetry_tick=stamp
  self._telemetry_cache=dict(height=d.root_pos_w[:,2],velocity=d.root_lin_vel_b,angular=d.root_ang_vel_b,tilt=tilt,torso_tilt=torso_tilt,torso_g=torso_g,contact=contact,foot_force=force[:,:,2],foot_velocity=footv,foot_position=footp,clearance=clearance,slip=slip,nonfoot=nonfoot,invalid=invalid,crossing=(rel[:,1,1]-rel[:,0,1])<.08,drift=(d.root_pos_w[:,:2]-self.start_xy).norm(dim=-1),relfeet=rel,com=com)
  return self._telemetry_cache

 def _get_dones(self):
  m=self.telemetry();self.metrics=m
  finite=torch.isfinite(self.robot.data.joint_pos).all(1)&torch.isfinite(m['height'])
  # Preserve the 60m excess-travel margin; commanded locomotion is allowed.
  # Charge the exact action command once per control tick, across switches.
  if self._command_travel_tick!=self.common_step_counter:
   command=getattr(self,'action_commands',self.commands)
   self.command_travel_budget.add_(command[:,:2].norm(dim=1)*self.step_dt)
   self._command_travel_tick=self.common_step_counter
  self.termination_reasons=dict(low_height=m['height']<.58,pelvis_tilt=m['tilt']>1.05,
   torso_tilt=m['torso_tilt']>1.10,invalid_differential=m['invalid'],nonfinite=~finite,
   excessive_drift=m['drift']>(60.+self.command_travel_budget),nonfoot_impact=m['nonfoot']>self.total_weight*.65)
  bad=torch.stack(list(self.termination_reasons.values()),1).any(1)
  return bad,self.episode_length_buf>=self.max_episode_length-1

 def _get_rewards(self):
  m=self.metrics;d=self.robot.data;v=self.commands[:,0]
  upright=torch.exp(-6*m['tilt'].square());torso=torch.exp(-5*m['torso_tilt'].square())
  height=torch.exp(-((m['height']-self.reference_height)/.06).square())
  healthy=upright*height*torso
  tracking=torch.exp(-((m['velocity'][:,:2]-self.commands[:,:2]).square().sum(1))/.20)
  quiet=torch.exp(-m['velocity'][:,:2].square().sum(1)/.025-m['angular'].square().sum(1)/.20)
  stopped=v<=STOP_SPEED;both=m['contact'].all(1)
  ref_error=(d.joint_pos-self.reference).square().mean(1)
  ph=torch.stack([self.phase,(self.phase+.5)%1.],1)
  expected=ph<self.reference_duty[:,None];expected=torch.where(stopped[:,None],True,expected)
  contact_match=(expected==m['contact']).float().mean(1)
  forward=quat_apply(d.root_quat_w,torch.tensor([1.,0.,0.],device=self.device).expand(self.num_envs,-1))[:,:2]
  forward=forward/forward.norm(dim=-1,keepdim=True).clamp_min(.1)
  slip_limit=torch.where(v>RUN_SPEED,.25,.18)
  eligible=(~self.reset_terminated)&(~m['invalid'])&(~m['crossing'])&(m['nonfoot']<25.)&(m['slip']<slip_limit)&(healthy>.20)
  events=self.step_tracker.update(m['contact'],m['foot_position'][:,:,:2],m['clearance'],d.root_pos_w[:,:2],forward,v,eligible,self.step_dt)
  leg_error=(d.joint_pos[:,self.leg_ids]-self.reference[:,self.leg_ids]).square().mean(1)
  fresh,budget,recent=self.forward_progress.update(d.root_pos_w[:,:2],forward,v,events,self.step_dt)
  walk=walking_terms(v,m['velocity'],healthy*eligible.float(),m['slip'],m['contact'],expected,m['clearance'],m['relfeet'],self.reference_feet,leg_error,events,self.step_dt,advance_fraction=fresh,swing_budget=budget,recent_step=recent)
  posture=walk.pop('posture_scale');progress_fraction=walk.pop('progress_fraction');swing_quality=walk.pop('swing_quality')
  progress_activity=walk.pop('progress_activity')
  # Continuous torque cost is in motor space for differential pairs.
  torque_cost=(d.applied_torque/self.continuous.clamp_min(.5)).square().mean(1)
  for key,state in self.actuator.last_motor_states.items():
   cont=15. if key=='waist' else 18.
   torque_cost+=.25*(state['motor_applied']/cont).square().mean(1)
  action_rate=(self.actions-self.previous_actions).square().mean(1)
  accel=((d.joint_vel-self.previous_vel)*.10).square().mean(1);self.previous_vel.copy_(d.joint_vel)
  power=(d.applied_torque*d.joint_vel).abs().sum(1)/MASS['total_mass_kg']
  limit_cost=((d.joint_pos-self.target_lo).clamp_max(0).square()+(d.joint_pos-self.target_hi).clamp_min(0).square()).sum(1)
  terms=dict(track=2.5*tracking*healthy,upright=.65*upright,torso=.40*torso,height=.70*height,
    stand=.85*quiet*healthy*both*stopped,reference=.70*torch.exp(-ref_error/.10)*healthy,gait=.55*contact_match*healthy,
    yaw=-.15*(m['angular'][:,2]-self.commands[:,2]).square(),vertical=-.15*d.root_lin_vel_b[:,2].square(),
    slip=-.80*m['slip'],nonfoot=-1.5*(m['nonfoot']>25.).float(),crossing=-1.0*m['crossing'].float(),
    torque=-.035*torque_cost,action_rate=-.045*action_rate,accel=-.005*accel,power=-.004*power,limits=-5.*limit_cost,
    termination=fall_penalty(self.reset_terminated,self.step_dt))
  # Preserve the settled standing reward; replace only commanded locomotion.
  for key,value in walk.items():terms[key]=torch.where(stopped,terms.get(key,torch.zeros_like(v)),value)
  for key in ['upright','torso','height']:terms[key]*=posture
  # Before the first accepted step, transfer guidance is a finite event bank,
  # not an expiring rate. After real steps the phase rate can guide continued gait.
  terms['load_transfer']=load_transfer(v,m['foot_force'],expected,healthy*eligible,budget,self.total_weight)*recent
  terms['step_discovery']=self.step_discovery.update(m['foot_force'],m['foot_position'][:,:,:2],m['clearance'],d.root_pos_w[:,:2],forward,v,eligible,healthy,events,self.total_weight,self.step_dt)
  relcom=quat_apply_inverse(d.root_quat_w,m['com']-d.root_pos_w)
  stop_drift=self.standing_anchor.update(d.root_pos_w[:,:2],v)
  terms.update(standing_support(v,m['contact'],relcom,m['relfeet'],stop_drift))
  reward=sum(terms.values())*self.step_dt
  self.extras['log']={k:x.mean().detach() for k,x in terms.items()}
  self.extras['log'].update({'height_m':m['height'].mean().detach(),'forward_speed':m['velocity'][:,0].mean().detach(),'foot_slip':m['slip'].mean().detach(),'invalid_fraction':m['invalid'].float().mean().detach()})
  moving=(~stopped).float();denom=moving.sum().clamp_min(1.)
  audit=dict(moving_fraction=moving.mean(),moving_forward_speed=(m['velocity'][:,0]*moving).sum()/denom,
    moving_progress_fraction=(progress_fraction*moving).sum()/denom,completed_steps_per_s=events.sum()/self.num_envs/self.step_dt,
    actual_swing_quality=(swing_quality*moving).sum()/denom,foot_slip=m['slip'].mean(),
    missing_swing_fraction=((~expected)&m['contact']).float().mean(),falls_per_s=self.reset_terminated.float().mean()/self.step_dt,
    new_territory_fraction=(fresh*moving).sum()/denom,unloaded_foot_fraction=((~m['contact']).any(1).float()*moving).sum()/denom,
    stand_forward_speed=(m['velocity'][:,0]*stopped).sum()/stopped.sum().clamp_min(1),
    stand_falls_per_s=(self.reset_terminated*stopped).float().sum()/stopped.sum().clamp_min(1)/self.step_dt,
    healthy_fraction=(healthy>.5).float().mean(),reward_eligible_fraction=eligible.float().mean(),
    load_transfer_rate=terms['load_transfer'].mean(),flight_rate=terms['flight'].mean(),
    step_discovery_rate=terms['step_discovery'].mean(),motion_deficit_rate=terms['motion_deficit'].mean(),
    walking_flight_rate=terms['walking_flight'].mean(),stand_support_rate=terms['stand_support'].mean(),stand_anchor_rate=terms['stand_anchor'].mean(),
    validated_progress_activity=(progress_activity*moving).sum()/denom,
    discovery_bank_remaining=(1.-self.step_discovery.best).sum(1).mean())
  for k,x in self.termination_reasons.items():audit['termination_'+k+'_per_s']=x.float().mean()/self.step_dt
  for k,x in audit.items():self.reward_audit[k]=self.reward_audit.get(k,0.)+x.detach()
  self.reward_audit_ticks+=1
  # Only 23 reduced values per field, not an ever-growing 8192-environment trace.
  saturation=(self.actuator.computed_effort-self.actuator.applied_effort).abs()>1e-3
  fields=dict(q_mean=d.joint_pos.mean(0),target_mean=self.targets.mean(0),
   tracking_abs_error=(self.targets-d.joint_pos).abs().mean(0),target_clip_fraction=self.target_clipped.float().mean(0),
   slew_fraction=self.target_slew_limited.float().mean(0),torque_saturation_fraction=saturation.float().mean(0),
   continuous_torque_exceeded_fraction=(d.applied_torque.abs()>self.continuous).float().mean(0))
  for k,x in fields.items():self.joint_audit[k]=self.joint_audit.get(k,0.)+x.detach()
  self.joint_audit_ticks+=1
  return reward

 def read_reward_audit(self):
  result={k:float(x/max(self.reward_audit_ticks,1)) for k,x in self.reward_audit.items()}
  self.reward_audit={};self.reward_audit_ticks=0
  return result

 def read_joint_audit(self):
  values={k:(x/max(self.joint_audit_ticks,1)).cpu().tolist() for k,x in self.joint_audit.items()}
  result={n:{k:v[i] for k,v in values.items()} for i,n in enumerate(self.robot.joint_names)}
  self.joint_audit={};self.joint_audit_ticks=0
  return result

 def _get_observations(self):
  m=self.telemetry();d=self.robot.data
  phase=torch.stack([torch.sin(2*torch.pi*self.phase),torch.cos(2*torch.pi*self.phase)],1)*(self.commands[:,0:1]>.08)
  frame=torch.cat([d.root_ang_vel_b*.25,d.projected_gravity_b,self.commands,(d.joint_pos-self.stand),d.joint_vel*.08,self.actions,phase],1)
  assert frame.shape[1]==FRAMES,frame.shape
  noisy=frame
  if not self.cfg.evaluation:
   scales=torch.full((FRAMES,),.005,device=self.device);scales[:3]=.025;scales[3:6]=.012;scales[6:9]=0.;scales[32:55]=.02;scales[55:]=0.
   noisy=frame+(torch.rand_like(frame)*2-1)*scales
  if self._last_history_tick!=self.common_step_counter:
   self.history=torch.cat([self.history[:,1:],noisy[:,None]],1);self._last_history_tick=self.common_step_counter
  critic=torch.cat([frame,d.root_lin_vel_b,(m['height']-self.reference_height)[:,None],m['torso_g'],m['contact'].float(),m['foot_velocity'].reshape(self.num_envs,-1),self.domain],1)
  return {'policy':self.history.reshape(self.num_envs,-1),'critic':critic}

 def _reset_idx(self,env_ids):
  self._telemetry_tick=-1
  if env_ids is None:env_ids=self.robot._ALL_INDICES
  super()._reset_idx(env_ids)
  if not hasattr(self,'stand'):return
  n=len(env_ids);q=self.stand.expand(n,-1).clone();qd=torch.zeros_like(q)
  root=self.robot.data.default_root_state[env_ids].clone();root[:,:3]+=self.scene.env_origins[env_ids]
  root[:,7:]=0.
  if not self.cfg.evaluation:
   q+=torch.randn_like(q)*.012
   angles=(torch.rand((n,3),device=self.device)-.5)*torch.tensor([.04,.04,6.2],device=self.device)
   root[:,3:7]=quat_from_euler_xyz(angles[:,0],angles[:,1],angles[:,2])
   root[:,7:9]=(torch.rand((n,2),device=self.device)-.5)*.05
   self.domain[env_ids,0]=.9+.2*torch.rand(n,device=self.device)
   self.domain[env_ids,1]=.9+.2*torch.rand(n,device=self.device)
   self.domain[env_ids,2]=(torch.rand(n,device=self.device)>.5).float()
   self._sample_commands(env_ids)
  else:self.domain[env_ids]=torch.tensor([1.,1.,0.,1.,.8],device=self.device);self.commands[env_ids]=0.
  self.robot.write_root_pose_to_sim(root[:,:7],env_ids);self.robot.write_root_velocity_to_sim(root[:,7:],env_ids)
  self.robot.write_joint_state_to_sim(q,qd,env_ids=env_ids)
  self.targets[env_ids]=q;self.previous_targets[env_ids]=q;self.reference[env_ids]=q
  self.actions[env_ids]=0.;self.previous_actions[env_ids]=0.;self.history[env_ids]=0.;self.previous_vel[env_ids]=0.
  self.start_xy[env_ids]=root[:,:2];self.phase[env_ids]=0.;self.command_travel_budget[env_ids]=0.
  self.step_tracker.reset(env_ids)
  self.forward_progress.reset(env_ids)
  self.step_discovery.reset(env_ids)
  self.standing_anchor.reset(env_ids)
  if not self.cfg.evaluation:
   self.actuator.stiffness[env_ids]=self.actuator._stiffness_baseline*self.domain[env_ids,0:1]
   self.actuator.damping[env_ids]=self.actuator._damping_baseline*self.domain[env_ids,1:2]

 def initialize_randomization(self):
  self.actuator._stiffness_baseline=self.actuator.stiffness[0].clone()
  self.actuator._damping_baseline=self.actuator.damping[0].clone()
  if not self.cfg.evaluation:
   # Startup mass+inertia scaling preserves each COM and inertia ratios.
   # Nominal and perturbed promotion tests use a separate nominal process.
   view=self.robot.root_physx_view;ids=torch.arange(self.num_envs,dtype=torch.int32)
   scale=.90+.20*torch.rand(self.num_envs)
   masses=view.get_masses().clone();inertias=view.get_inertias().clone()
   view.set_masses(masses*scale[:,None],ids)
   view.set_inertias(inertias*scale[:,None,None],ids)
   material=view.get_material_properties().clone();mu=.70+.30*torch.rand(self.num_envs)
   material[:,:,0]=mu[:,None]+.05;material[:,:,1]=mu[:,None];material[:,:,2]=0.
   view.set_material_properties(material,ids)
   self.domain[:,3]=scale.to(self.device);self.domain[:,4]=mu.to(self.device)
