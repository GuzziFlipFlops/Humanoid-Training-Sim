"""Unitree reward port on unchanged v29 physical assets and motor constraints."""
from types import SimpleNamespace
import torch
from environment import LocomotionEnv as PhysicalEnv, LocomotionCfg as PhysicalCfg, META, MASS, MOTION
from isaaclab.utils import configclass
from unitree_reward_port import Scene,gait_parameters
from stage_rewards import CurriculumRewards
from heading_control import yaw_from_quat,heading_yaw_command,wrap_angle,COMMAND_VERSION

FRAMES=80
HISTORY=5

@configclass
class LocomotionCfg(PhysicalCfg):
 observation_space=FRAMES*HISTORY

class LocomotionEnv(PhysicalEnv):
 def __init__(self,cfg,**kw):
  super().__init__(cfg,**kw)
  self.reward_port=CurriculumRewards(self.robot.joint_names,self.robot.body_names,self.contacts.body_names,
   float(MOTION['root_height']),float(self.robot.data.body_link_pos_w[0,self.feet,2].mean()),'g1')
  self.reward_proxy=SimpleNamespace(scene=Scene(self.robot,self.contacts),device=self.device,num_envs=self.num_envs,step_dt=self.step_dt,
   command_manager=SimpleNamespace(get_command=lambda name:self.commands),
   action_manager=SimpleNamespace(action=self.actions,prev_action=self.previous_actions),
   termination_manager=SimpleNamespace(terminated=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)),
   episode_length_buf=self.episode_length_buf)
  # Unitree default-offset target action; physical limit/speed projection remains.
  self.scales.fill_(.25)
  self.nominal_stand_snapshot=self.stand.clone()
  self.history=torch.zeros((self.num_envs,HISTORY,FRAMES),device=self.device)
  self.critic_history=torch.zeros((self.num_envs,HISTORY,FRAMES+3),device=self.device)
  self._last_history_tick=-1
  self.defer_command_resampling=True
  self.heading_target=yaw_from_quat(self.robot.data.root_quat_w).clone()
  self._command_tick=-1
  self.fixed_evaluation_speed=None
  self.fixed_evaluation_heading=None
  self.action_commands=self.commands.clone()

 def configure_fixed_commands(self,speeds,heading):
  assert self.cfg.evaluation,'Fixed commands are reserved for independent evaluation'
  self.fixed_evaluation_speed=speeds.clone()
  self.fixed_evaluation_heading=heading.clone()

 def _update_observation_command(self):
  # Isaac executes reward before the next observation. Command switches and
  # heading feedback therefore belong here, not after sampling a policy action.
  if self._command_tick==self.common_step_counter:return
  if not self.cfg.evaluation:
   self.command_ticks-=1
   ids=(self.command_ticks<=0).nonzero(as_tuple=False).flatten()
   if ids.numel():self._sample_commands(ids)
  self.commands[:,2]=heading_yaw_command(self.robot.data.root_quat_w,self.heading_target,
    self.commands[:,:2].norm(dim=1)>.1)
  self._command_tick=self.common_step_counter

 def _reference(self):
  # Resets assign individual reference rows. They must never write through an
  # expanded view into the single immutable nominal action/observation origin.
  self.reference=self.stand.expand(self.num_envs,-1).clone()
  self.reference_height.fill_(float(MOTION['root_height']))
  self.reference_feet=self.foot_clips[0,0].expand(self.num_envs,-1,-1)
  self.reference_duty=torch.full((self.num_envs,),.55,device=self.device)

 def _pre_physics_step(self,actions):
  self.action_commands=self.commands.clone()
  super()._pre_physics_step(actions)
  period,_=gait_parameters(self.commands)
  self.phase=torch.remainder(self.episode_length_buf.float()*self.step_dt/period,1.)

 def _get_observations(self):
  self._update_observation_command()
  period,_=gait_parameters(self.commands)
  self.phase=torch.remainder(self.episode_length_buf.float()*self.step_dt/period,1.)
  d=self.robot.data
  phase=torch.stack([torch.sin(2*torch.pi*self.phase),torch.cos(2*torch.pi*self.phase)],1)*(self.commands[:,0:1]>.1)
  frame=torch.cat([d.root_ang_vel_b*.2,d.projected_gravity_b,self.commands,d.joint_pos-self.stand,d.joint_vel*.05,self.actions,phase],1)
  assert frame.shape[1]==FRAMES
  noisy=frame
  if not self.cfg.evaluation:
   noise=torch.zeros(FRAMES,device=self.device)
   noise[:3]=.04;noise[3:6]=.05;noise[9:32]=.01;noise[32:55]=.075
   noisy=frame+(torch.rand_like(frame)*2-1)*noise
  if self._last_history_tick!=self.common_step_counter:
   self.history=torch.cat([self.history[:,1:],noisy[:,None]],1)
   self.critic_history=torch.cat([self.critic_history[:,1:],torch.cat([d.root_lin_vel_b,frame],1)[:,None]],1)
   self._last_history_tick=self.common_step_counter
  # RSL-RL retains this observation until after env.step(), when _reset_idx
  # clears history rows in place. Return owned snapshots: a reset must never
  # rewrite the state associated with an already-sampled action/log-prob.
  return {'policy':self.history.reshape(self.num_envs,-1).clone(),
          'critic':self.critic_history.reshape(self.num_envs,-1).clone()}

 def _reset_idx(self,env_ids):
  super()._reset_idx(env_ids)
  ids=self.robot._ALL_INDICES if env_ids is None else env_ids
  if hasattr(self,'heading_target'):
   self.heading_target[ids]=yaw_from_quat(self.robot.data.root_quat_w[ids])
   if self.fixed_evaluation_speed is not None:
    self.commands[ids]=0.
    self.commands[ids,0]=self.fixed_evaluation_speed[ids]
    self.heading_target[ids]=self.fixed_evaluation_heading[ids]
   self._command_tick=-1
   if len(ids)==self.num_envs:self._last_history_tick=-1
  if hasattr(self,'critic_history'):
   self.critic_history[ids]=0.

 def _get_rewards(self):
  p=self.reward_proxy
  p.action_manager.action=self.actions;p.action_manager.prev_action=self.previous_actions
  p.termination_manager.terminated=self.reset_terminated;p.episode_length_buf=self.episode_length_buf
  period,_=gait_parameters(self.commands)
  reward_phase=torch.remainder(self.episode_length_buf.float()*self.step_dt/period,1.)
  heading_error=wrap_angle(self.heading_target-yaw_from_quat(self.robot.data.root_quat_w))
  p.cad_metrics=dict(self.metrics,reference_height=self.reference_height,phase=reward_phase,heading_error_rad=heading_error)
  reward,raw,weighted=self.reward_port.evaluate(p)
  assert reward.shape==(self.num_envs,)
  self.last_reward_terms=weighted
  m=self.metrics;d=self.robot.data;moving=(self.commands[:,0]>.1);stopped=~moving
  self.extras['log']={key:value.mean().detach() for key,value in weighted.items()}
  self.extras['log'].update(height_m=m['height'].mean().detach(),forward_speed=m['velocity'][:,0].mean().detach(),foot_slip=m['slip'].mean().detach())
  audit=dict(moving_fraction=moving.float().mean(),moving_forward_speed=(m['velocity'][:,0]*moving).sum()/moving.sum().clamp_min(1),
   foot_slip=m['slip'].mean(),falls_per_s=self.reset_terminated.float().mean()/self.step_dt,
   stand_forward_speed=(m['velocity'][:,0]*stopped).sum()/stopped.sum().clamp_min(1),
   stand_falls_per_s=(self.reset_terminated*stopped).float().sum()/stopped.sum().clamp_min(1)/self.step_dt,
   healthy_fraction=((m['tilt']<.25)&(m['height']>.7)).float().mean(),
   flight_fraction=(~m['contact'].any(1)).float().mean(),both_contact_fraction=m['contact'].all(1).float().mean())
  audit.update({'reward_'+key:value.mean() for key,value in weighted.items()})
  audit.update({'shaping_'+key:value.mean() for key,value in self.reward_port.last_diagnostics.items()})
  audit.update({'termination_'+key+'_per_s':value.float().mean()/self.step_dt for key,value in self.termination_reasons.items()})
  audit['action_command_difference']=(self.commands-self.action_commands).abs().amax()
  audit['heading_command_abs_rad_s']=self.commands[:,2].abs().mean()
  audit['heading_error_abs_rad']=heading_error.abs().mean()
  for key,value in audit.items():self.reward_audit[key]=self.reward_audit.get(key,0.)+value.detach()
  self.reward_audit_ticks+=1
  saturation=(self.actuator.computed_effort-self.actuator.applied_effort).abs()>1e-3
  fields=dict(q_mean=d.joint_pos.mean(0),target_mean=self.targets.mean(0),tracking_abs_error=(self.targets-d.joint_pos).abs().mean(0),
   target_clip_fraction=self.target_clipped.float().mean(0),slew_fraction=self.target_slew_limited.float().mean(0),
   torque_saturation_fraction=saturation.float().mean(0),continuous_torque_exceeded_fraction=(d.applied_torque.abs()>self.continuous).float().mean(0))
  for key,value in fields.items():self.joint_audit[key]=self.joint_audit.get(key,0.)+value.detach()
  self.joint_audit_ticks+=1
  return reward
