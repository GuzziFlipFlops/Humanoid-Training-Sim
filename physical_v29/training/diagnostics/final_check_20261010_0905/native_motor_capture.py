"""Passive actual 5ms shaft samples. Does not alter inputs, actions or cache."""
import json,sys,os
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[1]/'automation'))
from motor_shaft_metrics import installed_motor_arrays,summarize_motor_arrays
class NativeCapture:
 def __init__(self,raw,folder):
  self.raw=raw;self.folder=Path(folder);self.rows={};self.count=0;self.alive=np.ones(raw.num_envs,dtype=bool);self.ids=None
  self.original=raw.actuator.compute;raw.actuator.compute=self.compute
 def copy(self,x):return x.detach().cpu().numpy().copy()
 def put(self,k,x):self.rows.setdefault(k,[]).append(x)
 def compute(self,control_action,joint_pos,joint_vel):
  control,q,qd=control_action,joint_pos,joint_vel
  raw=self.raw
  # Copies before the original mutates the action container.
  target=None if control.joint_positions is None else self.copy(control.joint_positions)
  result=self.original(control,q,qd)
  if self.ids is None:
   command=self.copy(raw.action_commands[:,0]);self.ids=np.arange(raw.num_envs) if os.environ.get('FINAL_EPISODE')=='1' else np.flatnonzero(np.isclose(command,1.)|np.isclose(command,3.5))
   self.command=command[self.ids]
  ids=self.ids;a=raw.actuator
  states={key:{k:self.copy(v)[ids] for k,v in st.items() if hasattr(v,'detach')} for key,st in a.last_motor_states.items()}
  data=installed_motor_arrays(a.joint_names,a.records,[(k,ii) for k,ii,_ in a.pairs],self.copy(q)[ids],self.copy(qd)[ids],self.copy(a.computed_effort)[ids],self.copy(a.applied_effort)[ids],states)
  self.names=data['motor_names'];self.refs=data['shaft_references']
  for k in ['position_rad','velocity_rad_s','velocity_RPM','requested_torque_Nm','applied_torque_Nm','signed_mechanical_power_W','valid']:self.put(k,data[k].astype(np.float32) if k!='valid' else data[k])
  for key,value in [('joint_pos_rad',q),('joint_velocity_rad_s',qd),('requested_pivot_torque_Nm',a.computed_effort),('applied_pivot_torque_Nm',a.applied_effort),('root_state_w',raw.robot.data.root_state_w),('body_pos_w',raw.robot.data.body_pos_w),('body_quat_w',raw.robot.data.body_quat_w),('foot_force_w',raw.contacts.data.net_forces_w[:,raw.contact_feet]),('com_w',(raw.robot.data.body_com_pos_w*raw.link_masses[None,:,None]).sum(1)/raw.link_masses.sum()),('action',raw.actions),('action_command',raw.action_commands)]:self.put(key,self.copy(value)[ids])
  from isaaclab.utils.math import quat_apply
  footp=raw.robot.data.body_link_pos_w[:,raw.feet,:];footq=raw.robot.data.body_link_quat_w[:,raw.feet,:]
  vertices=quat_apply(footq[:,:,None,:].expand(-1,-1,8,-1).reshape(-1,4),raw.sole_corners[None].expand(raw.num_envs,-1,-1,-1).reshape(-1,3)).reshape(raw.num_envs,2,8,3)
  self.put('sole_clearance_m',self.copy((footp[:,:,2]+vertices[:,:,:,2].amin(2)).clamp_min(0.))[ids])
  self.put('target_rad',target[ids] if target is not None else np.full_like(self.copy(q)[ids],np.nan))
  self.put('alive',self.alive[ids].copy());self.put('invalid_transmission',self.copy(a.invalid_transmission)[ids])
  self.put('terminated',np.zeros(len(ids),dtype=bool));self.put('truncated',np.zeros(len(ids),dtype=bool))
  for pair,st in states.items():
   for key,value in st.items():self.put('cad_'+pair+'_'+key,value)
  self.count+=1
  return result
 def done(self,terminated,truncated):
  term=self.copy(terminated).astype(bool);trunc=self.copy(truncated).astype(bool);self.alive &= ~(term|trunc)
  if self.ids is not None and (term|trunc).any():
   for j in range(max(0,self.count-4),self.count):
    self.rows['terminated'][j]=term[self.ids].copy();self.rows['truncated'][j]=trunc[self.ids].copy();self.rows['alive'][j]&=~(term|trunc)[self.ids]
 def save(self):
  if not self.count:return
  arr={k:np.stack(v) for k,v in self.rows.items()};arr['environment_ids']=self.ids;arr['command_m_s']=self.command
  dt=float(self.raw.physics_dt);assert abs(dt-.005)<1e-9
  arr['interval_start_s']=np.arange(self.count)*dt;arr['interval_end_s']=arr['interval_start_s']+dt
  np.savez_compressed(self.folder/'native_motor_200hz.npz',**arr)
  mapping=dict(motor_names=self.names,shaft_references=self.refs,joint_names=self.raw.actuator.joint_names,body_names=self.raw.robot.body_names,physics_dt_s=dt,torque_state_time='pre-application interval start',root_contact_time='pre-application current articulation/sensor sample',power='applied shaft torque times simultaneous pre-application shaft angular velocity',power_scope='simulation derived mechanical shaft power, not electrical battery power',relative_zero_uncalibrated=True,position_unit='rad',velocity_units=['rad/s','RPM'],torque_unit='N*m',power_unit='W',command=self.command.tolist(),environments=self.ids.tolist(),sample_count=self.count,healthy_statistics_exclude_first_2s_and_any_terminal_reset=True)
  (self.folder/'motor_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n',encoding='utf-8')
  summaries={}
  for c in sorted(set(self.command.tolist())):
   idx=np.isclose(self.command,c);selection=arr['alive'][:,idx]&(arr['interval_start_s'][:,None]>=2.)
   d={k:arr[k][:,idx] for k in ['applied_torque_Nm','velocity_RPM','signed_mechanical_power_W','valid']};d.update(motor_names=self.names,shaft_references=self.refs)
   summaries[str(c)]=summarize_motor_arrays(d,selection)
  (self.folder/'motor_summary.json').write_text(json.dumps(summaries,indent=2)+'\n',encoding='utf-8')
