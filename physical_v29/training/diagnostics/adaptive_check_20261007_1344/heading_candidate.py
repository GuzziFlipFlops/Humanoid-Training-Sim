"""Diagnostic truthful high-speed course feedback; never certified by old hashes."""
import builtins,runpy,sys
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
from heading_control import yaw_from_quat,wrap_angle

def course_command(quat,target,velocity):
 heading_error=wrap_angle(target-yaw_from_quat(quat))
 course_error=wrap_angle(target-torch.atan2(velocity[:,1],velocity[:,0]))
 course_error=torch.where(velocity[:,:2].norm(dim=1)>.2,course_error,torch.zeros_like(course_error))
 return (.5*heading_error+course_error).clamp(-.2,.2)

def patch_environment(cls):
 original=cls._update_observation_command
 def update(self):
  if self._command_tick==self.common_step_counter:return
  original(self)
  selected=self.commands[:,0]>3.
  if bool(selected.any()):
   rate=course_command(self.robot.data.root_quat_w,self.heading_target,self.robot.data.root_lin_vel_w)
   self.commands[selected,2]=rate[selected]
 cls._update_observation_command=update

def run_worker():
 original_import=builtins.__import__;installed=False
 def install(name,*args,**kwargs):
  nonlocal installed
  result=original_import(name,*args,**kwargs)
  if name=='environment_unitree' and not installed:
   installed=True;patch_environment(sys.modules[name].LocomotionEnv);builtins.__import__=original_import
  return result
 builtins.__import__=install
 runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')

if __name__=='__main__':run_worker()
