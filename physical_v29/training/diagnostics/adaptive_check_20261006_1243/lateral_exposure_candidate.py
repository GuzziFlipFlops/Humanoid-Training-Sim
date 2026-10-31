"""Diagnostic training-only frontier lateral exposure. Never used in inference.

Original runtime files and strict checkpoint/teacher validators are unmodified.
This explicit wrapper changes the sampled truthful command distribution only.
Evaluation and all earlier-command cohorts retain zero lateral commands.
"""
import builtins,json,runpy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
original_import=builtins.__import__;installed=False
def install_import(name,*args,**kwargs):
 global installed
 result=original_import(name,*args,**kwargs)
 if name=='environment_unitree' and not installed:
  installed=True
  import torch
  cls=sys.modules[name].LocomotionEnv;original=cls._sample_commands
  def sample(self,ids):
   original(self,ids)
   if not self.cfg.evaluation and self.cfg.stage_speed==3.5:
    frontier=self.commands[ids,0]>3.+1e-6
    lateral=.1*(2.*torch.rand(len(ids),device=self.device)-1.)
    self.commands[ids,1]=torch.where(frontier,lateral,torch.zeros_like(lateral))
  cls._sample_commands=sample
  builtins.__import__=original_import
 return result
builtins.__import__=install_import
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
