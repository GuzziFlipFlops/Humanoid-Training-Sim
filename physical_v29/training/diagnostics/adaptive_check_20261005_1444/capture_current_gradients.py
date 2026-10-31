"""Passive real-PPO capture of conditioned labels and pre-auxiliary derivatives."""
import json,runpy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'offline_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
import torch
from upper_body_refinement import LocomotionOnlyRetention
original=LocomotionOnlyRetention._regularize;captured=0
def audit(self,module,inputs,mean):
 global captured
 obs=self._observations
 if self.command_range is None or obs is None or not mean.requires_grad or captured>=4:
  return original(self,module,inputs,mean)
 raw=obs['policy'];index=raw.shape[-1]-80+6;speed=raw[:,index:index+2].norm(dim=1)
 ids=((speed>self.command_range[0])&(speed<=self.command_range[1]+1e-6)).nonzero().flatten()
 if not ids.numel():return original(self,module,inputs,mean)
 stride=max(1,(len(ids)+self.max_samples-1)//self.max_samples);ids=ids[::stride][:self.max_samples]
 record=dict(mlp_input=inputs[0].detach().cpu().clone(),mean=mean.detach().cpu().clone(),ids=ids.cpu(),
  raw_observations=raw.detach().cpu().clone(),upper_indices=self.upper,
  state={k:v.detach().cpu().clone() for k,v in module.state_dict().items()})
 number=captured;captured+=1
 def capture(gradient):
  record['incoming_gradient']=gradient.detach().cpu().clone()
  torch.save(record,OUT/f'current_gradient_{number:02d}.pt')
  return gradient
 mean.register_hook(capture)
 teacher_forward=self.teacher.forward
 def teacher_call(observations,*args,**kwargs):
  record['teacher_inputs']=observations['policy'].detach().cpu().clone()
  result=teacher_forward(observations,*args,**kwargs)
  record['teacher_raw_targets']=result.detach().cpu().clone()
  return result
 self.teacher.forward=teacher_call
 try:return original(self,module,inputs,mean)
 finally:self.teacher.forward=teacher_forward
LocomotionOnlyRetention._regularize=audit
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
