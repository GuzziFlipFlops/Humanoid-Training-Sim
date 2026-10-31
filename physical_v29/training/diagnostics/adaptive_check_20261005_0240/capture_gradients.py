"""Observe six genuine PPO/prior/upper gradient batches without changing gradients."""
import json,runpy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;d=json.loads((OUT/'diagnostic_pause.json').read_text());TRAIN=Path(d['run'])/'source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
import torch
from upper_body_refinement import LocomotionOnlyRetention
original=LocomotionOnlyRetention._regularize
captured=0

def audited(self,module,inputs,mean):
 global captured
 obs=self._observations
 if self.command_range is None or captured>=6 or obs is None or not mean.requires_grad:return original(self,module,inputs,mean)
 raw=obs['policy'];index=raw.shape[-1]-80+6;speed=raw[:,index:index+2].norm(dim=1)
 ids=((speed>self.command_range[0])&(speed<=self.command_range[1]+1e-6)).nonzero().flatten()
 if not ids.numel():return original(self,module,inputs,mean)
 stride=max(1,(ids.numel()+self.max_samples-1)//self.max_samples);ids=ids[::stride][:self.max_samples]
 with torch.no_grad():
  target=self.teacher(obs[ids]).clamp(-3,3);difference=mean[ids].detach()-target;difference[:,self.upper]=0
  derivative=2*self.weight*difference/difference.numel()
 saved=dict(mlp_input=inputs[0].detach().cpu().clone(),mean=mean.detach().cpu().clone(),ids=ids.cpu(),prior_derivative=derivative.cpu(),upper_indices=self.upper,
   mlp_state={k:v.detach().cpu().clone() for k,v in module.state_dict().items()},latest_commands=raw[:,index:index+3].detach().cpu().clone())
 number=captured;captured+=1
 def capture(gradient):
  saved['incoming_gradient']=gradient.detach().cpu().clone();torch.save(saved,OUT/f'gradient_batch_{number:02d}.pt');return gradient
 mean.register_hook(capture)
 return original(self,module,inputs,mean)
LocomotionOnlyRetention._regularize=audited
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
