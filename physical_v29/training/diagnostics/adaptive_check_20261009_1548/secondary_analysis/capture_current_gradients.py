"""Passive real-PPO capture of conditioned labels and pre-auxiliary derivatives."""
import json,runpy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
import torch
from upper_body_refinement import LocomotionOnlyRetention
from rsl_rl.storage.rollout_storage import RolloutStorage
original_generator=RolloutStorage.mini_batch_generator
current_batch=None
def passive_batches(self,*args,**kwargs):
 global current_batch
 for batch in original_generator(self,*args,**kwargs):
  current_batch=batch
  yield batch
 current_batch=None
RolloutStorage.mini_batch_generator=passive_batches
original=LocomotionOnlyRetention._regularize;captured=0;pure_PPO={}
def audit(self,module,inputs,mean):
 global captured
 obs=self._observations
 if self.command_range is None and obs is not None and mean.requires_grad and captured<4:
  # Capture before the standing hook is registered, preserving its exact order.
  # Earlier CPU subtraction had uncertainty; this reads the real incoming PPO
  # derivative and returns that same tensor without numerical modification.
  key=id(mean)
  def before_standing(gradient):
   pure_PPO[key]=gradient.detach().cpu().clone()
   return gradient
  mean.register_hook(before_standing)
  return original(self,module,inputs,mean)
 if self.command_range is None or obs is None or not mean.requires_grad or captured>=4:
  return original(self,module,inputs,mean)
 raw=obs['policy'];index=raw.shape[-1]-80+6;speed=raw[:,index:index+2].norm(dim=1)
 ids=((speed>self.command_range[0])&(speed<=self.command_range[1]+1e-6)).nonzero().flatten()
 if not ids.numel():return original(self,module,inputs,mean)
 stride=max(1,(len(ids)+self.max_samples-1)//self.max_samples);ids=ids[::stride][:self.max_samples]
 record=dict(mlp_input=inputs[0].detach().cpu().clone(),mean=mean.detach().cpu().clone(),ids=ids.cpu(),
  raw_observations=raw.detach().cpu().clone(),upper_indices=self.upper,
  state={k:v.detach().cpu().clone() for k,v in module.state_dict().items()})
 assert current_batch is not None
 record['advantages']=current_batch.advantages.detach().cpu().clone()
 record['returns']=current_batch.returns.detach().cpu().clone()
 record['sampled_actions']=current_batch.actions.detach().cpu().clone()
 record['old_log_prob']=current_batch.old_actions_log_prob.detach().cpu().clone()
 record['old_distribution_params']=[p.detach().cpu().clone() for p in current_batch.old_distribution_params]
 number=captured;captured+=1
 def capture(gradient):
  record['incoming_gradient']=gradient.detach().cpu().clone()
  record['incoming_PPO_gradient']=pure_PPO.pop(id(mean))
  record['actual_standing_hook_derivative']=record['incoming_gradient']-record['incoming_PPO_gradient']
  record['current_distribution_params']=[p.detach().cpu().clone() for p in self.actor.output_distribution_params]
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
