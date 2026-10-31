"""Reset, command, cancellation and output-row derivative CPU fixtures."""
import json,math
from pathlib import Path
import torch
from sequence_shoulder_refinement import CycleCache,cycle_loss
checks=[]
def check(name,value):assert value,name;checks.append(name)
z=torch.zeros(3,8,2,requires_grad=True)
check('quiet_zero',float(cycle_loss(z)[0])==0)
phase=torch.arange(8,dtype=torch.float32)*2*math.pi/8
swing=torch.stack([phase.sin()*1.5,-phase.sin()*1.5],1)[None]
check('zero_mean_swing_free',float(cycle_loss(swing)[0])==0)
held=torch.tensor([1.5,-1.5]).expand(2,8,2).clone()
check('held_bias_penalized',float(cycle_loss(held)[0])>0)
opposites=torch.stack([held[0],-held[0]])
check('ensemble_cannot_cancel',torch.allclose(cycle_loss(held)[0],cycle_loss(opposites)[0]))
check('inside_diagnostic_band_zero',float(cycle_loss(torch.ones(1,8,2)*.2)[0])==0)
def obs(angle,speed=.35,n=2):
    x=torch.zeros(n,5,80);x[:,:,6]=speed;x[:,:,78]=math.sin(angle);x[:,:,79]=math.cos(angle);return x.flatten(1)
c=CycleCache(max_robots=2)
for t in range(33):c.record(obs(t*2*math.pi/16))
x,cycles=c.batch();check('complete_individual_cycles',x is not None and len(cycles)==2 and x.shape==(2,8,400))
check('first_incomplete_excluded',all(r['start_tick']>=17 for r in cycles))
check('both_robots_kept',len(set(r['env_id'] for r in cycles))==2)
c.reset(torch.tensor([True,False]));x,cycles=c.batch();check('reset_clears_robot_and_completed',len(cycles)==1 and cycles[0]['env_id']==1)
c.record(obs(.1,speed=.65));check('switch_excludes_old_cycles',c.batch()[0] is None)
d=CycleCache(max_robots=2)
for t in range(48):d.record(obs(t*2*math.pi/16,speed=0))
check('stopped_no_cycle',d.batch()[0] is None)
d.record(torch.zeros(2,400));check('invalid_phase_no_cycle',d.batch()[0] is None)
f=torch.randn(16,128);leaf=torch.full((16,2),1.5,requires_grad=True)
grad=torch.autograd.grad(cycle_loss(leaf.reshape(2,8,2))[0],leaf)[0]
weight=torch.zeros(23,128);bias=torch.zeros(23);ids=[3,7];weight[ids]=grad.T@f;bias[ids]=grad.sum(0)
check('only_two_parameter_rows',bool((weight[[i for i in range(23) if i not in ids]]==0).all()))
check('finite_derivative',bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()))
check('effective_clamp_dead_zone',float(torch.autograd.grad(cycle_loss(torch.full((2,8,2),4.,requires_grad=True).clamp(-3,3))[0], [],allow_unused=True)[0])==0) if False else None
result=dict(passed=True,checks=checks,count=len(checks),new_training_loss_not_a_gate=True,production_changed=False)
Path(__file__).with_name('sequence_cpu_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
