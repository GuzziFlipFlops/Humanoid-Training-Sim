import sys,json,copy
from pathlib import Path
import torch
from torch import nn
OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT))
sys.path.insert(0,str(OUT.parents[1]))
from shoulder_mean_refinement import phase_mean_loss,ShoulderMeanRefinement
torch.set_num_threads(4);torch.manual_seed(97);checks=[]
def check(name,cond):assert cond,name;checks.append(name)
N=256;phase=(torch.arange(N,dtype=torch.float64)+.5)/N*2*torch.pi
raw=torch.zeros(N,400,dtype=torch.float64);frames=raw.reshape(N,5,80);frames[:,:,6]=3.5;frames[:,:,78]=phase.sin()[:,None];frames[:,:,79]=phase.cos()[:,None]
quiet=torch.zeros(N,2,dtype=torch.float64);held=quiet.clone();held[:,0]=-1.5;held[:,1]=1.5
sine=torch.stack([1.5*phase.sin(),-1.5*phase.sin()],1)
check('quiet_zero',phase_mean_loss(quiet,raw)[0]==0)
check('opposite_held_penalized',phase_mean_loss(held,raw)[0]>.01)
check('zero_mean_sine_free',phase_mean_loss(sine,raw)[0]<1e-12)
# Population cancellation is an explicit limitation, not individual cycle proof.
cancel=torch.cat([held,-held]);rr_cancel=torch.cat([raw,raw])
check('opposite_individual_bias_can_cancel_in_population',phase_mean_loss(cancel,rr_cancel)[0]==0)
# Unequal phase sampling must not turn the balanced sine into a DC target.
ids=torch.cat([torch.arange(N),torch.arange(32).repeat(20)])
check('phase_occupancy_invariant',phase_mean_loss(sine[ids],raw[ids])[0]<1e-12)
for transform,label in [(lambda x:x[:32],'partial_cycle_skip'),(lambda x:x.clone().fill_(0),'invalid_phase_skip')]:
    rr=transform(raw);yy=held[:len(rr)]
    check(label,phase_mean_loss(yy,rr)[0]==0)
rr=raw.clone();rr.reshape(N,5,80)[:,0,6]=3.
check('command_history_transition_skip',phase_mean_loss(held,rr)[0]==0)
rr=raw.clone();rr.reshape(N,5,80)[:,:,7]=.1
check('lateral_command_skip',phase_mean_loss(held,rr)[0]==0)
leaf=held.clone().requires_grad_(True);loss=phase_mean_loss(leaf,raw)[0];grad=torch.autograd.grad(loss,leaf)[0]
check('held_gradient_points_toward_zero',bool((grad[:,0]<0).all() and (grad[:,1]>0).all()))
delta=1e-5;finite=(phase_mean_loss(held+delta,raw)[0]-phase_mean_loss(held-delta,raw)[0])/(2*delta)
check('finite_difference_total_direction',abs(float(finite-grad.sum()))<1e-8)
leaf=held.clone().fill_(4).requires_grad_(True);gd=torch.autograd.grad(phase_mean_loss(leaf.clamp(-3,3),raw)[0],leaf)[0]
check('effective_clip_plateau_zero_derivative',bool((gd==0).all()))
class Actor(nn.Module):
    def __init__(self):super().__init__();self.mlp=nn.Sequential(nn.Linear(400,16),nn.ELU(),nn.Linear(16,23))
names=['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']+[f'body_{i}' for i in range(15)]
actor=Actor().double();control=copy.deepcopy(actor)
with torch.no_grad():actor.mlp[-1].bias[2:4]=torch.tensor([-1.5,1.5]);control.load_state_dict(actor.state_dict())
aux=ShoulderMeanRefinement(actor,names,apply=True,max_samples=4096)
g=torch.randn(N,23,dtype=torch.float64)*.01
y=actor.mlp(raw);z=control.mlp(raw);check('same_forward',torch.equal(y,z))
(y*g).sum().backward();(z*g).sum().backward()
for i,(pa,pc) in enumerate(zip(actor.parameters(),control.parameters())):
    if i<2:check(f'shared_parameter_{i}_no_direct_change',torch.equal(pa.grad,pc.grad))
    else:
        mask=torch.ones(23,dtype=torch.bool);mask[2:4]=False
        check(f'lower_head_roll_elbow_{i}_no_direct_change',torch.equal(pa.grad[mask],pc.grad[mask]))
audit=aux.read_audit();s=audit['statistics'];check('actual_gradient_cap',s['bounded_gradient_norm']<=.100001*s['actual_PPO_shoulder_norm'])
check('hook_one_batch',audit['batches']==1);aux.close();check('pending_empty',not aux.pending)
passive=copy.deepcopy(control);passive.zero_grad();control.zero_grad()
probe=ShoulderMeanRefinement(passive,names,apply=False,max_samples=4096)
(passive.mlp(raw)*g).sum().backward();(control.mlp(raw)*g).sum().backward()
check('passive_control_all_parameter_gradients_bitwise_equal',all(torch.equal(a.grad,b.grad) for a,b in zip(passive.parameters(),control.parameters())))
probe.read_audit();probe.close()
(OUT/'shoulder_mean_cpu_tests.json').write_text(json.dumps(dict(passed=True,checks=checks,count=len(checks),audit=audit,synthetic_not_motion_reference=True),indent=2)+'\n')
print(json.dumps(dict(passed=True,count=len(checks),statistics=s),indent=2))
