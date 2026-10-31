"""Counterfactual gradient, inference and state-preservation tests on CPU."""
import sys,json,copy
from pathlib import Path
import torch
from tensordict import TensorDict
OUT=Path(__file__).resolve().parent;STAGED=OUT/'validation_source/physical_v29'
sys.path[:0]=[str(STAGED/'training'),str(STAGED)]
from upper_body_refinement import UpperBodyRefinement,LocomotionOnlyRetention,upper_indices,posture_loss
from standing_retention import StandingRetention
from curriculum_provenance import training_reward_signature
torch.set_num_threads(1);torch.manual_seed(31)
names=json.loads((OUT.parents[1]/'videos/full_body_gui_20261004_2340/rendered_mesh_inventory.json').read_text())['runtime_joint_names']
ids=upper_indices(names);lower=[i for i in range(23) if i not in ids]
class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.mlp=torch.nn.Sequential(torch.nn.Linear(400,512),torch.nn.ELU(),
            torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,23))
    def forward(self,td):return self.mlp(td['policy'])

checks=[]
def check(condition,name):assert condition,name;checks.append(name)
zero=torch.zeros(64,8);check(float(posture_loss(zero)[0])==0,'neutral_zero_auxiliary_loss')
tilt=zero.clone();tilt[:,0]=3.;check(posture_loss(tilt)[0]>posture_loss(zero)[0],'turned_head_not_rewarded')
left=zero.clone();left[:,2]=1.;left[:,3]=-1.
wrong=left.clone();wrong[:,3]=1.;check(posture_loss(wrong)[0]>posture_loss(left)[0],'opposite_arm_swing_allowed')
wild=zero.clone();wild[:,4]=3.;check(posture_loss(wild)[0]>posture_loss(zero)[0],'sideways_arm_excess_discouraged')
extreme=torch.full((64,8),1e6,requires_grad=True);loss,_=posture_loss(extreme);grad=torch.autograd.grad(loss,extreme)[0]
check(bool(torch.isfinite(grad).all()) and float(grad.abs().max())<.01,'bounded_extreme_gradient')
raw=torch.randn(64,400);raw[:,326:329]=torch.tensor([.65,0.,0.]);td=TensorDict({'policy':raw},batch_size=[64])
base=Actor();refined=copy.deepcopy(base)
with torch.no_grad():refined.mlp[-1].bias[ids[:2]]=torch.tensor([3.,-2.]);base.load_state_dict(refined.state_dict())
hook=UpperBodyRefinement(refined,names)
out0=base(td);out1=refined(td);check(torch.equal(out0,out1),'initial_action_bitwise_equal')
out0.square().mean().backward();out1.square().mean().backward()
for (n,p),(m,q) in zip(base.named_parameters(),refined.named_parameters()):
    if n not in ['mlp.6.weight','mlp.6.bias']:check(torch.equal(p.grad,q.grad),'no_auxiliary_shared_gradient_'+n)
    else:check(torch.equal(p.grad[lower],q.grad[lower]),'lower_output_gradient_bitwise_equal_'+n)
check(not torch.equal(base.mlp[-1].bias.grad[ids],refined.mlp[-1].bias.grad[ids]),'upper_gradient_present')
check(hook.read_audit()['batches']==1,'audited_single_backward')
# Auxiliary-only update changes only upper output rows and preserves all lower predictions.
refined.zero_grad();before={k:v.clone() for k,v in refined.state_dict().items()}
before_actions=refined(td).detach();before_loss=float(posture_loss(before_actions[:,ids])[0]);(refined(td).sum()*0.).backward()
with torch.no_grad():
    for p in refined.parameters():p-=.05*p.grad
after_actions=refined(td).detach();check(torch.equal(before_actions[:,lower],after_actions[:,lower]),'auxiliary_update_lower_actions_bitwise_equal')
check(float(posture_loss(after_actions[:,ids])[0])<before_loss,'auxiliary_update_reduces_posture_loss')
for n,p in refined.state_dict().items():
    if n not in ['mlp.6.weight','mlp.6.bias']:check(torch.equal(before[n],p),'auxiliary_update_keeps_hidden_'+n)
hook.read_audit();hook.close()
check(torch.equal(refined(td),after_actions),'hook_removal_no_inference_effect')

# Qualified lower-body direct teacher derivatives are exactly the previous ones.
for command,interval in [(0.,None),(.65,(.1,1.)),(1.5,(.1,1.))]:
    raw[:,326:329]=torch.tensor([command,0.,0.]);td=TensorDict({'policy':raw.clone()},batch_size=[64])
    actor=Actor();new=copy.deepcopy(actor);teacher=Actor()
    old_hook=StandingRetention(actor,copy.deepcopy(teacher),command_range=interval)
    new_hook=LocomotionOnlyRetention(new,copy.deepcopy(teacher),command_range=interval,joint_names=names)
    a=actor(td);b=new(td);(a.sum()*0.).backward();(b.sum()*0.).backward()
    check(torch.equal(actor.mlp[-1].bias.grad[lower],new.mlp[-1].bias.grad[lower]),f'exact_prior_lower_derivative_{command}')
    check(bool((new.mlp[-1].bias.grad[ids]==0).all()),f'prior_does_not_copy_upper_{command}')
    old_hook.close();new_hook.close()
try:upper_indices(names[:-1])
except AssertionError:checks.append('reject_wrong_joint_inventory')
else:raise AssertionError('Wrong joint inventory accepted')
prior=Path(json.loads((OUT/'candidate_source_proof.json').read_text())['prior_run'])
check(training_reward_signature(STAGED/'training')==training_reward_signature(prior/'source/physical_v29/training'),'reward_inference_signature_unchanged')
result={'passed':True,'checks':checks,'count':len(checks),'torch_version':torch.__version__,
    'training_only_refinement':True,'initial_actor_and_optimizer_not_modified':True,
    'lower_direct_prior_gradient_bitwise_equal':True,'auxiliary_shared_gradient_zero':True,
    'new_posture_qualified':False,'physical_PPO_full_load_still_required':True}
(OUT/'cpu_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
