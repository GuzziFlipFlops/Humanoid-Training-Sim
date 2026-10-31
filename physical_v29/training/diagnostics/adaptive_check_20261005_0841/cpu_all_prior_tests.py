import copy,sys,json,importlib.util,hashlib
from pathlib import Path
import torch
from tensordict import TensorDict
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'all_prior_source/physical_v29/training';sys.path.insert(0,str(TRAIN))
torch.set_num_threads(2)
from upper_body_refinement import LocomotionOnlyRetention,UPPER_JOINTS
from teacher_head_context import context_inputs
spec=importlib.util.spec_from_file_location('original_upper',Path(json.loads((OUT/'diagnostic_pause.json').read_text())['run'])/'source/physical_v29/training/upper_body_refinement.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
names=['left_hip_pitch','right_hip_pitch','waist_yaw','left_hip_roll','right_hip_roll','waist_pitch','left_hip_yaw','right_hip_yaw','waist_roll','left_knee_pitch','right_knee_pitch','head_yaw','left_shoulder_pitch','right_shoulder_pitch','left_ankle_pitch','right_ankle_pitch','head_pitch','left_shoulder_roll','right_shoulder_roll','left_ankle_roll','right_ankle_roll','left_elbow_pitch','right_elbow_pitch']
class Actor(torch.nn.Module):
    def __init__(self):super().__init__();self.mlp=torch.nn.Sequential(torch.nn.Linear(400,32),torch.nn.ELU(),torch.nn.Linear(32,23))
    def forward(self,d):return self.mlp(d['policy'])
torch.manual_seed(71);base=Actor();teacher=Actor();context=json.loads((TRAIN/'teacher_head_context.json').read_text());checks=[]
def observations(speed):
    raw=torch.randn(32,5,80)*.1;raw[:,:,6]=speed;raw[:,:,7:9]=0.;raw[:,:,[20,25]]=0.
    phase=torch.rand(32,5)*2*torch.pi;raw[:,:,78]=phase.sin();raw[:,:,79]=phase.cos()
    return TensorDict({'policy':raw.reshape(-1,400),'critic':torch.randn(32,80)},batch_size=[32])
for command,range_ in [(0.,None),(.15,(.1,1.)),(.35,(.1,1.)),(.65,(.1,1.)),(1.,(.1,1.)),(1.5,(.1,1.))]:
    a,b=copy.deepcopy(base),copy.deepcopy(base);teach=copy.deepcopy(teacher)
    oldhook=old.LocomotionOnlyRetention(a,copy.deepcopy(teach),joint_names=names,command_range=range_)
    newhook=LocomotionOnlyRetention(b,copy.deepcopy(teach),joint_names=names,command_range=range_)
    data=observations(command);owned=data.clone()
    before=a(data);after=b(data);assert torch.equal(before,after);checks.append('Same inference mean '+str(command))
    before.square().mean().backward();after.square().mean().backward()
    assert torch.equal(data['policy'],owned['policy']) and torch.equal(data['critic'],owned['critic']);checks.append('Input ownership '+str(command))
    eligible=command in [.15,.35,.65,1.]
    if eligible:
        reference=copy.deepcopy(base);inputs=data.clone()
        inputs['policy']=context_inputs(inputs['policy'],torch.tensor(context['contexts'][str(command)]),context['columns'])
        target=teach(inputs).detach().clamp(-3.,3.);upper=[names.index(n) for n in UPPER_JOINTS]
        mean=reference(data);difference=mean-target;difference[:,upper]=0.
        (mean.square().mean()+.25*difference.square().mean()).backward()
        assert all(torch.allclose(v.grad,dict(reference.named_parameters())[k].grad,atol=2e-7) for k,v in b.named_parameters())
        checks.append('Unchanged exact .25 MSE formula with declared conditioned target '+str(command))
        assert not torch.equal(a.mlp[-1].weight.grad,b.mlp[-1].weight.grad)
    else:
        assert all(torch.equal(v.grad,dict(a.named_parameters())[k].grad) for k,v in b.named_parameters())
        checks.append('Unmatched derivative bitwise identical '+str(command))
    for k in ['weight','bias']:
        upper=[names.index(n) for n in UPPER_JOINTS]
        assert torch.equal(getattr(a.mlp[-1],k).grad[upper],getattr(b.mlp[-1],k).grad[upper])
    checks.append('No direct upper row teacher copying '+str(command))
    audit=newhook.read_audit();assert audit['teacher_phase_conditioned_samples']==(32 if eligible else 0)
    oldhook.close();newhook.close()
for reason in ['mixed_history','nonneutral_head','invalid_phase']:
    d=observations(.35);x=d['policy'].reshape(-1,5,80)
    if reason=='mixed_history':x[:,0,6]=.15
    if reason=='nonneutral_head':x[:,-1,20]=.5
    if reason=='invalid_phase':x[:,0,78:80]=0.
    a,b=copy.deepcopy(base),copy.deepcopy(base)
    h1=old.LocomotionOnlyRetention(a,copy.deepcopy(teacher),joint_names=names,command_range=(.1,1.))
    h2=LocomotionOnlyRetention(b,copy.deepcopy(teacher),joint_names=names,command_range=(.1,1.))
    a(d).sum().backward();b(d).sum().backward()
    assert all(torch.equal(v.grad,dict(a.named_parameters())[k].grad) for k,v in b.named_parameters())
    assert h2.read_audit()['teacher_phase_conditioned_samples']==0;h1.close();h2.close();checks.append('No conditioning for '+reason)
record=dict(passed=True,count=len(checks),checks=checks,student_observations_truthful=True,standing_bitwise_gradient_equal=True,
 qualified_prior_certificate_does_not_qualify_conditioned_controller=True,production_changed=False)
(OUT/'cpu_all_prior_tests.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
