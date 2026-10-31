"""Verify exact gradient scope, observational ownership and inference neutrality."""
import copy,json,sys,hashlib
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'validation_source/physical_v29/training'
sys.path.insert(0,str(TRAIN));torch.set_num_threads(2)
from phase_transfer import RunningRowTransfer,context_inputs,UPPER
names=['left_hip_pitch','right_hip_pitch','waist_yaw','left_hip_roll','right_hip_roll','waist_pitch','left_hip_yaw','right_hip_yaw','waist_roll','left_knee_pitch','right_knee_pitch','head_yaw','left_shoulder_pitch','right_shoulder_pitch','left_ankle_pitch','right_ankle_pitch','head_pitch','left_shoulder_roll','right_shoulder_roll','left_ankle_roll','right_ankle_roll','left_elbow_pitch','right_elbow_pitch']
context=json.loads((TRAIN/'phase_context.json').read_text());checks=[]
class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.mlp=torch.nn.Sequential(torch.nn.Linear(400,32),torch.nn.ELU(),torch.nn.Linear(32,23))
    def forward(self,observations):return self.mlp(observations['policy'])
torch.manual_seed(71);base=Actor();teacher=Actor().eval()
raw=torch.randn(64,5,80)*.1;raw[:,:,6]=1.5;raw[:,:,7:9]=0.;raw[:,:,[20,25]]=0.
angle=torch.rand(64,5)*2*torch.pi;raw[:,:,78]=angle.sin();raw[:,:,79]=angle.cos();raw=raw.reshape(64,400)
table=torch.tensor(context['contexts']['1.5']);columns=context['columns']
modified=context_inputs(raw,table,columns)
mask=torch.ones(5,80,dtype=torch.bool);mask[:,columns]=False
assert torch.equal(modified.reshape(-1,5,80)[:,mask],raw.reshape(-1,5,80)[:,mask]);checks.append('Only teacher head channels transformed')
original=raw.clone();plain=copy.deepcopy(base);candidate=copy.deepcopy(base)
hook=RunningRowTransfer(candidate,teacher,names,context)
p=plain({'policy':raw});c=candidate({'policy':raw});assert torch.equal(p,c);checks.append('Before backward actor output bitwise identical')
p.square().mean().backward();c.square().mean().backward()
assert torch.equal(raw,original);checks.append('Student observations unchanged')
for name,param in candidate.named_parameters():
    baseline=dict(plain.named_parameters())[name].grad
    if 'mlp.2' not in name:assert torch.equal(param.grad,baseline);checks.append('Shared trunk gradient bitwise unchanged '+name)
    else:
        upper=[names.index(n) for n in UPPER]
        assert torch.equal(param.grad[upper],baseline[upper]);checks.append('Upper row gradient bitwise unchanged '+name)
        assert not torch.equal(param.grad[hook.lower],baseline[hook.lower]);checks.append('Auxiliary lower gradient present '+name)
with torch.no_grad():
    features=plain.mlp[:2](raw)
    target=teacher({'policy':modified}).clamp(-3,3)[:,hook.lower]
    expected=.25*(p.detach()[:,hook.lower]-target).clamp(-1,1)/(len(raw)*23)
    assert torch.allclose(candidate.mlp[-1].weight.grad[hook.lower]-plain.mlp[-1].weight.grad[hook.lower],expected.T@features,atol=2e-8)
    assert torch.allclose(candidate.mlp[-1].bias.grad[hook.lower]-plain.mlp[-1].bias.grad[hook.lower],expected.sum(0),atol=2e-8)
    checks+=['Explicit bounded Huber weight derivative matches','Explicit bias derivative matches']
assert all(p.grad is None and not p.requires_grad for p in teacher.parameters());checks.append('Frozen teacher has no derivative')
audit=hook.read_audit();assert audit['batches']==1;checks.append('Auxiliary audit consumed only actual backward batch')
candidate.eval();candidate.zero_grad();mean=candidate({'policy':raw});mean.square().mean().backward()
assert hook.read_audit()['batches']==0;checks.append('No inference auxiliary or context access')
hook.close()
for label,mutate in [('stopped',lambda x:x[:,:,6].zero_()),('prior',lambda x:x[:,:,6].fill_(1.)),
 ('mixed history',lambda x:x[:,0,6].zero_()),('invalid phase',lambda x:x[:,0,78:80].zero_()),
 ('nonneutral actual head',lambda x:x[:,-1,20].fill_(.5))]:
    actor=copy.deepcopy(base);teach=copy.deepcopy(teacher);ref=copy.deepcopy(base)
    inputs=raw.clone().reshape(-1,5,80);mutate(inputs);inputs=inputs.reshape(-1,400)
    loss=RunningRowTransfer(actor,teach,names,context)
    actor({'policy':inputs}).sum().backward();ref({'policy':inputs}).sum().backward()
    assert all(torch.equal(v.grad,dict(ref.named_parameters())[k].grad) for k,v in actor.named_parameters())
    assert loss.read_audit()['batches']==0;loss.close();checks.append('No auxiliary for '+label)
record=dict(passed=True,checks=checks,count=len(checks),student_inputs_truthful=True,
 inference_unchanged=True,upper_and_shared_aux_gradient_zero=True,production_changed=False,
 measured_sample=audit,implementation_sha256=hashlib.sha256((TRAIN/'phase_transfer.py').read_bytes()).hexdigest())
(OUT/'cpu_transfer_tests.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
