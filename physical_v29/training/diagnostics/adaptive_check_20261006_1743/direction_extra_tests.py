import json,torch,math
from pathlib import Path
from direction_kernel import score
out=Path(__file__).resolve().parent
checks=[]
c=torch.tensor([[3.5,0.,0.]])
for name,v in [('stationary',[0.,0.,0.]),('reverse',[-3.5,0.,0.]),('pure_lateral',[0.,3.5,0.])]:
 assert score(c,torch.tensor([v]),torch.zeros(1),.5).item()<1e-10;checks.append(name)
torch.manual_seed(91);v=torch.randn(400,3);h=torch.randn(400);cmd=c.repeat(400,1)
assert torch.allclose(score(cmd,v,h,.5),score(cmd,v,h+2*math.pi,.5),atol=2e-6,rtol=2e-5);checks.append('wrap_invariance')
v=torch.tensor([[3.5,.3,0.]],requires_grad=True);score(c,v,torch.zeros(1),.5).sum().backward();assert v.grad[0,1]<0;checks.append('corrective_lateral_derivative')
v=torch.tensor([[3.5,0.,0.]],requires_grad=True);h=torch.tensor([.15],requires_grad=True);score(c,v,h,.5).sum().backward();assert h.grad<0;checks.append('corrective_heading_derivative')
assert all(torch.isfinite(x).all() for x in [v.grad,h.grad]);checks.append('finite_gradients')
(out/'direction_extra_tests.json').write_text(json.dumps({'checks':checks,'passed':len(checks),'scope':'Synthetic counterexamples/autograd only, no physical or convergence certification.'},indent=2))
print(checks)
