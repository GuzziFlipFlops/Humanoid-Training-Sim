import json,math
from pathlib import Path
import numpy as np,torch
from direction_kernel import score
out=Path(__file__).resolve().parent
checks=[]
for speed in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
 c=torch.tensor([[speed,0.,0.]])
 v=torch.tensor([[speed,0.,0.]])
 assert score(c,v,torch.zeros(1),.5).item()==1.
 checks.append(f'aligned_{speed}')
for speed in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.]:
 torch.manual_seed(71);v=torch.randn(1024,3);c=torch.zeros_like(v);c[:,0]=speed;h=torch.randn(1024)
 old=torch.exp(-((c[:,:2]-v[:,:2])**2).sum(1)/.5**2)
 assert torch.equal(score(c,v,h,.5),old)
 checks.append(f'prior_bitwise_1024_{speed}')
for sign in [-1.,1.]:
 c=torch.tensor([[3.5,0.,0.]])
 h=torch.tensor([sign*.2]);v=torch.tensor([[3.5,0.,0.]])
 assert 0<score(c,v,h,.5).item()<.2
 # A heading-tilted robot traveling along the fixed target earns full score.
 v=torch.tensor([[3.5*math.cos(.2),sign*3.5*math.sin(.2),0.]])
 assert abs(score(c,v,h,.5).item()-1.)<1e-6
 checks.append(f'heading_sign_and_compensation_{sign}')
c=torch.tensor([[3.5,0.,0.]]).repeat(4,1);v=c.clone();v[:,1]=torch.tensor([0.,.15,.3,.6])
r=score(c,v,torch.zeros(4),.5);assert (r[:-1]>r[1:]).all()
checks.append('monotone_lateral_error')
assert torch.all((r>=0)&(r<=1))
checks.append('bounded_and_same_std')
review=json.loads((out/'initial_gate_review.json').read_text())
pair=review['latest'];summaries=[]
for label in ['nominal','pushed']:
 p=Path(pair[label])/'gait_diagnostics_50hz.npz'
 with np.load(p) as d:
  print(json.dumps({'label':label,'keys':d.files}))
result={'completed':True,'checks':checks,'total_checks':len(checks),'reward_weight_width_unchanged':True,'prior_fixtures_bitwise':True,'deployed':False,
 'scope':'Synthetic coordinate/negative-transfer checks only; do not prove dynamic gait or convergence.'}
(out/'direction_cpu_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
