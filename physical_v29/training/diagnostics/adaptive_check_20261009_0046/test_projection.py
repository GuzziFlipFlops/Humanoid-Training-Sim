"""Meaningful boundary, constraint-order and paired-state diagnostic fixtures."""
import json
import copy
from pathlib import Path
import torch
from tensordict import TensorDict
from action_projection import project_targets, ramp, clone_observation_mapping
OUT=Path(__file__).resolve().parent
cases=[]
def check(name, condition):
    assert bool(condition), name
    cases.append(name)
n=6
a=torch.tensor([[0.,0.],[20.,-20.],[1.,-1.],[2.,2.],[0.,0.],[.8,.8]])
previous=torch.zeros_like(a);reference=torch.zeros(2);scale=torch.full((2,),.25)
lo=torch.full((2,),-.5);hi=-lo;slew=torch.full((2,),100.)
t,c,s,i=project_targets(a,reference,scale,lo,hi,previous,slew,.02)
check('quiet zero',torch.equal(t[0],torch.zeros(2)))
check('action clamp precedes joint clamp',torch.equal(t[1],torch.tensor([.5,-.5])) and c[1].all())
check('interior target unchanged',torch.equal(t[2],torch.tensor([.25,-.25])))
check('exact joint boundary not clipped',not c[3].any())
t,c,s,i=project_targets(a,reference,scale,lo,hi,previous,torch.full((2,),4.),.02)
check('slew rate bound',t.abs().max()<=.080001 and s[1].all())
check('quiet state does not slew',not s[0].any())
class InvalidPair:
    def state(self,q):
        valid=q.sum(-1)<=.05
        return None,None,valid,None
t,c,s,i=project_targets(a,reference,scale,lo,hi,previous,slew,.02,[('pair',[0,1],InvalidPair())])
check('invalid coupled target holds previous',torch.equal(t[3],previous[3]) and i[3].all())
check('valid coupled pose retained',torch.equal(t[2],torch.tensor([.25,-.25])) and not i[2].any())
check('original input unmodified',a[1,0]==20 and not previous.any())
check('ramp zero before intervention',ramp(4.99)==0 and ramp(5.)==0)
check('ramp midpoint',ramp(5.25)==.5)
check('ramp bounded and smooth endpoints',ramp(100)==1 and 0<ramp(5.01)<ramp(5.02)<1)
g=torch.Generator().manual_seed(57)
a=torch.randn(128,23,generator=g)*5
ref=torch.randn(23,generator=g)*.1;scale=torch.full((23,),.25)
prev=torch.randn(128,23,generator=g)*.1;lo=torch.full((23,),-.5);hi=-lo;slew=torch.full((23,),12.)
actual=project_targets(a,ref,scale,lo,hi,prev,slew,.02)[0]
expected=ref+scale*a.clamp(-3,3)
expected=torch.maximum(torch.minimum(expected,hi),lo)
expected=prev+(expected-prev).clamp(-slew*.02,slew*.02)
check('frozen scalar-map random parity',torch.equal(actual,expected))
check('map repeated call is nonmutating',torch.equal(actual,project_targets(a,ref,scale,lo,hi,prev,slew,.02)[0]))
for kind in ['dict','TensorDict']:
    original={'policy':torch.randn(4,400,generator=g),'critic':torch.randn(4,400,generator=g)}
    if kind=='TensorDict':original=TensorDict(original,batch_size=[4])
    before={k:v.clone() for k,v in original.items()}
    owned=clone_observation_mapping(original)
    owned['policy'][:,0]=37.
    check(kind+' teacher observation storage isolated',all(torch.equal(original[k],before[k]) for k in before))
    check(kind+' teacher observation other fields exact',torch.equal(owned['critic'],before['critic']))
configuration={'algorithm':{'class_name':'PPO','learning_rate':1e-4},'policy':{'class_name':'ActorCritic','actor_hidden_dims':[512,256]}}
first=copy.deepcopy(configuration);second=copy.deepcopy(configuration)
first['algorithm'].pop('class_name');first['policy']['actor_hidden_dims'][0]=17
check('separate runner configuration ownership',second==configuration and second['algorithm']['class_name']=='PPO')
(OUT/'projection_cpu_tests.json').write_text(json.dumps(dict(passed=True,count=len(cases),cases=cases,
    scope='Diagnostic reconstruction and bounded arm ramp only; not a production gradient or physical identification.'),indent=2)+'\n')
print('Passed',len(cases),'CPU fixtures.')
