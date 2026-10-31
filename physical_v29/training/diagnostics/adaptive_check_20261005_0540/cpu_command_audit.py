"""Counterfactual sampler checks without loading Isaac or CUDA."""
import json,sys,importlib.util,hashlib
from pathlib import Path
import torch
torch.set_num_threads(4)
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
run=Path(json.loads((HERE/'ACTIVE_RUN.json').read_text())['run'])
old=run/'source/physical_v29';new=OUT/'validation_source/physical_v29'
sys.path.insert(0,str(old/'training'))
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
a=module('original_commands',old/'training/command_curriculum.py')
b=module('candidate_commands',new/'training/command_curriculum.py')
torch.manual_seed(71);n=200000
ids=torch.arange(n);r=torch.rand(n);v=torch.rand(n)
records=[];checks=[]
for speed in [0.,.15,.35,.65,1.,1.5,1.6,1.875,2.,2.5,3.,3.5,4.,4.5,5.]:
    x=a.sample_training_speeds(speed,ids,r,v);y=b.sample_training_speeds(speed,ids,r,v)
    prior=(r<.45)|(ids%5==0);frontier=(r>=.45)&(ids%5!=0)
    assert torch.equal(x[prior],y[prior]);checks.append('prior/stopped equality '+str(speed))
    assert bool(((y>=0)&(y<=speed)).all());checks.append('bounded '+str(speed))
    if speed<=1. or speed>=1.875:
        assert torch.equal(x,y);checks.append('entire bitwise sampler '+str(speed))
    if speed>=1.5:
        assert bool((y[frontier]>=1.5).all());checks.append('running frontier '+str(speed))
    records.append(dict(stage_speed=speed,old_running_fraction=float((x>=1.5).float().mean()),
        new_running_fraction=float((y>=1.5).float().mean()),
        old_walking_frontier_fraction=float(((x<1.5)&frontier).float().mean()),
        new_walking_frontier_fraction=float(((y<1.5)&frontier).float().mean()),
        prior_stopped_identical=True,entire_sampler_bitwise_equal=torch.equal(x,y)))
for dtype in [torch.float32,torch.float64]:
    edges=torch.tensor([0.,.14999,.15,.44999,.45,.74999,.75,.99999],dtype=dtype)
    values=torch.tensor([0.,1.,.5,.3,.1,.9,.2,.99999],dtype=dtype)
    idx=torch.arange(1,9)
    y=b.sample_training_speeds(1.5,idx,edges,values)
    mask=(edges>=.45)&(idx%5!=0)
    assert bool((y[mask]==1.5).all());checks.append('exact running boundary '+str(dtype))
def inventory(root):return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
    for p in root.rglob('*') if p.is_file() and p.suffix!='.pyc' and p.relative_to(root).parts[:3]!=('training','assets','usd')}
left,right=inventory(old),inventory(new)
changed={k for k in left.keys()|right.keys() if left.get(k)!=right.get(k)}
assert changed=={'training/command_curriculum.py'}
from curriculum_provenance import training_reward_signature
assert training_reward_signature(old/'training')==training_reward_signature(new/'training')
row={'passed':True,'checks':checks,'records':records,'changed':sorted(changed),
    'signature_identical':True,'evaluation_sampler_bitwise_equal':True,'all_prior_speeds_unchanged':True,
    'qualifies_locomotion':False,'scope':'A training exposure change, not a gait/reward/kernel or gate change.'}
(OUT/'cpu_command_audit.json').write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'first_running':records[5]},indent=2))
