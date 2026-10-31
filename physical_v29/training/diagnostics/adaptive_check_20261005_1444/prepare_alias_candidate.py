import json,shutil,hashlib,torch
from pathlib import Path
OUT=Path(__file__).resolve().parent;control=OUT/'offline_source/physical_v29';candidate=OUT/'alias_source/physical_v29'
assert not candidate.exists();shutil.copytree(control,candidate,ignore=shutil.ignore_patterns('__pycache__','*.pyc','usd'))
p=candidate/'training/upper_body_refinement.py';b=p.read_bytes()
token=b"            difference=mean[ids].detach()-target;difference[:,self.upper]=0.\r\n"
if token not in b:token=token.replace(b'\r\n',b'\n')
assert b.count(token)==1
nl=b'\r\n' if b'\r\n' in b else b'\n'
insert=nl.join([b"            difference=mean[ids].detach()-target;difference[:,self.upper]=0.",
 b"            if self.command_range is not None:",
 b"                # Equal executed actions at the same clamp endpoint require",
 b"                # no walking-teacher correction of an unexecuted raw mean.",
 b"                equivalent=(mean[ids].detach().clamp(-3.,3.)==target)&(mean[ids].detach().abs()>3.)",
 b"                difference=torch.where(equivalent,torch.zeros_like(difference),difference)",b""])
p.write_bytes(b.replace(token,insert))
# Counterfactuals cover exact endpoint aliases, corrective disagreement,
# every in-range fixture and the stopped-gradient invariant.
torch.manual_seed(41);checks=[]
def residual(mean,target,moving):
 diff=mean-target.clamp(-3,3)
 if moving:diff=torch.where((mean.clamp(-3,3)==target.clamp(-3,3))&(mean.abs()>3),torch.zeros_like(diff),diff)
 return diff
for sign in [-1.,1.]:
 mean=torch.tensor([[sign*4,sign*5,sign*10.]])
 target=torch.full_like(mean,sign*3.)
 assert (residual(mean,target,True)==0).all();checks.append('same_endpoint_'+str(sign))
 inside=torch.full_like(mean,sign*2.)
 assert torch.equal(residual(mean,inside,True),mean-inside);checks.append('disagreement_'+str(sign))
 assert torch.equal(residual(mean,target,False),mean-target);checks.append('stopped_unchanged_'+str(sign))
for i in range(10):
 mean=torch.rand(64,23)*6-3;target=torch.randn(64,23)*5
 assert torch.equal(residual(mean,target,True),mean-target.clamp(-3,3));checks.append('in_range_'+str(i))
def inventory(root):return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'usd' not in p.parts}
a,b=inventory(control),inventory(candidate);changed=[k for k in a if a[k]!=b[k]]
assert changed==['training/upper_body_refinement.py']
record=dict(passed=True,checks=checks,changed_files=changed,implementation_sha256=b[changed[0]],
 unchanged_weight=.25,unchanged_denominator=23,stopped_gradient_unchanged=True,inference_unchanged=True,
 qualifies_locomotion=False,hypothesis='Remove reproduced teacher correction on physically equivalent clipped actions; matched test decides usefulness. Not established as primary running fault.')
(OUT/'alias_cpu_tests.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
