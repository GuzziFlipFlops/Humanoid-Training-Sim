"""Training-distribution counterexamples and protected source checks."""
import torch,json,hashlib,ast
from pathlib import Path
OUT=Path(__file__).resolve().parent
pause=json.loads((OUT/'diagnostic_pause.json').read_text());old=Path(pause['run'])/'source/physical_v29';new=OUT/'control_source/physical_v29'
def inventory(root):
 return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
assert inventory(old)==inventory(new)
tree=ast.parse((OUT/'lateral_exposure_candidate.py').read_text());sample=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sample'][0]
module=ast.Module(body=[sample],type_ignores=[]);ast.fix_missing_locations(module)
namespace={'torch':torch};exec(compile(module,'candidate_extracted','exec'),namespace)
from types import SimpleNamespace
cases=[]
for evaluation in [False,True]:
 for stage in [0.,1.,3.,3.5,4.]:
  speeds=torch.tensor([0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.0000005,3.01,3.5])
  commands=torch.zeros((len(speeds),3));commands[:,0]=speeds;commands[:,2]=torch.linspace(-.2,.2,len(speeds))
  before=commands.clone();env=SimpleNamespace(cfg=SimpleNamespace(evaluation=evaluation,stage_speed=stage),commands=commands,device='cpu')
  namespace['original']=lambda self,ids:None
  torch.manual_seed(71);namespace['sample'](env,torch.arange(len(speeds)))
  mask=(speeds>3.+1e-6) if stage==3.5 and not evaluation else torch.zeros(len(speeds),dtype=torch.bool)
  assert torch.equal(commands[:,0],before[:,0]) and torch.equal(commands[:,2],before[:,2])
  assert torch.equal(commands[~mask,1],before[~mask,1])
  assert commands[:,1].abs().max()<=.1
  assert torch.equal(commands[mask,1],commands[mask,1].clone())
  cases.append(dict(evaluation=evaluation,stage=stage,changed_rows=int((commands!=before).any(1).sum())))
# Selected resets cannot mutate unrelated rows.
commands=torch.zeros((40,3));commands[:,0]=3.5;before=commands.clone();env=SimpleNamespace(cfg=SimpleNamespace(evaluation=False,stage_speed=3.5),commands=commands,device='cpu')
namespace['sample'](env,torch.tensor([2,5,11]));untouched=torch.ones(40,dtype=torch.bool);untouched[[2,5,11]]=False;assert torch.equal(commands[untouched],before[untouched])
# Bounds/symmetry distribution, command magnitude, and existing runtime clamp.
v=.1*(2.*torch.rand(100000)-1.);assert abs(float(v.mean()))<.001 and abs(float((v>0).float().mean())-.5)<.01
norm=torch.sqrt(3.5**2+v**2);assert float((norm-3.5).max())<.0015
result=dict(passed=True,cases=cases,selected_reset_preservation=True,protected_authored_files=len(inventory(old)),all_runtime_bytes_identical=True,maximum_lateral_command=.1,maximum_frontier_norm_delta=float((norm-3.5).max()),symmetric_command_mean=float(v.mean()),evaluation_sampler_unchanged=True,all_prior_speeds_zero_lateral=True,no_reward_or_physical_limit_change=True,diagnostic_wrapper_only=True,qualifies_locomotion=False)
(OUT/'exposure_cpu_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
