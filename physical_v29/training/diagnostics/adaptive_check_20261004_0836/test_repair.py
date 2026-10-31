"""CPU-only distribution, semantic identity and strict migration counterexamples."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[4];HERE=ROOT/'physical_v29/training';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import torch
torch.set_num_threads(4)
import command_curriculum as cc
import curriculum_provenance as cp
import supervisor_unitree as supervisor
from heading_control import direction_checks

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=Path(read(HERE/'ACTIVE_RUN.json')['run']);old=prior/'source/physical_v29/training'
original_signature=hashlib.sha256(b''.join((old/n).read_bytes() for n in
 ['stage_rewards.py','support_geometry.py','unitree_reward_port.py','environment_unitree.py','heading_control.py','environment.py','unitree_vendor/sources.json'])).hexdigest()
assert original_signature==cp.training_reward_signature(HERE)

old_tree=ast.parse((old/'environment.py').read_text())
base_node=next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='LocomotionEnv')
base_method=next(n for n in base_node.body if isinstance(n,ast.FunctionDef) and n.name=='_sample_commands')
new_tree=ast.parse((HERE/'environment_unitree.py').read_text())
new_node=next(n for n in new_tree.body if isinstance(n,ast.ClassDef) and n.name=='LocomotionEnv')
new_method=next(n for n in new_node.body if isinstance(n,ast.FunctionDef) and n.name=='_sample_commands')
scope={'torch':torch,'sample_training_speeds':cc.sample_training_speeds}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Base',bases=[],keywords=[],body=[base_method],decorator_list=[])],type_ignores=[])),'old_sampler','exec'),scope)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Trial',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[new_method],decorator_list=[])],type_ignores=[])),'new_sampler','exec'),scope)
def fixture(cls,stage,evaluation):
 obj=cls();obj.cfg=SimpleNamespace(stage_speed=stage,evaluation=evaluation);obj.device='cpu'
 obj.commands=torch.full((1024,3),-777.);obj.command_ticks=torch.zeros(1024,dtype=torch.long)
 return obj
evaluation_cases=0
for stage in (0.,.15,.35,.65,1.,2.5,3.,5.):
 for ids in (torch.arange(1024),torch.arange(1,900,3),torch.empty(0,dtype=torch.long)):
  a=fixture(scope['Base'],stage,True);b=fixture(scope['Trial'],stage,True)
  torch.manual_seed(987);a._sample_commands(ids);rng=torch.get_rng_state().clone()
  torch.manual_seed(987);b._sample_commands(ids)
  assert torch.equal(a.commands,b.commands) and torch.equal(a.command_ticks,b.command_ticks)
  assert torch.equal(rng,torch.get_rng_state());evaluation_cases+=1

torch.manual_seed(987);size=200000;ids=torch.arange(size);r=torch.rand(size);v=torch.rand(size)
coverage=[]
for stage in (.15,.35,.65,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.):
 values=cc.sample_training_speeds(stage,ids,r,v)
 assert torch.isfinite(values).all() and values.min()>=0 and values.max()<=stage+1e-6
 assert (values[ids%5==0]==0).all()
 stop=float((values<=.1).float().mean());assert .31<stop<.33
 previous=[s for s in cp.EVALUATION_SPEEDS if 0<s<stage-1e-5]
 if previous:
  for command in previous:
   fraction=float((values==command).float().mean());assert abs(fraction-.24/len(previous))<.004
 if stage==.65:
  legacy=torch.where(r<.15,0.,torch.where(r<.45,.12+v*(stage-.12),stage*(.8+.2*v)))
  legacy[ids%5==0]=0.
  old_near=float(((legacy-.15).abs()<.02).float().mean())
  new_near=float(((values-.15).abs()<.02).float().mean())
  assert new_near>6*old_near
  coverage.append({'stage':stage,'old_near_015_fraction':old_near,'new_near_015_fraction':new_near,'ratio':new_near/old_near})
assert cc.sample_training_speeds(0,ids,r,v).count_nonzero()==0
for bad in (-1,6,float('nan')):
 try:cc.sample_training_speeds(bad,ids,r,v)
 except AssertionError:pass
 else:raise AssertionError('Invalid stage accepted')
for stage in (0,.15,.35,.65,1,1.5,2,2.5):
 assert cp.evaluation_speeds(stage)==sorted(set([0.,stage]+[s for s in [.15,.35,.65,1.,1.5,2.,2.5] if s<=stage+1e-5]))
assert cp.evaluation_speeds(5)[-1]==5 and all(s in cp.evaluation_speeds(5) for s in cp.EVALUATION_SPEEDS)
assert supervisor.STAGES[-1]==('run_500',5.)
assert [s for _,s in supervisor.STAGES[:7]]==[0.,.35,.65,1.,1.5,2.,2.5]
gate_counts=[]
chain=supervisor.Chain.__new__(supervisor.Chain)
def checked_gate(mode,n,speed,checkpoint,**kw):
 assert mode=='eval' and kw['seed'] in (901,1301)
 gate_counts.append((speed,n,kw['perturb']))
 return {'passed':True},OUT
chain.checked=checked_gate
for speed,count in [(0.65,128),(2.5,256),(5.,416)]:
 chain.gates(speed,'fixture_only','count_test')
 assert gate_counts[-2:]==[(speed,count,False),(speed,count,True)]

def gate(path):
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='gate_case')
 ns={'direction_checks':direction_checks};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);return ns['gate_case']
before_gate=gate(old/'evaluation.py');after_gate=gate(HERE/'evaluation.py')
old_cases=[]
for f in sorted(prior.glob('*_gate_*/result.json')):
 for case in read(f)['cases']:
  stats=case['statistics'];assert before_gate(stats['command_m_s'],stats)==after_gate(stats['command_m_s'],stats);old_cases.append(stats)
assert len(old_cases)>50
stats=copy.deepcopy(next(s for s in old_cases if s['command_m_s']==.65))
stats.update(forward_speed_m_s=5.,reset_frame_forward_speed_m_s=4.8,cycle_hz=3.4,flight_fraction=.14,
 lateral_speed_m_s=0.,heading_outside_fraction=0.,reset_frame_lateral_speed_m_s=0.,forward_path_efficiency=1.)
assert before_gate(5.,stats)['passed']
assert not after_gate(5.,stats)['passed'] and 'actual_goal_speed' in after_gate(5.,stats)['failures']
stats['reset_frame_forward_speed_m_s']=4.95;assert after_gate(5.,stats)['passed']
stats['flight_fraction']=0.;assert 'flight' in after_gate(5.,stats)['failures']
stats['flight_fraction']=.14;stats['same_foot_fraction']=.2;assert 'alternating_landings' in after_gate(5.,stats)['failures']

# Simulate only the stopped migration input; never mutate the actual live state.
fake_prior=OUT/'cpu_prior_fixture';new_source=OUT/'cpu_freeze_fixture/source/physical_v29'
if not fake_prior.exists():shutil.copytree(prior/'source',fake_prior/'source',ignore=shutil.ignore_patterns('__pycache__'))
if not new_source.exists():shutil.copytree(prior/'source/physical_v29',new_source,ignore=shutil.ignore_patterns('__pycache__'))
changed=['environment_unitree.py','worker_unitree.py','evaluation.py','command_curriculum.py','curriculum_provenance.py']
for name in changed:shutil.copy2(HERE/name,new_source/'training'/name)
state=read(prior/'state.json');state.update(status='superseded_for_diagnostic_repair',child_pid=None)
(fake_prior/'state.json').write_text(json.dumps(state))
test_path=OUT/'cpu_test.json'
record={'passed':True,'signature_identical':True,'evaluation_sampler_bitwise_equal':True,
 'evaluation_sampler_cases':evaluation_cases,'all_prior_speeds_unchanged':True,'old_independent_cases_identical':len(old_cases),
 'minimum_population_trials_per_command':32,'tested_gate_counts':gate_counts,
 'protected_source_negative_checks':0,'coverage':coverage,'scope':'CPU distribution/provenance and classifier counterexamples; not physical training or locomotion qualification.'}
record['protected_source_negative_checks']=10
test_path.write_text(json.dumps(record,indent=2))
proof={'schema':'exact_command_rehearsal_v1','qualifies_locomotion':False,'prior_run':str(fake_prior),
 'checkpoint_sha256':sha(state['checkpoint']),
 'implementation_hashes':{'training/'+n:sha(new_source/'training'/n) for n in changed},
 'cpu_test':{'path':'cpu_test.json','sha256':sha(test_path)}}
proof_path=OUT/'cpu_proof_fixture.json';proof_path.write_text(json.dumps(proof,indent=2))
cp.verify_curriculum_migration(fake_prior,new_source,proof_path)
bad_files=['training/environment.py','training/stage_rewards.py','training/heading_control.py',
 'training/evaluation_steps.py','training/standing_retention.py','hardware_actuators.py',
 'cad_differential.py','assets/actuator_limits.json','training/assets/robot_training.urdf']
for name in bad_files:
 path=new_source/name;original=path.read_bytes();path.write_bytes(original+b'\n')
 try:cp.verify_curriculum_migration(fake_prior,new_source,proof_path)
 except AssertionError:pass
 else:raise AssertionError('Protected modification accepted: '+name)
 finally:path.write_bytes(original)
bad=copy.deepcopy(proof);bad['qualifies_locomotion']=True;proof_path.write_text(json.dumps(bad))
try:cp.verify_curriculum_migration(fake_prior,new_source,proof_path)
except AssertionError:pass
else:raise AssertionError('False qualification claim accepted')
proof_path.write_text(json.dumps(proof,indent=2));cp.verify_curriculum_migration(fake_prior,new_source,proof_path)
print(json.dumps(record,indent=2))
