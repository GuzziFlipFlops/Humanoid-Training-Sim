"""Counterfactual gradients, qualification binding, and protected runtime tests."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
from tensordict import TensorDict
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
pause=json.loads((OUT/'diagnostic_pause.json').read_text());RUN=Path(pause['run']);NEW=OUT/'validation_source/physical_v29'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
new=module(NEW/'training/standing_retention.py','candidate');old=module(RUN/'source/physical_v29/training/standing_retention.py','prior')
sys.path.insert(0,str(NEW/'training'))
from curriculum_provenance import training_reward_signature
import ast
GATE_VERSION=next(ast.literal_eval(n.value) for n in ast.parse((NEW/'training/evaluation.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GATE_VERSION' for t in n.targets))
signature=training_reward_signature(NEW/'training')
assert signature==training_reward_signature(RUN/'source/physical_v29/training')
tests=[]
def check(name,condition):assert condition,name;tests.append(name)
class Actor(torch.nn.Module):
 def __init__(self):
  super().__init__();self.mlp=torch.nn.Linear(400,23)
 def forward(self,x):return self.mlp(x['policy'])
torch.manual_seed(71)
base=Actor();teacher=Actor()
raw=torch.randn(7,400)
raw.reshape(7,5,80)[:,:,6:9]=0.
commands=[0.,.15,.35,.65,1.,1.5,2.]
raw[:,326]=torch.tensor(commands)
observed=TensorDict({'policy':raw},[7])
actor=copy.deepcopy(base);prior=new.StandingRetention(actor,copy.deepcopy(teacher),command_range=(.1,1.))
actor.train();means=actor(observed);means.retain_grad();(means.sum()*0).backward()
with torch.no_grad():expected=torch.zeros_like(means);expected[1:5]=.5*(means[1:5]-teacher(observed[1:5]).clamp(-3,3))/92
check('exact_prior_gradient',torch.allclose(means.grad,expected,atol=1e-8))
check('standing_and_running_rows_unregularized',torch.equal(means.grad[[0,5,6]],torch.zeros((3,23))))
check('prior_rows_have_nonzero_gradient',bool((means.grad[1:5].abs().sum(1)>0).all()))
check('latest_frame_not_old_stopped_frame',prior.read_audit()['batches']==1)
check('teacher_frozen',all(not p.requires_grad for p in prior.teacher.parameters()))
prior.close();check('hooks_removed',not actor._forward_pre_hooks and not actor.mlp._forward_hooks)
gradients=[]
for impl in [old,new]:
 a=copy.deepcopy(base);h=impl.StandingRetention(a,copy.deepcopy(teacher));a.train();m=a(observed);m.retain_grad();(m.sum()*0).backward();gradients.append(m.grad.clone());h.close()
check('original_standing_gradient_bitwise_equal',torch.equal(*gradients))
a=copy.deepcopy(base);h1=new.StandingRetention(a,copy.deepcopy(teacher));h2=new.StandingRetention(a,copy.deepcopy(teacher),command_range=(.1,1.))
a.train();m=a(observed);m.retain_grad();(m.sum()*0).backward()
check('two_hooks_add_exactly_and_disjointly',torch.allclose(m.grad,gradients[0]+expected,atol=1e-8))
h1.close();h2.close()
for training,enabled in [(True,False),(False,True)]:
 a=copy.deepcopy(base);h=new.StandingRetention(a,copy.deepcopy(teacher),command_range=(.1,1.));a.train(training)
 with torch.set_grad_enabled(enabled):a(observed)
 check('inference_no_teacher_'+str(training),h.read_audit()['batches']==0);h.close()
a=copy.deepcopy(base);h=new.StandingRetention(a,copy.deepcopy(teacher),command_range=(.1,1.))
with torch.no_grad():before=base(observed);after=a(observed)
check('inference_outputs_bitwise_equal',torch.equal(before,after));h.close()
baseline=RUN/'002_ppo_smoke/transferred_actor.pt'
cert=new.qualification_certificate(RUN,baseline,1.,signature,GATE_VERSION)
check('qualified_teacher_all_commands',new.validate_qualification(cert,baseline,signature,GATE_VERSION)==1.)
(OUT/'qualified_teacher.json').write_text(json.dumps(cert,indent=2)+'\n')
for field,value in [('checkpoint_sha256','0'*64),('maximum_command_m_s',1.5),('reward_signature','bad'),('gate_version','old')]:
 bad=copy.deepcopy(cert);bad[field]=value
 try:new.validate_qualification(bad,baseline,signature,GATE_VERSION)
 except AssertionError:check('reject_'+field,True)
 else:raise AssertionError(field)
bad=copy.deepcopy(cert);bad['gates']*=2
try:new.validate_qualification(bad,baseline,signature,GATE_VERSION)
except AssertionError:check('reject_duplicate_flags',True)
else:raise AssertionError('duplicate')
selected=RUN/'028_run_150_train_004/candidate.pt'
for name in ['029_run_150_gate_005_nominal','030_run_150_gate_005_push']:
 c=read(RUN/name/'result.json')['cases'][-1]
 check('selected_running_case_'+name,c['passed'] and all(c['checks'].values()) and c['statistics']['trials']>=32)
check('changed_objective_not_needed',read(OUT/'physical_response/result.json')['groups'][4]['mean_reward_rate']>read(OUT/'physical_response/result.json')['groups'][0]['mean_reward_rate'])
changed=[];protected=[]
for p in (RUN/'source/physical_v29').rglob('*'):
 if not p.is_file() or '__pycache__' in p.parts or 'usd' in p.parts:continue
 rel=p.relative_to(RUN/'source/physical_v29').as_posix();q=NEW/rel
 (protected if p.read_bytes()==q.read_bytes() else changed).append(rel)
check('only_two_training_files_changed',set(changed)=={'training/standing_retention.py','training/worker_unitree.py'})
check('reward_signature_identical',signature==training_reward_signature(RUN/'source/physical_v29/training'))
result=dict(passed=True,tests=tests,count=len(tests),changed=changed,protected=protected,reward_signature=signature,
 inference_identical=True,original_standing_gradient_identical=True,training_only=True,qualifies_locomotion=False)
(OUT/'cpu_tests.json').write_text(json.dumps(result,indent=2)+'\n')
e=read(OUT/'evidence.json');updates=sum(b['updates'] for b in e['completed_main_blocks'] if Path(b['folder']).name<='028_run_150_train_004')
proof=dict(schema='prior_command_retention_continuation_v1',prior_run=str(RUN),drained_checkpoint_sha256=pause['checkpoint_sha256'],
 selected_checkpoint=str(selected),selected_checkpoint_sha256=sha(selected),selected_main_updates=updates,
 qualified_teacher=cert,reward_signature=signature,gate_version=GATE_VERSION,
 implementation_hashes={p:sha(NEW/p) for p in changed},supervisor_sha256=sha(OUT/'candidate_supervisor_unitree.py'),
 cpu_tests=dict(path='cpu_tests.json',sha256=sha(OUT/'cpu_tests.json')),
 reason='Restore a physically reproduced actor with passing running cases; preserve and retrain lost prior walking through bounded training-only teacher gradients. Reward and every inference/physical classifier remain identical.',
 qualifies_locomotion=False)
(OUT/'retention_continuation_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(passed=True,count=len(tests),protected=len(protected),selected_updates=updates,teacher_sha256=cert['checkpoint_sha256']),indent=2))
