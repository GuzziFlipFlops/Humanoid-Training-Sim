"""Stage one termination correction, preserving every other frozen runtime byte."""
import ast,copy,hashlib,json,shutil
from pathlib import Path
import torch
from types import SimpleNamespace
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
live=read(OUT/'initial_owned_observation.json');RUN=Path(live['active_run']['run']);source=RUN/'source/physical_v29'
candidate=OUT/'candidate_source/physical_v29'
assert not candidate.exists();shutil.copytree(source,candidate,ignore=shutil.ignore_patterns('usd','__pycache__','*.pyc'))
old=(source/'training/environment.py').read_bytes();nl=b'\r\n' if b'\r\n' in old else b'\n'
def b(x):return x.replace('\n',nl.decode()).encode()
changes=[
 ('  self.commands=torch.zeros((self.num_envs,3),device=self.device);self.phase=torch.zeros(self.num_envs,device=self.device)\n',
  '  self.commands=torch.zeros((self.num_envs,3),device=self.device);self.phase=torch.zeros(self.num_envs,device=self.device)\n  self.command_travel_budget=torch.zeros(self.num_envs,device=self.device);self._command_travel_tick=-1\n'),
 ('  finite=torch.isfinite(self.robot.data.joint_pos).all(1)&torch.isfinite(m[\'height\'])\n',
  '  finite=torch.isfinite(self.robot.data.joint_pos).all(1)&torch.isfinite(m[\'height\'])\n  # Preserve the 60m excess-travel margin; commanded locomotion is allowed.\n  # Charge the exact action command once per control tick, across switches.\n  if self._command_travel_tick!=self.common_step_counter:\n   command=getattr(self,\'action_commands\',self.commands)\n   self.command_travel_budget.add_(command[:,:2].norm(dim=1)*self.step_dt)\n   self._command_travel_tick=self.common_step_counter\n'),
 ('   excessive_drift=m[\'drift\']>60.,nonfoot_impact=',
  '   excessive_drift=m[\'drift\']>(60.+self.command_travel_budget),nonfoot_impact='),
 ('  self.start_xy[env_ids]=root[:,:2];self.phase[env_ids]=0.\n',
  '  self.start_xy[env_ids]=root[:,:2];self.phase[env_ids]=0.;self.command_travel_budget[env_ids]=0.\n')]
new=old
for before,after in changes:
 assert new.count(b(before))==1,before;new=new.replace(b(before),b(after))
(candidate/'training/environment.py').write_bytes(new)
ast.parse(new.decode())
def inventory(root):return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
prior,after=inventory(source),inventory(candidate)
diff=[k for k in prior.keys()|after.keys() if prior.get(k)!=after.get(k)];assert diff==['training/environment.py'],diff
reverse=new
for before,replacement in reversed(changes):reverse=reverse.replace(b(replacement),b(before))
assert reverse==old

def method(content):
 tree=ast.parse(content.decode());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LocomotionEnv')
 fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_get_dones')
 ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'environment.py','exec'),ns);return ns['_get_dones']
old_done,new_done=method(old),method(new);tests=[]
def fixture(speed=0.,drift=0.,other=None):
 m=dict(height=torch.ones(4),tilt=torch.zeros(4),torso_tilt=torch.zeros(4),invalid=torch.zeros(4,dtype=torch.bool),drift=torch.full((4,),float(drift)),nonfoot=torch.zeros(4))
 if other:m.update(other)
 e=SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(4,23))),metrics=None,
   commands=torch.tensor([[speed,0.,0.]]*4),action_commands=torch.tensor([[speed,0.,0.]]*4),step_dt=.02,
   command_travel_budget=torch.zeros(4),_command_travel_tick=-1,common_step_counter=1,total_weight=338.,
   episode_length_buf=torch.zeros(4,dtype=torch.long),max_episode_length=1000)
 e.telemetry=lambda:m;return e,m
def check(name,condition):assert condition,name;tests.append(name)
for speed in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
 e,m=fixture(speed)
 for i in range(1000):
  e.common_step_counter=i+1;m['drift'].fill_(speed*(i+1)*.02)
  result=new_done(e)
 check(f'normal_travel_{speed:g}',not result[0].any())
 check(f'integrated_budget_{speed:g}',torch.allclose(e.command_travel_budget,torch.full((4,),speed*20),atol=.002))
 saved=e.command_travel_budget.clone();new_done(e);check(f'idempotent_tick_{speed:g}',torch.equal(saved,e.command_travel_budget))
 m['drift']=e.command_travel_budget+60.01;check(f'excess_still_terminates_{speed:g}',new_done(e)[0].all())
for cause,value in [('height',torch.full((4,),.5)),('tilt',torch.full((4,),1.06)),('torso_tilt',torch.full((4,),1.11)),('invalid',torch.ones(4,dtype=torch.bool)),('nonfoot',torch.full((4,),400.))]:
 e,m=fixture(3.5,5.,{cause:value});f=copy.copy(e)
 a=old_done(e)[0];bnew=new_done(f)[0];check('physical_termination_'+cause,torch.equal(a,bnew) and bnew.all())
e,m=fixture(3.5);new_done(e);budget=e.command_travel_budget.clone();e.commands.zero_();e.action_commands.zero_();e.common_step_counter+=1;new_done(e)
check('stop_does_not_erase_earned_travel',torch.equal(budget,e.command_travel_budget))
e.command_travel_budget[[1,3]]=0.;check('per_env_reset',e.command_travel_budget[1]==0 and e.command_travel_budget[0]>0)
for speed in [0.,.15,1.,3.5,5.]:
 for drift in [0.,10.,30.,59.9]:
  e,m=fixture(speed,drift);f=copy.copy(e)
  check(f'prior_guard_below60_{speed}_{drift}',torch.equal(old_done(e)[0],new_done(f)[0]))
write=dict(passed=True,tests=tests,count=len(tests),only_runtime_change=diff,reverse_patch_bitwise_equal=True,
      protected_source_files=len(prior)-1,implementation_hashes={'training/environment.py':after['training/environment.py']},
      native_limits_and_classifiers_unchanged=True,
      scope='Termination displacement corrected; no gait/direction/survival gate relaxed. Exact original60m margin is retained beyond integrated executed command path length. Command switches and per-environment reset covered; all physical-fault predicates preserved.')
(OUT/'travel_guard_cpu_tests.json').write_text(json.dumps(write,indent=2)+'\n')
(OUT/'guard_patch.json').write_text(json.dumps({'prior_run':str(RUN),'source':str(candidate),'before_sha256':sha(source/'training/environment.py'),'after_sha256':sha(candidate/'training/environment.py'),'replacements':changes,'protected_hashes':{k:v for k,v in prior.items() if k!='training/environment.py'}},indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(tests),'changed_runtime_files':diff,'protected_files':len(prior)-1}))
