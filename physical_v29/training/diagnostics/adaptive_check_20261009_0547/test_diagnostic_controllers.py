"""Reject cross-robot replay, aliasing and unqualified teacher selection."""
import copy,json
from pathlib import Path
import torch
from diagnostic_controllers import cohorts,combine_actions,MODES
from action_projection import clone_observation_mapping,project_targets
checks=[]
groups=cohorts();assert len(groups)==36 and len({g['name'] for g in groups})==36
checks.append('36 uniquely named paired cohorts')
assert all(g['mode']=='current' for g in groups if g['command']==3.5)
checks.append('No3.5 teacher invocation/selection')
upper=[0,1,15,16,17,18,21,22];lower=[i for i in range(23) if i not in upper]
current=torch.arange(6*2*23).reshape(12,23).float()/17
teacher=current+100;context=current+200;qualified=torch.ones(12,dtype=torch.bool)
actual=combine_actions(current,teacher,context,MODES,2,upper,qualified)
assert torch.equal(actual[:2],current[:2]);checks.append('Current actions unchanged')
assert torch.equal(actual[2:4],teacher[2:4]);checks.append('Teacher actions same own rows')
assert torch.equal(actual[4:6,upper],current[4:6,upper]) and torch.equal(actual[4:6,lower],teacher[4:6,lower]);checks.append('Hybrid retains current upper and same-state teacher body')
assert torch.equal(actual[6:8,upper],teacher[6:8,upper]) and torch.equal(actual[6:8,lower],current[6:8,lower]);checks.append('Inverse hybrid retains current body')
assert torch.equal(actual[8:10],context[8:10]);checks.append('Existing label context controller explicit')
assert torch.equal(actual[10:,upper],current[10:,upper]) and torch.equal(actual[10:,lower],context[10:,lower]);checks.append('Context body hybrid is explicit')
before=current.clone();actual.fill_(-99);assert torch.equal(current,before);checks.append('No aliasing or donor robot actions')
qualified[4]=False
try:combine_actions(current,teacher,context,MODES,2,upper,qualified)
except AssertionError:checks.append('Unqualified teacher use rejected')
else:raise AssertionError('Unqualified teacher accepted')
mapping={'policy':torch.ones(3,400),'critic':torch.ones(3,440)}
cloned=clone_observation_mapping(mapping);cloned['policy'][0,0]=7
assert mapping['policy'][0,0]==1;checks.append('Teacher-context storage cannot change student observations')
cfg={'actor':{'class_name':'MLP'}};clone=copy.deepcopy(cfg);clone['actor'].pop('class_name')
assert cfg['actor']['class_name']=='MLP';checks.append('Runner configs independently owned')
stopped=torch.zeros(2,23);target,clip,slew,invalid=project_targets(stopped,stopped,torch.ones(23),torch.full((23,),-1.),torch.ones(23),stopped,torch.ones(23),.02)
assert torch.equal(target,stopped) and not clip.any() and not slew.any() and not invalid.any();checks.append('Native projection identity fixture')
record=dict(passed=True,count=len(checks),cases=checks,external_diagnostic_only=True,production_changes=False)
(Path(__file__).resolve().parent/'controller_cpu_tests.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
