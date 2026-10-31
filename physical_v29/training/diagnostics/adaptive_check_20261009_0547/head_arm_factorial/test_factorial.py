import json, torch
from pathlib import Path
from diagnostic_controllers import cohorts,combine_actions,MODES
out=Path(__file__).resolve().parent
groups=cohorts();per=2;n=len(groups)*per
current=torch.arange(n*23,dtype=torch.float32).reshape(n,23)
teacher=-current-100
upper=[1,7,2,3,4,5,6,8]
eligible=torch.tensor([g['command']<=3 for g in groups]).repeat_interleave(per)
result=combine_actions(current,teacher,teacher,[g['mode'] for g in groups],per,upper,eligible)
checks=[]
for mode in MODES:
    for i,g in enumerate(groups):
        if g['mode']!=mode:continue
        ids=slice(i*per,(i+1)*per)
        expected=current[ids].clone() if mode=='current' else teacher[ids].clone()
        selected=upper if mode=='teacher_body_current_upper' else upper[:2] if mode=='teacher_current_head' else upper[2:] if mode=='teacher_current_arms' else []
        if selected:expected[:,selected]=current[ids,selected]
        assert torch.equal(result[ids],expected)
    checks.append(mode+':same-own-row head/arm selection')
assert len(groups)==22 and n==44
checks.append('Two seeds;22 cohorts;3.5 sentinel current-only')
try:combine_actions(current,teacher,teacher,[g['mode'] for g in groups],per,upper,torch.zeros(n,dtype=torch.bool))
except AssertionError:checks.append('Teacher outside certificate rejected')
else:raise AssertionError('Eligibility bypass')
assert torch.equal(current,torch.arange(n*23,dtype=torch.float32).reshape(n,23))
checks.append('Inputs unchanged')
assert result.data_ptr()!=current.data_ptr()
checks.append('No input storage alias')
(out/'CPU_tests.json').write_text(json.dumps(dict(passed=True,count=len(checks),checks=checks),indent=2))
print('Passed',len(checks),'factorial CPU checks.')
