"""CPU fixtures for exact diagnostic head/body preservation and row selection."""
from pathlib import Path
import sys,json,hashlib,torch
OUT=Path(__file__).resolve().parent/'high_speed_extrapolation';sys.path.insert(0,str(OUT))
from diagnostic_arm_rows import replace_arm_rows
torch.set_num_threads(4)
x=torch.arange(23*12,dtype=torch.float32).reshape(12,23);y=-x-100
rows=torch.zeros(23,dtype=torch.bool);rows[[4,5,7,10,16,22]]=True
checks=[]
for selected in [torch.zeros(12,dtype=torch.bool),torch.ones(12,dtype=torch.bool),torch.arange(12)%2==0]:
    z=replace_arm_rows(x,y,rows,selected)
    assert torch.equal(z[~selected],x[~selected])
    assert torch.equal(z[:,~rows],x[:,~rows])
    assert torch.equal(z[selected][:,rows],y[selected][:,rows])
    assert torch.equal(x,torch.arange(23*12,dtype=torch.float32).reshape(12,23))
    checks.append('Exact role/head/body row preservation, selected='+str(int(selected.sum())))
for a,b,r,m in [(x,y[:1],rows,torch.ones(12,dtype=torch.bool)),(x,y,rows[:-1],torch.ones(12,dtype=torch.bool)),(x,y,rows,torch.ones(11,dtype=torch.bool))]:
    try:replace_arm_rows(a,b,r,m)
    except AssertionError:checks.append('Invalid shape rejected')
    else:raise AssertionError('Invalid row shape accepted')
receipt=dict(passed=True,count=len(checks),checks=checks,helper_sha256=hashlib.sha256((OUT/'diagnostic_arm_rows.py').read_bytes()).hexdigest(),
    scope='CPU row-selection fixtures; physical closed-loop compatibility and locomotion qualification remain untested.')
(OUT/'inverse_arm_CPU_tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
