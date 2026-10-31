"""Compare the original saved actor under the old and new independent verifier."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
OLD=HERE/'runs/locomotion_20261004_r30_exact_graph_execution/079_walk_065_gate_020_nominal'
NEW=HERE/'runs/locomotion_20261004_r31_exact_command_rehearsal/007_walk_065_gate_001_nominal'
def read(p):return json.loads(p.read_text())
assert (NEW/'result.json').exists(),'Wait for the complete independent baseline'
old_cfg=read(OLD/'run_config.json');new_cfg=read(NEW/'run_config.json')
assert old_cfg['arguments']['num_envs']==new_cfg['arguments']['num_envs']==128
for key in ['checkpoint','seed','duration','speed','perturb']:
    assert old_cfg['arguments'][key]==new_cfg['arguments'][key],key
old_result=read(OLD/'result.json');new_result=read(NEW/'result.json')
files=[]
for name in ['telemetry.npz','gait_diagnostics_50hz.npz','startup_diagnostics.npz']:
    with np.load(OLD/name,allow_pickle=False) as a,np.load(NEW/name,allow_pickle=False) as b:
        assert a.files==b.files,(name,a.files,b.files)
        arrays=[]
        for key in a.files:
            x,y=a[key],b[key]
            assert x.shape==y.shape and x.dtype==y.dtype,(name,key)
            equal=np.array_equal(x,y,equal_nan=True) if x.dtype.kind in 'fc' else np.array_equal(x,y)
            delta=float(np.nanmax(np.abs(x-y))) if x.size and x.dtype.kind in 'fci' else None
            arrays.append(dict(key=key,shape=list(x.shape),bitwise_equal=bool(equal),maximum_abs_difference=delta))
        files.append(dict(name=name,all_arrays_equal=all(x['bitwise_equal'] for x in arrays),arrays=arrays))
value=dict(passed=all(f['all_arrays_equal'] for f in files) and old_result['cases']==new_result['cases'],
           old=str(OLD),new=str(NEW),files=files,independent_cases_identical=old_result['cases']==new_result['cases'],
           scope='Same saved checkpoint/seed128-robot20-second nominal baseline; unchanged evaluation/physics at current commands, not proof training will improve')
(OUT/'evaluation_parity.json').write_text(json.dumps(value,indent=2))
print(json.dumps(dict(passed=value['passed'],files=[dict(name=f['name'],all_arrays_equal=f['all_arrays_equal']) for f in files],independent_cases_identical=value['independent_cases_identical'])))
