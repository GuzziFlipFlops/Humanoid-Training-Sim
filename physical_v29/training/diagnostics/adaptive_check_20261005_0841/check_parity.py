import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;pause=json.loads((OUT/'diagnostic_pause.json').read_text())
baseline=Path(pause['completed_eval']);candidate=OUT/'candidate_same_actor_replay'
records=[]
for name in ['telemetry.npz','gait_diagnostics_50hz.npz']:
    a,b=np.load(baseline/name),np.load(candidate/name)
    matches={k:(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes()) for k in a.files}
    records.append(dict(file=name,keys_equal=set(a.files)==set(b.files),arrays=matches))
cases=json.loads((baseline/'evaluation.json').read_text())['cases']==json.loads((candidate/'evaluation.json').read_text())['cases']
result=dict(passed=cases and all(r['keys_equal'] and all(r['arrays'].values()) for r in records),cases_equal=cases,
  arrays=records,baseline=str(baseline),candidate=str(candidate),scope='Same exact actor,192robots20s,source change only training auxiliary. No qualification inferred.')
(OUT/'physical_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
