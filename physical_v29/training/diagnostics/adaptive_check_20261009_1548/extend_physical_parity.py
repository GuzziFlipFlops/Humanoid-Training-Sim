"""Verify every saved synchronized array, including action/target histories."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
paths=[OUT/'unchanged_physical_replay.json',OUT/'coupled_training_trial/matched_coupled_teacher/unchanged_actor_parity.json']
receipts=[]
for proof_file in paths:
    proof=json.loads(proof_file.read_text());original=Path(proof['original']);replay=Path(proof['replay'])
    arrays={}
    for filename in ['telemetry.npz','gait_diagnostics_50hz.npz','startup_diagnostics.npz']:
        with np.load(original/filename) as a,np.load(replay/filename) as b:
            assert set(a.files)==set(b.files)
            checks={key:np.array_equal(a[key],b[key],equal_nan=True) if a[key].dtype.kind in 'fc' else np.array_equal(a[key],b[key]) for key in a.files}
            assert all(checks.values());arrays[filename]=checks
    assert json.loads((original/'evaluation.json').read_text())==json.loads((replay/'evaluation.json').read_text())
    receipts.append(dict(passed=True,source_proof=str(proof_file),checkpoint=proof.get('checkpoint',proof.get('exact_checkpoint')),
        original=str(original),replay=str(replay),arrays=arrays,total_arrays=sum(len(x) for x in arrays.values()),all_case_reports_bitwise_equal=True))
(OUT/'full_physical_parity.json').write_text(json.dumps(dict(passed=True,receipts=receipts),indent=2)+'\n')
print(json.dumps([dict(checkpoint=r['checkpoint'],arrays=r['total_arrays'],passed=True) for r in receipts]))
