"""Verify both basic and synchronized representative replay arrays independently."""
import json,numpy as np
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_full_state'
baseline=json.loads((ROOT/'current_baseline_parity.json').read_text())
original=Path(baseline['original']);replay=ROOT/'current_nominal_replay';files={}
for name in ['telemetry.npz','gait_diagnostics_50hz.npz']:
 with np.load(original/name) as a,np.load(replay/name) as b:
  assert set(a.files)==set(b.files)
  files[name]={k:bool(np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind in 'fc' else np.array_equal(a[k],b[k])) for k in a.files}
passed=all(v for fields in files.values() for v in fields.values())
d=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=passed,array_count=sum(map(len,files.values())),files=files,
 original=str(original),replay=str(replay),checkpoint_sha256=baseline['checkpoint_sha256'],
 scope='The driver had already verified 13 basic telemetry arrays and all evaluation cases. This separate CPU check adds the actual synchronized high-rate representative fields. No source, training or behavior change.')
(OUT/'full_baseline_parity.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(dict(passed=passed,array_count=d['array_count'],counts={k:len(v) for k,v in files.items()},differences={k:[n for n,p in v.items() if not p] for k,v in files.items()})))
assert passed
