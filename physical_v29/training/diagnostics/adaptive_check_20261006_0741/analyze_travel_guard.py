"""CPU-only failure attribution, before deciding whether to change a guard."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
g=json.loads((OUT/'initial_gate_review.json').read_text());records=[]
for pair in g['gate_pairs'][-12:]:
 for test in ['nominal','pushed']:
  folder=Path(pair[test]);d=np.load(folder/'gait_diagnostics_50hz.npz');ids=np.flatnonzero(np.isclose(d['commands'],3.5))
  if not len(ids):continue
  failures=d['failure_names'].tolist();reason=failures.index('excessive_drift')
  for robot in ids:
   ticks=np.flatnonzero(d['done'][:,robot]);events=[]
   for tick in ticks:
    prev=max(0,tick-1);q=d['root_state'][prev,robot];start=d['root_state'][0,robot,:2]
    causes=[failures[k] for k in np.flatnonzero(d['failure_causes'][tick,robot])]
    events.append(dict(time_s=float(d['time'][tick]),causes=causes,
     previous_root_height_m=float(q[2]),previous_root_xy=q[:2].tolist(),
     previous_displacement_from_first_tick_m=float(np.linalg.norm(q[:2]-start)),
     previous_linear_velocity_m_s=q[7:10].tolist(),
     drift_only=bool(d['failure_causes'][tick,robot,reason] and d['failure_causes'][tick,robot].sum()==1)))
   records.append(dict(checkpoint=pair['checkpoint'],test=test,robot=int(robot),events=events))
summary=dict(records=records,total_done_events=sum(len(r['events']) for r in records),
 drift_only_events=sum(e['drift_only'] for r in records for e in r['events']),
 source_guard="Frozen environment.py: excessive_drift=m['drift']>60. regardless of commanded travel",
 interpretation='A drift-only done is an artificial displacement truncation, not evidence of a physical fall. Post-done pose is reset; prior tick supplies physical context. Full-population gates remain failed until fresh matching corrected rollouts.')
(OUT/'travel_guard_attribution.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2));print(json.dumps(records[-2:],indent=2))
