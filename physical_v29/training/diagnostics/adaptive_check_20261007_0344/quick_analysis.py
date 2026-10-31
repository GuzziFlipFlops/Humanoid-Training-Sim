import json,hashlib
from pathlib import Path
out=Path(__file__).resolve().parent;read=lambda p:json.loads(Path(p).read_text())
g=read(out/'initial_gate_review.json');rows=[]
for x in g['gate_pairs'][-12:]:
 rows.append(dict(checkpoint=Path(x['checkpoint']).parent.name,current=[{k:x[key][-1][k] for k in ['command','survival','signed_speed','lateral_speed','flight','failures']} for key in ['nominal_cases','pushed_cases']],prior_failures={key:[dict(command=c['command'],failures=c['failures']) for c in x[key][:-1] if not c['passed']] for key in ['nominal_cases','pushed_cases']}))
(out/'recent_gate_trends.json').write_text(json.dumps(rows,indent=2))
v=read(out/'initial_upper/upper_joint_time_audit.json');print('upper keys',list(v))
print(json.dumps(rows[-1],indent=2))
