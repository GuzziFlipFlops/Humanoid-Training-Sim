"""Compact exact-actor behavioral comparison, preserving nonmonotonic passes."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent;read=lambda p:json.loads(Path(p).read_text())
latest=read(OUT/'closing_gate_review.json');old=read(OUT.parent/'adaptive_check_20261008_1946/final_owned_gate_review.json')
pairs={Path(p['checkpoint']).parent.name:p for p in latest['gate_pairs']}
pairs.update({Path(p['checkpoint']).parent.name:p for p in old['gate_pairs']})
wanted=['605','794','797','827','833','836','839']
selected=[]
for prefix in wanted:
 for name,p in pairs.items():
  if name.startswith(prefix+'_'):
   row=dict(actor=name,checkpoint=p['checkpoint'],sha256=p['sha256'])
   for test in ['nominal','pushed']:
    row[test]=[{k:c[k] for k in ['command','passed','failures','signed_speed','lateral_speed','flight']} for c in p[test+'_cases']]
   selected.append(row)
(OUT/'gate_trends.json').write_text(json.dumps(dict(completed=True,actors=selected,
 flight_field='Existing geometric_flight_fraction; separate controlled probe true flight requires both sole clearance>1mm and force<1N.',
 conclusion='Retention and direction are nonmonotonic.833 recovers3m in both tests;839 nominal3m narrowly regresses.3.5 current-case passes never certify all-prior stage or newer partial actors.'),indent=2)+'\n')
print('Saved compact trends for',len(selected),'exact actors.')
