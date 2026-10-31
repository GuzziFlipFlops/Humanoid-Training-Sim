"""Refresh exact comparison review and emit compact independent results."""
import json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(OUT/'review_state_trial.py')],check=True,stdout=subprocess.DEVNULL)
d=json.loads((OUT/'matched_full_state_review.json').read_text());rows=[]
for r in d['records']:
 for name,t in r['tests'].items():
  row=dict(seed=r['seed'],branch=r['branch'],updates=r['updates'],test=name,passed=t['passed'])
  row['failures']={str(c['command']):c['failures'] for c in t['cases'] if not c['passed']}
  for s in [.15,1.5,3.5]:
   c=next(c for c in t['cases'] if c['command']==s)
   row[str(s)]=dict(signed=round(c['signed_speed'],4),lateral=round(c['lateral_speed'],4),flight_percent=round(c['true_flight']*100,4))
  rows.append(row)
print(json.dumps(dict(complete=d['all_expected_components_complete'],results=rows)))
