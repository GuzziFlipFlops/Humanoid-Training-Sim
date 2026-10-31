import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
r=read(OUT/'initial_gate_review.json');previous=read(OUT.parent/'adaptive_check_20261007_0844/initial_gate_review.json')
for p in r['promotions']:
 if p['stage']=='run_300':p['matching_pair']=next(x for x in previous['promotions'] if x['stage']=='run_300')['matching_pair']
(OUT/'initial_gate_review.json').write_text(json.dumps(r,indent=2)+'\n')
pair=r['latest'];trend=[]
for p in r['gate_pairs'][-12:]:
 trend.append({'actor':Path(p['checkpoint']).parent.name,'current':[{k:c[k] for k in ['passed','failures','signed_speed','lateral_speed','flight']} for c in [p['nominal_cases'][-1],p['pushed_cases'][-1]]],'prior_pass':all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in p[k][:-1])})
summary={'latest':Path(pair['checkpoint']).parent.name,'current':[{k:c[k] for k in ['passed','failures','signed_speed','lateral_speed','flight']} for c in [pair['nominal_cases'][-1],pair['pushed_cases'][-1]]],'prior_failures':{k:[{'command':c['command'],'failures':c['failures']} for c in pair[k][:-1] if not c['passed']] for k in ['nominal_cases','pushed_cases']},'head_current':[r['latest_head'][k][-1] for k in ['nominal','pushed']],'last12pairs':trend}
(OUT/'initial_gate_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
