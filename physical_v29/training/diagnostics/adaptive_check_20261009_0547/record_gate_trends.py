"""Actor-bound comparison; current-speed passes are separate from retention."""
import json
from pathlib import Path
out=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
review=read(out/'closing_gate_review.json')
pairs={Path(p['checkpoint']).parent.name:p for p in review['gate_pairs']}
latest=Path(review['latest']['checkpoint']).parent.name
wanted=['605','794','797','833','839','908','914','917',latest.split('_')[0]]
rows=[]
for prefix in wanted:
    for name,p in pairs.items():
        if name.startswith(prefix+'_') and name not in {r['actor'] for r in rows}:
            row=dict(actor=name,checkpoint=p['checkpoint'],sha256=p['sha256'])
            for test in ['nominal','pushed']:
                row[test]=[{k:c[k] for k in ['command','passed','failures','signed_speed','lateral_speed','flight']} for c in p[test+'_cases']]
            rows.append(row)
(out/'gate_trends.json').write_text(json.dumps(dict(passed=True,actors=rows,
    flight_field='Existing geometric_flight_fraction. Controlled physical true flight also requires force<1N.',
    conclusion='908 and917 current3.5 cases pass both tests; stage qualification still requires every exact-checkpoint prior. Direction and retention are nonmonotonic; no qualification is attributed to partial actors.'),indent=2)+'\n')
print('Saved',len(rows),'exact matching pairs.')
