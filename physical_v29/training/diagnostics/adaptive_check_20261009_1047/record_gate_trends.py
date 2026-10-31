"""CPU-only actor-bound comparison of exact complete nominal/pushed pairs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
review=json.loads((ROOT/'closing_gate_review.json').read_text(encoding='utf-8'))
latest=Path(review['latest']['checkpoint']).parent.name
wanted={'605','794','833','908','917','923','992','998','1001',latest.split('_')[0]}
rows=[]
for pair in review['gate_pairs']:
    actor=Path(pair['checkpoint']).parent.name
    if actor.split('_')[0] not in wanted:continue
    row=dict(actor=actor,checkpoint=pair['checkpoint'],sha256=pair['sha256'])
    for test in ['nominal','pushed']:
        row[test]=[{k:c[k] for k in ['command','passed','failures','signed_speed','lateral_speed','flight']}
                   for c in pair[test+'_cases']]
    rows.append(row)
assert latest in {r['actor'] for r in rows}
result=dict(completed=True,actors=rows,latest_reviewed_actor=latest,
    conclusion='Current-speed and prior retention are nonmonotonic. Saved passing pairs remain actor-specific; '
      'every exact-checkpoint prior and current case must pass before promotion. No qualification extends to newer partial actors.',
    flight_field='Original gate geometric_flight_fraction. Physical probes additionally require both sole forces below1N.')
(ROOT/'gate_trends.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('Saved',len(rows),'exact matching actor pairs.')
