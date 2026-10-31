"""Compare exact complete gate pairs; preserve checkpoint-specific qualification."""
import json
from pathlib import Path
from datetime import datetime, timezone
OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
files = [('previous closing', OUT.parent / 'adaptive_check_20261009_1548/closing_newer_pair.json'),
         ('initial', OUT / 'initial_gate_review.json'),
         ('closing after genuine resume', OUT / 'closing_gate_review.json')]
records = []
for label, file in files:
    data = read(file)
    pair = data.get('latest', data.get('newer_matching_pair', data))
    if 'nominal_cases' not in pair:
        # The older compact record retains full per-test summaries rather than the full review schema.
        continue
    records.append(dict(label=label, source=str(file), checkpoint=pair['checkpoint'], sha256=pair['sha256'],
        tests={t: dict(all_commands_pass=all(c['passed'] for c in pair[t + '_cases']), cases=pair[t + '_cases'])
               for t in ['nominal', 'pushed']}))
# Read the previous full review when its newer compact artifact has a different schema.
if not any(r['label'] == 'previous closing' for r in records):
    data = read(OUT.parent / 'adaptive_check_20261009_1548/closeout_gate_review.json')['latest']
    records.insert(0, dict(label='previous fully reviewed',
        source=str(OUT.parent / 'adaptive_check_20261009_1548/closeout_gate_review.json'),
        checkpoint=data['checkpoint'], sha256=data['sha256'],
        tests={t: dict(all_commands_pass=data[t + '_all_pass'], cases=data[t + '_cases'])
               for t in ['nominal', 'pushed']}))
result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), records=records,
    scope='Exact saved complete nominal/pushed pairs only. The earlier compact and full review may bind different actors; each is labelled. Neither a matching pair nor posture or motor attribution qualifies newer partial learning.',
    endpoint_established=False)
(OUT / 'saved_gate_trends.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps([dict(label=r['label'], checkpoint=Path(r['checkpoint']).parent.name,
    fast=[{k: c[k] for k in ['signed_speed', 'lateral_speed', 'passed']}
          for t in ['nominal', 'pushed'] for c in r['tests'][t]['cases'] if c['command'] == 3.5])
    for r in records]))
