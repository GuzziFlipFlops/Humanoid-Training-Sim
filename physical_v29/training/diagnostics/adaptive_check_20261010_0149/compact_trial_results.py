"""Small readout of new saved tests; missing tests remain explicit."""
import json, subprocess, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(OUT/'review_state_trial.py')], check=True, stdout=subprocess.DEVNULL)
d = json.loads((OUT/'matched_full_state_review.json').read_text())
rows = []
for r in d['records']:
    row = {k:r[k] for k in ['seed', 'branch', 'updates']}
    row['tests'] = {}
    for label, t in r['tests'].items():
        cases = t['cases']
        fast = next(c for c in cases if c['command'] == 3.5)
        flight = next(c for c in cases if c['command'] == 1.5)
        row['tests'][label] = dict(passed=t['passed'],
            failed_commands={str(c['command']):c['failures'] for c in cases if not c['passed']},
            signed_3_5=round(fast['signed_speed'],4), lateral_3_5=round(fast['lateral_speed'],4),
            flight_1_5_percent=round(100*flight['true_flight'],4))
    rows.append(row)
print(json.dumps(dict(complete=d['all_expected_components_complete'],records=rows)))
