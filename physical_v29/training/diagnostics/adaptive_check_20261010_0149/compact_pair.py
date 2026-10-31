import argparse, json
from pathlib import Path
OUT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument('file'); a = p.parse_args()
r = json.loads((OUT / a.file).read_text())['latest']
out = dict(actor=Path(r['checkpoint']).parent.name, sha256=r['sha256'])
for label in ['nominal', 'pushed']:
    out[label] = [{k: c[k] for k in ['command', 'passed', 'failures', 'signed_speed', 'lateral_speed', 'flight']}
        for c in r[label + '_cases']]
print(json.dumps(out, indent=2))
