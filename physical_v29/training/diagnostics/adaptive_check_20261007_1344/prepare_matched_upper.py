"""Build explicit exact-actor upper-timeline input for each completed branch."""
import json,hashlib,argparse
from pathlib import Path
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=128);a=p.parse_args()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
base=read(OUT/'initial_gate_review.json')
for branch in ['control','candidate']:
 root=OUT/'matched_heading';cp=root/f'{branch}_{a.updates:03d}_train/candidate.pt'
 nominal=root/f'{branch}_{a.updates:03d}_nominal';pushed=root/f'{branch}_{a.updates:03d}_push'
 if not all((x/'result.json').exists() and read(x/'result.json')['completed'] for x in [cp.parent,nominal,pushed]):continue
 pair=dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(nominal),pushed=str(pushed))
 output=OUT/f'{branch}_upper_{a.updates:03d}';output.mkdir(exist_ok=True)
 (output/'gate_input.json').write_text(json.dumps(dict(run=base['run'],latest=pair,promotions=[]),indent=2))
 print(str(output))
