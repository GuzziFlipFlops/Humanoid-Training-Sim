import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_context'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
base=read(OUT/'initial_gate_review.json');cert=read(Path(read(OUT/'gradient_smoke/run_config.json')['arguments']['locomotion_prior_certificate']))
reference=dict(checkpoint=cert['checkpoint'],sha256=cert['checkpoint_sha256'],nominal=str(ROOT/'truthful_teacher_nominal'),pushed=str(ROOT/'truthful_teacher_push'))
for branch in ['control','candidate']:
 cp=ROOT/f'{branch}_128_train/candidate.pt';pair=dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(ROOT/f'{branch}_128_nominal'),pushed=str(ROOT/f'{branch}_128_push'))
 assert all(read(Path(pair[k])/'result.json')['completed'] for k in ['nominal','pushed'])
 for label in ['nominal','pushed']:
  pair[label+'_cases']=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],statistics=c['statistics']) for c in read(Path(pair[label])/'evaluation.json')['cases']]
 review=dict(run=base['run'],latest=pair,promotions=[dict(stage='run_300',speed=3.,checkpoint=cert['checkpoint'],matching_pair=reference)])
 (OUT/f'{branch}_gate_review.json').write_text(json.dumps(review,indent=2))
print('Saved exact128-update actor pairs and fresh truthful349 reference')
