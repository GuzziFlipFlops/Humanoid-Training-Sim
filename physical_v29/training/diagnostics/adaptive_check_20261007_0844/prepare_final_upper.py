import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
r=read(OUT/'closing_gate_review.json');initial=read(OUT/'initial_gate_review.json')
for promoted in r['promotions']:
    if promoted['stage']=='run_300':
        original=next(p for p in initial['promotions'] if p['stage']=='run_300')
        assert Path(promoted['checkpoint']).resolve()==Path(original['checkpoint']).resolve()
        promoted['matching_pair']=original['matching_pair']
(OUT/'closing_gate_review.json').write_text(json.dumps(r,indent=2)+'\n')
print('Attached exact preserved349 reference to latest completed production pair',Path(r['latest']['checkpoint']).parent.name)
