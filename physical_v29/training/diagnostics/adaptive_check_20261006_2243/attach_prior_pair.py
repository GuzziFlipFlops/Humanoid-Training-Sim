import json
from pathlib import Path
out=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
p=out/'initial_gate_review.json';v=read(p)
old=read(out.parent/'adaptive_check_20261006_1243/closing_gate_review.json')
trusted=next(x for x in old['promotions'] if x['stage']=='run_300')['matching_pair']
assert trusted['nominal_all_pass'] and trusted['pushed_all_pass']
for x in v['promotions']:
 if x['stage']=='run_300':
  assert Path(x['checkpoint']).resolve()==Path(trusted['checkpoint']).resolve()
  x['matching_pair']=trusted
p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
print('Attached preserved exact349 matching nominal/pushed qualification; no gate changed.')
