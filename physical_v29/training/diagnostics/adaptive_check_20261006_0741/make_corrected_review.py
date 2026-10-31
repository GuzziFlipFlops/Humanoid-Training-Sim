import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
review=read(OUT/'initial_gate_review.json')
def pair(nom,push):
 cp=Path(read(nom/'run_config.json')['arguments']['checkpoint'])
 assert str(cp)==read(push/'run_config.json')['arguments']['checkpoint']
 row=dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(nom),pushed=str(push))
 for label,folder in [('nominal',nom),('pushed',push)]:
  r=read(folder/'result.json');row[label+'_all_pass']=r['passed'];row[label+'_cases']=[]
  for c in r['cases']:
   z=c['statistics'];row[label+'_cases'].append(dict(command=z['command_m_s'],passed=c['passed'],failures=c['failures'],statistics=z))
 return row
review['latest']=pair(OUT/'101_current_run_350_gate_nominal',OUT/'102_current_run_350_gate_push')
review['promotions'][-1]['matching_pair']=pair(OUT/'103_teacher_run_300_gate_nominal',OUT/'103_teacher_run_300_gate_push')
review['scope']='Corrected same-actor tests; current3.5m unqualified, fresh349 teacher all prior cases passes.'
(OUT/'corrected_gate_review.json').write_text(json.dumps(review,indent=2)+'\n')
