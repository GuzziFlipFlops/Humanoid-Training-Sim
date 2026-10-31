"""Review the latest owned pair and attach the preserved exact3m qualification."""
import subprocess,json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(OUT/'review_gates.py'),'--snapshot','closing_cpu/live_handoff.json','--output','closing_gate_review.json'],check=True)
p=OUT/'closing_gate_review.json';v=json.loads(p.read_text(encoding='utf-8-sig'))
initial=json.loads((OUT/'initial_gate_review.json').read_text(encoding='utf-8-sig'))
trusted=next(x for x in initial['promotions'] if x['stage']=='run_300')['matching_pair']
for x in v['promotions']:
    if x['stage']=='run_300':
        assert Path(x['checkpoint']).resolve()==Path(trusted['checkpoint']).resolve()
        x['matching_pair']=trusted
p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
