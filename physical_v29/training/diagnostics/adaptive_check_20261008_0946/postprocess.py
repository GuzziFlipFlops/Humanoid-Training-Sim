import subprocess,sys,json,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
deadline=time.monotonic()+400
while not (OUT/'diagnostic_resume.json').exists():
    assert time.monotonic()<deadline,'Diagnostic resume receipt unavailable'
    time.sleep(2)
assert json.loads((OUT/'diagnostic_completion.json').read_text())['completed']
for script in ['analyze_current_gradients.py','analyze_individual_cycles.py']:
    with (OUT/(script+'.log')).open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-u','-X','utf8',str(OUT/script)],check=True,stdout=f,stderr=subprocess.STDOUT)
subprocess.run([sys.executable,'-u','-X','utf8',str(OUT/'refresh_evidence.py'),'--tag','closing'],
    check=True,stdout=(OUT/'closing_summary.log').open('w'),stderr=subprocess.STDOUT)
print('Both CPU analyses and fresh owned/gate/full-state review complete.')
