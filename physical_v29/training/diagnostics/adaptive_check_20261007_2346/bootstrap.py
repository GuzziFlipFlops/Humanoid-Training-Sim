"""Fresh CPU-only evidence; preserve immutable prior diagnostics."""
import json,shutil,re,subprocess,sys,importlib.util
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1]
PREV=OUT.parent/'adaptive_check_20261007_1845'
for name in ['review_gates.py','verify_recovery.py','upper_joint_time_audit.py','upper_bias_audit.py','refresh_evidence.py']:
    shutil.copy2(PREV/name,OUT/name)
p=OUT/'verify_recovery.py';text=p.read_text()
start=text.index('    for resume_path in [');end=text.index(']:',start)
text=text[:start]+"    for resume_path in [OUT.parent/'adaptive_check_20261007_0844/diagnostic_resume.json',OUT.parent/'adaptive_check_20261007_1344/diagnostic_resume.json',OUT.parent/'adaptive_check_20261007_1845/attempt01_initial_cache_failure/diagnostic_resume.json',OUT.parent/'adaptive_check_20261007_1845/diagnostic_resume.json',OUT/'diagnostic_resume.json'"+text[end:]
p.write_text(text)
spec=importlib.util.spec_from_file_location('owned_recorder',TRAINING/'record_owned_observation_handoff.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT/'initial_cpu';m.main()
for script,dest in [('review_gates.py','initial_gate_review.json'),('verify_recovery.py','initial_learning_verification.json')]:
    subprocess.run([sys.executable,'-u','-X','utf8',str(OUT/script),'--snapshot',str(OUT/'initial_cpu/live_handoff.json'),'--output',str(OUT/dest)],check=True,stdout=(OUT/(script+'.log')).open('w'),stderr=subprocess.STDOUT)
p=OUT/'initial_gate_review.json';review=json.loads(p.read_text());old=json.loads((PREV/'liveness_gate_review.json').read_text())
for promotion in review['promotions']:
    if promotion['stage']=='run_300':promotion['matching_pair']=next(v['matching_pair'] for v in old['promotions'] if v['stage']=='run_300')
p.write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(dict(checkpoint=review['latest']['checkpoint'],cases={k:[dict(command=c['command'],passed=c['passed'],failures=c['failures'],signed=c['signed_speed'],lateral=c['lateral_speed'],flight=c['flight']) for c in review['latest'][k]] for k in ['nominal_cases','pushed_cases']}),indent=2))
