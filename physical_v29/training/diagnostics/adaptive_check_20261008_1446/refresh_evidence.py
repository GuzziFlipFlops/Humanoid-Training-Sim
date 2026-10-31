import argparse,json,subprocess,sys,importlib.util
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1]
p=argparse.ArgumentParser();p.add_argument('--tag',default='closing');a=p.parse_args()
folder=OUT/(a.tag+'_cpu')
spec=importlib.util.spec_from_file_location('owned_recorder',TRAINING/'record_owned_observation_handoff.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=folder;m.main()
for script,destination in [('review_gates.py',a.tag+'_gate_review.json'),('verify_recovery.py',a.tag+'_learning_verification.json')]:
    subprocess.run([sys.executable,'-u','-X','utf8',str(OUT/script),'--snapshot',str(folder/'live_handoff.json'),
      '--output',str(OUT/destination)],check=True,stdout=(OUT/(a.tag+'_'+script+'.log')).open('w'),stderr=subprocess.STDOUT)
f=OUT/(a.tag+'_gate_review.json');review=json.loads(f.read_text());old=json.loads((OUT.parent/'adaptive_check_20261007_2346/post_resume_gate_review.json').read_text())
for promotion in review['promotions']:
    if promotion['stage']=='run_300':promotion['matching_pair']=next(v['matching_pair'] for v in old['promotions'] if v['stage']=='run_300')
f.write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(dict(tag=a.tag,checkpoint=review['latest']['checkpoint'],
    cases={k:[dict(command=c['command'],passed=c['passed'],failures=c['failures'],signed=c['signed_speed'],lateral=c['lateral_speed'],flight=c['flight']) for c in review['latest'][k]] for k in ['nominal_cases','pushed_cases']}),indent=2))
