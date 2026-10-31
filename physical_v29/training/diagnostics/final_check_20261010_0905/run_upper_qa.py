import json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;OLD=OUT.parents[1]/'diagnostics/adaptive_check_20261010_0149'
for label in ['latest','selected']:
 review=OUT/(label+'_gate_review.json')
 for test in ['nominal','pushed']:
  for band,speeds in [('slow',[0,.15,.35,.65]),('middle',[1,1.5,2]),('fast',[2.5,3,3.5])]:
   folder=OUT/(label+'_upper')/test/band
   cmd=[sys.executable,str(OLD/'upper_joint_time_audit.py'),'--gate-review',str(review),'--output-dir',str(folder),'--trace-test',test,'--trace-speeds']+list(map(str,speeds))
   folder.mkdir(parents=True,exist_ok=True)
   with (folder/'audit.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  folder=OUT/(label+'_upper')/test/'bias'
  folder.mkdir(parents=True,exist_ok=True)
  with (folder/'audit.log').open('w') as log:subprocess.run([sys.executable,str(OLD/'upper_bias_audit.py'),'--gate-review',str(review),'--output-dir',str(folder)],stdout=log,stderr=subprocess.STDOUT,check=True)
print('CPU upper actual angle/target/velocity/phase and bias audits complete.')
