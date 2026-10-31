import sys,subprocess,json,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
records={}
for kind,source in [('control',OUT/'offline_source/physical_v29'),('candidate',OUT/'alias_source/physical_v29')]:
 cp=OUT/f'{kind}_064_train/candidate.pt'
 train_name='control_128_train_retry01' if kind=='control' else 'candidate_128_train'
 for mode in ['train','nominal','push']:
  name=train_name if mode=='train' else f'{kind}_128_{mode}'
  selected=cp if mode=='train' else OUT/train_name/'candidate.pt'
  cmd=[sys.executable,'-u','-X','utf8',str(OUT/'run_worker.py'),'--source',str(source),'--folder',name,
   '--checkpoint',str(selected),'--mode','train' if mode=='train' else 'eval',
   '--environments','4096' if mode=='train' else '192','--iterations','64',
   '--seed','71' if mode=='train' else '901' if mode=='nominal' else '1301']
  if mode=='push':cmd+=['--perturb']
  print(json.dumps(dict(start=name,time=time.time())),flush=True)
  result=subprocess.run(cmd);assert result.returncode==0,(name,result.returncode)
  records[name]=str(OUT/name)
  (OUT/'matched_extension_completed.json').write_text(json.dumps(records,indent=2)+'\n')
print('MATCHED 128 UPDATE EXTENSION COMPLETE',flush=True)
