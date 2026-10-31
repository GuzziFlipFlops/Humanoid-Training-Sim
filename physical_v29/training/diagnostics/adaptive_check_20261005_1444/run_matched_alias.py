"""Matched full-state control/candidate at 64 and 128 updates; diagnostic only."""
import json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'initial_evidence.json').read_text());initial=Path(r['checkpoint'])
records={}
for boundary in [64,128]:
 for kind,source in [('control',OUT/'offline_source/physical_v29'),('candidate',OUT/'alias_source/physical_v29')]:
  cp=initial if boundary==64 else OUT/f'{kind}_064_train/candidate.pt'
  for mode in ['train','nominal','push']:
   name=f'{kind}_{boundary:03d}_{mode}'
   selected=cp if mode=='train' else OUT/f'{kind}_{boundary:03d}_train/candidate.pt'
   cmd=[sys.executable,'-u','-X','utf8',str(OUT/'run_worker.py'),'--source',str(source),'--folder',name,
    '--checkpoint',str(selected),'--mode','train' if mode=='train' else 'eval',
    '--environments','4096' if mode=='train' else '192','--iterations','64',
    '--seed','71' if mode=='train' else '901' if mode=='nominal' else '1301']
   if mode=='push':cmd+=['--perturb']
   print(json.dumps(dict(start=name,time=time.time())),flush=True)
   result=subprocess.run(cmd);assert result.returncode==0,(name,result.returncode)
   records[name]=str(OUT/name)
   (OUT/'matched_completed_operations.json').write_text(json.dumps(records,indent=2)+'\n')
print('MATCHED EXPERIMENT COMPLETE',flush=True)
