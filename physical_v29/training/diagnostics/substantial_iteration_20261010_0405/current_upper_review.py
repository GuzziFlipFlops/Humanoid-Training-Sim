"""CPU actual upper timelines for the freshest complete original pair."""
import json,hashlib,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];OLD=TRAIN/'diagnostics/adaptive_check_20261010_0149'
RUN=TRAIN/'runs/locomotion_20261007_r40_goal_budget_full_state'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
pairs={}
for p in RUN.glob('*_gate_*/result.json'):
 if not read(p).get('completed'):continue
 config=read(p.parent/'run_config.json')['arguments'];cp=config['checkpoint']
 pairs.setdefault(cp,{})['pushed' if config['perturb'] else 'nominal']=str(p.parent)
complete=[(cp,folders) for cp,folders in pairs.items() if set(folders)=={'nominal','pushed'}]
cp,folders=max(complete,key=lambda x:int(Path(x[0]).parent.name.split('_')[0]))
pair=dict(checkpoint=cp,sha256=hashlib.sha256(Path(cp).read_bytes()).hexdigest(),**folders)
review=OUT/'current_complete_gate_review.json';review.write_text(json.dumps(dict(run=str(RUN),latest=pair,promotions=[]),indent=2)+'\n')
folder=OUT/'current_original_upper'
for tool in ('upper_joint_time_audit.py','upper_bias_audit.py'):
 with (OUT/(tool+'.log')).open('w') as log:
  subprocess.run([sys.executable,str(OLD/tool),'--gate-review',str(review),'--output-dir',str(folder)],check=True,stdout=log,stderr=subprocess.STDOUT)
with (OUT/'upper_pushed.log').open('w') as log:
 subprocess.run([sys.executable,str(OLD/'upper_joint_time_audit.py'),'--gate-review',str(review),'--output-dir',str(folder/'pushed'),'--trace-test','pushed'],check=True,stdout=log,stderr=subprocess.STDOUT)
print(json.dumps(pair,indent=2))
