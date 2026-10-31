import json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
review=json.loads((OUT/'matched_shoulder_review.json').read_text())
for row in review['rows']:
    if row['updates']!=128:continue
    folder=OUT/(row['branch']+'_upper');folder.mkdir(exist_ok=True)
    existing=folder/'upper_joint_time_audit.json'
    if existing.exists() and json.loads(existing.read_text())['records'][0]['checkpoint']==row['checkpoint']:
        continue
    gate=json.loads((OUT/'initial_gate_review.json').read_text());gate['latest']=row
    path=OUT/(row['branch']+'_128_gate_review.json');path.write_text(json.dumps(gate,indent=2)+'\n')
    for name in ['upper_joint_time_audit.py','upper_bias_audit.py','upper_target_bias_audit.py']:
        subprocess.run([sys.executable,'-u','-X','utf8',str(OUT/name),'--gate-review',str(path),'--output-dir',str(folder)],check=True,
            stdout=(OUT/(row['branch']+'_'+name+'.log')).open('w'),stderr=subprocess.STDOUT)
print('Saved current-state control/candidate128 upper joint/target/velocity/bias timelines.')
