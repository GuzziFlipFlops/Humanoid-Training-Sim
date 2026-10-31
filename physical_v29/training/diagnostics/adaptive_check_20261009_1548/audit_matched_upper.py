"""CPU actual-motion comparison for completed exact128-update branches."""
import json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRIAL=OUT/'coupled_training_trial';ROOT=TRIAL/'matched_coupled_teacher'
read=lambda p:json.loads(Path(p).read_text())
review=read(OUT/'matched_coupled_teacher_review.json');done=[]
for row in review['records']:
    if row['updates']!=128 or not {'nominal','push'}.issubset(row['tests']):continue
    label=f"seed{row['seed']}_{row['branch']}_upper";folder=OUT/label
    if (folder/'upper_joint_time_audit.json').exists():continue
    pair=dict(checkpoint=row['checkpoint'],sha256=row['checkpoint_sha256'],
        nominal=row['tests']['nominal']['folder'],pushed=row['tests']['push']['folder'])
    gate=OUT/(label+'_gate_review.json');gate.write_text(json.dumps(dict(latest=pair,promotions=[],run=read(TRIAL/'diagnostic_pause.json')['run']),indent=2)+'\n')
    folder.mkdir(exist_ok=True)
    common=['--gate-review',str(gate),'--output-dir',str(folder)]
    for script in ['upper_joint_time_audit.py','upper_bias_audit.py','upper_target_bias_audit.py']:
        with (folder/(script+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/script)]+common+
                (['--trace-speeds','.15','1.5','2.5','3.5'] if script=='upper_joint_time_audit.py' else []),check=True,stdout=log,stderr=subprocess.STDOUT)
    done.append(label)
print(json.dumps(dict(audited=done,plots_need_explicit_visual_QA=True)))
