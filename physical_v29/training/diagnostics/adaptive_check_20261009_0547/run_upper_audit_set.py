"""CPU-only actual-motion audit, with optional all-command nominal/pushed plots."""
import argparse,json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--gate-review',required=True);p.add_argument('--output',required=True);p.add_argument('--all-traces',action='store_true');a=p.parse_args()
review=(OUT/a.gate_review).resolve();folder=(OUT/a.output).resolve()
assert folder.is_relative_to(OUT) and review.is_file();folder.mkdir(exist_ok=True,parents=True)
common=['--gate-review',str(review),'--output-dir',str(folder)]
for name in ['upper_joint_time_audit.py','upper_bias_audit.py','upper_target_bias_audit.py']:
    with (folder/(name+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/name)]+common,check=True,stdout=log,stderr=subprocess.STDOUT)
supplements=[]
if a.all_traces:
    for label,speeds,test in [('slow',[0,.15,.35,.65],'nominal'),('middle',[1,1.5,2,2.5],'nominal'),
        ('pushed_fast',[1,2.5,3,3.5],'pushed'),('pushed_slow',[0,.15,.35,.65],'pushed'),('pushed_middle',[1,1.5,2,2.5],'pushed')]:
        sub=folder/label
        with (folder/(label+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'upper_joint_time_audit.py'),'--gate-review',str(review),
                '--output-dir',str(sub),'--trace-speeds']+[str(s) for s in speeds]+['--trace-test',test],check=True,stdout=log,stderr=subprocess.STDOUT)
        supplements.append(dict(folder=str(sub),speeds=speeds,test=test,needs_visual_QA=True))
(folder/'audit_set.json').write_text(json.dumps(dict(completed=True,gate_review=str(review),supplements=supplements,
    all_commands_explicit_nominal_and_pushed_traces=a.all_traces,plots_need_visual_QA=True),indent=2)+'\n')
print('Saved CPU joint/target/velocity and bias audits:',folder)
