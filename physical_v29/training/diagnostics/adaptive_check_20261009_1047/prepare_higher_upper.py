"""CPU-only upper timelines for the explicitly tested higher commands."""
from pathlib import Path
import json, subprocess, sys
ROOT=Path(__file__).resolve().parent
HIGH=ROOT/'high_speed_extrapolation'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
pause=read(HIGH/'diagnostic_pause.json')
for label in ['nominal','pushed','diverse_nominal','diverse_pushed']:
    assert read(HIGH/label/'result.json')['completed']
    assert read(HIGH/label/'run_config.json')['arguments']['checkpoint']==pause['checkpoint']
review=dict(run=pause['run'],promotions=[],latest=dict(checkpoint=pause['checkpoint'],sha256=pause['checkpoint_sha256'],
    nominal=str(HIGH/'nominal'),pushed=str(HIGH/'pushed')))
(ROOT/'higher_upper_gate_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
for label in ['nominal','pushed']:
    target=HIGH/('upper_'+label)
    assert not (target/'upper_joint_time_audit.json').exists()
    with (HIGH/('upper_'+label+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'upper_joint_time_audit.py'),
            '--gate-review',str(ROOT/'higher_upper_gate_review.json'),'--output-dir',str(target),
            '--trace-test',label,'--trace-speeds','3.5','4','4.5','5'],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Saved higher-speed head/shoulder/elbow angle, target and velocity timelines; visual inspection pending.')
