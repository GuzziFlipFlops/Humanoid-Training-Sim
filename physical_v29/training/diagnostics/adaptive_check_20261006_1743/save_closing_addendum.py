import json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
d=json.loads((OUT/'closing_gate_review.json').read_text());pair=d['latest']
assert Path(pair['checkpoint']).parent.name=='267_run_350_train_086'
assert all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in pair[k] if c['command']<3.5)
entry=f'''\n\n<!-- adaptive_check_20261006_1743_closing267 -->\nClosing evidence {datetime.now(timezone.utc).isoformat()}: exact post-resume267 now passes all stopped and prior cases through3m/s in268 nominal/269 pushed, including recovered1.5 geometric flight.3.5 signed3.489/3.496 and lateral0.516/0.488 still fail direction. Preserve this improving original full learner. Fresh267 actual angle-target-velocity plots were saved and visually inspected; older264 review/plots/report remain in the dated folder. No diagnostic actor/source/PD/cap/reset deployed. Updated same native5h focus and current report: {OUT/'REPORT.md'}. Exact pair SHA{pair['sha256']}. Newer partial actors are separately unqualified.\n'''
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
 if 'adaptive_check_20261006_1743_closing267' not in p.read_text(encoding='utf-8'):
  with p.open('a',encoding='utf-8') as f:f.write(entry)
head=f'''<!-- closing267_20261006_1743 -->\nLatest exact267/268nominal/269pushed: all stopped/prior cases through3m/s pass,1.5 flight recovered;3.5 direction still fails. Same original R39 full learner running,55hashes,existing upper refinement. Fresh267 upper timelines inspected; previous264 evidence preserved. Report {OUT/'REPORT.md'}, then fresh ACTIVE_RUN/state. Next preserve this useful actor; never infer newer qualification. Same active5h/verified5 or qualified4+native limit endpoint,computeron.\n<!-- /closing267_20261006_1743 -->\n\n'''
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
 if '<!-- closing267_20261006_1743 -->' not in p.read_text(encoding='utf-8'):p.write_text(head+p.read_text(encoding='utf-8'),encoding='utf-8')
print('Closing267 evidence appended without replacing earlier entries.')
