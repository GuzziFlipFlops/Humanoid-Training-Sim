"""Synchronize the existing dated text with the already reviewed closing focus."""
import json
from pathlib import Path
out=Path(__file__).resolve().parent;train=out.parents[1];root=train.parents[1]
old='Reviewed 920 also fails 2/2.5/3 m/s lateral direction and nominal 0.35 m/s bilateral stance.'
new='Reviewed 920 failed 2/2.5/3 m/s lateral direction; newer 923 recovers all three in both tests. Both 0.35 m/s stance and 0.65 m/s speed now fail, alongside the persistent 0.15 path and 1.5 flight failures. Preserve that actor-specific recovery and audit these fresh slow cases too.'
for p in [out/'REPORT.md',train/'ITERATION_HISTORY.md',root/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
 text=p.read_text(encoding='utf-8');assert text.count(old) in (0,1)
 p.write_text(text.replace(old,new),encoding='utf-8')
p=out/'evidence.json';e=json.loads(p.read_text());e['decision']=json.loads((out/'decision.json').read_text());p.write_text(json.dumps(e,indent=2)+'\n')
print('Closing focus is consistent in report, journals, references and evidence.')
