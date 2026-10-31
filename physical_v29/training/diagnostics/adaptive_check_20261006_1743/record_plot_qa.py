import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
files=[f'{d}/upper_joint_{p}.png' for d in ['upper','control_upper','candidate_upper','closing_upper'] for p in ['angles','velocities']]+['arm_response.png','arm_frequency_response.png','world_head_axis.png']
notes=dict(upper='Walking has substantial pitch swing, unequal bias/elbow coordination. Fast pitch swing attenuates and roll/head oscillation remains. Actual/target overlay inspected; not natural-motion certification.',
 control_upper='128 unchanged-control pitch bias and fast attenuation remain, head/arm velocities are jagged at50Hz.',
 candidate_upper='Direction candidate preserves visible alternating legs, but fast upper bias remains and native head RMS slightly exceeds matched control; candidate gait gates fail elsewhere.',
 closing_upper='Original full-state post-resume267 shoulders at1m counter-swing but unequal bias/elbows;3.5 pitch nearly fixed with stronger roll and gait-locked head oscillation.',
 arm_response='Higher damping attenuates pitch; free-base effects included, no open-loop identification.',
 arm_frequency_response='Small stopped sines produce frequency-dependent gains/phase and base motion; no gain selected for production.',
 world_head_axis='Modeled neck+waist+root forward axis oscillates more than native head angles; no CAD optical-axis calibration.')
records=[]
for f in files:
 p=OUT/f;assert p.exists();key=p.parent.name if '/' in f else p.stem
 records.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),visually_inspected=True,notes=notes[key]))
(OUT/'plot_qa.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),files=records,scope='Tool view_image inspection by the assistant; statistics/population/reference limits remain in the underlying audits.'),indent=2)+'\n')
print(json.dumps(dict(inspected_pngs=len(records))))
