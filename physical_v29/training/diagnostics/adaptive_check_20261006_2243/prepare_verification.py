import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
out=Path(__file__).resolve().parent
files=[]
for f in ['initial_upper/upper_joint_angles.png','initial_upper/upper_joint_velocities.png','final_upper/upper_joint_angles.png','final_upper/upper_joint_velocities.png','arm_response.png','actual_head_world_axis.png']:
 p=out/f;files.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),visually_inspected=True))
(out/'plot_qa.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),files=files,notes='Axes/labels/readability/actual-versus-target traces inspected. Persistent fast shoulder bias/asymmetry and gait-locked head/velocity oscillation documented; no human-motion certification.'),indent=2)+'\n')
t=(out.parent/'adaptive_check_20261006_1743/verify_finish.py').read_text().replace('adaptive_check_20261006_1743','adaptive_check_20261006_2243')
(out/'verify_finish.py').write_text(t)
