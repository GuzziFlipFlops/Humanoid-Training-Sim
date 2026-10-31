"""Record visual inspection only after viewing every listed plot."""
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--notes',required=True);a=p.parse_args()
folder=Path(a.folder)
names=['upper_joint_angles.png','upper_joint_velocities.png','upper_bias_decomposition.png','upper_target_bias.png']
assert all((folder/n).is_file() for n in names)
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),visually_inspected=True,
 plots=[dict(path=str(folder/n),sha256=hashlib.sha256((folder/n).read_bytes()).hexdigest(),visually_inspected=True) for n in names],
 notes=a.notes,scope='Focal nominal representative plots; all-command nominal/pushed population statistics saved separately. No human/Unitree motion or optical-gaze certification.')
(folder/'visual_inspection.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(folder=str(folder),plots=len(names),visually_inspected=True)))
