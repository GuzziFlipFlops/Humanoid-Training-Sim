"""Record hashes only after the calling agent visually inspects both plots."""
import json,hashlib,argparse
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--note',required=True);a=p.parse_args()
qa=json.loads((OUT/'plot_qa.json').read_text(encoding='utf-8-sig'))
for name in ['upper_joint_angles.png','upper_joint_velocities.png']:
 f=OUT/a.folder/name;assert f.exists()
 qa['files'].append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),visually_inspected=True,inspection_note=a.note))
qa['recorded_utc']=datetime.now(timezone.utc).isoformat()
(OUT/'plot_qa.json').write_text(json.dumps(qa,indent=2))
print(json.dumps(dict(folder=a.folder,inspected_files=2)))
