"""CPU metadata and checkpoint lineage verification before physical rollout."""
import json, hashlib
from pathlib import Path
import torch
out=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
run=out.parents[1]/'runs/locomotion_20261007_r40_goal_budget_full_state'
cp=Path(read(run/'state.json')['checkpoint'])
args=read(cp.parent/'run_config.json')['arguments']
cert=read(args['locomotion_prior_certificate'])
teacher=Path(cert['checkpoint'])
a=torch.load(cp,map_location='cpu',weights_only=False)['infos']
b=torch.load(teacher,map_location='cpu',weights_only=False)['infos']
record=dict(student=str(cp),teacher=str(teacher),student_keys=list(a),teacher_keys=list(b),
    matches={k:a.get(k)==b.get(k) for k in ['contract','asset_sha256']},
    present={k:k in a and k in b for k in ['contract','asset_sha256']},
    teacher_sha_verified=hashlib.sha256(teacher.read_bytes()).hexdigest()==cert['checkpoint_sha256'])
(out/'teacher_checkpoint_metadata_CPU.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
