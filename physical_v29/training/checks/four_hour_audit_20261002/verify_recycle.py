"""Check saved Adam continuity across the first real v5 worker recycle."""
import json
from pathlib import Path
import torch

out=Path(__file__).resolve().parent
run=out.parents[1]/'runs/locomotion_20261002_r15_motion_discovery'
source=run/'010_walk_015_train_001/resource_recovery.pt'
continued=run/'011_walk_015_train_001_part_02/model_0.pt'
a=torch.load(source,weights_only=False,map_location='cpu',mmap=True)
b=torch.load(continued,weights_only=False,map_location='cpu',mmap=True)
assert a['infos']['reward_version']==b['infos']['reward_version']=='motion_deficit_discovery_v5'
assert Path(b['infos']['source']).resolve()==source.resolve()
def steps(saved):return sorted(set(int(v['step']) for v in saved['optimizer_state_dict']['state'].values()))
before,after=steps(a),steps(b)
assert before and len(before)==len(after)==1 and before[0]>0 and after[0]==before[0]+32,(before,after)
report=dict(passed=True,source=str(source),continued_checkpoint=str(continued),
    saved_Adam_steps=before,continued_Adam_steps=after,expected_minibatch_updates_per_PPO_iteration=32,
    saved_learning_rates=[p['lr'] for p in a['optimizer_state_dict']['param_groups']],
    continued_learning_rates=[p['lr'] for p in b['optimizer_state_dict']['param_groups']],
    reward_version=b['infos']['reward_version'],
    scope='First PPO update after restart retained and advanced Adam; source actor equality is also asserted by worker before learning.')
(out/'optimizer_continuation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
