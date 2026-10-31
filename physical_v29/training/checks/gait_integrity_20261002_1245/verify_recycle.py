"""CPU verification of saved v6 optimizer continuity across RAM recycling."""
import json
from pathlib import Path
import torch

OUT=Path(__file__).resolve().parent
RUN=OUT.parents[1]/'runs/locomotion_20261002_r17_gait_integrity'
source=RUN/'010_walk_015_train_001/resource_recovery.pt'
continued=RUN/'011_walk_015_train_001_part_02/model_0.pt'
a=torch.load(source,weights_only=False,map_location='cpu',mmap=True)
b=torch.load(continued,weights_only=False,map_location='cpu',mmap=True)
assert a['infos']['reward_version']==b['infos']['reward_version']=='gait_integrity_v6'
assert Path(b['infos']['source']).resolve()==source.resolve()
def steps(saved):
    return sorted(set(int(v['step']) for v in saved['optimizer_state_dict']['state'].values()))
before,after=steps(a),steps(b)
assert before and len(before)==len(after)==1 and before[0]>0 and after[0]==before[0]+32,(before,after)
lr_a=[p['lr'] for p in a['optimizer_state_dict']['param_groups']]
lr_b=[p['lr'] for p in b['optimizer_state_dict']['param_groups']]
assert lr_a==lr_b
report=dict(passed=True,source=str(source),continued_checkpoint=str(continued),
    saved_Adam_steps=before,continued_Adam_steps=after,
    minibatch_updates_per_PPO_iteration=32,saved_learning_rates=lr_a,continued_learning_rates=lr_b,
    scope='Within-v6 restart retains and advances Adam. Critic/Adam were reset only at the v5-to-v6 reward migration.')
(OUT/'optimizer_continuation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
