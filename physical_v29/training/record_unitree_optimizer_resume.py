"""Verify an actual RAM-recycle continuation from saved CPU checkpoint tensors."""
from pathlib import Path
import json
import torch

HERE = Path(__file__).parent
active = json.loads((HERE / 'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
run = Path(active['run'])
state = json.loads((run / 'state.json').read_text())
recycle = state['resource_recycles'][0]
source = Path(recycle['checkpoint'])
continuation = run / '009_stand_train_001_part_02' / 'model_0.pt'
old = torch.load(source, map_location='cpu', weights_only=False)
new = torch.load(continuation, map_location='cpu', weights_only=False)
assert new['infos']['source'] == str(source), 'Wrong continuation source'
for key in ['contract', 'reward_version', 'reward_signature', 'asset_sha256', 'motion_sha256', 'actuator_sha256']:
    assert old['infos'][key] == new['infos'][key], key
assert new['infos']['contract'] == 'physical_v29_history5_unitree_defaultoffset_v2'
old_opt, new_opt = old['optimizer_state_dict'], new['optimizer_state_dict']
assert set(old_opt['state']) == set(new_opt['state'])
steps = []
for key, value in old_opt['state'].items():
    before = float(value['step'])
    after = float(new_opt['state'][key]['step'])
    assert after == before + 20, (key, before, after)
    assert torch.isfinite(new_opt['state'][key]['exp_avg']).all()
    assert torch.isfinite(new_opt['state'][key]['exp_avg_sq']).all()
    steps.append({'parameter': key, 'before': before, 'after': after})
assert any(not torch.equal(value, new['actor_state_dict'][key]) for key, value in old['actor_state_dict'].items())
record = dict(passed=True, run=str(run), source=str(source), continuation=str(continuation),
    saved_full_updates=recycle['completed_iterations'], optimizer_steps=steps,
    provenance_fields_equal=True, actor_updated=True, expected_optimizer_updates_per_ppo=20,
    scope='Saved actual continuation after the first new PPO update; Adam counters advanced rather than restarting. Not a behavior gate.')
output = HERE / 'diagnostics/unitree_termination_20261002/optimizer_resume_r22.json'
output.write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(dict(passed=True, parameters=len(steps), before_steps=sorted({row['before'] for row in steps}),
    after_steps=sorted({row['after'] for row in steps}), evidence=str(output)), indent=2))
