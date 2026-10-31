"""Explicit actor-only transfer across this reward repair; no qualification reuse."""
from pathlib import Path
import hashlib

CONTRACT = 'physical_v29_history5_unitree_defaultoffset_v2'
SOURCE_REWARD = 'unitree_g1_lifetime_v2'
INITIAL_EXPLORATION_STD = .35


def validate_actor_transfer(saved, training_dir):
    here = Path(training_dir)
    info = saved.get('infos',{})
    assert info.get('contract')==CONTRACT, 'Actor observation/action contract differs'
    assert info.get('reward_version')==SOURCE_REWARD, 'Only the audited R22 actor is accepted for this repair'
    assert float(info.get('speed',-1))==0., 'Use the selected standing actor'
    for key,path in [('asset_sha256',here/'assets/robot_training.urdf'),
                     ('motion_sha256',here/'assets/motion_reference.npz'),
                     ('actuator_sha256',here.parent/'assets/actuator_limits.json')]:
        assert info.get(key)==hashlib.sha256(path.read_bytes()).hexdigest(), key+' differs'
    assert saved['actor_state_dict']['mlp.0.weight'].shape[1]==400
    last = max(int(k.split('.')[1]) for k in saved['actor_state_dict'] if k.startswith('mlp.') and k.endswith('.weight'))
    assert saved['actor_state_dict'][f'mlp.{last}.weight'].shape[0]==23
    return dict(source_reward=SOURCE_REWARD,contract=CONTRACT,actor_only=True,
        critic_loaded=False,optimizer_loaded=False,qualification_reused=False,
        initial_exploration_std=INITIAL_EXPLORATION_STD)
