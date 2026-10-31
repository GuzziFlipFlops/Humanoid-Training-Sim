"""Explicit actor-only transfer across this reward repair; no qualification reuse."""
from pathlib import Path
import hashlib

CONTRACT = 'physical_v29_history5_unitree_defaultoffset_v2'
SOURCE_REWARD = 'cad_standing_g1_curriculum_v3'
GAIT_SOURCE_REWARD = 'cad_phase_feedback_retention_v4'
GAIT_SOURCE_SIGNATURE = '2fdf54a05c1ca88522d432d2831b09c17369373e088535f5f3ba95b27c031b6f'
RUNNING_SOURCE_REWARD = 'cad_heading_alignment_retention_v6'
RUNNING_SOURCE_SIGNATURE = '482dce78353420d47fb35a6db21bb1b8c6c11944f1e8519f1462aa77f75d0b2b'
INITIAL_EXPLORATION_STD = .35


def validate_actor_transfer(saved, training_dir):
    here = Path(training_dir)
    info = saved.get('infos',{})
    assert info.get('contract')==CONTRACT, 'Actor observation/action contract differs'
    source = info.get('reward_version')
    assert source in (SOURCE_REWARD,GAIT_SOURCE_REWARD,RUNNING_SOURCE_REWARD), 'Only explicitly source-bound v29 actor transfers are accepted'
    if source==SOURCE_REWARD:
        assert float(info.get('speed',-1))==0., 'Use the selected standing actor'
    elif source==RUNNING_SOURCE_REWARD:
        assert float(info.get('speed',-1))==1.5, 'Use the physically audited R31 running actor'
        assert info.get('reward_signature')==RUNNING_SOURCE_SIGNATURE, 'v6 source differs from audited R31'
    else:
        assert float(info.get('speed',-1)) in (0.,.35,.65,1.,1.5,2.,2.5), 'Unrecognized v4 stage'
        assert info.get('reward_signature')==GAIT_SOURCE_SIGNATURE, 'v4 reward/observation signature differs from audited R25'
    for key,path in [('asset_sha256',here/'assets/robot_training.urdf'),
                     ('motion_sha256',here/'assets/motion_reference.npz'),
                     ('actuator_sha256',here.parent/'assets/actuator_limits.json')]:
        assert info.get(key)==hashlib.sha256(path.read_bytes()).hexdigest(), key+' differs'
    assert saved['actor_state_dict']['mlp.0.weight'].shape[1]==400
    last = max(int(k.split('.')[1]) for k in saved['actor_state_dict'] if k.startswith('mlp.') and k.endswith('.weight'))
    assert saved['actor_state_dict'][f'mlp.{last}.weight'].shape[0]==23
    return dict(source_reward=source,contract=CONTRACT,actor_only=True,
        critic_loaded=False,optimizer_loaded=False,qualification_reused=False,
        initial_exploration_std=INITIAL_EXPLORATION_STD if source==SOURCE_REWARD else None,
        preserve_trained_exploration=source in (GAIT_SOURCE_REWARD,RUNNING_SOURCE_REWARD),
        command_adaptation='Desired yaw-rate channel retained; heading feedback newly supplies its target')
