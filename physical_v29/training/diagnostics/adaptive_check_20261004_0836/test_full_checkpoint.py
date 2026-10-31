"""CPU load of the real production actor, critic and complete Adam state."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE))
from memory_guard import restored_learning_rate
from curriculum_provenance import training_reward_signature
torch.set_num_threads(2)
pointer=json.loads((HERE/'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
checkpoint=Path(pointer['initial_full_checkpoint'])
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert saved['infos']['reward_signature']==training_reward_signature(HERE)
obs=TensorDict(dict(policy=torch.zeros(2,400),critic=torch.zeros(2,415)),batch_size=[2])
actor=MLPModel(obs,{'actor':['policy']},'actor',23,hidden_dims=[512,256,128],activation='elu',obs_normalization=False,
               distribution_cfg=dict(class_name='GaussianDistribution',init_std=.35,std_type='log'))
critic=MLPModel(obs,{'critic':['critic']},'critic',1,hidden_dims=[512,256,128],activation='elu',obs_normalization=False)
storage=RolloutStorage('rl',2,1,obs,[23])
alg=PPO(actor,critic,storage,learning_rate=.0001,schedule='adaptive')
alg.load(saved,load_cfg={'actor':True,'critic':True,'optimizer':True,'iteration':False},strict=True)
assert all(torch.equal(v,actor.state_dict()[k]) for k,v in saved['actor_state_dict'].items())
assert all(torch.equal(v,critic.state_dict()[k]) for k,v in saved['critic_state_dict'].items())
loaded=alg.optimizer.state_dict()
assert loaded['param_groups']==saved['optimizer_state_dict']['param_groups']
assert loaded['state'].keys()==saved['optimizer_state_dict']['state'].keys()
tensors=0
for key,original in saved['optimizer_state_dict']['state'].items():
    assert loaded['state'][key].keys()==original.keys()
    for name,value in original.items():
        if isinstance(value,torch.Tensor):
            assert torch.equal(value,loaded['state'][key][name]);tensors+=1
        else:assert value==loaded['state'][key][name]
rate=restored_learning_rate(alg.optimizer,.0001)
record=dict(passed=True,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    actor_tensors=len(saved['actor_state_dict']),critic_tensors=len(saved['critic_state_dict']),Adam_tensors_equal=tensors,
    saved_learning_rates=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],initial_worker_learning_rate=rate,
    learning_rate_note='Existing unchanged worker ceiling of1e-4 bounds the saved1.5e-4 rate; moments and learned tensors are preserved exactly',
    scope='CPU full-state equality before training, separate from real CUDA PPO/reload smoke and locomotion gates')
Path(__file__).with_name('full_checkpoint_load.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
