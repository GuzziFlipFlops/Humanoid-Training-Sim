import requests,json,hashlib
from pathlib import Path
root=Path(__file__).parent/'research';root.mkdir(exist_ok=True)
files={
 'unitree_g1_cfg.py':'https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py',
 'unitree_ppo_cfg.py':'https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py',
 'unitree_rewards.py':'https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py',
 'asimov_motion_dataset.py':'https://raw.githubusercontent.com/menloresearch/cyclotron/main/source/cyclotron/cyclotron/tasks/locomotion/motion_dataset.py',
 'asimov_walk.npz':'https://raw.githubusercontent.com/menloresearch/cyclotron/main/source/cyclotron/cyclotron/tasks/locomotion/motions/policy_delay_walk_slow.npz',
 'asimov_ppo_cfg.py':'https://raw.githubusercontent.com/menloresearch/cyclotron/main/source/cyclotron/cyclotron/tasks/locomotion/agents/rsl_rl_ppo_cfg.py',
 'asimov_amp_ppo.py':'https://raw.githubusercontent.com/menloresearch/cyclotron/main/source/cyclotron/cyclotron/algorithms/amp_ppo.py',
 'humanoidgym_reference.py':'https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_env.py',
}
manifest=[]
for filename,url in files.items():
 r=requests.get(url,timeout=30);r.raise_for_status();(root/filename).write_bytes(r.content)
 manifest.append(dict(file=filename,url=url,sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content)))
(root/'sources.json').write_text(json.dumps(manifest,indent=2))
import numpy as np
z=np.load(root/'asimov_walk.npz');print('Motion',[(k,z[k].shape,str(z[k].dtype)) for k in z.files]);print({k:str(z[k])[:700] for k in z.files if z[k].size<100})
