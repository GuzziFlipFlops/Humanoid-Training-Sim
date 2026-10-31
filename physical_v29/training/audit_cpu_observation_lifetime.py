"""Controlled CPU-only PPO observation-lifetime experiment, no simulator.

Checks whether Python-visible observation references accumulate with the
installed PPO, actual history methods and optional new bounded update audit.
Cannot diagnose native PhysX/Kit or CUDA allocator ownership by itself.
"""
import gc
import json
from pathlib import Path
import weakref

import psutil
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from ppo_update_audit import PpoUpdateAudit
from test_observation_snapshot import compiled, fake, read_method, reset_function

HERE = Path(__file__).resolve().parent
OUT = HERE / 'diagnostics/walk_regression_20261003/cpu_lifetime.json'


def experiment(audited):
    torch.manual_seed(7103)
    env = fake()
    get = compiled(read_method(HERE / 'environment_unitree.py', '_get_observations'))
    clear = reset_function()
    obs = TensorDict(get(env), batch_size=[8])
    actor = MLPModel(obs, {'actor': ['policy']}, 'actor', 23,
        hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False,
        distribution_cfg=dict(class_name='GaussianDistribution', init_std=.35, std_type='log'))
    critic = MLPModel(obs, {'critic': ['critic']}, 'critic', 1,
        hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False)
    storage = RolloutStorage('rl', 8, 24, obs, [23])
    alg = PPO(actor, critic, storage, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=.001, schedule='adaptive', desired_kl=.01, gamma=.99, lam=.95)
    audit = PpoUpdateAudit(alg) if audited else None
    references = []
    rows = []
    proc = psutil.Process()
    for update in range(32):
        with torch.inference_mode():
            for tick in range(24):
                references.extend(weakref.ref(obs[key]) for key in ['policy', 'critic'])
                action = alg.act(obs)
                env.actions = action.clone()
                env.robot.data.joint_pos = torch.randn(8, 23) * .1
                env.common_step_counter += 1
                env.episode_length_buf += 1
                dones = torch.zeros(8)
                if tick == 12:
                    clear(env, torch.tensor([0, 2, 6]))
                    dones[[0, 2, 6]] = 1.
                obs = TensorDict(get(env), batch_size=[8])
                alg.process_env_step(obs, torch.full((8,), .2), dones, {})
            alg.compute_returns(obs)
        alg.update()
        gc.collect()
        info = proc.memory_info()
        rows.append(dict(update=update, live_tracked_tensors=sum(ref() is not None for ref in references),
            tracked_snapshots=len(references), private_mb=info.private / 2**20,
            pre_update_kl=audit.latest['pre_update_kl'] if audit else None))
    return dict(audited=audited, rows=rows,
        alive_first=rows[0]['live_tracked_tensors'], alive_last=rows[-1]['live_tracked_tensors'],
        live_reference_growth=rows[-1]['live_tracked_tensors']-rows[0]['live_tracked_tensors'],
        private_mb_growth_last16=rows[-1]['private_mb']-rows[16]['private_mb'])


def main():
    torch.set_num_threads(1)
    result = dict(experiments=[experiment(False), experiment(True)],
        scope='CPU installed-PPO/history ownership experiment, 8 fake environments and actual network dimensions. '
              'No Isaac/CUDA worker or saved-policy changes; cannot certify native/CUDA memory stability.')
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(scope=result['scope'], experiments=[{k:v for k,v in r.items() if k!='rows'} for r in result['experiments']])))


if __name__ == '__main__':
    main()
