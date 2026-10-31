"""Isolate RSL-RL allocation from Isaac/PhysX without using a second GPU.

Synthetic observations are not a robot test. Same network/PPO config, short CPU
updates, bounded RSS/commit samples; compare with real worker telemetry.
"""
import ast,copy,gc,json,time,contextlib,psutil
from pathlib import Path
from types import SimpleNamespace
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

here=Path(__file__).resolve().parent
training=here.parents[1]
tree=ast.parse((training/'worker.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='agent_config')
scope={'args':SimpleNamespace(learning_rate=.0002)}
exec(compile(ast.Module(body=[function],type_ignores=[]),'agent_config','exec'),scope)
torch.set_num_threads(1)

class SyntheticEnv:
    device='cpu';num_envs=128;num_actions=23;max_episode_length=500;cfg={}
    def __init__(self):self.episode_length_buf=torch.zeros(self.num_envs,dtype=torch.long)
    def get_observations(self):
        return TensorDict({'policy':torch.randn(self.num_envs,320),'critic':torch.randn(self.num_envs,100)},batch_size=[self.num_envs])
    def step(self,actions):
        self.episode_length_buf+=1
        done=self.episode_length_buf%50==0
        return self.get_observations(),-.01*actions.square().mean(1),done,{'log':{'synthetic':actions.mean().detach()}}

samples=[];start=time.monotonic()
with (here/'stdout.log').open('w') as log,contextlib.redirect_stdout(log):
    runner=OnPolicyRunner(SyntheticEnv(),copy.deepcopy(scope['agent_config']()),log_dir=str(here/'logs'),device='cpu')
    original=runner.logger.log
    def progress(**kwargs):
        original(**kwargs)
        before=psutil.Process().memory_info()
        collected=gc.collect() if kwargs['it']%4==0 else 0
        after=psutil.Process().memory_info()
        samples.append(dict(iteration=kwargs['it'],rss_mb=after.rss/2**20,
            private_mb=getattr(after,'private',after.rss)/2**20,
            gc_collected=collected,gc_rss_change_mb=(after.rss-before.rss)/2**20))
    runner.logger.log=progress
    runner.learn(32)
(here/'result.json').write_text(json.dumps(dict(scope='Synthetic CPU PPO, no Isaac physics and no GPU allocation',
    iterations=32,seconds=time.monotonic()-start,samples=samples),indent=2))
print(json.dumps(dict(seconds=time.monotonic()-start,first=samples[0],last=samples[-1])))
