"""CPU-only installed PPO memory isolation, with/without local audit hooks."""
import copy
import gc
import json
from pathlib import Path
import sys
import time

import psutil
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

HERE=Path(__file__).resolve().parent
TRAINING=HERE.parents[1]
sys.path.insert(0,str(TRAINING))
from ppo_update_audit import PpoUpdateAudit
from standing_retention import StandingRetention


def sample(iteration):
    proc=psutil.Process();info=proc.memory_info()
    return dict(iteration=iteration,rss_mb=info.rss/2**20,
        private_mb=getattr(info,"private",info.rss)/2**20,garbage_collected=gc.collect())


def run(hooks):
    torch.manual_seed(615);n=128;ticks=24
    def observation():
        p=torch.randn(n,400)*.05
        p[:,6::80]=.35
        p[::3,6::80]=0.
        return TensorDict({"policy":p,"critic":torch.randn(n,415)*.05},batch_size=[n])
    obs=observation()
    actor=MLPModel(obs,{"actor":["policy"]},"actor",23,hidden_dims=[512,256,128],activation="elu",obs_normalization=False,
        distribution_cfg=dict(class_name="GaussianDistribution",init_std=.35,std_type="log"))
    critic=MLPModel(obs,{"critic":["critic"]},"critic",1,hidden_dims=[512,256,128],activation="elu",obs_normalization=False)
    storage=RolloutStorage("rl",n,ticks,obs,[23],device="cpu")
    alg=PPO(actor,critic,storage,num_learning_epochs=5,num_mini_batches=4,learning_rate=.0001,schedule="adaptive",desired_kl=.01)
    audit=PpoUpdateAudit(alg) if hooks else None
    retention=StandingRetention(actor,copy.deepcopy(actor)) if hooks else None
    records=[]
    for iteration in range(32):
        with torch.inference_mode():
            for tick in range(ticks):
                actions=alg.act(obs)
                obs=observation()
                rewards=.20-actions.square().mean(1)*.001
                done=torch.zeros(n,dtype=torch.long)
                if tick%12==0:done[::11]=1
                alg.process_env_step(obs,rewards,done,{})
            alg.compute_returns(obs)
        alg.update()
        if retention:retention.read_audit()
        records.append(sample(iteration))
        if iteration%8==0:print(json.dumps(dict(hooks=hooks,**records[-1])),flush=True)
    if audit:audit.close()
    if retention:retention.close()
    return dict(hooks=hooks,records=records,
        post_warmup_private_growth_mb=records[-1]["private_mb"]-records[7]["private_mb"],
        post_warmup_rss_growth_mb=records[-1]["rss_mb"]-records[7]["rss_mb"])


def main():
    torch.set_num_threads(2)
    start=time.monotonic();cases=[]
    for hooks in [False,True]:
        cases.append(run(hooks));gc.collect()
    result=dict(cases=cases,seconds=time.monotonic()-start,
        scope="Synthetic owned observations and installed CPU PPO, same architecture, no Isaac/CUDA. Does not isolate native simulation or logger.")
    (HERE/"ppo_memory_isolation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"seconds":result["seconds"],"cases":[{k:c[k] for k in ["hooks","post_warmup_private_growth_mb","post_warmup_rss_growth_mb"]} for c in cases]}),flush=True)


if __name__=="__main__":main()
