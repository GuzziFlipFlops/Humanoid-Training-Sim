"""Passive capture of truthful policy inputs; returns identical actions."""
import sys,runpy,json,hashlib,builtins
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
original=None
def capture(env,policy,speed,duration,seed,perturb,out,play=False):
    count=0;population=[];rep=[];actions=[];times=[];reptimes=[]
    def passive(obs):
        nonlocal count
        action=policy(obs)
        if count%5==0:
            population.append(obs['policy'].detach().cpu().numpy().copy())
            actions.append(action.detach().cpu().numpy().copy())
            times.append(count*env.unwrapped.step_dt)
        rep.append(obs['policy'][:16].detach().cpu().numpy().copy());reptimes.append(count*env.unwrapped.step_dt)
        count+=1
        return action
    result=original(env,passive,speed,duration,seed,perturb,out,play)
    np.savez_compressed(Path(out)/'truthful_policy_states.npz',population_observations=np.stack(population),
        population_raw_actions=np.stack(actions),population_time=np.asarray(times),
        representative_observations=np.stack(rep),representative_time=np.asarray(reptimes))
    (Path(out)/'passive_capture.json').write_text(json.dumps(dict(actual_inputs=True,unchanged_actions=True,
        policy_calls=count,population_hz=10,representative_hz=50,representative_robots=16,
        phase_timing='Input before action at t; gait_diagnostics physical sample is after the action at t+20ms',
        production_change=False),indent=2))
    return result
original_import=builtins.__import__;installed=False
def install(name,*args,**kwargs):
    global original,installed
    result=original_import(name,*args,**kwargs)
    if name=='evaluation' and not installed:
        installed=True;original=result.evaluate;result.evaluate=capture
        builtins.__import__=original_import
    return result
builtins.__import__=install
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
