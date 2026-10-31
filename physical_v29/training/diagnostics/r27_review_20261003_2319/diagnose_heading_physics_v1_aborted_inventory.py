"""Isolated frozen-asset steering/memory/profile experiment; never a gate."""
import argparse
from collections import defaultdict
import warnings
import functools
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType

import psutil

parser=argparse.ArgumentParser()
parser.add_argument("--run",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--checkpoint",required=True)
parser.add_argument("--duration",type=float,default=7.)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
for proc in psutil.process_iter(["name","cmdline"]):
    if (proc.info["name"] or "").lower() in ("python.exe","pythonw.exe"):
        assert not any(Path(a).name=="worker_unitree.py" for a in (proc.info["cmdline"] or [])), "Live Isaac worker must be drained first"
run=Path(args.run).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
assert not (out/"result.json").exists(),"Do not overwrite a completed experiment"
source=run/"source/physical_v29"
sys.path.insert(0,str(source));sys.path.insert(0,str(source/"training"))
app=AppLauncher(args)

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from environment_unitree import LocomotionEnv,LocomotionCfg
from cad_differential import CadDifferential


def memory(label,tick):
    p=psutil.Process().memory_info()
    return dict(label=label,tick=tick,private_mb=getattr(p,"private",p.rss)/2**20,rss_mb=p.rss/2**20,
        host_available_mb=psutil.virtual_memory().available/2**20,
        cuda_allocated_mb=torch.cuda.memory_allocated()/2**20,python_gc_count=len(gc.get_objects()))


def tensor_inventory():
    shapes=defaultdict(lambda:[0,0]);seen=set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if not isinstance(obj,torch.Tensor):continue
                storage=obj.untyped_storage();identity=(str(obj.device),storage.data_ptr())
                key=str((str(obj.device),tuple(obj.shape),str(obj.dtype)))
                shapes[key][0]+=1
                if identity not in seen:
                    shapes[key][1]+=storage.nbytes();seen.add(identity)
            except (RuntimeError,ReferenceError):pass
    return {k:dict(objects=v[0],unique_storage_mb=v[1]/2**20)
            for k,v in sorted(shapes.items(),key=lambda item:item[1][1],reverse=True)}


def actor_from(checkpoint):
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    state=saved["actor_state_dict"];indexes=sorted(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith("weight"))
    layers=[]
    for idx in indexes:
        weight=state[f"mlp.{idx}.weight"];layers.append(torch.nn.Linear(weight.shape[1],weight.shape[0]))
        if idx!=indexes[-1]:layers.append(torch.nn.ELU())
    actor=torch.nn.Sequential(*layers)
    actor.load_state_dict({k.removeprefix("mlp."):v for k,v in state.items() if k.startswith("mlp.")})
    return actor.to("cuda:0").eval()


def main():
    torch.set_num_threads(4)
    cfg=LocomotionCfg();cfg.scene.num_envs=72;cfg.robot_usd=str(source/"training/assets/usd/robot_training.usd")
    assert Path(cfg.robot_usd).exists()
    cfg.sim.device="cuda:0";cfg.stage_speed=.35;cfg.evaluation=True;cfg.seed=901;cfg.episode_length_s=30.
    for key,value in dict(gpu_found_lost_aggregate_pairs_capacity=2**22,gpu_total_aggregate_pairs_capacity=2**20,
        gpu_max_rigid_contact_count=2**20,gpu_max_rigid_patch_count=2**17,gpu_heap_capacity=2**26,gpu_collision_stack_size=2**25).items():
        setattr(cfg.sim.physx,key,value)
    raw=LocomotionEnv(cfg);raw.initialize_randomization();env=RslRlVecEnvWrapper(raw,clip_actions=3.)
    labels=np.arange(72)%6;speeds=np.array([.15,.15,.15,.35,.35,.35],dtype=np.float32)
    yaw_commands=np.array([-.2,0.,.2,-.2,0.,.2],dtype=np.float32)
    targets=torch.tensor(yaw_commands[labels],device=raw.device)
    raw.configure_fixed_commands(torch.tensor(speeds[labels],device=raw.device),torch.zeros(72,device=raw.device))
    def constant(self):
        self.commands[:,2]=targets
        self._command_tick=self.common_step_counter
    raw._update_observation_command=MethodType(constant,raw)
    active_profile=False;profile=defaultdict(list)
    def wrap(original,name):
        @functools.wraps(original)
        def timed(*pos,**kw):
            if not active_profile:return original(*pos,**kw)
            start=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            value=original(*pos,**kw);b.record()
            profile[name].append((time.perf_counter()-start,a,b))
            return value
        return timed
    CadDifferential.state=wrap(CadDifferential.state,"cad_state")
    raw._pre_physics_step=wrap(raw._pre_physics_step,"pre_physics")
    raw.actuator.compute=wrap(raw.actuator.compute,"actuator_compute")
    raw.sim.step=wrap(raw.sim.step,"simulation_step")
    raw._get_rewards=wrap(raw._get_rewards,"rewards")
    raw._get_observations=wrap(raw._get_observations,"observations")
    results=[];mem=[];inventories=[]
    for label,checkpoint in [("baseline",run/"002_ppo_smoke/transferred_actor.pt"),("192_updates",Path(args.checkpoint))]:
        actor=actor_from(checkpoint);env.reset();alive=torch.ones(72,dtype=torch.bool,device=raw.device)
        roots=[];valid=[];frames=[];n=round(args.duration/raw.step_dt);started=time.monotonic()
        mem.append(memory(label,-1))
        inventories.append(dict(label=label,tick=-1,tensors=tensor_inventory()))
        with torch.inference_mode():
            for tick in range(n):
                active_profile=label=="192_updates" and tick>=n-8
                obs=env.get_observations();actions=actor(obs["policy"]);_,_,done,_=env.step(actions);alive&=~done.bool()
                roots.append(raw.robot.data.root_state_w.cpu().numpy().copy());valid.append(alive.cpu().numpy().copy())
                if tick%5==0:
                    m=raw.telemetry();frames.append(dict(t=(tick+1)*.02,q=raw.robot.data.joint_pos[:16].cpu().numpy().copy(),
                        root=roots[-1][:16].copy(),contact=m["contact"][:16].cpu().numpy().copy(),
                        clearance=m["clearance"][:16].cpu().numpy().copy()))
                if tick%50==0:
                    row=memory(label,tick);mem.append(row);print("DIAGNOSTIC_PROGRESS "+json.dumps(row),flush=True)
        active_profile=False;torch.cuda.synchronize()
        roots=np.stack(roots);valid=np.stack(valid);w,x,y,z=np.moveaxis(roots[...,3:7],-1,0)
        heading=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)),axis=0)
        cases=[];start=100
        for group in range(6):
            ids=(labels==group);mask=valid[start:,ids];pairs=valid[start+1:,ids]&valid[start:-1,ids]
            xy=roots[start:,ids,:2];v=np.diff(xy,axis=0)/.02
            cases.append(dict(speed=float(speeds[group]),desired_yaw_rate=float(yaw_commands[group]),
                survival=float(valid[-1,ids].mean()),body_yaw_velocity=float(roots[start:,ids,12][mask].mean()),
                net_heading_rate=float(((heading[-1,ids]-heading[start,ids])/((n-1-start)*.02))[valid[-1,ids]].mean()),
                forward_speed=float(v[...,0][pairs].mean()),lateral_abs_speed=float(np.abs(v[...,1][pairs]).mean()),
                final_heading_deg=float(np.rad2deg(heading[-1,ids][valid[-1,ids]]).mean())))
        results.append(dict(label=label,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            cases=cases,wall_seconds=time.monotonic()-started,steps=n))
        np.savez_compressed(out/f"{label}_counterfactual.npz",root=roots,alive=valid,labels=labels,
            speed=speeds,yaw_commands=yaw_commands,joint_names=np.array(raw.robot.joint_names),
            frame_time=np.array([f["t"] for f in frames]),q=np.stack([f["q"] for f in frames]),
            frame_root=np.stack([f["root"] for f in frames]),contact=np.stack([f["contact"] for f in frames]),
            clearance=np.stack([f["clearance"] for f in frames]))
        del actor;gc.collect();mem.append(memory(label,n))
        inventories.append(dict(label=label,tick=n,tensors=tensor_inventory()))
    timing={name:dict(calls=len(rows),host_dispatch_seconds=sum(r[0] for r in rows),
        cuda_elapsed_seconds=sum(r[1].elapsed_time(r[2]) for r in rows)/1000) for name,rows in profile.items()}
    report=dict(completed=True,results=results,memory=mem,tensor_inventories=inventories,profile=timing,profile_control_ticks=8,
        scope="One isolated Isaac/CUDA worker, exact R27 frozen geometry/actuation, constant yaw-command interventions at identical resets. Diagnostic only; seven-second response cannot qualify locomotion. Profile is inclusive and instrumented.")
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n");print("DIAGNOSTIC_COMPLETE "+json.dumps(results),flush=True)
    env.close()


try:main()
finally:app.app.close()
