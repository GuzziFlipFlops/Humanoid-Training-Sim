"""CPU actor and reward counterfactuals from synchronized saved measurements.

Reconstruct and validate actual five-frame actor inputs before varying a command.
This is an input-sensitivity test, not a physical steering qualification.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]
sys.path.insert(0, str(TRAINING))
from unitree_reward_port import FUNCTIONS
from types import SimpleNamespace as NS


def inverse(q, v):
    xyz=q[..., 1:]
    t=2*np.cross(xyz, v)
    return v-q[..., :1]*t+np.cross(xyz, t)


def actor_from(checkpoint):
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    state=saved["actor_state_dict"]
    indexes=sorted(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith("weight"))
    layers=[]
    for idx in indexes:
        weight=state[f"mlp.{idx}.weight"]
        layers.append(torch.nn.Linear(weight.shape[1],weight.shape[0]))
        if idx!=indexes[-1]:layers.append(torch.nn.ELU())
    actor=torch.nn.Sequential(*layers).eval()
    actor.load_state_dict({k.removeprefix("mlp."):v for k,v in state.items() if k.startswith("mlp.")})
    return actor, saved["infos"]


def main():
    assert psutil.virtual_memory().available>2500*2**20,"Wait for host headroom before CPU torch audit"
    torch.set_num_threads(2)
    run=Path(json.loads((TRAINING/"ACTIVE_RUN.json").read_text())["run"])
    baseline=run/"002_ppo_smoke/transferred_actor.pt"
    plans=[("baseline","011_walk_035_gate_001_nominal",baseline),
           ("64_updates","027_walk_035_gate_002_nominal",run/"026_walk_035_train_001_part_14/candidate.pt"),
           ("128_updates","038_walk_035_gate_003_nominal",run/"037_walk_035_train_002_part_09/candidate.pt")]
    motion=np.load(run/"source/physical_v29/training/assets/motion_reference.npz",allow_pickle=False)
    records=[]
    for label,folder,checkpoint in plans:
        with np.load(run/folder/"gait_diagnostics_50hz.npz",allow_pickle=False) as z:
            names=z["joint_names"].tolist()
            stand=motion["stand"][[motion["names"].tolist().index(n) for n in names]]
            data={k:z[k].copy() for k in z.files}
        actor,info=actor_from(checkpoint)
        root=data["root_state"]
        frames=np.zeros((len(root),16,80),dtype=np.float32)
        # Input for action t uses post-step state/action from t-1. Exclude
        # startup and terminated rows so unavailable reset state is never guessed.
        frames[1:,:,:3]=inverse(root[:-1,:,3:7],root[:-1,:,10:13])*.2
        frames[1:,:,3:6]=inverse(root[:-1,:,3:7],np.broadcast_to(np.array([0,0,-1.],dtype=np.float32),root[:-1,:,10:13].shape))
        frames[:,:,6:9]=data["action_command"]
        frames[1:,:,9:32]=data["q"][:-1]-stand
        frames[1:,:,32:55]=data["qd"][:-1]*.05
        frames[1:,:,55:78]=data["applied_action"][:-1]
        phase=np.remainder(np.arange(len(root))*.02/.8,1.)
        moving=data["commands"]>.1
        frames[:,:,78]=np.sin(2*np.pi*phase)[:,None]*moving[None,:]
        frames[:,:,79]=np.cos(2*np.pi*phase)[:,None]*moving[None,:]
        histories=np.stack([frames[i-4:i+1].transpose(1,0,2).reshape(16,400) for i in range(5,len(frames))])
        ticks=np.arange(5,len(frames))
        alive=np.cumsum(data["done"],axis=0)==0
        valid=(data["time"][ticks,None]>=2.) & alive[ticks] & moving[None,:]
        # Use all saved moving frames. Strides can alias both interleaved
        # command groups and the 40-tick walking phase.
        flat_ids=np.flatnonzero(valid.reshape(-1))
        observations=torch.from_numpy(histories.reshape(-1,400)[flat_ids]).clone()
        actual=torch.from_numpy(data["raw_action"][ticks].reshape(-1,23)[flat_ids])
        with torch.inference_mode():
            expected=actor(observations)
            error=(actual-expected).abs()
            reconstruction=float(error.max())
            assert reconstruction<2e-4, f"History reconstruction disagrees with measured actor: {label}: {reconstruction}"
            results=[]
            for scope in ["latest_frame","all_five_frames"]:
                actions=[]
                for cmd in [-.2,0.,.2]:
                    altered=observations.clone()
                    indexes=[328] if scope=="latest_frame" else [8,88,168,248,328]
                    altered[:,indexes]=cmd
                    actions.append(actor(altered))
                delta=actions[2]-actions[0]
                results.append(dict(scope=scope,
                    mean_action_delta_rms=float(delta.square().mean().sqrt()),
                    joint_action_delta_mean={name:float(delta[:,i].mean()) for i,name in enumerate(names)},
                    joint_action_delta_abs_mean={name:float(delta[:,i].abs().mean()) for i,name in enumerate(names)}))
        # Exact pinned angular kernel on recorded body-frame angular velocity.
        body=inverse(root[...,3:7],root[...,10:13])
        angular=torch.from_numpy(body[ticks].reshape(-1,3)[flat_ids])
        commands=torch.from_numpy(data["action_command"][ticks].reshape(-1,3)[flat_ids])
        proxy=NS(scene={"robot":NS(data=NS(root_ang_vel_b=angular))},
            command_manager=NS(get_command=lambda name:commands))
        with torch.inference_mode():
            reward=FUNCTIONS["track_ang_vel_z_exp"](proxy,.5,"base_velocity")
            bias=angular[:,2].mean()-commands[:,2].mean()
            scores={}
            for fraction in [0.,.25,.5,1.]:
                adjusted=angular.clone();adjusted[:,2]-=fraction*bias
                proxy.scene["robot"].data.root_ang_vel_b=adjusted
                scores[str(fraction)]=float(.5*FUNCTIONS["track_ang_vel_z_exp"](proxy,.5,"base_velocity").mean())
        records.append(dict(label=label,checkpoint=str(checkpoint),trace=str(run/folder/"gait_diagnostics_50hz.npz"),
            sampled_observations=len(observations),reconstruction_max_abs_error=reconstruction,
            source_reward_version=info["reward_version"],counterfactuals=results,
            angular=dict(measured_mean_rad_s=float(angular[:,2].mean()),desired_mean_rad_s=float(commands[:,2].mean()),
                mean_bias_rad_s=float(bias),kernel_mean=float(reward.mean()),
                score_by_removed_mean_bias_fraction=scores,
                scope="Recorded-state counterfactual, keeps phase oscillations; no claim of action-to-yaw causal gradient")))
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=True,records=records,
        scope="CPU only, exact measured-input reconstruction and pinned kernel, no optimizer or simulator")
    (HERE/"actor_command_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":True,"records":[{k:r[k] for k in ["label","sampled_observations","reconstruction_max_abs_error","angular"]} for r in records]},indent=2))


if __name__=="__main__":main()
