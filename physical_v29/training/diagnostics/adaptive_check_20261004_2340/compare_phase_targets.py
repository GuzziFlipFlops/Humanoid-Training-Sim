"""CPU-only paired-actor response on exact saved histories and phase/contact traces.

This holds observed state fixed. It cannot establish closed-loop effects, gradient
interference, a takeoff repair or a motor ceiling. No training file is imported
with simulator startup, and no live checkpoint/source is written.
"""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import audit_running as audit

torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent
RUN=audit.RUN
folders=[RUN/'039_run_150_gate_008_nominal', RUN/'063_run_150_gate_016_nominal',RUN/'066_run_150_gate_017_nominal']

def model(path):
    state=torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict']
    actor=torch.nn.Sequential(torch.nn.Linear(400,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),
        torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,23))
    actor.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')})
    return actor.eval()

def histories(folder):
    d=np.load(folder/'gait_diagnostics_50hz.npz',allow_pickle=False)
    assert not d['done'][2:].any()
    motion=np.load(audit.FROZEN/'training/assets/motion_reference.npz',allow_pickle=False)
    names=list(d['joint_names']);stand=torch.from_numpy(motion['stand'][[list(motion['names']).index(n) for n in names]])
    roots=torch.from_numpy(d['root_state'][:-1]);quat=roots[...,3:7]
    commands=torch.from_numpy(d['observed_command'][1:])
    period,_=audit.gait_parameters(commands.reshape(-1,3));period=period.reshape(commands.shape[:2])
    phase=torch.remainder(torch.arange(1,len(d['time']),dtype=torch.float32)[:,None]*.02/period,1.)
    ph=torch.stack([torch.sin(2*torch.pi*phase),torch.cos(2*torch.pi*phase)],-1)*(commands[...,0:1]>.1)
    gravity=torch.zeros_like(roots[...,10:13]);gravity[...,2]=-1.
    frames=torch.cat([audit.inverse(quat,roots[...,10:13])*.2,audit.inverse(quat,gravity),commands,
        torch.from_numpy(d['q'][:-1])-stand,torch.from_numpy(d['qd'][:-1])*.05,
        torch.from_numpy(d['applied_action'][:-1]),ph],-1)
    n=len(d['commands'])
    h=torch.stack([frames[i-4:i+1].transpose(0,1).reshape(n,400) for i in range(4,len(frames))])[20::7]
    labels=np.broadcast_to(d['commands'],h.shape[:2]).reshape(-1)
    expected=torch.from_numpy(d['raw_action'][5:])[20::7].reshape(-1,23)
    return d,names,h.reshape(-1,400),labels,expected

actors={};checkpoints={}
for folder in folders:
    assert folder.exists(),folder
    ck=Path(audit.read(folder/'run_config.json')['arguments']['checkpoint'])
    actors[folder.name]=model(ck);checkpoints[folder.name]={'path':str(ck),'sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}
rows=[]
with torch.inference_mode():
    for folder in folders:
        d,names,h,labels,expected=histories(folder)
        own=actors[folder.name](h)
        error=float((own-expected).abs().max());assert error<.0005,error
        mask=torch.from_numpy(labels==1.5)
        selected=h[mask];ref=own[mask];ref_clip=ref.clamp(-3,3)
        comparisons=[]
        for name,actor in actors.items():
            action=actor(selected);clipped=action.clamp(-3,3);joints=[]
            for j,joint in enumerate(names):
                if not any(x in joint for x in ['hip','knee','ankle']):continue
                joints.append({'joint':joint,'raw_clip_fraction':float((action[:,j].abs()>3).float().mean()),
                    'raw_RMS_difference':float((action[:,j]-ref[:,j]).square().mean().sqrt()),
                    'clipped_RMS_difference':float((clipped[:,j]-ref_clip[:,j]).square().mean().sqrt()),
                    'pre_CAD_target_RMS_difference_rad':float(.25*(clipped[:,j]-ref_clip[:,j]).square().mean().sqrt())})
            comparisons.append({'actor':name,'samples':len(selected),'joints':joints,
                'all_outputs_clip_fraction':float((action.abs()>3).float().mean()),
                'effective_clipped_RMS_difference':float((clipped-ref_clip).square().mean().sqrt())})
        trace=audit.trace_stats(folder)['commands'][-1]
        rows.append({'saved_trace':folder.name,'exact_action_reconstruction_max_error':error,
            'sampled_running_trace':trace,'fixed_state_actor_comparisons':comparisons})

value={'recorded_utc':datetime.now(timezone.utc).isoformat(),'checkpoints':checkpoints,'rows':rows,
    'scope':'Same fixed reconstructed400-input histories, raw versus runtime-clipped actions. Pre-CAD targets use the unchanged0.25 scale; coupled CAD projection is excluded and no target is applied in physics.',
    'decision':'Preserve the separately useful flight actor and latest retained-speed actor. Use measured leg-output clipping and intended-flight force timing for controlled physical/PPO cost-gradient tests before another repair. These cross-state responses cannot establish that retention causes flight loss.'}
(OUT/'phase_target_comparison.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps([{'trace':r['saved_trace'],'reconstruction_error':r['exact_action_reconstruction_max_error'],
    'genuine_clearance_fraction':r['sampled_running_trace']['both_soles_above_1mm_fraction'],
    'loaded_leg_in_flight':r['sampled_running_trace']['loaded_leg_in_expected_flight_fraction'],
    'own_leg_clipping':[(j['joint'],round(j['raw_clip_fraction'],3)) for j in next(c for c in r['fixed_state_actor_comparisons'] if c['actor']==r['saved_trace'])['joints']]} for r in rows],indent=2))
