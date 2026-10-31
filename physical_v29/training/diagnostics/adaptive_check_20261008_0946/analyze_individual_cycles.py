"""Truthful individual-cycle coverage and final-row gradients; CPU only."""
import json,csv,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
torch.set_num_threads(4)
sys.path.insert(0,str(OUT/'control_source/physical_v29/training'))
from upper_body_refinement import posture_loss
pause=json.loads((OUT/'diagnostic_pause.json').read_text())
saved=torch.load(pause['checkpoint'],map_location='cpu',weights_only=False)
def model(state):
    m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
    values={k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')} if any(k.startswith('mlp.') for k in state) else state
    m.load_state_dict(values);m.eval()
    return m
m=model(saved['actor_state_dict'])
with np.load(OUT/'truthful_nominal/truthful_policy_states.npz') as z:
    pop=z['population_observations'];pop_action=z['population_raw_actions'];pop_time=z['population_time'];rep=z['representative_observations'];time=z['representative_time']
with np.load(OUT/'truthful_nominal/gait_diagnostics_50hz.npz') as z:
    names=z['joint_names'].tolist();done=z['done'];actual=z['q'];target=z['target']
ids=[names.index(n) for n in ['left_shoulder_pitch','right_shoulder_pitch']]
def forwards(x):
    with torch.no_grad():return torch.cat([m(torch.tensor(a)) for a in np.array_split(x, max(1,len(x)//4096))])
pred=forwards(pop.reshape(-1,400)).reshape(*pop.shape[:2],23)
error=float((pred-torch.tensor(pop_action)).abs().max());assert error<6e-5,error
rp=forwards(rep.reshape(-1,400)).reshape(*rep.shape[:2],23).numpy()
frames=rep.reshape(*rep.shape[:2],5,80)
ph=np.remainder(np.arctan2(frames[:,:,-1,78],frames[:,:,-1,79]),2*np.pi)
cycle_rows=[];cycle_indices=[];commands=frames[0,:,-1,6]
for robot,command in enumerate(commands):
    if command<=.1:continue
    end=np.flatnonzero(done[:,robot])
    terminal=int(end[0]) if len(end) else len(time)
    wraps=np.flatnonzero(np.diff(ph[:terminal,robot]) < -np.pi)+1
    for lo,hi in zip(wraps[:-1],wraps[1:]):
        if time[lo]<5 or time[hi]>17:continue
        stable=np.all(np.abs(frames[lo:hi,robot,:,6]-command)<1e-6)
        if not stable:continue
        bins=(ph[lo:hi,robot]*(8/(2*np.pi))).astype(int)
        if len(set(bins))<8:continue
        raw=rp[lo:hi,robot][:,ids];effective=np.clip(raw,-3,3)*.25*180/np.pi
        measured=actual[lo:hi,robot][:,ids]*180/np.pi;wanted=target[lo:hi,robot][:,ids]*180/np.pi
        counts=np.bincount(bins,minlength=8)
        weights=1/(8*counts[bins])
        cycle_indices.append((robot,int(lo),int(hi),weights))
        cycle_rows.append(dict(robot=robot,command=float(command),start_s=float(time[lo]),end_s=float(time[hi]),samples=int(hi-lo),
            policy_balanced_mean_deg=(weights[:,None]*effective).sum(0).tolist(),
            actual_time_mean_deg=measured.mean(0).tolist(),target_time_mean_deg=wanted.mean(0).tolist(),
            effective_p95_p05_deg=(np.quantile(effective,.95,axis=0)-np.quantile(effective,.05,axis=0)).tolist(),
            raw_clamp_fraction=float((abs(raw)>3).mean())))
assert len(cycle_rows)>100
summary=[]
for command in sorted(set(r['command'] for r in cycle_rows)):
    rows=[r for r in cycle_rows if r['command']==command]
    means=np.array([r['policy_balanced_mean_deg'] for r in rows]);q=np.array([r['actual_time_mean_deg'] for r in rows])
    summary.append(dict(command=command,cycles=len(rows),robots=len(set(r['robot'] for r in rows)),
        ensemble_policy_mean_deg=means.mean(0).tolist(),individual_cycle_policy_mean_RMS_deg=np.sqrt((means**2).mean(0)).tolist(),
        individual_actual_cycle_mean_RMS_deg=np.sqrt((q**2).mean(0)).tolist(),
        cycle_mean_abs_above5deg_fraction=(abs(means)>5).mean(0).tolist(),
        cycle_mean_p05_p95_deg=np.quantile(means,[.05,.95],axis=0).tolist()))
np.savez_compressed(OUT/'individual_cycle_means.npz',command=np.array([r['command'] for r in cycle_rows]),
    robot=np.array([r['robot'] for r in cycle_rows]),mean_deg=np.array([r['policy_balanced_mean_deg'] for r in cycle_rows]))
(OUT/'individual_cycle_rows.json').write_text(json.dumps(cycle_rows,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
for side,ax in enumerate(axes):
    for command in sorted(set(r['command'] for r in cycle_rows)):
        rows=[r for r in cycle_rows if r['command']==command]
        ax.plot([(r['start_s']+r['end_s'])/2 for r in rows],[r['policy_balanced_mean_deg'][side] for r in rows],'.',label=f'{command:g}m/s',ms=3)
    ax.axhspan(-5,5,color='gray',alpha=.15);ax.set_title(['Left shoulder','Right shoulder'][side]);ax.set_xlabel('Simulation time(s)');ax.set_ylabel('Effective cycle mean(deg)');ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
fig.suptitle('Per-robot completed gait cycles: held shoulder target offset\n50Hz truthful representative inputs; diagnostic5deg band, no human curve')
fig.tight_layout();fig.savefig(OUT/'individual_cycle_means.png',dpi=150);plt.close(fig)
# One noncancelling, detached-feature cycle gradient, only2 final rows. It is a
# diagnostic proposed derivative, never installed into a PPO worker.
flat=torch.tensor(rep.reshape(-1,400))
with torch.no_grad():hidden=m[:6](flat)
weight=m[-1].weight[ids].detach().clone().requires_grad_(True);bias=m[-1].bias[ids].detach().clone().requires_grad_(True)
out=(hidden@weight.T+bias).clamp(-3,3).reshape(len(time),16,2)
means=torch.stack([(out[lo:hi,robot]*torch.tensor(w[:,None],dtype=out.dtype)).sum(0) for robot,lo,hi,w in cycle_indices])
band=(5*np.pi/180)/.25;excess=(means.abs()-band).clamp_min(0)
loss=.025*torch.where(excess<=1,.5*excess.square(),excess-.5).mean()
gw,gb=torch.autograd.grad(loss,[weight,bias]);gcycle=torch.cat([gw.flatten(),gb.flatten()])
def gradient_feature(y,features,der):return torch.cat([(der.T@features).flatten(),der.sum(0).flatten()])
def cosine(a,b):return float((a@b/(a.norm()*b.norm()).clamp_min(1e-12)).clamp(-1,1))
gradient_rows=[];training_summary=[]
for file in sorted(OUT.glob('current_gradient_*.pt')):
    d=torch.load(file,map_location='cpu',weights_only=False);bm=model(d['state']);x=d['mlp_input'];raw=d['raw_observations'].reshape(-1,5,80)
    with torch.no_grad():h=bm[:6](x);y=bm[-1](h)
    assert float((y-d['mean']).abs().max())<6e-5
    speed=raw[:,-1,6];gppo=gradient_feature(y,h,d['incoming_PPO_gradient'][:,ids])
    stride=max(1,(len(y)+255)//256);selected=torch.arange(0,len(y),stride)[:256]
    upper=y[selected][:,d['upper_indices']].detach().requires_grad_(True)
    oldloss=posture_loss(upper)[0];ud=torch.autograd.grad(oldloss,upper)[0]
    shoulder_cols=[d['upper_indices'].index(i) for i in ids]
    gold=gradient_feature(y[selected],h[selected],ud[:,shoulder_cols])
    command_rows=[]
    for command in sorted(set(commands.tolist()))+["frontier"]:
        mask=(speed>3) if command=='frontier' else (speed-float(command)).abs()<1e-6
        if not mask.any():continue
        sm=mask[selected];pder=d['incoming_PPO_gradient'][:,ids]*mask[:,None]
        gp=gradient_feature(y,h,pder);gu=gradient_feature(y[selected],h[selected],ud[:,shoulder_cols]*sm[:,None])
        eff=y[mask][:,ids].clamp(-3,3)*.25*180/np.pi
        phase=torch.remainder(torch.atan2(raw[mask,-1,78],raw[mask,-1,79]),2*torch.pi)
        phasebins=(phase*(8/(2*torch.pi))).long().clamp(0,7)
        counts=torch.bincount(phasebins,minlength=8)
        binmean=torch.stack([eff[phasebins==i].mean(0) if (phasebins==i).any() else torch.full((2,),float('nan')) for i in range(8)])
        command_rows.append(dict(command=command,samples=int(mask.sum()),existing_upper_selected=int(sm.sum()),
            actual_PPO_shoulder_final_row_norm=float(gp.norm()),existing_upper_shoulder_final_row_norm=float(gu.norm()),
            PPO_existing_upper_cosine=cosine(gp,gu),PPO_eval_cycle_gradient_cosine=cosine(gp,gcycle),
            phase_counts=counts.tolist(),phase_balanced_effective_mean_deg=binmean.mean(0).tolist(),
            mean_per_state_absolute_effective_offset_deg=eff.abs().mean(0).tolist(),
            same_truthful_command_all_history_fraction=float(((raw[mask,:,6]-speed[mask,None]).abs()<1e-6).all(1).float().mean()),
            target_derivative_scope='Final two rows before global clipping/Adam; cross-dataset cycle gradient from saved deterministic evaluation, not an actual PPO loss or causal intervention.'))
    gradient_rows.append(dict(batch=file.name,existing_shoulder_norm=float(gold.norm()),actual_PPO_shoulder_norm=float(gppo.norm()),
        old_cycle_cosine=cosine(gold,gcycle),PPO_cycle_cosine=cosine(gppo,gcycle),per_command=command_rows))
# Counterexamples explain why a population/temporal average can hide drift.
t=torch.linspace(0,2*torch.pi,64+1)[:-1]
examples={"quiet":torch.zeros(64,2),"zero_mean_opposed_swing":torch.stack([torch.sin(t),-torch.sin(t)],1),
          "held_opposite21deg":torch.tensor([-1.5,1.5]).expand(64,2),
          "opposite_robots_cancellation":torch.stack([torch.tensor([-1.5,1.5]),torch.tensor([1.5,-1.5])])}
assert examples['zero_mean_opposed_swing'].mean(0).abs().max()<1e-6
assert examples['opposite_robots_cancellation'].mean(0).abs().max()==0
assert examples['opposite_robots_cancellation'].square().mean()>2
fixtures=3
result=dict(checkpoint=pause['checkpoint'],sha256=pause['checkpoint_sha256'],CPU_action_replay_max_error=error,
    synthetic_counterexamples_passed=fixtures,cycles=len(cycle_rows),per_command=summary,
    proposed_cycle_gradient_loss=float(loss),proposed_cycle_gradient_norm=float(gcycle.norm()),gradient_batches=gradient_rows,
    scope='16 representative robots at50Hz; one or two percommand, first-episode complete8-bin cycles only,5-17s. Population10Hz inputs captured separately. Exact native .25rad/action clamp. Actual angle/physical target samples occur20ms after input; time means cannot establish motor identification. Existing upper auxiliary on its actual256-row subsample versus incoming actual PPO derivative. Frozen snapshot features are detached for proposed2-row cycle VJP. No trunk, head or lower-row derivative and no diagnostic loss deployed.',
    qualification='No motion-likeness, world-gaze, policy-fault, gradient-interference or convergence claim; matched learning/physical controls still required before a continuation.',production_changed=False)
(OUT/'individual_cycle_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(replay_max_error=error,cycles=len(cycle_rows),per_command=summary,gradient_batches=len(gradient_rows)),indent=2))
