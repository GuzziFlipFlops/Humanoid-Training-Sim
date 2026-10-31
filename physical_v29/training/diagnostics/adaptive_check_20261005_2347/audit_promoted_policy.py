"""CPU replay of actual saved inputs, prior-label provenance and native motor mapping.

No simulator, CUDA, production hook, optimizer update or observation rewrite.
Counterfactual teacher inputs are evaluated separately from truthful actor replay.
"""
import json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import psutil

OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=read(OUT/'initial_gate_review.json');RUN=Path(review['run']);FROZEN=RUN/'source/physical_v29'
sys.path[:0]=[str(FROZEN),str(FROZEN/'training')]
from cad_differential import CadDifferential
from teacher_head_context import context_inputs
torch.set_num_threads(2)
assert psutil.virtual_memory().available>2500*2**20

def inverse(q,v):
    t=2*np.cross(q[...,1:],v)
    return v-q[...,:1]*t+np.cross(q[...,1:],t)

def actor_from(cp):
    saved=torch.load(cp,map_location='cpu',weights_only=False);state=saved['actor_state_dict']
    indexes=sorted(int(k.split('.')[1]) for k in state if k.startswith('mlp.') and k.endswith('weight'))
    layers=[]
    for i in indexes:
        w=state[f'mlp.{i}.weight'];layers.append(torch.nn.Linear(w.shape[1],w.shape[0]))
        if i!=indexes[-1]:layers.append(torch.nn.ELU())
    actor=torch.nn.Sequential(*layers).eval()
    actor.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')})
    return actor,saved

def reconstruct(folder,actor):
    with np.load(folder/'gait_diagnostics_50hz.npz') as z:d={k:z[k].copy() for k in z.files}
    assert not d['done'].any(),'Do not guess reset histories'
    assert np.array_equal(d['observed_command'],d['action_command'])
    names=d['joint_names'].tolist();m=np.load(FROZEN/'training/assets/motion_reference.npz')
    stand=m['stand'][[m['names'].tolist().index(n) for n in names]]
    root=d['root_state'];frames=np.zeros((len(root),root.shape[1],80),np.float32)
    frames[1:,:,:3]=inverse(root[:-1,:,3:7],root[:-1,:,10:13])*.2
    frames[1:,:,3:6]=inverse(root[:-1,:,3:7],np.broadcast_to(np.array([0,0,-1.],np.float32),root[:-1,:,10:13].shape))
    frames[:,:,6:9]=d['action_command'];frames[1:,:,9:32]=d['q'][:-1]-stand
    frames[1:,:,32:55]=d['qd'][:-1]*.05;frames[1:,:,55:78]=d['applied_action'][:-1]
    frequency=np.where(d['commands']>=1.5,.9+.5*d['commands'],1.25)
    # Native float32 phase clock, action i observes the previous post-step state.
    phase=np.remainder(np.arange(len(root),dtype=np.float32)[:,None]*np.float32(.02)/(1/frequency).astype(np.float32)[None,:],1.)
    frames[:,:,78]=np.sin(np.float32(2*np.pi)*phase)*(d['commands']>.1)
    frames[:,:,79]=np.cos(np.float32(2*np.pi)*phase)*(d['commands']>.1)
    ticks=np.arange(5,len(root));histories=np.stack([frames[i-4:i+1].transpose(1,0,2).reshape(root.shape[1],400) for i in ticks])
    keep=(d['time'][ticks,None]>=2.)&np.broadcast_to(d['commands'][None,:]>.1,(len(ticks),len(d['commands'])))
    ids=np.flatnonzero(keep.ravel());obs=torch.from_numpy(histories.reshape(-1,400)[ids]).clone()
    actual=torch.from_numpy(d['raw_action'][ticks].reshape(-1,23)[ids])
    with torch.inference_mode():pred=actor(obs);error=float((actual-pred).abs().max())
    assert error<2e-4,('Native input reconstruction mismatch',folder.name,error)
    commands=np.broadcast_to(d['commands'][None,:],keep.shape).reshape(-1)[ids]
    return d,names,obs,actual,commands,error

context=read(FROZEN/'training/teacher_head_context.json')
promoted=review['promotions'][-1]['matching_pair'];latest=review['latest']
teacher_cp=Path(promoted['checkpoint']);teacher,teacher_saved=actor_from(teacher_cp)
plans=[('qualified_1.5',promoted),('current_2.0',latest)]
records=[]
for label,pair in plans:
    folder=Path(pair['nominal']);actor,saved=actor_from(pair['checkpoint'])
    assert not read(folder/'agent.json')['actor']['obs_normalization']
    d,names,obs,actual,commands,error=reconstruct(folder,actor)
    changed=obs.clone();f=changed.reshape(-1,5,80);eligibility=torch.zeros(len(obs),dtype=torch.bool)
    for command,values in context['contexts'].items():
        command=float(command)
        if command>1.:continue
        eligible=((f[:,:,6]-command).abs()<1e-6).all(1)
        eligible&=(f[:,:,78:80].square().sum(-1)>.5).all(1)
        eligible&=(f[:,-1,context['columns'][:2]].abs()<.174533).all(1)
        selected=eligible.nonzero().flatten();eligibility|=eligible
        if len(selected):changed[selected]=context_inputs(changed[selected],torch.tensor(values,dtype=obs.dtype),context['columns'])
    upper=['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']
    lower=[i for i,n in enumerate(names) if n not in upper]
    with torch.inference_mode():truth=teacher(obs).clamp(-3,3);conditioned=teacher(changed).clamp(-3,3)
    rows=[]
    for command in sorted(set(commands)):
        mask=torch.from_numpy(np.isclose(commands,command));a=actual[mask].clamp(-3,3)
        delta=(conditioned[mask]-truth[mask])[:,lower]
        rows.append(dict(command_m_s=float(command),samples=int(mask.sum()),conditioned_fraction=float(eligibility[mask].float().mean()),
            clipped_lower_label_change_RMS=float(delta.square().mean().sqrt()),clipped_lower_label_change_max=float(delta.abs().max()),
            student_to_truthful_teacher_lower_RMS=float((a[:,lower]-truth[mask][:,lower]).square().mean().sqrt()),
            student_to_conditioned_teacher_lower_RMS=float((a[:,lower]-conditioned[mask][:,lower]).square().mean().sqrt()),
            context_changes_current_1_5_labels=bool(command==1.5 and delta.abs().max()>0)))
    assert all(not r['context_changes_current_1_5_labels'] for r in rows)
    frequency=np.where(d['commands']>=1.5,.9+.5*d['commands'],1.25)
    post=d['time']>=2.;phase=np.remainder(d['time'][:,None]*frequency[None,:],1.)
    flight=(d['clearance']>.001).all(-1)&(abs(d['foot_force'])<1).all(-1)
    events=[]
    for command in [1.5,2.]:
        ids=np.flatnonzero(np.isclose(d['commands'],command))
        if not len(ids):continue
        window=(np.remainder(phase[:,ids],.5)>=.43)&post[:,None]
        actualflight=flight[:,ids]&post[:,None];clearance=d['clearance'][:,ids].min(-1)
        run_lengths=[]
        for j in range(len(ids)):
            v=np.r_[False,actualflight[:,j],False];edges=np.diff(v.astype(int))
            run_lengths.extend((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)).tolist())
        events.append(dict(command_m_s=command,robots=len(ids),post_warmup_flight_fraction=float(flight[post][:,ids].mean()),
            intended_flight_loaded_foot_fraction=float((np.max(abs(d['foot_force'][:,ids]),-1)>1)[window].mean()),
            intended_window_mean_total_force_N=float(abs(d['foot_force'][:,ids]).sum(-1)[window].mean()),
            flight_sample_clearance_p50_mm=float(np.median(clearance[actualflight])*1000) if actualflight.any() else None,
            sampled_flight_event_ticks=run_lengths,sampled_flight_event_max_ms=20*max(run_lengths,default=0),
            raw_action_clamp_fraction=float((abs(d['raw_action'][post][:,ids])>3).mean())))
    records.append(dict(label=label,checkpoint=pair['checkpoint'],sha256=sha(pair['checkpoint']),trace=str(folder),
        actor_samples=len(obs),actor_reconstruction_max_error=error,teacher_labels=rows,phase_contacts=events))

# Pose-dependent CAD Jacobians on CURRENT full-population 10Hz telemetry.
z=np.load(Path(latest['nominal'])/'telemetry.npz');names=z['joint_names'].tolist();q=torch.from_numpy(z['q']).reshape(-1,23);qd=torch.from_numpy(z['qd']).reshape_as(q)
labels=np.tile(z['case_labels'],len(z['q']));specs=read(FROZEN/'assets/differential_geometry.json');caps=read(FROZEN/'assets/actuator_limits.json');motors={};paired=set();jacobian_checks=[]
for name,spec in specs.items():
    ix=[names.index(n) for n in spec['joint_names']];paired.update(spec['joint_names']);model=CadDifferential(spec)
    _,J,valid,closure=model.state(q[:,ix]);assert bool(valid.all()) and float(closure.max())<.01
    velocity=(J@qd[:,ix,None]).squeeze(-1).numpy();cap=caps[spec['joint_names'][0]]['motor_speed_cap_RPM']*2*np.pi/60
    for i in range(2):motors[f'{name}/motor_{i+1}']=(velocity[:,i],cap,'factory-geared output; exact J(q) qdot')
    sel=torch.arange(0,len(q),max(1,len(q)//64))[:64];pose=q[sel][:,ix].double();v=qd[sel][:,ix].double();h=1e-6
    plus=model.state(pose+h*v)[0];minus=model.state(pose-h*v)[0];finite=torch.atan2(torch.sin(plus-minus),torch.cos(plus-minus))/(2*h)
    analytic=(model.state(pose)[1]@v[...,None]).squeeze(-1);err=float(abs(finite-analytic).max());assert err<1e-5
    jacobian_checks.append(dict(pair=name,poses=len(sel),maximum_rad_s_error=err))
for i,name in enumerate(names):
    if name in paired:continue
    a=caps[name];reduction=a.get('reduction',a.get('physical_external_reduction',1.))
    motors[name]=(qd[:,i].numpy()*reduction,a['velocity_rad_s']*reduction,'native joint times declared reduction; cap at same shaft')
motor_rows=[]
for ci,speed in enumerate(z['case_speeds']):
    ids=labels==ci;items=[]
    for name,(v,cap,units) in motors.items():
        ratio=abs(v[ids])/cap
        items.append(dict(motor=name,cap_rad_s=cap,p99_cap_ratio=float(np.quantile(ratio,.99)),maximum_cap_ratio=float(ratio.max()),
            at_or_above_cap_fraction=float((ratio>=1).mean()),above_90_percent_fraction=float((ratio>=.9).mean()),units=units))
    motor_rows.append(dict(command_m_s=float(speed),samples_per_motor=int(ids.sum()),motors=sorted(items,key=lambda a:a['p99_cap_ratio'],reverse=True)))
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=True,records=records,qualified_teacher=str(teacher_cp),
    qualified_teacher_sha256=sha(teacher_cp),condition_table_sha256=sha(FROZEN/'training/teacher_head_context.json'),
    condition_table_metadata={k:v for k,v in context.items() if k!='contexts'},motor_mapping=motor_rows,jacobian_checks=jacobian_checks,
    production_change=False,actual_PPO_advantage_gradients_measured=False,
    scope='CPU actual observed-input replay and fixed-state teacher-label sensitivity only. Source applies old head context to the new neutral-head teacher for <=1 commands. This changes labels but not 1.5 labels; it does not establish causation or justify repair without physical/matched learning tests. Current full-population motor samples are 10Hz, not driving-clip attribution or a sustained velocity ceiling. Phase contacts use the bounded 16-robot 50Hz trace, not all-population qualification; event times are sampled.')
(OUT/'promoted_policy_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(passed=True,records=[{k:r[k] for k in ['label','actor_samples','actor_reconstruction_max_error','teacher_labels','phase_contacts']} for r in records],
    motor_top=[dict(command=r['command_m_s'],top=r['motors'][:3]) for r in motor_rows[-2:]]),indent=2))
