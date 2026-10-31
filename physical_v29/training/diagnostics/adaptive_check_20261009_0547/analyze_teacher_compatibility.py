"""CPU paired physical/label/motor analysis; no locomotion gate substitution."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
result=read(OUT/'teacher_compatibility/result.json')
with np.load(OUT/'teacher_compatibility/telemetry_50hz.npz') as z:
    data={k:z[k] for k in z.files}
with np.load(OUT/'teacher_compatibility/teacher_targets_10hz.npz') as z:
    label={k:z[k] for k in z.files}
groups=result['groups'];per=int(data['robots_per_group']);names=data['joint_names'].tolist()
upper=[names.index(k) for k in ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch',
    'left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']]
lower=label['lower_indices'];shoulder=[names.index('left_shoulder_pitch'),names.index('right_shoulder_pitch')]
checks=[];rows=[];target_rows=[]
def require(c,n):
    assert c,n
    checks.append(n)
require(result['maximum_action_command_difference']==0,'Observed and action commands identical')
require(result['maximum_reward_term_sum_error']<1e-3,'All reward terms reconstruct full reward')
require(result['maximum_physical_target_reconstruction_error_rad']<1e-6,'Native physical target map identical')
require(result['teacher_calls_outside_certificate']==0,'No teacher invocation above3m/s certificate')
require(not result['prescribed_body_action_replay'] and not result['production_controller_deployed'],
    'Own-state closed-loop diagnostic only; no donor-body replay or production hybrid')
require(per==32 and len(groups)==36,'1152 robots with32 independent states/cohort')
for key in ['q','qd','tau','target','root','com','foot_position','foot_velocity','force','clearance','current_raw_action','applied_raw_action']:
    require(np.isfinite(data[key]).all(),'Finite synchronized '+key)
for i,g in enumerate(groups):
    ids=slice(i*per,(i+1)*per)
    initial=np.concatenate([data['initial_q'][ids],data['initial_qd'][ids]],1)
    require(len({x.tobytes() for x in initial})==32,g['name']+':32 individually seeded states')
    control_i=next(j for j,c in enumerate(groups) if c['seed']==g['seed'] and c['command']==g['command']
        and c['initial_command']==g['initial_command'] and c['transition']==g['transition'] and c['mode']=='current')
    other=slice(control_i*per,(control_i+1)*per)
    require(np.array_equal(data['initial_q'][ids],data['initial_q'][other]) and
        np.array_equal(data['initial_qd'][ids],data['initial_qd'][other]),g['name']+':paired current control states')
    raw=data['applied_raw_action'][::5,ids]
    if g['mode']=='current':expected=label['current_raw'][:,ids]
    elif g['mode']=='teacher':expected=label['truthful_teacher_raw'][:,ids]
    elif g['mode']=='context_teacher':expected=label['context_teacher_raw'][:,ids]
    elif g['mode']=='current_body_teacher_upper':
        expected=label['current_raw'][:,ids].copy();expected[:,:,upper]=label['truthful_teacher_raw'][:,ids][:,:,upper]
    else:
        expected=label['current_raw'][:,ids].copy()
        selected=label['context_teacher_raw'] if g['mode']=='context_teacher_body_current_upper' else label['truthful_teacher_raw']
        expected[:,:,lower]=selected[:,ids][:,:,lower]
    require(np.array_equal(raw,expected),g['name']+':selected action equals same-own-state controller rows')
    if g['mode']!='current':require(label['teacher_label_valid'][:,ids].all(),g['name']+':teacher only inside certificate')
    window=data['time'] >= (8. if g['transition'] else 2.)
    alive=data['alive'][window,ids];root=data['root'][window,ids]
    heading=data['reference_heading'][ids][None,:]
    forward=root[:,:,7]*np.cos(heading)+root[:,:,8]*np.sin(heading)
    lateral=-root[:,:,7]*np.sin(heading)+root[:,:,8]*np.cos(heading)
    flight=(data['clearance'][window,ids]>.001).all(-1)&(data['force'][window,ids]<1.).all(-1)&alive
    per_robot=lambda x:np.where(alive,x,np.nan).mean(0)
    quant=lambda x:dict(mean=float(np.nanmean(x)),p05=float(np.nanquantile(x,.05)),p95=float(np.nanquantile(x,.95)))
    q=data['q'][window,ids];target=data['target'][window,ids]
    row=dict(name=g['name'],seed=g['seed'],command=g['command'],mode=g['mode'],transition=g['transition'],
        survival=g['survival'],first_fall_seconds=float(data['time'][np.flatnonzero(~data['alive'][:,ids].all(1))[0]])
            if not data['alive'][:,ids].all() else None,
        signed_speed_m_s=quant(per_robot(forward)),absolute_lateral_m_s=quant(per_robot(abs(lateral))),
        genuine_flight_fraction=quant(flight.mean(0)),reward_rate=quant(per_robot(data['reward_rate'][window,ids])),
        actual_shoulder_mean_deg=np.rad2deg(q[:,:,shoulder].mean((0,1))).tolist(),
        physical_shoulder_target_mean_deg=np.rad2deg(target[:,:,shoulder].mean((0,1))).tolist(),
        head_RMS_deg=float(np.rad2deg(np.sqrt(np.square(q[:,:,upper[:2]]).sum(-1).mean()))),
        full_population=True,survivor_conditioned_motion_statistics=not bool(alive.all()))
    row['paired_control_difference']={}
    control_root=data['root'][window,other];c_forward=control_root[:,:,7]*np.cos(heading)+control_root[:,:,8]*np.sin(heading)
    c_lateral=-control_root[:,:,7]*np.sin(heading)+control_root[:,:,8]*np.cos(heading)
    c_flight=(data['clearance'][window,other]>.001).all(-1)&(data['force'][window,other]<1.).all(-1)&data['alive'][window,other]
    if alive.all() and data['alive'][window,other].all():
        for key,values in [('signed_speed',forward-c_forward),('absolute_lateral',abs(lateral)-abs(c_lateral)),
                ('flight',flight.astype(float)-c_flight.astype(float)),('complete_reward_rate',data['reward_rate'][window,ids]-data['reward_rate'][window,other])]:
            row['paired_control_difference'][key]=quant(values.mean(0))
    rows.append(row)
    for window_name,lo,hi in [('settled',8. if g['transition'] else 2.,20.),('switch_first_second',6.,7.)] if g['transition'] else [('settled',2.,20.)]:
        selected=(label['time']>=lo)&(label['time']<hi)
        good=label['alive'][selected,ids]&label['teacher_label_valid'][selected,ids]
        if not good.any():continue
        phase=label['phase'][selected,ids]%1.
        error=label['current_projected'][selected,ids][:,:,lower]-label['teacher_projected'][selected,ids][:,:,lower]
        context_error=label['current_projected'][selected,ids][:,:,lower]-label['context_teacher_projected'][selected,ids][:,:,lower]
        bins=[]
        for bucket in range(16):
            valid=good&(phase>=bucket/16)&(phase<(bucket+1)/16)
            bins.append(dict(phase_lo=bucket/16,samples=int(valid.sum()),lower_target_RMS_deg=float(np.rad2deg(np.sqrt(np.square(error)[valid].mean()))) if valid.any() else None))
        target_rows.append(dict(name=g['name'],seed=g['seed'],command=g['command'],mode=g['mode'],transition=g['transition'],window=window_name,
            lower_target_RMS_deg=float(np.rad2deg(np.sqrt(np.square(error)[good].mean()))),
            conditioned_lower_target_RMS_deg=float(np.rad2deg(np.sqrt(np.square(context_error)[good].mean()))),
            context_eligibility_fraction=float(label['context_eligible'][selected,ids][good].mean()),
            teacher_joint_clip_fraction=float(label['teacher_joint_clip'][selected,ids][:,:,lower][good].mean()),
            teacher_slew_fraction=float(label['teacher_slew'][selected,ids][:,:,lower][good].mean()),
            teacher_CAD_invalid_fraction=float(label['teacher_CAD_invalid'][selected,ids][:,:,lower][good].mean()),phase_bins=bins))
require(all(g['survival']==1 for g in groups if g['mode']=='current'),'All actual current controls survive')
record=dict(passed=True,checkpoint=result['checkpoint'],sha256=result['checkpoint_sha256'],
    teacher_checkpoint=result['teacher_checkpoint'],teacher_sha256=result['teacher_sha256'],rows=rows,
    validation_count=len(checks),validation_cases=checks,production_controllers_deployed=False,
    scope='Own-state closed-loop physical compatibility,paired seed/initial states.32nominal-physics states/cohort/two seeds. Full independent locomotion gates and learning stability are separate. Motion/reward averages conditional on survival where any robot falls.')
(OUT/'teacher_compatibility_analysis.json').write_text(json.dumps(record,indent=2)+'\n')
(OUT/'teacher_target_history_analysis.json').write_text(json.dumps(dict(checkpoint=result['checkpoint'],sha256=result['checkpoint_sha256'],
    rows=target_rows,teacher_outputs_above_certificate_used=False,clock='Pre-actionphase and next20ms physical response; not a proven optimal shift',
    scope='10Hz passive same-state current/teacher effective target projections through original action/joint/slew/CAD maps; teacher context eligibility is explicit. These instantaneous errors are not a validated learning gradient.'),indent=2)+'\n')
fig,ax=plt.subplots(4,2,figsize=(15,12),sharex=True)
colors={'current':'#222222','teacher':'#2563eb','teacher_body_current_upper':'#d97706','current_body_teacher_upper':'#16a34a'}
for col,speed in enumerate([.15,1.5]):
    for mode,color in colors.items():
        i=next(i for i,g in enumerate(groups) if g['seed']==4401 and not g['transition'] and g['command']==speed and g['mode']==mode)
        ids=slice(i*per,(i+1)*per);heading=data['reference_heading'][ids][None,:];root=data['root'][:,ids];live=data['alive'][:,ids]
        forward=root[:,:,7]*np.cos(heading)+root[:,:,8]*np.sin(heading)
        lateral=-root[:,:,7]*np.sin(heading)+root[:,:,8]*np.cos(heading)
        flight=(data['clearance'][:,ids]>.001).all(-1)&(data['force'][:,ids]<1.).all(-1)&live
        ax[0,col].plot(data['time'],np.where(live,forward,np.nan).mean(1),color=color,label=mode)
        ax[1,col].plot(data['time'],np.where(live,abs(lateral),np.nan).mean(1),color=color)
        bins=np.arange(0,len(data['time']),10)
        ax[2,col].plot(data['time'][bins],np.array([flight[b:b+10].mean()*100 for b in bins]),color=color)
        q=data['q'][:,ids][:,:,shoulder];ax[3,col].plot(data['time'],np.rad2deg(np.where(live[:,:,None],q,np.nan).mean(1)[:,0]),color=color)
    ax[0,col].set_title(f'Command{speed}m/s;seed4401;32paired robots/controller')
    ax[0,col].legend(fontsize=7);ax[3,col].set_xlabel('Simulatedtime(s)')
for row,label_text in enumerate(['Signedforwardm/s','Abslateral m/s','Trueflight %;0.2s bins','Actualleftshoulder pitchdeg']):
    ax[row,0].set_ylabel(label_text)
    for col in range(2):ax[row,col].grid(alpha=.2)
fig.suptitle('Qualified349/body/current-upper compatibility — original gains/caps,each controller own actual feedback\nSelected seed means;all individual/two-seed statistics saved. Diagnostic compositions never deployed.',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'teacher_compatibility.png',dpi=150);plt.close(fig)
del data,label
with np.load(OUT/'teacher_compatibility/motors_200hz.npz') as z:
    native={k:z[k] for k in z.files}
motor_names=native['motor_names'].tolist();ids=native['baseline_robot_ids'];motor_rows=[]
require(len(native['cap_ratio'])==4000,'4000 actual5ms native actuator calls')
require(np.all(native['applied']<=native['available_positive_Nm']+1e-4) and
    np.all(native['applied']>=-native['available_negative_Nm']-1e-4),'Applied motor torque within actual directional native envelope')
for i,g in enumerate(groups):
    samples=np.flatnonzero((ids>=i*per)&(ids<(i+1)*per))
    if not len(samples):continue
    ratio=native['cap_ratio'][:,samples];clips=native['driving_clip'][:,samples];alive=native['alive'][:,samples]
    longest=[]
    for motor in range(len(motor_names)):
        best=0
        for robot in range(len(samples)):
            x=clips[:,robot,motor]&alive[:,robot];edges=np.diff(np.r_[False,x,False].astype(np.int8))
            starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            if len(starts):best=max(best,int((ends-starts).max()))
        longest.append(best*.005)
    braking=(ratio>=1)&(native['requested'][:,samples]*native['velocity'][:,samples]<=0)&alive[:,:,None]
    motor_rows.append(dict(name=g['name'],seed=g['seed'],command=g['command'],mode=g['mode'],robots=len(samples),
        driving_clip_fraction=float((clips&alive[:,:,None]).sum()/max(int(alive.sum())*len(motor_names),1)),
        longest_driving_clip_seconds=max(longest),max_p99_cap_ratio=float(np.quantile(ratio[alive],.99,axis=0).max()),
        at_above_cap_braking_fraction=float(braking.mean()),peak_abs_applied_motor_Nm=float(abs(native['applied'][:,samples][alive]).max()),
        per_motor=[dict(name=name,p99_cap_ratio=float(np.quantile(ratio[:,:,j][alive],.99)),driving_clip_fraction=float(clips[:,:,j][alive].mean()),
            longest_driving_seconds=longest[j]) for j,name in enumerate(motor_names)]))
(OUT/'current_substep_motor_attribution.json').write_text(json.dumps(dict(passed=True,checkpoint=result['checkpoint'],sha256=result['checkpoint_sha256'],
    groups=motor_rows,sample_rate_Hz=200,actual_native_calls=4000,available_torque_scope='Actual frozen directional flat-current/hard-speed BLDC and servo linear envelopes. User-assumed model,not measured voltage/current/thermal curves.',
    requested_and_applied_torque_saved=True,directional_available_torque_saved=True,native_limits_changed=False,
    locomotion_relevant_ceiling_proven=False,scope='Exact current and qualified teacher/static body-current-upper diagnostic cohorts only. Sparse later gate samples cannot inherit this attribution. No4/5endpoint.'),indent=2)+'\n')
print(json.dumps([dict(name=r['name'],survival=r['survival'],speed=r['signed_speed_m_s']['mean'],lateral=r['absolute_lateral_m_s']['mean'],flight=r['genuine_flight_fraction']['mean'])
    for r in rows if not r['transition']],indent=2))
