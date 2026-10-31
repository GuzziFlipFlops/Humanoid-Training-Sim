from pathlib import Path
OUT=Path(__file__).resolve().parent;old=OUT.parent/'adaptive_check_20261005_1142/physical_takeoff_probe.py'
b=old.read_text()
b=b.replace("drain=json.loads((OUT/'diagnostic_pause.json').read_text())","drain=json.loads((OUT/'initial_evidence.json').read_text());drain['run']=drain['active']['run'];drain['worker_drained']=not drain['processes']")
b=b.replace("FROZEN=RUN/'source/physical_v29'","FROZEN=OUT/'offline_source/physical_v29'")
b=b.replace("out=OUT/'physical_response'","out=OUT/'physical_speed_response'")
start=b.index('    # name,command,');end=b.index('    cfg=LocomotionCfg()',start)
b=b[:start]+"    # Only truthful commanded speed changes; no pulse or input rewrite.\n    groups=[(f'speed_{speed}',speed,0.,.40,0.) for speed in [.65,1.,1.5,1.75,2.,2.5]]\n    per=32;n=len(groups)*per\n"+b[end:]
b=b.replace('cfg.stage_speed=1.5','cfg.stage_speed=2.5')
token='    raw=LocomotionEnv(cfg);raw.initialize_randomization();env=RslRlVecEnvWrapper(raw,clip_actions=3.)'
assert b.count(token)==1;b=b.replace(token,"    from offline_ground import install_local_ground\n    install_local_ground()\n"+token)
start=b.index('        action=action.clone()');end=b.index('        observation,reward,done,_=env.step(action)',start)
b=b[:start]+"        phase=raw.phase.clone()\n"+b[end:]
token='    reward_terms={}'
insert='''    motor_names=[];motor_units=[];pair_indices=set()
    for key,ids,_ in raw.actuator.pairs:
        pair_indices.update(ids);motor_names.extend([key+'_motor_1',key+'_motor_2']);motor_units.extend(['CAD output motor rad/s']*2)
    ordinary=[i for i in range(23) if i not in pair_indices]
    motor_names.extend([raw.actuator.joint_names[i] for i in ordinary]);motor_units.extend(['native output joint rad/s']*len(ordinary))
    substep_ratios=[];substep_speed_clip=[];original_compute=raw.actuator.compute
    def passive_compute(control_action,joint_pos,joint_vel):
        result=original_compute(control_action,joint_pos,joint_vel)
        ratios=[];clipped=[]
        for key,ids,_ in raw.actuator.pairs:
            state=raw.actuator.last_motor_states[key]
            cap=raw.actuator.records[raw.actuator.joint_names[ids[0]]]['motor_speed_cap_RPM']*2*torch.pi/60
            velocity=state['motor_velocity'];requested=state['motor_requested'];applied=state['motor_applied']
            ratios.append(velocity.abs()/cap)
            clipped.append((velocity.abs()>=cap)&(requested*velocity>0)&(applied.abs()<1e-8)&state['valid'][:,None])
        velocity=joint_vel[:,ordinary];request=raw.actuator.computed_effort[:,ordinary]
        ratios.append(velocity.abs()/raw.actuator._speed[ordinary])
        clipped.append((velocity.abs()>raw.actuator._speed[ordinary]+1e-6)&(request*velocity>0)&(~raw.actuator._linear[ordinary])[None,:])
        substep_ratios.append(torch.cat(ratios,1).detach().cpu().numpy().copy())
        substep_speed_clip.append(torch.cat(clipped,1).detach().cpu().numpy().copy())
        return result
    raw.actuator.compute=passive_compute
'''
assert b.count(token)==1;b=b.replace(token,insert+token)
token='    np.savez_compressed(out/\'telemetry.npz\''
idx=b.index(token)
insert='''    ratios=np.asarray(substep_ratios);clips=np.asarray(substep_speed_clip)
    assert len(ratios)==4000,'Expected four 5ms actuator calls per 20ms control step'
    motor_rows=[]
    for i,g in enumerate(groups):
        ids=slice(i*per,(i+1)*per);x=ratios[1000:,ids];c=clips[1000:,ids]
        streak=np.zeros((per,len(motor_names)),dtype=np.int32);longest=streak.copy()
        for tick in c:streak=np.where(tick,streak+1,0);longest=np.maximum(longest,streak)
        motors=[dict(name=name,unit=motor_units[j],p99_cap_ratio=float(np.quantile(x[:,:,j],.99)),
            maximum_cap_ratio=float(x[:,:,j].max()),speed_clipped_fraction=float(c[:,:,j].mean()),
            maximum_contiguous_speed_clip_seconds=float(longest[:,j].max()*.005)) for j,name in enumerate(motor_names)]
        motor_rows.append(dict(command=g[1],motors=motors))
    np.savez_compressed(out/'substep_motor_200hz.npz',cap_ratio=ratios,speed_clip=clips,
        motor_names=np.asarray(motor_names),motor_units=np.asarray(motor_units),dt=.005,robots_per_group=per)
    (out/'substep_motor_audit.json').write_text(json.dumps(dict(completed=True,sample_rate_Hz=200,
        total_substeps=len(ratios),commands=motor_rows,source='Exact deployed HardwareActuator last_motor_states/Jacobians; passive post-compute copies',
        endpoint_proven=False,scope='This actor is trained only through 1.5 commands. Extrapolation/falls/torque failure never establish a hardware velocity ceiling.'),indent=2)+'\\n')
'''
b=b[:idx]+insert+b[idx:]
b=b.replace('normal_motor_limits=True,root_impulses=False,pose_writes=False,','normal_motor_limits=True,root_impulses=False,pose_writes=False,observation_rewrites=False,action_pulses=False,')
b=b.replace('"""Controlled normal-actuator interventions; never a qualification or trainer."""','"""Truthful speed-response and passive 5ms motor audit; diagnostic only."""')
(OUT/'physical_speed_probe.py').write_text(b);compile(b,str(OUT/'physical_speed_probe.py'),'exec');print('Prepared truthful speed/substep probe; not run yet')
