"""Truthful speed-response and passive 5ms motor audit; diagnostic only."""
import argparse,hashlib,json,sys,time,traceback,copy
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
drain=json.loads((OUT/'diagnostic_pause.json').read_text())
RUN=Path(drain['run']);FROZEN=OUT/'control_source/physical_v29';TRAIN=FROZEN/'training'
sys.path[:0]=[str(TRAIN),str(FROZEN)]
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args)
import numpy as np,torch,psutil
torch.set_num_threads(4)
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment_unitree import LocomotionCfg,LocomotionEnv
from curriculum_provenance import training_reward_signature
from heading_control import yaw_from_quat,wrap_angle
import unitree_reward_port
from unitree_reward_port import gait_parameters
from stage_rewards import running_unload_quality

def main():
    assert drain['worker_drained']
    for proc in psutil.process_iter(['pid','name','cmdline']):
        a=proc.info['cmdline'] or []
        assert not any(Path(x).name=='worker_unitree.py' for x in a),'Another Isaac worker is active'
    out=OUT/'motor_retention_response';out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists(),'Preserve completed physical response'
    checkpoint=Path(drain['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==drain['checkpoint_sha256']
    converter=UrdfConverter(UrdfConverterCfg(asset_path=str(TRAIN/'assets/robot_training.urdf'),
        usd_dir=str(TRAIN/'assets/usd'),usd_file_name='robot_training.usd',fix_base=False,root_link_name='pelvis',
        merge_fixed_joints=True,make_instanceable=True,collision_from_visuals=False,self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
    # Same seeded initial joint states across actors and command laws.
    config_args=json.loads((checkpoint.parent/'run_config.json').read_text(encoding='utf-8'))['arguments']
    certificate=json.loads(Path(config_args['locomotion_prior_certificate']).read_text(encoding='utf-8'))
    actor_files={'current':checkpoint,'qualified_prior':Path(certificate['checkpoint'])}
    actor_hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in actor_files.items()}
    groups=[]
    for seed in [4491,4511]:
        for actor in ['current','qualified_prior']:
            for speed in ([.15,.35,.65,1.,1.5,3.,3.5] if actor=='current' else [.15,.35,.65,1.,1.5,3.]):
                groups.append((f'seed{seed}_{actor}_{speed}',speed,0.,.5,None,actor,seed,'baseline'))
    per=32;n=len(groups)*per
    cfg=LocomotionCfg();cfg.scene.num_envs=n;cfg.scene.env_spacing=3.;cfg.evaluation=True
    cfg.stage_speed=3.5;cfg.robot_usd=converter.usd_path;cfg.sim.device=args.device or 'cuda:0'
    cfg.seed=71;cfg.episode_length_s=30.
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity=2**22
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity=2**20
    cfg.sim.physx.gpu_max_rigid_contact_count=2**20
    from offline_ground import install_local_ground
    install_local_ground()
    raw=LocomotionEnv(cfg);raw.initialize_randomization();env=RslRlVecEnvWrapper(raw,clip_actions=3.)
    policies={}
    for label,file in actor_files.items():
        config=json.loads((file.parent/'agent.json').read_text())
        runner=OnPolicyRunner(env,config,log_dir=None,device=raw.device)
        saved=torch.load(file,map_location=raw.device,weights_only=False)
        if label=='current':
            assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)
        else:
            from standing_retention import validate_qualification
            from evaluation import GATE_VERSION
            validate_qualification(certificate,str(file),training_reward_signature(TRAIN),GATE_VERSION)
            assert actor_hashes[label]==certificate['checkpoint_sha256']
            (out/'qualified_teacher_receipt.json').write_text(json.dumps(dict(certificate=certificate,
                actor_sha256=actor_hashes[label],original_saved_signature=saved['infos']['reward_signature'],
                current_gate_signature=training_reward_signature(TRAIN),scope='Exact freshly gate-certified teacher; diagnostic inference only. This does not admit an experimental actor or authorize generic signature bypass.'),indent=2),encoding='utf-8')
        runner.load(str(file),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
        assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in saved['actor_state_dict'].items())
        policies[label]=copy.deepcopy(runner.get_inference_policy(raw.device))
    actor_labels=np.array([g[5] for g in groups]).repeat(per)
    masks={label:torch.tensor(actor_labels==label,device=raw.device) for label in policies}
    speeds=torch.tensor([g[1] for g in groups],device=raw.device).repeat_interleave(per)
    heading_gain=torch.tensor([g[3] for g in groups],device=raw.device).repeat_interleave(per)
    course_gain=torch.tensor([g[2] for g in groups],device=raw.device).repeat_interleave(per)
    requested_yaw=torch.tensor([float('nan') if g[4] is None else g[4] for g in groups],device=raw.device).repeat_interleave(per)
    original_update=raw._update_observation_command
    law_kind=np.array([g[7] for g in groups]).repeat(per)
    low_mask=torch.tensor(law_kind=='lowpass120ms',device=raw.device)
    cycle_mask=torch.tensor(law_kind=='cycle380ms',device=raw.device)
    filtered_velocity=None;velocity_ring=[];law_tick=-1
    def truthful_requested_commands():
        nonlocal filtered_velocity,law_tick,velocity_ring
        original_update()
        velocity=raw.robot.data.root_lin_vel_w[:,:2]
        if law_tick!=raw.common_step_counter:
            if filtered_velocity is None:filtered_velocity=velocity.clone()
            else:filtered_velocity=filtered_velocity+(1.-__import__('math').exp(-raw.step_dt/.12))*(velocity-filtered_velocity)
            velocity_ring.append(velocity.clone())
            if len(velocity_ring)>19:velocity_ring.pop(0)
            law_tick=raw.common_step_counter
        average=torch.stack(velocity_ring).mean(0)
        measured=torch.where(low_mask[:,None],filtered_velocity,velocity)
        measured=torch.where(cycle_mask[:,None],average,measured)
        heading_error=wrap_angle(raw.heading_target-yaw_from_quat(raw.robot.data.root_quat_w))
        course_error=wrap_angle(raw.heading_target-torch.atan2(measured[:,1],measured[:,0]))
        course_error=torch.where(measured.norm(dim=1)>.2,course_error,torch.zeros_like(course_error))
        rate=(heading_gain*heading_error+course_gain*course_error).clamp(-.2,.2)
        baseline=course_gain==0
        assert torch.equal(raw.commands[baseline,2],rate[baseline]),'Baseline heading command parity failed'
        raw.commands[:,2]=torch.where(raw.commands[:,:2].norm(dim=1)>.1,rate,torch.zeros_like(rate))
    raw._update_observation_command=truthful_requested_commands
    raw.configure_fixed_commands(speeds,yaw_from_quat(raw.robot.data.root_quat_w).clone())
    raw.reset()
    offsets=[];velocities=[]
    for g in groups:
        rng=np.random.default_rng(g[6])
        offsets.append(rng.uniform(-.008,.008,(per,23)))
        velocities.append(rng.uniform(-.01,.01,(per,23)))
    q=raw.stand.expand(n,-1).clone()+torch.tensor(np.concatenate(offsets),device=raw.device,dtype=raw.stand.dtype)
    qd=torch.tensor(np.concatenate(velocities),device=raw.device,dtype=q.dtype)
    assert ((q>=raw.target_lo)&(q<=raw.target_hi)).all()
    raw.robot.write_joint_state_to_sim(q,qd)
    raw.targets[:]=q;raw.previous_targets[:]=q;raw.previous_vel[:]=qd
    raw.scene.update(raw.step_dt)
    # Rebuild truthful reset history after the diagnostic initial-state write.
    raw.history.zero_();raw.critic_history.zero_()
    raw._command_tick=-1;raw._last_history_tick=-1
    filtered_velocity=None;velocity_ring=[];law_tick=-1
    observation=raw._get_observations();reference_heading=raw.heading_target.clone();alive=torch.ones(n,dtype=torch.bool,device=raw.device)
    initial_q=q.cpu().numpy().copy();initial_qd=qd.cpu().numpy().copy()
    assert len({r.tobytes() for r in np.concatenate([initial_q[:per],initial_qd[:per]],1)})==per

    motor_names=[];motor_units=[];pair_indices=set()
    for key,ids,_ in raw.actuator.pairs:
        pair_indices.update(ids);motor_names.extend([key+'_motor_1',key+'_motor_2']);motor_units.extend(['CAD output motor rad/s']*2)
    ordinary=[i for i in range(23) if i not in pair_indices]
    motor_names.extend([raw.actuator.joint_names[i] for i in ordinary]);motor_units.extend(['native output joint rad/s']*len(ordinary))
    substep_ratios=[];substep_speed_clip=[];motor_requests=[];motor_applied=[];motor_velocities=[];original_compute=raw.actuator.compute
    def passive_compute(control_action,joint_pos,joint_vel):
        result=original_compute(control_action,joint_pos,joint_vel)
        ratios=[];clipped=[];requested_all=[];applied_all=[];velocities_all=[]
        for key,ids,_ in raw.actuator.pairs:
            state=raw.actuator.last_motor_states[key]
            cap=raw.actuator.records[raw.actuator.joint_names[ids[0]]]['motor_speed_cap_RPM']*2*torch.pi/60
            velocity=state['motor_velocity'];requested=state['motor_requested'];applied=state['motor_applied']
            ratios.append(velocity.abs()/cap)
            requested_all.append(requested);applied_all.append(applied);velocities_all.append(velocity)
            clipped.append((velocity.abs()>=cap)&(requested*velocity>0)&(applied.abs()<1e-8)&state['valid'][:,None])
        velocity=joint_vel[:,ordinary];request=raw.actuator.computed_effort[:,ordinary]
        ratios.append(velocity.abs()/raw.actuator._speed[ordinary])
        applied=raw.actuator.applied_effort[:,ordinary]
        requested_all.append(request);applied_all.append(applied);velocities_all.append(velocity)
        clipped.append((velocity.abs()>=raw.actuator._speed[ordinary])&(request*velocity>0)&(applied*velocity<=0))
        substep_ratios.append(torch.cat(ratios,1).detach().cpu().numpy().copy())
        substep_speed_clip.append(torch.cat(clipped,1).detach().cpu().numpy().copy())
        motor_requests.append(torch.cat(requested_all,1).detach().cpu().numpy().copy())
        motor_applied.append(torch.cat(applied_all,1).detach().cpu().numpy().copy())
        motor_velocities.append(torch.cat(velocities_all,1).detach().cpu().numpy().copy())
        return result
    raw.actuator.compute=passive_compute
    substate={k:[] for k in ['joint_position','joint_velocity','target','phase']}
    original_scene_update=raw.scene.update
    physics_tick=0
    representative_ids=torch.arange(0,n,per,device=raw.device)
    def post_physics_update(dt):
        nonlocal physics_tick
        original_scene_update(dt)
        assert abs(dt-.005)<1e-9
        physics_tick+=1
        values=dict(joint_position=raw.robot.data.joint_pos[representative_ids],
            joint_velocity=raw.robot.data.joint_vel[representative_ids],target=raw.targets[representative_ids],phase=raw.phase[representative_ids])
        for k,v in values.items():substate[k].append(v.detach().cpu().numpy().copy())
    raw.scene.update=post_physics_update
    reward_terms={}
    history={k:[] for k in ['time','phase','root','q','qd','tau','target','raw_action','effective_action','force','clearance','contact','alive','reward_rate','unload_score','failures','reward_phase','observed_command','action_command','target_clipped','target_slew','torque_saturation','healthy']}
    failed=torch.zeros(n,dtype=torch.int64,device=raw.device)
    for tick in range(1000):
        t=tick*raw.step_dt
        observed_commands=observation['policy'].reshape(n,5,80)[:,-1,6:9].clone()
        with torch.inference_mode():
            action=torch.empty((n,23),device=raw.device)
            for label,policy in policies.items():action[masks[label]]=policy(observation)[masks[label]]
        phase=raw.phase.clone()
        observation,reward,done,_=env.step(action)
        done=done.bool()
        alive&=~done;failed+=done.long();m=raw.telemetry()
        legphase=torch.remainder(raw.reward_proxy.cad_metrics['phase'][:,None]+torch.tensor([0.,.5],device=raw.device),1.)
        _,duty=gait_parameters(raw.action_commands)
        swing=legphase>=duty[:,None]
        unload=running_unload_quality(m['foot_force'],swing,raw.total_weight)
        row=dict(time=t+raw.step_dt,phase=phase,root=raw.robot.data.root_state_w,
            q=raw.robot.data.joint_pos,qd=raw.robot.data.joint_vel,tau=raw.robot.data.applied_torque,
            target=raw.targets,raw_action=action,effective_action=raw.actions,force=m['foot_force'],
            clearance=m['clearance'],contact=m['contact'],alive=alive,reward_rate=reward/raw.step_dt,
            unload_score=unload,failures=done,reward_phase=raw.reward_proxy.cad_metrics['phase'],
            observed_command=observed_commands,action_command=raw.action_commands,
            target_clipped=raw.target_clipped,target_slew=raw.target_slew_limited,
            torque_saturation=(raw.actuator.computed_effort-raw.actuator.applied_effort).abs()>1e-3,
            healthy=raw.reward_port.last_diagnostics['healthy'])
        for k,v in raw.last_reward_terms.items():reward_terms.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
        for k,v in row.items():history[k].append(v.detach().cpu().numpy().copy() if torch.is_tensor(v) else v)
        if tick%100==0:print(json.dumps({'tick':tick,'alive':int(alive.sum()),'height':float(m['height'].mean())}),flush=True)
    data={k:np.asarray(v) for k,v in history.items()};window=data['time']>=5.
    command_difference=float(abs(data['observed_command']-data['action_command']).max())
    term_sum=sum(np.asarray(v) for v in reward_terms.values())
    reward_sum_error=float(abs(term_sum-data['reward_rate']).max())
    assert command_difference==0.,command_difference
    assert reward_sum_error<1e-3,reward_sum_error
    summaries=[]
    for i,g in enumerate(groups):
        ids=slice(i*per,(i+1)*per);cl=data['clearance'][window,ids];force=data['force'][window,ids]
        contact=data['contact'][window,ids];good=data['alive'][window,ids]
        geometric=(cl>.001).all(-1);actual=geometric&(force<1.).all(-1)&good
        body_velocity=np.array([unitree_reward_port.quat_apply_inverse(torch.from_numpy(q),torch.from_numpy(v)).numpy() for q,v in zip(data['root'][window,ids,3:7],data['root'][window,ids,7:10])])
        stats=dict(mean_lateral_body_velocity=float(body_velocity[:,:,1][good].mean()),mean_yaw_rate=float(data['root'][window,ids,12][good].mean()),mean_absolute_world_lateral_velocity=float(abs(data['root'][window,ids,8])[good].mean()),name=g[0],command=g[1],
            heading_feedback_gain=g[3],course_feedback_gain=g[2],requested_yaw_rate=g[4],actor=g[5],robots=per,
            yaw_command_saturation_fraction=float((np.abs(data['action_command'][window,ids,2])>=.199999).mean()),
            mean_absolute_yaw_command=float(abs(data['action_command'][window,ids,2]).mean()),survival=float(alive[ids].float().mean()),resets=int(failed[ids].sum()),
            force_threshold_flight_fraction=float((~contact.any(-1)&good).mean()),
            both_sole_above_1mm_fraction=float((geometric&good).mean()),
            both_sole_above_5mm_fraction=float(((cl>.005).all(-1)&good).mean()),
            unloaded_geometric_flight_fraction=float(actual.mean()),
            max_minimum_sole_clearance_m=float(cl.min(-1)[good].max()) if good.any() else None,
            mean_forward_world_speed=float((data['root'][window,ids,7]*np.cos(reference_heading[ids].cpu().numpy())[None,:]+data['root'][window,ids,8]*np.sin(reference_heading[ids].cpu().numpy())[None,:])[good].mean()) if good.any() else None,
            reward_terms={k:float(np.asarray(v)[window,ids][good].mean()) for k,v in reward_terms.items()},
            head_RMS_deg=float(np.rad2deg(np.sqrt((data['q'][window,ids][:,:,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]**2).sum(-1)[good].mean()))),
            target_clip_fraction=float(data['target_clipped'][window,ids].mean()),
            target_slew_fraction=float(data['target_slew'][window,ids].mean()),
            torque_saturation_fraction=float(data['torque_saturation'][window,ids].mean()),
            healthy_mean=float(data['healthy'][window,ids][good].mean()),
            mean_reward_rate=float(data['reward_rate'][window,ids][good].mean()) if good.any() else None,
            mean_unload_score=float(data['unload_score'][window,ids][good].mean()) if good.any() else None,
            action_clip_fraction=float((np.abs(data['raw_action'][window,ids])>3).mean()))
        summaries.append(stats)
    ratios=np.asarray(substep_ratios);clips=np.asarray(substep_speed_clip)
    assert len(ratios)==4000,'Expected four 5ms actuator calls per 20ms control step'
    motor_rows=[]
    for i,g in enumerate(groups):
        ids=slice(i*per,(i+1)*per);x=ratios[1000:,ids];c=clips[1000:,ids]
        streak=np.zeros((per,len(motor_names)),dtype=np.int32);longest=streak.copy()
        for tick in c:streak=np.where(tick,streak+1,0);longest=np.maximum(longest,streak)
        motors=[dict(name=name,unit=motor_units[j],p99_cap_ratio=float(np.quantile(x[:,:,j],.99)),
            maximum_cap_ratio=float(x[:,:,j].max()),speed_clipped_fraction=float(c[:,:,j].mean()),
            maximum_contiguous_speed_clip_seconds=float(longest[:,j].max()*.005)) for j,name in enumerate(motor_names)]
        motor_rows.append(dict(name=g[0],command=g[1],motors=motors))
    assert physics_tick==4000
    np.savez_compressed(out/'representative_joint_200hz.npz',**{k:np.asarray(v) for k,v in substate.items()},
        dt=.005,robots_per_group=per,groups=np.array([g[0] for g in groups]))
    np.savez_compressed(out/'substep_motor_200hz.npz',cap_ratio=ratios,speed_clip=clips,
        motor_names=np.asarray(motor_names),motor_units=np.asarray(motor_units),dt=.005,robots_per_group=per,requested_torque=np.asarray(motor_requests),applied_torque=np.asarray(motor_applied),native_velocity=np.asarray(motor_velocities))
    (out/'substep_motor_audit.json').write_text(json.dumps(dict(completed=True,sample_rate_Hz=200,
        total_substeps=len(ratios),commands=motor_rows,source='Exact deployed HardwareActuator last_motor_states/Jacobians; passive post-compute copies',
        endpoint_proven=False,scope='Current actor and qualified prior, unchanged fixed commands at two initial-state seeds. Actual requested/applied torque, native speed ratio and driving clips at5ms. No command/action/observation intervention; one test cannot prove a high-speed ceiling.'),indent=2)+'\n')
    np.savez_compressed(out/'telemetry.npz',**data,**{'term_'+k:np.asarray(v) for k,v in reward_terms.items()},reference_heading=reference_heading.cpu().numpy(),initial_q=initial_q,initial_qd=initial_qd,joint_names=np.array(raw.robot.joint_names),groups=np.array([g[0] for g in groups]),robots_per_group=per)
    result=dict(completed=True,actor_checkpoints={k:str(v) for k,v in actor_files.items()},actor_sha256=actor_hashes,
        paired_initial_states=True,independent_initial_states_per_cohort=per,initial_state_seeds=[4491,4511],
        original_signature_does_not_certify_external_command_law=False,diagnostic_actors_deployed=False,checkpoint=str(checkpoint),checkpoint_sha256=drain['checkpoint_sha256'],groups=summaries,
        gravity=True,fixed_base=False,normal_motor_limits=True,root_impulses=False,runtime_pose_writes=False,independent_initial_joint_state_writes=True,observation_rewrites=False,action_pulses=False,
        actor_loaded_exact=True,qualifies_locomotion=False,maximum_command_difference=command_difference,maximum_reward_term_sum_error=reward_sum_error,
        scope='832 physical robots,20s,32 paired independent initial states per cohort at each of two seeds. Exact current actor versus gate-certified prior; unchanged native fixed commands/heading feedback. Actual5ms requested/applied torque, motor speeds/driving clips and representative physical q/qd/target/phase. No action offsets,support,phase/input rewrite or learned diagnostic actor. This is controlled diagnosis, not complete independent locomotion qualification.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();(OUT/'physical_probe_failure.json').write_text(json.dumps({'error':traceback.format_exc()}));raise
    finally:app.app.close()
