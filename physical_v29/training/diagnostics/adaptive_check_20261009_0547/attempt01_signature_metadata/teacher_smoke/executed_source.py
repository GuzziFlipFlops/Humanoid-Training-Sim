"""Truthful closed-loop compatibility of qualified teacher body/current upper.

All compositions are diagnostic. No prescribed donor-robot action replay, runtime
pose writes, support, physical/reward/gain/cap changes or production controllers.
"""
import argparse,copy,hashlib,json,sys,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent
pause=json.loads((OUT/'diagnostic_pause.json').read_text())
TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();app=AppLauncher(args)
import numpy as np,torch,psutil
torch.set_num_threads(4)
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from curriculum_provenance import training_reward_signature
from standing_retention import validate_qualification
from evaluation import GATE_VERSION
from environment_unitree import LocomotionCfg,LocomotionEnv
from heading_control import yaw_from_quat
from teacher_head_context import context_inputs
from action_projection import project_targets,clone_observation_mapping
from diagnostic_controllers import cohorts,combine_actions
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert pause['worker_drained']
    for process in psutil.process_iter(['cmdline']):
        assert not any(Path(a).name=='worker_unitree.py' for a in process.info['cmdline'] or []), 'Live Isaac worker exists'
    out=OUT/('teacher_smoke' if args.smoke else 'teacher_compatibility')
    assert out.is_dir() and not (out/'result.json').exists()
    cp=Path(pause['checkpoint']);assert sha(cp)==pause['checkpoint_sha256']
    configuration=json.loads((cp.parent/'run_config.json').read_text())['arguments']
    cert=json.loads(Path(configuration['locomotion_prior_certificate']).read_text())
    teacher_cp=Path(cert['checkpoint']);signature=training_reward_signature(TRAIN)
    validate_qualification(cert,str(teacher_cp),signature,GATE_VERSION)
    assert cert['maximum_command_m_s']==3.
    converter=UrdfConverter(UrdfConverterCfg(asset_path=str(TRAIN/'assets/robot_training.urdf'),
        usd_dir=str(TRAIN/'assets/usd'),usd_file_name='robot_training.usd',fix_base=False,
        root_link_name='pelvis',merge_fixed_joints=True,make_instanceable=True,
        collision_from_visuals=False,self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
    groups=cohorts();per=2 if args.smoke else 32;n=len(groups)*per
    steps=100 if args.smoke else 1000;transition_tick=60 if args.smoke else 300
    cfg=LocomotionCfg();cfg.scene.num_envs=n;cfg.scene.env_spacing=3.
    cfg.evaluation=True;cfg.stage_speed=3.5;cfg.episode_length_s=30.
    cfg.robot_usd=converter.usd_path;cfg.sim.device=args.device or 'cuda:0';cfg.seed=71
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity=2**22
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity=2**20
    cfg.sim.physx.gpu_max_rigid_contact_count=2**20
    from offline_ground import install_local_ground
    install_local_ground();raw=LocomotionEnv(cfg);raw.initialize_randomization()
    env=RslRlVecEnvWrapper(raw,clip_actions=3.)
    config=json.loads((cp.parent/'agent.json').read_text())
    def load_policy(path):
        runner=OnPolicyRunner(env,copy.deepcopy(config),log_dir=None,device=raw.device)
        saved=torch.load(path,map_location=raw.device,weights_only=False)
        assert saved['infos']['reward_signature']==signature
        runner.load(str(path),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
        assert all(torch.equal(value,runner.alg.actor.state_dict()[key]) for key,value in saved['actor_state_dict'].items())
        return copy.deepcopy(runner.get_inference_policy(raw.device))
    policy=load_policy(cp);teacher=load_policy(teacher_cp)
    context=json.loads((TRAIN/'teacher_head_context.json').read_text())
    tables={float(k):torch.tensor(v,device=raw.device) for k,v in context['contexts'].items() if float(k)<=1.}
    speeds=torch.tensor([g['initial_command'] for g in groups],device=raw.device).repeat_interleave(per)
    raw.configure_fixed_commands(speeds,yaw_from_quat(raw.robot.data.root_quat_w).clone());raw.reset()
    q0=[];qd0=[]
    for g in groups:
        for robot in range(per):
            rng=np.random.default_rng(np.random.SeedSequence([g['seed'],int(g['initial_command']*100),robot]))
            q0.append(rng.uniform(-.008,.008,23));qd0.append(rng.uniform(-.01,.01,23))
    q=raw.stand.expand(n,-1).clone()+torch.tensor(np.asarray(q0),device=raw.device,dtype=raw.stand.dtype)
    qd=torch.tensor(np.asarray(qd0),device=raw.device,dtype=q.dtype)
    assert ((q>=raw.target_lo)&(q<=raw.target_hi)).all()
    raw.robot.write_joint_state_to_sim(q,qd);raw.targets[:]=q;raw.previous_targets[:]=q;raw.previous_vel[:]=qd
    raw.scene.update(raw.step_dt);raw.history.zero_();raw.critic_history.zero_();raw._command_tick=-1;raw._last_history_tick=-1
    observation=raw._get_observations();reference_heading=raw.heading_target.clone()
    initial_q=q.cpu().numpy().copy();initial_qd=qd.cpu().numpy().copy()
    names=list(raw.robot.joint_names)
    upper=[names.index(k) for k in ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch',
        'left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']]
    lower=[i for i in range(23) if i not in upper]
    modes=[g['mode'] for g in groups]
    transitions=torch.tensor([g['transition'] for g in groups],device=raw.device).repeat_interleave(per)
    final_speeds=torch.tensor([g['command'] for g in groups],device=raw.device).repeat_interleave(per)
    alive=torch.ones(n,dtype=torch.bool,device=raw.device);failed=torch.zeros(n,dtype=torch.long,device=raw.device)
    update=raw._update_observation_command;switched=False
    def transition_before_observation():
        nonlocal switched
        if raw.common_step_counter>=transition_tick and not switched:
            raw.commands[transitions,0]=final_speeds[transitions];raw.fixed_evaluation_speed[transitions]=final_speeds[transitions];switched=True
        update()
    raw._update_observation_command=transition_before_observation
    native_ids=torch.tensor([i*per+r for i,g in enumerate(groups)
        if not g['transition'] and g['mode'] in ['current','teacher','teacher_body_current_upper'] for r in range(per)],device=raw.device)
    pair_indices=set();motor_names=[]
    for key,indices,model in raw.actuator.pairs:
        pair_indices.update(indices);motor_names += [key+'_motor_1',key+'_motor_2']
    ordinary=[i for i in range(23) if i not in pair_indices]
    motor_names += [raw.actuator.joint_names[i] for i in ordinary]
    native={k:[] for k in ['cap_ratio','driving_clip','requested','applied','velocity','alive','available_positive_Nm','available_negative_Nm']}
    original_compute=raw.actuator.compute
    def passive_compute(control,joint_pos,joint_vel):
        result=original_compute(control,joint_pos,joint_vel)
        ratios=[];clips=[];requests=[];applications=[];velocities=[];positive=[];negative=[]
        for key,indices,model in raw.actuator.pairs:
            state=raw.actuator.last_motor_states[key]
            cap=raw.actuator.records[raw.actuator.joint_names[indices[0]]]['motor_speed_cap_RPM']*2*torch.pi/60
            velocity,request,applied=state['motor_velocity'],state['motor_requested'],state['motor_applied']
            ratios.append(velocity.abs()/cap);requests.append(request);applications.append(applied);velocities.append(velocity)
            clips.append((velocity.abs()>=cap)&(request*velocity>0)&(applied.abs()<1e-8)&state['valid'][:,None])
            peak=raw.actuator.records[raw.actuator.joint_names[indices[0]]]['motor_peak_Nm']
            positive.append(torch.where((velocity<cap)&state['valid'][:,None],torch.full_like(velocity,peak),0.))
            negative.append(torch.where((velocity>-cap)&state['valid'][:,None],torch.full_like(velocity,peak),0.))
        velocity=joint_vel[:,ordinary];request=raw.actuator.computed_effort[:,ordinary];applied=raw.actuator.applied_effort[:,ordinary]
        ratios.append(velocity.abs()/raw.actuator._speed[ordinary]);requests.append(request);applications.append(applied);velocities.append(velocity)
        clips.append((velocity.abs()>=raw.actuator._speed[ordinary])&(request*velocity>0)&(applied*velocity<=0))
        peak=raw.actuator._peak[ordinary];speed=raw.actuator._speed[ordinary]
        linear_positive=(peak*(1-velocity/speed)).clamp(-peak,peak)
        linear_negative=-(peak*(-1-velocity/speed)).clamp(-peak,peak)
        positive.append(torch.where(raw.actuator._linear[ordinary],linear_positive.clamp_min(0.),
            torch.where(velocity>speed+1e-6,torch.zeros_like(velocity),peak)))
        negative.append(torch.where(raw.actuator._linear[ordinary],linear_negative.clamp_min(0.),
            torch.where(velocity<-speed-1e-6,torch.zeros_like(velocity),peak)))
        for key,value in [('cap_ratio',torch.cat(ratios,1)),('driving_clip',torch.cat(clips,1)),
            ('requested',torch.cat(requests,1)),('applied',torch.cat(applications,1)),('velocity',torch.cat(velocities,1)),
            ('available_positive_Nm',torch.cat(positive,1)),('available_negative_Nm',torch.cat(negative,1))]:
            native[key].append(value[native_ids].detach().cpu().numpy().copy())
        native['alive'].append(alive[native_ids].cpu().numpy().copy())
        return result
    raw.actuator.compute=passive_compute
    history={};labels={};terms={};parity=0.;teacher_calls_outside_certificate=0
    for tick in range(steps):
        t=tick*raw.step_dt;obs_command=observation['policy'].reshape(n,5,80)[:,-1,6:9].clone()
        qualified=(obs_command[:,:2].norm(dim=1)<=cert['maximum_command_m_s']+1e-6)
        ids=qualified.nonzero().flatten()
        with torch.inference_mode():
            current=policy(observation)
            truthful=current.clone();truthful[ids]=teacher({k:v[ids] for k,v in observation.items()})
            label_observation=clone_observation_mapping(observation)
            label_input=label_observation['policy'];frames=label_input.reshape(n,5,80)
            eligible_all=torch.zeros(n,dtype=torch.bool,device=raw.device)
            for command,table in tables.items():
                eligible=((frames[:,:,6]-command).abs()<1e-6).all(1)
                eligible &= (frames[:,:,78:80].square().sum(-1)>.5).all(1)
                eligible &= (frames[:,-1,context['columns'][:2]].abs()<.174533).all(1)
                eligible &= qualified
                selected=eligible.nonzero().flatten()
                if len(selected):label_input[selected]=context_inputs(label_input[selected],table,context['columns'])
                eligible_all |= eligible
            conditioned=current.clone();conditioned[ids]=teacher({k:v[ids] for k,v in label_observation.items()})
        assert bool(qualified[ids].all())
        action=combine_actions(current,truthful,conditioned,modes,per,upper,qualified)
        previous=raw.targets.clone();phase=raw.phase.clone()
        projected,clipped,slew,invalid=project_targets(action,raw.stand.expand(n,-1),raw.scales,raw.target_lo,raw.target_hi,
            previous,raw.slew,raw.step_dt,raw.actuator.pairs)
        if tick%5==0:
            true_target,true_clip,true_slew,true_invalid=project_targets(truthful,raw.stand.expand(n,-1),raw.scales,raw.target_lo,raw.target_hi,
                previous,raw.slew,raw.step_dt,raw.actuator.pairs)
            context_target,context_clip,context_slew,context_invalid=project_targets(conditioned,raw.stand.expand(n,-1),raw.scales,raw.target_lo,raw.target_hi,
                previous,raw.slew,raw.step_dt,raw.actuator.pairs)
            current_target=project_targets(current,raw.stand.expand(n,-1),raw.scales,raw.target_lo,raw.target_hi,
                previous,raw.slew,raw.step_dt,raw.actuator.pairs)[0]
            row=dict(time=t,phase=phase,observed_history=observation['policy'],previous_target=previous,
                current_raw=current,truthful_teacher_raw=truthful,context_teacher_raw=conditioned,
                current_projected=current_target,teacher_projected=true_target,context_teacher_projected=context_target,
                teacher_joint_clip=true_clip,teacher_slew=true_slew,teacher_CAD_invalid=true_invalid,
                context_teacher_joint_clip=context_clip,context_teacher_slew=context_slew,
                context_teacher_CAD_invalid=context_invalid,context_eligible=eligible_all,
                teacher_label_valid=qualified,alive=alive)
            for key,value in row.items():labels.setdefault(key,[]).append(value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value)
        observation,reward,done,_=env.step(action)
        done=done.bool();alive &= ~done;failed += done.long()
        if (~done).any():parity=max(parity,float((projected[~done]-raw.targets[~done]).abs().max()))
        m=raw.telemetry()
        row=dict(time=t+raw.step_dt,phase=phase,reward_phase=raw.reward_proxy.cad_metrics['phase'],
            root=raw.robot.data.root_state_w,q=raw.robot.data.joint_pos,qd=raw.robot.data.joint_vel,tau=raw.robot.data.applied_torque,
            target=raw.targets,current_raw_action=current,applied_raw_action=action,effective_action=raw.actions,
            force=m['foot_force'],clearance=m['clearance'],com=m['com'],foot_position=m['foot_position'],
            foot_velocity=m['foot_velocity'],contact=m['contact'],alive=alive,failures=done,reward_rate=reward/raw.step_dt,
            observed_command=obs_command,action_command=raw.action_commands,target_clipped=raw.target_clipped,target_slew=raw.target_slew_limited)
        for key,value in row.items():history.setdefault(key,[]).append(value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value)
        for key,value in raw.last_reward_terms.items():terms.setdefault(key,[]).append(value.detach().cpu().numpy().copy())
        if tick%100==0:print(json.dumps(dict(tick=tick,alive=int(alive.sum()),target_parity=parity)),flush=True)
    data={k:np.asarray(v) for k,v in history.items()};native_data={k:np.asarray(v) for k,v in native.items()}
    assert len(native_data['cap_ratio'])==steps*4
    command_error=float(np.abs(data['observed_command']-data['action_command']).max())
    sum_error=float(np.abs(sum(np.asarray(v) for v in terms.values())-data['reward_rate']).max())
    assert command_error==0 and sum_error<1e-3 and parity<1e-6
    summaries=[]
    for i,g in enumerate(groups):
        cohort=slice(i*per,(i+1)*per)
        window=data['time'] >= ((1.4 if g['transition'] else .8) if args.smoke else (8. if g['transition'] else 2.))
        good=data['alive'][window,cohort];root=data['root'][window,cohort]
        flight=(data['clearance'][window,cohort]>.001).all(-1)&(data['force'][window,cohort]<1.).all(-1)&good
        heading=reference_heading[cohort].cpu().numpy()[None,:]
        signed=root[:,:,7]*np.cos(heading)+root[:,:,8]*np.sin(heading)
        lateral=-root[:,:,7]*np.sin(heading)+root[:,:,8]*np.cos(heading)
        head=data['q'][window,cohort][:,:,upper[:2]]
        summaries.append(dict(**g,robots=per,survival=float(alive[cohort].float().mean()),resets=int(failed[cohort].sum()),
            signed_forward_m_s=float(signed[good].mean()) if good.any() else None,
            absolute_lateral_m_s=float(abs(lateral)[good].mean()) if good.any() else None,
            genuine_flight_fraction=float(flight.mean()),mean_reward_rate=float(data['reward_rate'][window,cohort][good].mean()) if good.any() else None,
            head_RMS_deg=float(np.rad2deg(np.sqrt(np.square(head).sum(-1)[good].mean()))) if good.any() else None,
            reward_terms={k:float(np.asarray(v)[window,cohort][good].mean()) if good.any() else None for k,v in terms.items()}))
    np.savez_compressed(out/'telemetry_50hz.npz',**data,**{'term_'+k:np.asarray(v) for k,v in terms.items()},
        groups=np.array([g['name'] for g in groups]),joint_names=np.array(names),robots_per_group=per,
        reference_heading=reference_heading.cpu().numpy(),initial_q=initial_q,initial_qd=initial_qd)
    np.savez_compressed(out/'teacher_targets_10hz.npz',**{k:np.asarray(v) for k,v in labels.items()},
        joint_names=np.array(names),groups=np.array([g['name'] for g in groups]),robots_per_group=per,lower_indices=np.array(lower))
    np.savez_compressed(out/'motors_200hz.npz',**native_data,motor_names=np.array(motor_names),
        baseline_robot_ids=native_ids.cpu().numpy(),dt=.005)
    result=dict(completed=True,smoke_only=args.smoke,checkpoint=str(cp),checkpoint_sha256=sha(cp),actor_loaded_exact=True,
        teacher_checkpoint=str(teacher_cp),teacher_sha256=sha(teacher_cp),teacher_maximum_command_m_s=3.,
        teacher_calls_outside_certificate=teacher_calls_outside_certificate,reward_signature=signature,groups=summaries,
        seeds=[4401,4411],independent_initial_states_per_cohort=per,
        maximum_action_command_difference=command_error,maximum_reward_term_sum_error=sum_error,
        maximum_physical_target_reconstruction_error_rad=parity,original_gains=True,native_caps=True,
        gravity=True,free_base=True,self_collision=True,student_observations_truthful=True,
        teacher_truthful_feedback_explicit=True,conditioned_teacher_experiment_explicit=True,
        production_controller_deployed=False,hybrid_qualification_claimed=False,locomotion_qualified=False,
        runtime_pose_writes=False,root_support=False,prescribed_body_action_replay=False,
        scope=f'{n} robots,{steps*raw.step_dt}s,two seeds with{per} individually seeded paired initial states/cohort. Each policy reads its own cohort actual joint/history feedback. Qualified349/current body-upper compatibility and existing teacher-head-label context are diagnostic only.3.5 sentinel uses current actor only; no teacher extrapolation or production hybrid.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=True,commands=command_error,target_parity=parity)),flush=True);env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'teacher_probe_failure.json').write_text(json.dumps(dict(error=traceback.format_exc())))
        raise
    finally:app.app.close()
