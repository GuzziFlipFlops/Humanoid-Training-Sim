"""Truthful speed-response and passive 5ms motor audit; diagnostic only."""
import argparse,hashlib,json,sys,time,traceback
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
from heading_control import yaw_from_quat
import unitree_reward_port
from unitree_reward_port import gait_parameters
from stage_rewards import running_unload_quality

def main():
    assert drain['worker_drained']
    for proc in psutil.process_iter(['pid','name','cmdline']):
        a=proc.info['cmdline'] or []
        assert not any(Path(x).name=='worker_unitree.py' for x in a),'Another Isaac worker is active'
    out=OUT/'arm_response';out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists(),'Preserve completed physical response'
    checkpoint=Path(drain['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==drain['checkpoint_sha256']
    converter=UrdfConverter(UrdfConverterCfg(asset_path=str(TRAIN/'assets/robot_training.urdf'),
        usd_dir=str(TRAIN/'assets/usd'),usd_file_name='robot_training.usd',fix_base=False,root_link_name='pelvis',
        merge_fixed_joints=True,make_instanceable=True,collision_from_visuals=False,self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
    # Only truthful commanded speed changes; no pulse or input rewrite.
    groups=[(f'{speed}_'+kind,speed,0.,damping,None) for speed in [1.,3.5] for kind,damping in [('control',1.),('damping2',2.),('damping4',4.)]]
    groups += [(f'0.0_sine{frequency}',0.,0.,1.,None) for frequency in [.75,1.5,2.66,3.25]]
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
    config=json.loads((checkpoint.parent/'agent.json').read_text())
    runner=OnPolicyRunner(env,config,log_dir=None,device=raw.device)
    saved=torch.load(checkpoint,map_location=raw.device,weights_only=False)
    assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)
    runner.load(str(checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
    assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in saved['actor_state_dict'].items())
    policy=runner.get_inference_policy(raw.device)
    speeds=torch.tensor([g[1] for g in groups],device=raw.device).repeat_interleave(per)
    requested_vy=torch.tensor([g[2] for g in groups],device=raw.device).repeat_interleave(per)
    requested_yaw=torch.tensor([float('nan') if g[4] is None else g[4] for g in groups],device=raw.device).repeat_interleave(per)
    original_update=raw._update_observation_command
    def truthful_requested_commands():
        original_update()
        raw.commands[:,1]=requested_vy
        raw.commands[:,2]=torch.where(torch.isfinite(requested_yaw),requested_yaw,raw.commands[:,2])
    raw._update_observation_command=truthful_requested_commands
    raw.configure_fixed_commands(speeds,yaw_from_quat(raw.robot.data.root_quat_w).clone())
    raw.reset();reference_heading=raw.heading_target.clone();observation=raw._get_observations();alive=torch.ones(n,dtype=torch.bool,device=raw.device)
    shoulder_ids=[raw.lookup[x+'_shoulder_pitch'] for x in ['left','right']]
    original_damping=raw.actuator.damping.clone()
    values=torch.tensor([g[3] for g in groups],device=raw.device).repeat_interleave(per)
    raw.actuator.damping[:,shoulder_ids]=values[:,None]
    assert float(original_damping[:,shoulder_ids].min())==1. and float(original_damping[:,shoulder_ids].max())==1.
    frequencies=torch.tensor([float(g[0].split('sine')[1]) if 'sine' in g[0] else 0. for g in groups],device=raw.device).repeat_interleave(per)
    original_pre=raw._pre_physics_step
    def diagnostic_sine_targets(actions):
        original_pre(actions)
        phase_t=(raw.episode_length_buf.float()+1)*raw.step_dt
        offsets=.04*torch.sin(2*torch.pi*frequencies*phase_t)*(frequencies>0)
        desired=raw.targets[:,shoulder_ids]+offsets[:,None]*torch.tensor([1.,-1.],device=raw.device)
        desired=torch.maximum(torch.minimum(desired,raw.target_hi[shoulder_ids]),raw.target_lo[shoulder_ids])
        desired=raw.previous_targets[:,shoulder_ids]+(desired-raw.previous_targets[:,shoulder_ids]).clamp(-raw.slew[shoulder_ids]*raw.step_dt,raw.slew[shoulder_ids]*raw.step_dt)
        raw.targets[:,shoulder_ids]=desired
        raw.delayed_targets[:,shoulder_ids]=torch.where(raw.domain[:,2:3]>.5,raw.previous_targets[:,shoulder_ids],desired)
    raw._pre_physics_step=diagnostic_sine_targets

    motor_names=[];motor_units=[];pair_indices=set()
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
        applied=raw.actuator.applied_effort[:,ordinary]
        clipped.append((velocity.abs()>=raw.actuator._speed[ordinary])&(request*velocity>0)&(applied*velocity<=0))
        substep_ratios.append(torch.cat(ratios,1).detach().cpu().numpy().copy())
        substep_speed_clip.append(torch.cat(clipped,1).detach().cpu().numpy().copy())
        return result
    raw.actuator.compute=passive_compute
    reward_terms={}
    history={k:[] for k in ['time','phase','root','q','qd','tau','target','raw_action','effective_action','force','clearance','contact','alive','reward_rate','unload_score','failures','reward_phase','observed_command','action_command','target_clipped','target_slew','torque_saturation','healthy','heading_error','stiffness','damping']}
    failed=torch.zeros(n,dtype=torch.int64,device=raw.device)
    for tick in range(1000):
        t=tick*raw.step_dt
        observed_commands=observation['policy'].reshape(n,5,80)[:,-1,6:9].clone()
        with torch.inference_mode():action=policy(observation)
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
            healthy=raw.reward_port.last_diagnostics['healthy'],heading_error=raw.reward_proxy.cad_metrics['heading_error_rad'],stiffness=raw.actuator.stiffness,damping=raw.actuator.damping)
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
            requested_lateral_velocity=g[2],requested_yaw_rate=g[4],shoulder_damping_Nms_rad=g[3],diagnostic_sine_frequency_Hz=float(g[0].split('sine')[1]) if 'sine' in g[0] else None,robots=per,survival=float(alive[ids].float().mean()),resets=int(failed[ids].sum()),
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
    np.savez_compressed(out/'substep_motor_200hz.npz',cap_ratio=ratios,speed_clip=clips,
        motor_names=np.asarray(motor_names),motor_units=np.asarray(motor_units),dt=.005,robots_per_group=per)
    (out/'substep_motor_audit.json').write_text(json.dumps(dict(completed=True,sample_rate_Hz=200,
        total_substeps=len(ratios),commands=motor_rows,source='Exact deployed HardwareActuator last_motor_states/Jacobians; passive post-compute copies',
        endpoint_proven=False,scope='Current3.5 actor,truthful yaw/lateral command interventions. Driving clips are sampled exactly at5ms; one test cannot prove a high-speed ceiling.'),indent=2)+'\n')
    np.savez_compressed(out/'telemetry.npz',**data,**{'term_'+k:np.asarray(v) for k,v in reward_terms.items()},reference_heading=reference_heading.cpu().numpy(),joint_names=np.array(raw.robot.joint_names),groups=np.array([g[0] for g in groups]),robots_per_group=per)
    result=dict(completed=True,checkpoint=str(checkpoint),checkpoint_sha256=drain['checkpoint_sha256'],groups=summaries,
        gravity=True,fixed_base=False,normal_motor_limits=True,root_impulses=False,pose_writes=False,observation_rewrites=False,action_pulses=False,diagnostic_shoulder_target_sines=True,diagnostic_shoulder_damping_changes=True,production_deployed=False,
        actor_loaded_exact=True,qualifies_locomotion=False,maximum_command_difference=command_difference,maximum_reward_term_sum_error=reward_sum_error,
        scope='320 physical robots,20s: normal-limit shoulder damping1/2/4 at1/3.5 and0.04rad stopped shoulder sine tests0.75/1.5/2.66/3.25Hz. Truthful student observations, unchanged leg control/free base/gravity/native motor caps/joint/slew limits. Diagnostic only; no changed gain, targets or actor deployed.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();(OUT/'physical_probe_failure.json').write_text(json.dumps({'error':traceback.format_exc()}));raise
    finally:app.app.close()
