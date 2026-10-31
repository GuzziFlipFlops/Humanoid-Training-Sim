"""Controlled normal-actuator interventions; never a qualification or trainer."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
drain=json.loads((OUT/'diagnostic_pause.json').read_text())
RUN=Path(drain['run']);FROZEN=RUN/'source/physical_v29';TRAIN=FROZEN/'training'
sys.path[:0]=[str(TRAIN),str(FROZEN)]
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args)
import numpy as np,torch,psutil,copy
torch.set_num_threads(4)
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment_unitree import LocomotionCfg,LocomotionEnv
from curriculum_provenance import training_reward_signature
from heading_control import yaw_from_quat

def main():
    assert drain['worker_drained']
    for proc in psutil.process_iter(['pid','name','cmdline']):
        a=proc.info['cmdline'] or []
        assert not any(Path(x).name=='worker_unitree.py' for x in a),'Another Isaac worker is active'
    out=OUT/'head_causal_response';out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists(),'Preserve completed physical response'
    checkpoint=Path(drain['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==drain['checkpoint_sha256']
    converter=UrdfConverter(UrdfConverterCfg(asset_path=str(TRAIN/'assets/robot_training.urdf'),
        usd_dir=str(TRAIN/'assets/usd'),usd_file_name='robot_training.usd',fix_base=False,root_link_name='pelvis',
        merge_fixed_joints=True,make_instanceable=True,collision_from_visuals=False,self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
    groups=[('latest_run',1.5,0.,0.,'latest',.40,0.),
            ('preserved_flight_actor',1.5,0.,0.,'flight',.40,0.),
            ('old_body_neutral_head_actual_feedback',1.5,0.,0.,'flight',.40,0.),
            ('old_body_neutral_head_reference_head_feedback',1.5,0.,0.,'flight',.40,0.)]
    per=32;n=len(groups)*per
    cfg=LocomotionCfg();cfg.scene.num_envs=n;cfg.scene.env_spacing=3.;cfg.evaluation=True
    cfg.stage_speed=1.5;cfg.robot_usd=converter.usd_path;cfg.sim.device=args.device or 'cuda:0'
    cfg.seed=71;cfg.episode_length_s=30.
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity=2**22
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity=2**20
    cfg.sim.physx.gpu_max_rigid_contact_count=2**20
    raw=LocomotionEnv(cfg);raw.initialize_randomization();env=RslRlVecEnvWrapper(raw,clip_actions=3.)
    config=json.loads((checkpoint.parent/'agent.json').read_text())
    runner=OnPolicyRunner(env,config,log_dir=None,device=raw.device)
    saved=torch.load(checkpoint,map_location=raw.device,weights_only=False)
    assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)
    runner.load(str(checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
    assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in saved['actor_state_dict'].items())
    actors={}
    files={'latest':checkpoint,'flight':HERE/'runs/locomotion_20261004_r34_bilateral_swing_unloading/038_run_150_train_007/candidate.pt'}
    for name,path in files.items():
        value=torch.load(path,map_location=raw.device,weights_only=False)
        assert value['infos']['reward_signature']==training_reward_signature(TRAIN)
        actor=copy.deepcopy(runner.alg.actor);actor.load_state_dict(value['actor_state_dict']);actor.eval()
        actors[name]=actor

    speeds=torch.tensor([g[1] for g in groups],device=raw.device).repeat_interleave(per)
    amplitude=torch.tensor([g[2] for g in groups],device=raw.device).repeat_interleave(per)
    shift=torch.tensor([g[3] for g in groups],device=raw.device).repeat_interleave(per)
    retract_start=torch.tensor([g[5] for g in groups],device=raw.device).repeat_interleave(per)
    extension=torch.tensor([g[6] for g in groups],device=raw.device).repeat_interleave(per)
    raw.configure_fixed_commands(speeds,yaw_from_quat(raw.robot.data.root_quat_w).clone())
    raw.reset();observation=raw._get_observations();alive=torch.ones(n,dtype=torch.bool,device=raw.device)
    reward_terms={}
    history={k:[] for k in ['time','phase','root','q','qd','tau','target','raw_action','effective_action','force','clearance','contact','alive','reward_rate','unload_score','failures','observations']}
    failed=torch.zeros(n,dtype=torch.int64,device=raw.device)
    for tick in range(1000):
        t=tick*raw.step_dt
        observed=observation['policy'].clone().reshape(n,5,80)
        if t>=3.:
            ang=torch.atan2(observed[:,:,78],observed[:,:,79])+shift[:,None]*2*torch.pi
            observed[:,:,78]=torch.sin(ang);observed[:,:,79]=torch.cos(ang)
        with torch.inference_mode():
            action=torch.empty((n,23),device=raw.device)
            for gi,group in enumerate(groups):
                ids=slice(gi*per,(gi+1)*per)
                actor_input=observed[ids].clone()
                if gi==3 and t>=3.:
                    # Only a causal diagnostic: head q/qd/previous-action history
                    # from the simultaneous original-head cohort. Physical head
                    # control stays native zero in BOTH neutral-head cohorts.
                    columns=[9+raw.lookup[x] for x in ['head_yaw','head_pitch']]
                    columns += [32+raw.lookup[x] for x in ['head_yaw','head_pitch']]
                    columns += [55+raw.lookup[x] for x in ['head_yaw','head_pitch']]
                    actor_input[:,:,columns]=observed[per:2*per,:,columns]
                action[ids]=actors[group[4]]({'policy':actor_input.reshape(per,400)})
                if gi>=2 and t>=3.:
                    action[ids,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]=0.

        action=action.clone() # owned normal tensor for diagnostic action offsets
        phase=raw.phase.clone()
        # Shorten both legs only around the desired airborne phase. Anatomical
        # hip+knee+ankle offsets sum to zero; all ordinary projections remain.
        phasehalf=torch.remainder(phase,.5)
        pulse=torch.sin(((phasehalf-retract_start)/.07).clamp(0.,1.)*torch.pi)*(phasehalf>=retract_start)
        pulse*=float(t>=3.)
        # Diagnostic-only extension immediately before the intended flight,
        # followed by retraction; all normal action/CAD/torque/velocity limits apply.
        push=torch.sin(((phasehalf-.28)/.12).clamp(0.,1.)*torch.pi)*(phasehalf>=.28)*(phasehalf<=.40)
        delta=amplitude*pulse-extension*push*float(t>=3.)
        for side in ['left','right']:
            for joint,multiplier in [('hip_pitch',-.5),('knee_pitch',1.),('ankle_pitch',-.5)]:
                action[:,raw.lookup[side+'_'+joint]]+=multiplier*delta/.25
        observation,reward,done,_=env.step(action)
        done=done.bool()
        alive&=~done;failed+=done.long();m=raw.telemetry()
        legphase=torch.remainder(phase[:,None]+torch.tensor([0.,.5],device=raw.device),1.)
        swing=legphase>=.43
        unload=torch.exp(-(m['foot_force'].clamp_min(0.)*swing).sum(1)/(raw.total_weight*.25))
        row=dict(observations=observed.reshape(n,400),time=t+raw.step_dt,phase=phase,root=raw.robot.data.root_state_w,
            q=raw.robot.data.joint_pos,qd=raw.robot.data.joint_vel,tau=raw.robot.data.applied_torque,
            target=raw.targets,raw_action=action,effective_action=raw.actions,force=m['foot_force'],
            clearance=m['clearance'],contact=m['contact'],alive=alive,reward_rate=reward/raw.step_dt,
            unload_score=unload,failures=done)
        for k,v in raw.last_reward_terms.items():reward_terms.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
        for k,v in row.items():history[k].append(v.detach().cpu().numpy().copy() if torch.is_tensor(v) else v)
        if tick%100==0:print(json.dumps({'tick':tick,'alive':int(alive.sum()),'height':float(m['height'].mean())}),flush=True)
    data={k:np.asarray(v) for k,v in history.items()};window=data['time']>=3.
    summaries=[]
    for i,g in enumerate(groups):
        ids=slice(i*per,(i+1)*per);cl=data['clearance'][window,ids];force=data['force'][window,ids]
        contact=data['contact'][window,ids];good=data['alive'][window,ids]
        geometric=(cl>.001).all(-1);actual=geometric&(force<1.).all(-1)&good
        stats=dict(name=g[0],command=g[1],retraction_rad=g[2],policy_phase_shift_cycles=g[3],retract_start_cycles=g[5],extension_rad=g[6],
            robots=per,survival=float(alive[ids].float().mean()),resets=int(failed[ids].sum()),
            force_threshold_flight_fraction=float((~contact.any(-1)&good).mean()),
            both_sole_above_1mm_fraction=float((geometric&good).mean()),
            both_sole_above_5mm_fraction=float(((cl>.005).all(-1)&good).mean()),
            unloaded_geometric_flight_fraction=float(actual.mean()),
            max_minimum_sole_clearance_m=float(cl.min(-1)[good].max()) if good.any() else None,
            mean_forward_world_speed=float(data['root'][window,ids,7][good].mean()) if good.any() else None,
            mean_reward_rate=float(data['reward_rate'][window,ids][good].mean()) if good.any() else None,
            mean_unload_score=float(data['unload_score'][window,ids][good].mean()) if good.any() else None,
            reward_terms={k:float(np.asarray(v)[window,ids][good].mean()) for k,v in reward_terms.items()},
            action_clip_fraction=float((np.abs(data['raw_action'][window,ids])>3).mean()))
        head=data['q'][window,ids][:,:,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]
        stats['head_combined_rms_deg']=float(np.sqrt((head**2).sum(-1).mean())*180/np.pi)
        stats['head_target_max_abs_rad']=float(np.abs(data['target'][window,ids][:,:,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]).max())
        stats['mean_abs_world_lateral_speed']=float(np.abs(data['root'][window,ids,8][good]).mean())
        quat=data['root'][-1,ids,3:7];w,x,y,z=quat.T
        stats['mean_abs_final_yaw_deg']=float(np.abs(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))).mean()*180/np.pi)
        summaries.append(stats)
    np.savez_compressed(out/'telemetry.npz',**data,**{'term_'+k:np.asarray(v) for k,v in reward_terms.items()},joint_names=np.array(raw.robot.joint_names),groups=np.array([g[0] for g in groups]),robots_per_group=per)
    result=dict(completed=True,checkpoint=str(checkpoint),checkpoint_sha256=drain['checkpoint_sha256'],groups=summaries,
        gravity=True,fixed_base=False,normal_motor_limits=True,root_impulses=False,pose_writes=False,
        actor_loaded_exact=True,qualifies_locomotion=False,
        diagnostic_upper_action_decomposition=True,scope='128 physical robots, 20s; bounded diagnostic policy-phase or joint-action changes after3s. No external root support, no training source changes; automatic falls retained.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();(OUT/'head_causal_probe_failure.json').write_text(json.dumps({'error':traceback.format_exc()}));raise
    finally:app.app.close()
