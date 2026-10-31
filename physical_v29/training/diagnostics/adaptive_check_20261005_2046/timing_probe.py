"""Bounded diagnostic leg timing search; never deployed in training or inference."""
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent
pause=json.loads((OUT/'diagnostic_pause.json').read_text())
RUN=Path(pause['run']); TRAIN=RUN/'source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args(); app=AppLauncher(args)
import numpy as np, torch, psutil
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment_unitree import LocomotionCfg, LocomotionEnv
from curriculum_provenance import training_reward_signature
from heading_control import yaw_from_quat
from trajectory_parameters import ORIGIN, INITIAL_STD, NAMES, samples, offsets_numpy, update

def main():
    assert pause['worker_drained']
    sup=psutil.Process(pause['supervisor_pid'])
    assert sup.create_time()==pause['supervisor_create_time'] and sup.status()==psutil.STATUS_STOPPED
    for proc in psutil.process_iter(['cmdline']):
        assert not any(Path(x).name=='worker_unitree.py' for x in (proc.info['cmdline'] or [])), 'Active worker'
    out=OUT/'timing_probe'; out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists(), 'Preserve completed evidence'
    cp=Path(pause['checkpoint']); assert hashlib.sha256(cp.read_bytes()).hexdigest()==pause['checkpoint_sha256']
    torch.set_num_threads(4)
    converter=UrdfConverter(UrdfConverterCfg(asset_path=str(TRAIN/'assets/robot_training.urdf'),
        usd_dir=str(TRAIN/'assets/usd'),usd_file_name='robot_training.usd',fix_base=False,root_link_name='pelvis',
        merge_fixed_joints=True,make_instanceable=True,collision_from_visuals=False,self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
    from offline_ground import install_local_ground
    install_local_ground()
    per=32; count=6; n=per*count
    cfg=LocomotionCfg(); cfg.scene.num_envs=n; cfg.scene.env_spacing=3.; cfg.evaluation=True
    cfg.stage_speed=1.5; cfg.robot_usd=converter.usd_path; cfg.sim.device=args.device or 'cuda:0'
    cfg.seed=91; cfg.episode_length_s=30.
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity=2**22
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity=2**20
    cfg.sim.physx.gpu_max_rigid_contact_count=2**20
    raw=LocomotionEnv(cfg); raw.initialize_randomization(); env=RslRlVecEnvWrapper(raw,clip_actions=3.)
    config=json.loads((cp.parent/'agent.json').read_text())
    runner=OnPolicyRunner(env,config,log_dir=None,device=raw.device)
    saved=torch.load(cp,map_location=raw.device,weights_only=False)
    assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)
    runner.load(str(cp),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
    assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in saved['actor_state_dict'].items())
    policy=runner.get_inference_policy(raw.device)
    joint_ids=[raw.lookup[side+'_'+joint] for side in ['left','right'] for joint in ['hip_pitch','knee_pitch','ankle_pitch']]
    head_ids=[raw.lookup['head_yaw'],raw.lookup['head_pitch']]
    rng=np.random.default_rng(271); mean=ORIGIN.copy(); std=INITIAL_STD.copy(); rounds=[]
    started=time.time()
    for generation in range(1):
        parameters=np.asarray(json.loads((OUT/'timing_selection.json').read_text())['parameters']); repeated=np.repeat(parameters,per,axis=0)
        raw.configure_fixed_commands(torch.full((n,),1.5,device=raw.device),yaw_from_quat(raw.robot.data.root_quat_w).clone())
        raw.reset(); heading=raw.heading_target.clone().cpu().numpy(); obs=raw._get_observations()
        alive=torch.ones(n,dtype=torch.bool,device=raw.device)
        returns=np.zeros(n); terms={}; history={k:[] for k in ['time','phase','reward_phase','root','q','qd','target','force','clearance','alive','failure','raw_action','effective_action','offset','observed_command','action_command','reward','target_clip','target_slew']}
        for tick in range(1000):
            t=tick*raw.step_dt; phase=raw.phase.clone(); observed=obs['policy'].reshape(n,5,80)[:,-1,6:9].clone()
            with torch.inference_mode(): original=policy(obs)
            # Zero offsets execute exactly the base clamp; all feedback remains truthful.
            offset=torch.as_tensor(offsets_numpy(phase.cpu().numpy(),repeated,t),device=raw.device,dtype=original.dtype)
            adjusted=original.clamp(-3,3).clone()
            for j,idx in enumerate(joint_ids): adjusted[:,idx]+=offset[:,j%3]/raw.scales[idx]
            before=alive.clone(); obs,reward,done,_=env.step(adjusted); done=done.bool()
            returns+=np.where(before.cpu().numpy(),reward.cpu().numpy(),0.)
            alive&=~done; m=raw.telemetry()
            row=dict(time=t+raw.step_dt,phase=phase,reward_phase=raw.reward_proxy.cad_metrics['phase'],root=raw.robot.data.root_state_w,
                q=raw.robot.data.joint_pos,qd=raw.robot.data.joint_vel,target=raw.targets,force=m['foot_force'],clearance=m['clearance'],
                alive=alive,failure=done,raw_action=original,effective_action=raw.actions,offset=offset,observed_command=observed,
                action_command=raw.action_commands,reward=reward,target_clip=raw.target_clipped,target_slew=raw.target_slew_limited)
            for k,v in row.items(): history[k].append(v.detach().cpu().numpy().copy() if torch.is_tensor(v) else v)
            for k,v in raw.last_reward_terms.items(): terms.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
            if tick%150==0: print(json.dumps({'generation':generation,'tick':tick,'alive':int(alive.sum())}),flush=True)
        data={k:np.asarray(v) for k,v in history.items()}; window=data['time']>=5.; rows=[]
        for i in range(count):
            ids=slice(i*per,(i+1)*per); good=data['alive'][window,ids]; root=data['root'][window,ids]
            h=heading[ids][None,:]; fw=root[:,:,7]*np.cos(h)+root[:,:,8]*np.sin(h); lat=-root[:,:,7]*np.sin(h)+root[:,:,8]*np.cos(h)
            yaw=np.arctan2(2*(root[:,:,3]*root[:,:,6]+root[:,:,4]*root[:,:,5]),1-2*(root[:,:,5]**2+root[:,:,6]**2))
            err=np.arctan2(np.sin(yaw-h),np.cos(yaw-h))
            trueflight=(data['clearance'][window,ids]>.001).all(-1)&(data['force'][window,ids]<1.).all(-1)&good
            def avg(x): return float(x[good].mean()) if good.any() else None
            rows.append(dict(candidate=i,label=json.loads((OUT/'timing_selection.json').read_text())['labels'][i],parameters=dict(zip(NAMES,parameters[i].tolist())),mean_complete_return=float(returns[ids].mean()),
                survival=float(data['alive'][-1,ids].mean()),failures=int(data['failure'][:,ids].sum()),
                geometric_flight_fraction=float(trueflight.mean()),signed_forward_m_s=avg(fw),absolute_lateral_m_s=avg(abs(lat)),
                heading_RMS_deg=float(np.rad2deg(np.sqrt((err[good]**2).mean()))) if good.any() else None,
                head_RMS_deg=float(np.rad2deg(np.sqrt((data['q'][window,ids][:,:,head_ids]**2).sum(-1)[good].mean()))) if good.any() else None,
                mean_reward_rate=avg(data['reward'][window,ids]/raw.step_dt),
                reward_terms={k:avg(np.asarray(v)[window,ids]) for k,v in terms.items()},
                target_clip_fraction=float(data['target_clip'][window,ids].mean()),target_slew_fraction=float(data['target_slew'][window,ids].mean())))
        fitness=np.array([r['mean_complete_return'] for r in rows]); mean,std,elite=update(parameters,fitness)
        record=dict(generation=generation,rows=rows,elite_by_complete_return=elite.tolist(),next_mean=mean.tolist(),next_std=std.tolist(),
            max_command_difference=float(abs(data['observed_command']-data['action_command']).max()))
        (out/f'round_{generation:02d}.json').write_text(json.dumps(record,indent=2)+'\n')
        np.savez_compressed(out/f'round_{generation:02d}.npz',**data,**{'term_'+k:np.asarray(v) for k,v in terms.items()},
            parameters=parameters,heading=heading,joint_names=np.array(raw.robot.joint_names),returns=returns,robots_per_candidate=per)
        rounds.append(record); print(json.dumps({'generation':generation,'baseline':rows[0],'best_return':rows[int(fitness.argmax())],
            'max_flight':max(r['geometric_flight_fraction'] for r in rows)}),flush=True)
    result=dict(completed=True,checkpoint=str(cp),checkpoint_sha256=pause['checkpoint_sha256'],rounds=rounds,
        wall_seconds=time.time()-started,normal_limits=True,truthful_observations=True,actor_loaded_exact=True,
        diagnostic_offsets_only=True,production_source_changed=False,qualifies_locomotion=False,
        scope='Current exact actor,seed91: unchanged control,original best-flight offset,20/40/60ms advances and original best-return offset,32 robots each,20s. Complete return includes first terminal penalty; later reset rewards excluded. No root support, phase/pose/input rewrites or head override. Leg offsets are diagnostic only.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); env.close()

if __name__=='__main__':
    try: main()
    except BaseException:
        (OUT/'timing_probe_failure.json').write_text(json.dumps({'error':traceback.format_exc()},indent=2)); raise
    finally: app.app.close()
