"""Actual GUI/RTX capture: 500 control frames encoded as 10 seconds at 50 FPS."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
pause=json.loads((OUT/'video_pause.json').read_text());RUN=Path(pause['run']);TRAIN=RUN/'source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
assert not args.headless,'The user requested the GUI'
args.enable_cameras=True
launcher=AppLauncher(args)
import cv2,numpy as np,torch,psutil
from pxr import UsdGeom,Gf
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment_unitree import LocomotionCfg,LocomotionEnv
from curriculum_provenance import training_reward_signature
from heading_control import yaw_from_quat
torch.set_num_threads(4)
DEST=HERE/'videos/walking_gui_20261004'
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def verify_pause():
    supervisor=psutil.Process(pause['supervisor_pid'])
    assert supervisor.create_time()==pause['supervisor_create_time']
    assert supervisor.status()==psutil.STATUS_STOPPED,'Supervisor not suspended'
    assert json.loads((RUN/'state.json').read_text())['status']=='paused_for_video_capture'
    for p in psutil.process_iter(['cmdline']):
        assert not any(Path(a).name=='worker_unitree.py' for a in (p.info['cmdline'] or [])),'Training worker active'
def grid(stage):
    # Noncolliding visual guide lines; no physical materials/contact changes.
    curve=UsdGeom.BasisCurves.Define(stage,'/World/RecordingGuideGrid')
    points=[];counts=[]
    for x in range(-5,20):points.extend([Gf.Vec3f(x,-5,.002),Gf.Vec3f(x,5,.002)]);counts.append(2)
    for y in range(-5,6):points.extend([Gf.Vec3f(-5,y,.002),Gf.Vec3f(20,y,.002)]);counts.append(2)
    curve.CreateTypeAttr('linear');curve.CreateWrapAttr('nonperiodic')
    curve.CreatePointsAttr(points);curve.CreateCurveVertexCountsAttr(counts)
    curve.CreateWidthsAttr([.006]);curve.SetWidthsInterpolation('constant')
    curve.CreateDisplayColorAttr([Gf.Vec3f(.23,.27,.31)])
def camera(raw):
    p=raw.robot.data.root_pos_w[0].detach().cpu().numpy()
    raw.sim.set_camera_view([float(p[0])+2.8,float(p[1])-3.1,1.5],[float(p[0]),float(p[1]),.82])
def main():
    verify_pause();DEST.mkdir(parents=True,exist_ok=True)
    assert not (DEST/'manifest.json').exists(),'Preserve finished recording'
    checkpoint=Path(pause['checkpoint']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==pause['checkpoint_sha256']
    save(DEST/'capture_started.json',{'pid':psutil.Process().pid,'created_time':psutil.Process().create_time(),
         'gui':True,'checkpoint':str(checkpoint),'paused_supervisor':pause['supervisor_pid']})
    converter=UrdfConverter(UrdfConverterCfg(asset_path=str(TRAIN/'assets/robot_training.urdf'),
        usd_dir=str(TRAIN/'assets/usd'),usd_file_name='robot_training.usd',fix_base=False,root_link_name='pelvis',
        merge_fixed_joints=True,make_instanceable=True,collision_from_visuals=False,self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
    cfg=LocomotionCfg();cfg.scene.num_envs=1;cfg.evaluation=True;cfg.stage_speed=1.5
    cfg.robot_usd=converter.usd_path;cfg.sim.device=args.device or 'cuda:0';cfg.seed=901
    cfg.episode_length_s=60.;cfg.sim.render_interval=cfg.decimation;cfg.viewer.resolution=(1280,720)
    raw=LocomotionEnv(cfg,render_mode='rgb_array');raw.initialize_randomization()
    grid(raw.sim.stage)
    env=RslRlVecEnvWrapper(raw,clip_actions=3.)
    runner=OnPolicyRunner(env,json.loads((checkpoint.parent/'agent.json').read_text()),log_dir=None,device=raw.device)
    saved=torch.load(checkpoint,map_location=raw.device,weights_only=False)
    assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)
    runner.load(str(checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
    assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in saved['actor_state_dict'].items())
    policy=runner.get_inference_policy(raw.device);dt=float(raw.step_dt);frames=round(10./dt)
    assert frames==500 and abs(frames*dt-10)<1e-9
    # Create/warm the render product before any recorded physics frames.
    camera(raw)
    for _ in range(20):raw.render()
    manifest={'gui':True,'checkpoint':str(checkpoint),'checkpoint_sha256':pause['checkpoint_sha256'],
      'source_run':str(RUN),'duration_per_clip_s':10.,'frames_per_clip':frames,'fps':1/dt,
      'resolution':[1280,720],'camera':'Tracking front three-quarter view; visual 1m floor grid',
      'policy_mode':'Deterministic inference; no training, assistance or manual joint offsets',
      'physical_scope':'Same frozen v29 geometry, gravity, self-contact, CAD differential and motor limits; visual grid only',
      'qualification_scope':'Videos are demonstrations, not independent gates','clips':[]}
    if (DEST/'capture_progress.json').is_file():
        previous=json.loads((DEST/'capture_progress.json').read_text())
        assert previous['checkpoint_sha256']==manifest['checkpoint_sha256']
        manifest['clips']=previous['clips']
        for clip in manifest['clips']:
            cap=cv2.VideoCapture(clip['raw_video']);assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==500;cap.release()
    writer=None
    try:
        for speed in [.15,.35,.65,1.]:
            if any(clip['command_m_s']==speed for clip in manifest['clips']):continue
            verify_pause();label=f'walking_{round(speed*100):03d}'
            raw.configure_fixed_commands(torch.tensor([speed],device=raw.device),yaw_from_quat(raw.robot.data.root_quat_w).clone())
            # Isaac lazily creates some cached state under inference mode.
            # Reset those caches in the same context as the physical rollout.
            with torch.inference_mode():raw.reset();obs=raw._get_observations()
            # Two simulated seconds to settle into the commanded gait; not encoded.
            for _ in range(round(2/dt)):
                camera(raw)
                with torch.inference_mode():obs,_,_,_=env.step(policy(obs))
            raw_path=DEST/(label+'_mp4v.mp4');final_path=DEST/(label+'.mp4')
            assert not raw_path.exists() and not final_path.exists(),'Preserve previous video attempt'
            writer=cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*'mp4v'),1/dt,(1280,720))
            assert writer.isOpened()
            rows=[];start=time.perf_counter();first=raw.robot.data.root_pos_w[0].detach().cpu().numpy().copy()
            falls=0
            for tick in range(frames):
                with torch.inference_mode():obs,_,done,_=env.step(policy(obs))
                falls+=int(done[0]);camera(raw);frame=raw.render()
                assert frame.shape==(720,1280,3) and frame.std()>2.,'Empty renderer frame'
                rgb=np.ascontiguousarray(frame);bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                m=raw.telemetry();root=raw.robot.data.root_state_w[0].detach().cpu().numpy().copy()
                cv2.putText(bgr,f'Target {speed:.2f} m/s   |   {(tick+1)*dt:05.2f} / 10.00 s',(28,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),4,cv2.LINE_AA)
                cv2.putText(bgr,f'Target {speed:.2f} m/s   |   {(tick+1)*dt:05.2f} / 10.00 s',(28,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(240,240,240),2,cv2.LINE_AA)
                writer.write(bgr)
                rows.append({'time_s':(tick+1)*dt,'root':root.tolist(),'q':raw.robot.data.joint_pos[0].detach().cpu().tolist(),
                    'clearance':m['clearance'][0].detach().cpu().tolist(),'force':m['foot_force'][0].detach().cpu().tolist(),
                    'command':raw.action_commands[0].detach().cpu().tolist(),'terminated':int(done[0])})
                if tick in (0,249,499):cv2.imwrite(str(DEST/f'{label}_frame_{tick:03d}.jpg'),bgr)
                if tick%100==0:print(json.dumps({'clip':label,'frames':tick+1,'simulated_seconds':(tick+1)*dt,'wall_seconds':time.perf_counter()-start}),flush=True)
            writer.release();writer=None
            capture_wall=time.perf_counter()-start
            save(DEST/(label+'_telemetry.json'),rows)
            cap=cv2.VideoCapture(str(raw_path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
            assert count==500 and abs(count/fps-10.)<.001
            clip={'command_m_s':speed,'raw_video':str(raw_path),'video':str(final_path),'frames':count,'fps':fps,
                'duration_s':count/fps,'record_wall_seconds':capture_wall,'falls':falls,
                'mean_signed_world_forward_speed_m_s':float(np.mean([r['root'][7] for r in rows])),
                'net_forward_travel_m':float(rows[-1]['root'][0]-first[0]),
                'maximum_command_difference':max(abs(r['command'][0]-speed) for r in rows)}
            manifest['clips'].append(clip);save(DEST/'capture_progress.json',manifest)
            print(json.dumps({'clip_complete':clip}),flush=True)
        save(DEST/'manifest.json',manifest)
    finally:
        if writer is not None:writer.release()
        env.close()
    print(json.dumps({'completed':True,'manifest':str(DEST/'manifest.json')}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:launcher.app.close()
