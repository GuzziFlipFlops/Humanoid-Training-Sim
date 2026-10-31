"""Real GUI full-CAD visual playback; paused training, unchanged physical contract."""
import argparse,hashlib,json,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
pause=json.loads((OUT/'video_pause.json').read_text(encoding='utf-8-sig'))
capture=json.loads((OUT/'capture_config.json').read_text(encoding='utf-8-sig'))
RUN=Path(pause['run']);TRAIN=RUN/'source/physical_v29/training';DEST=Path(capture['destination'])
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
assert not args.headless;args.enable_cameras=True
launcher=AppLauncher(args)
import cv2,numpy as np,torch,psutil
from pxr import Usd,UsdGeom,UsdLux,UsdPhysics,UsdShade,Sdf,Gf
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment_unitree import LocomotionCfg,LocomotionEnv
from curriculum_provenance import training_reward_signature
from heading_control import yaw_from_quat
torch.set_num_threads(4)
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def verify_pause():
    p=psutil.Process(pause['supervisor_pid'])
    assert p.create_time()==pause['supervisor_create_time'] and p.status()==psutil.STATUS_STOPPED
    assert json.loads((RUN/'state.json').read_text())['status']=='paused_for_video_capture'
    for p in psutil.process_iter(['cmdline']):
        assert not any(Path(a).name=='worker_unitree.py' for a in (p.info['cmdline'] or []))
def camera(raw):
    p=raw.robot.data.root_pos_w[0].detach().cpu().numpy()
    raw.sim.set_camera_view([float(p[0])+2.2,float(p[1])-2.6,1.45],
                            [float(p[0])+.05,float(p[1]),.85])
def lights(stage):
    dome=UsdLux.DomeLight(stage.GetPrimAtPath('/World/Light'))
    dome.CreateIntensityAttr(650.);dome.CreateColorAttr(Gf.Vec3f(.90,.94,1.))
    dome.GetPrim().GetAttribute('visibleInPrimaryRay').Set(True)
    dome.CreateVisibilityAttr('inherited')
    key=UsdLux.DistantLight.Define(stage,'/World/RecordingKeyLight')
    key.CreateIntensityAttr(650.);key.CreateAngleAttr(12.)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-30.,-25.,-30.))
    grid=UsdGeom.BasisCurves.Define(stage,'/World/RecordingGuideGrid')
    points=[];counts=[]
    for x in range(-5,30):points.extend([Gf.Vec3f(x,-5,.002),Gf.Vec3f(x,5,.002)]);counts.append(2)
    for y in range(-5,6):points.extend([Gf.Vec3f(-5,y,.002),Gf.Vec3f(30,y,.002)]);counts.append(2)
    grid.CreateTypeAttr('linear');grid.CreateWrapAttr('nonperiodic')
    grid.CreatePointsAttr(points);grid.CreateCurveVertexCountsAttr(counts)
    grid.CreateWidthsAttr([.004]);grid.SetWidthsInterpolation('constant')
    grid.CreateDisplayColorAttr([Gf.Vec3f(.27,.31,.35)])
def restore_cad_materials(raw):
    # STL import can omit the URDF visual material binding. Restore the exact
    # original source colors on the owning visual Xform, including instances.
    root=raw.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
    uninstanced=[]
    for body in raw.robot.body_names:
        visual_root=raw.sim.stage.GetPrimAtPath(str(root.GetPath())+'/'+body+'/visuals')
        if visual_root and visual_root.IsInstance():
            visual_root.SetInstanceable(False);uninstanced.append(str(visual_root.GetPath()))
    save(DEST/'visual_only_instance_overrides.json',{'uninstanced_visual_roots':uninstanced,'physics_unchanged':True})
    visual_colors={v.get('name'):list(map(float,v.find('material/color').get('rgba').split()))
        for link in ET.parse(capture['visual_urdf']).getroot().findall('link') for v in link.findall('visual')}
    bindings=[]
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        if prim.GetTypeName()!='Mesh':continue
        owner=prim
        while owner and owner.GetName() not in visual_colors:owner=owner.GetParent()
        assert owner and not owner.IsInstanceProxy(),f'Unbindable visual {prim.GetPath()}'
        color=visual_colors[owner.GetName()]
        material=UsdShade.Material.Define(raw.sim.stage,'/World/RecordingMaterials/'+owner.GetName())
        shader=UsdShade.Shader.Define(raw.sim.stage,str(material.GetPath())+'/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.5)
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.05)
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(color[3])
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(owner).Bind(material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        bindings.append({'visual':owner.GetName(),'color_rgba':color,'bound_to':str(owner.GetPath())})
    assert len(bindings)==140
    save(DEST/'restored_material_bindings.json',bindings)
def mesh_inventory(raw):
    root=raw.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render],useExtentsHint=False)
    meshes=[]
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        if prim.GetTypeName()!='Mesh':continue
        ancestor=prim;body=None
        while ancestor and ancestor!=root:
            if ancestor.GetName() in raw.robot.body_names:
                body=ancestor.GetName();break
            ancestor=ancestor.GetParent()
        bbox=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        pts=UsdGeom.Mesh(prim).GetPointsAttr().Get()
        meshes.append({'path':str(prim.GetPath()),'body':body,'vertices':len(pts) if pts is not None else 0,
            'visibility':str(UsdGeom.Imageable(prim).ComputeVisibility()),
            'bbox_min':list(bbox.GetMin()),'bbox_max':list(bbox.GetMax())})
    save(DEST/'rendered_mesh_inventory.json',{'mesh_count':len(meshes),'source_expected':140,
        'runtime_body_names':raw.robot.body_names,'runtime_joint_names':raw.robot.joint_names,'meshes':meshes})
    assert len(meshes)==140 and all(m['body'] and m['vertices']>0 and m['visibility']!='invisible' for m in meshes), 'CAD meshes missing, unbound or hidden'
    return meshes
def main():
    verify_pause();DEST.mkdir(parents=True,exist_ok=True)
    assert not (DEST/'manifest.json').exists(),'Preserve completed recordings'
    save(DEST/'capture_started.json',{'pid':psutil.Process().pid,'created_time':psutil.Process().create_time(),
        'gui':True,'paused_supervisor':pause['supervisor_pid'],'run':str(RUN)})
    converter=UrdfConverter(UrdfConverterCfg(asset_path=capture['visual_urdf'],
        usd_dir=str(OUT/'playback_asset/usd'),usd_file_name='robot_full_visuals.usd',fix_base=False,root_link_name='pelvis',
        merge_fixed_joints=True,make_instanceable=True,collision_from_visuals=False,self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
    cfg=LocomotionCfg();cfg.scene.num_envs=1;cfg.evaluation=True;cfg.stage_speed=1.5
    cfg.robot_usd=converter.usd_path;cfg.sim.device=args.device or 'cuda:0';cfg.seed=901
    cfg.episode_length_s=60.;cfg.sim.render_interval=cfg.decimation;cfg.viewer.resolution=(1280,720)
    raw=LocomotionEnv(cfg,render_mode='rgb_array');raw.initialize_randomization();lights(raw.sim.stage);restore_cad_materials(raw)
    env=RslRlVecEnvWrapper(raw,clip_actions=3.)
    runner=OnPolicyRunner(env,json.loads((Path(capture['clips'][0]['checkpoint']).parent/'agent.json').read_text()),log_dir=None,device=raw.device)
    def load_actor(clip):
        checkpoint=Path(clip['checkpoint']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==clip['sha256']
        saved=torch.load(checkpoint,map_location=raw.device,weights_only=False)
        assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)
        runner.load(str(checkpoint),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
        assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in saved['actor_state_dict'].items())
        return runner.get_inference_policy(raw.device)
    dt=float(raw.step_dt);frames=round(10./dt);assert frames==500 and abs(frames*dt-10)<1e-9
    writer=None
    try:
        policy=load_actor(capture['clips'][0])
        raw.configure_fixed_commands(torch.tensor([0.],device=raw.device),yaw_from_quat(raw.robot.data.root_quat_w).clone())
        with torch.inference_mode():raw.reset();obs=raw._get_observations()
        for _ in range(80):
            with torch.inference_mode():obs,_,_,_=env.step(policy(obs))
            camera(raw)
        meshes=mesh_inventory(raw)
        for _ in range(20):raw.render()
        preview=raw.render();assert preview.shape==(720,1280,3)
        cv2.imwrite(str(DEST/'neutral_preview.png'),cv2.cvtColor(preview,cv2.COLOR_RGB2BGR))
        save(DEST/'preview_ready.json',{'usd':converter.usd_path,'meshes':len(meshes),'physics_parity_cpu':True,
            'joint_names':raw.robot.joint_names,'body_names':raw.robot.body_names})
        print('FULL_BODY_PREVIEW_READY',flush=True)
        deadline=time.monotonic()+600
        while not (OUT/'capture_approved.json').exists():
            assert time.monotonic()<deadline,'Preview review timed out; resume training'
            verify_pause();launcher.app.update();time.sleep(.2)
        assert json.loads((OUT/'capture_approved.json').read_text())['full_body_preview_passed']
        manifest={'gui':True,'source_run':str(RUN),'visual_urdf':capture['visual_urdf'],
            'visual_sha256':capture['visual_sha256'],'training_urdf_sha256':capture['training_urdf_sha256'],
            'mesh_count':len(meshes),'physics_joints_collisions_inertias_equal':True,
            'duration_per_clip_s':10.,'frames_per_clip':frames,'fps':1/dt,'resolution':[1280,720],
            'joint_names':raw.robot.joint_names,'body_names':raw.robot.body_names,
            'camera':'Tracking front/side three-quarter with full-body margin',
            'qualification_scope':'Per-speed saved actor demonstration; independent gate receipts retained',
            'control':'Exact deterministic trained actors, unaltered limits, no helper offsets or root support','clips':[]}
        for clip in capture['clips']:
            verify_pause();speed=clip['speed'];policy=load_actor(clip);label=f'locomotion_{round(speed*100):03d}'
            raw.configure_fixed_commands(torch.tensor([speed],device=raw.device),yaw_from_quat(raw.robot.data.root_quat_w).clone())
            with torch.inference_mode():raw.reset();obs=raw._get_observations()
            for _ in range(round(2/dt)):
                with torch.inference_mode():obs,_,_,_=env.step(policy(obs))
                camera(raw)
            raw_path=DEST/(label+'_mp4v.mp4');final_path=DEST/(label+'.mp4')
            assert not raw_path.exists() and not final_path.exists()
            writer=cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*'mp4v'),1/dt,(1280,720));assert writer.isOpened()
            rows=[];start=time.perf_counter();first=raw.robot.data.root_pos_w[0].detach().cpu().numpy().copy();falls=0
            for tick in range(frames):
                with torch.inference_mode():actions=policy(obs);obs,_,done,_=env.step(actions)
                falls+=int(done[0]);camera(raw);frame=raw.render()
                assert frame.shape==(720,1280,3) and frame.std()>2.
                bgr=cv2.cvtColor(np.ascontiguousarray(frame),cv2.COLOR_RGB2BGR)
                m=raw.telemetry();root=raw.robot.data.root_state_w[0].detach().cpu().numpy().copy()
                text=f'Command {speed:.2f} m/s | actual forward {root[7]:.2f} m/s | {(tick+1)*dt:05.2f} / 10.00 s'
                cv2.putText(bgr,text,(24,35),cv2.FONT_HERSHEY_SIMPLEX,.68,(20,20,20),4,cv2.LINE_AA)
                cv2.putText(bgr,text,(24,35),cv2.FONT_HERSHEY_SIMPLEX,.68,(245,245,245),2,cv2.LINE_AA)
                writer.write(bgr)
                rows.append({'time_s':(tick+1)*dt,'root':root.tolist(),'q':raw.robot.data.joint_pos[0].detach().cpu().tolist(),
                    'qd':raw.robot.data.joint_vel[0].detach().cpu().tolist(),'action':actions[0].detach().cpu().tolist(),
                    'body_pos':raw.robot.data.body_pos_w[0].detach().cpu().tolist(),
                    'body_quat':raw.robot.data.body_quat_w[0].detach().cpu().tolist(),
                    'clearance':m['clearance'][0].detach().cpu().tolist(),'force':m['foot_force'][0].detach().cpu().tolist(),
                    'command':raw.action_commands[0].detach().cpu().tolist(),'terminated':int(done[0])})
                if tick in (0,124,249,374,499):cv2.imwrite(str(DEST/f'{label}_frame_{tick:03d}.png'),bgr)
                if tick%100==0:print(json.dumps({'clip':label,'frames':tick+1,'wall_seconds':time.perf_counter()-start}),flush=True)
            writer.release();writer=None
            save(DEST/(label+'_telemetry.json'),rows)
            cap=cv2.VideoCapture(str(raw_path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
            assert count==500 and abs(count/fps-10)<.001
            result={**clip,'command_m_s':speed,'raw_video':str(raw_path),'video':str(final_path),
                'frames':count,'fps':fps,'duration_s':count/fps,'record_wall_seconds':time.perf_counter()-start,'falls':falls,
                'mean_signed_world_forward_speed_m_s':float(np.mean([r['root'][7] for r in rows])),
                'net_forward_travel_m':float(rows[-1]['root'][0]-first[0]),
                'maximum_command_difference':max(abs(r['command'][0]-speed) for r in rows),
                'geometric_flight_fraction':float(np.mean([min(r['clearance'])>.001 and max(r['force'])<1. for r in rows]))}
            manifest['clips'].append(result);save(DEST/'capture_progress.json',manifest)
            print(json.dumps({'clip_complete':result}),flush=True)
        save(DEST/'manifest.json',manifest)
    finally:
        if writer is not None:writer.release()
        env.close()
    print('FULL_BODY_CAPTURE_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:launcher.app.close()
