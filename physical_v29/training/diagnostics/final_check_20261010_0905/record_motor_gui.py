"""Real GUI full-CAD visual playback; paused training, unchanged physical contract."""
import argparse,hashlib,json,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
pause=json.loads((OUT/'stop_receipt.json').read_text(encoding='utf-8-sig'))
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
    assert not psutil.pid_exists(pause['supervisor_pid']), 'Owned supervisor must remain retired'
    assert json.loads((RUN/'state.json').read_text())['status']=='stopped_by_user_budget'
    return

def historical_unused_pause():
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
    # This importer instances the whole per-link visual scope. Open only visual
    # instances for source-color bindings; never deinstance collision/rigid scopes.
    visual_instances=[p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()) if p.IsInstance() and '/visuals' in str(p.GetPath()) and not p.IsInstanceProxy()]
    deinstanced=[]
    for p in visual_instances:
        deinstanced.append(str(p.GetPath()));p.SetInstanceable(False)
    save(DEST/'visual_instance_binding_repair.json',dict(paths=deinstanced,visual_only=True,physics_unchanged=True))
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
    from native_motor_capture import NativeCapture
    import os
    os.environ['FINAL_EPISODE']='1'
    verify_pause();DEST.mkdir(parents=True,exist_ok=True)
    assert not (DEST/'manifest.json').exists(),'Preserve completed recordings'
    save(DEST/'capture_started.json',dict(pid=psutil.Process().pid,created_time=psutil.Process().create_time(),gui=True,run=str(RUN),no_learning=True))
    converter=UrdfConverter(UrdfConverterCfg(asset_path=capture['visual_urdf'],usd_dir=str(OUT/'playback_asset/usd'),usd_file_name='robot_full_visuals.usd',fix_base=False,root_link_name='pelvis',merge_fixed_joints=True,make_instanceable=True,collision_from_visuals=False,self_collision=True,joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
    cfg=LocomotionCfg();cfg.scene.num_envs=1;cfg.evaluation=True;cfg.stage_speed=3.5;cfg.seed=7101;cfg.robot_usd=converter.usd_path;cfg.sim.device=args.device or 'cuda:0';cfg.episode_length_s=60.;cfg.sim.render_interval=cfg.decimation;cfg.viewer.resolution=(1280,720)
    from offline_ground import install_local_ground
    install_local_ground();raw=LocomotionEnv(cfg,render_mode='rgb_array');raw.initialize_randomization();lights(raw.sim.stage);restore_cad_materials(raw)
    env=RslRlVecEnvWrapper(raw,clip_actions=3.)
    cp=Path(capture['clips'][0]['checkpoint']);saved=torch.load(cp,map_location=raw.device,weights_only=False)
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==capture['clips'][0]['sha256'] and saved['infos']['reward_signature']==training_reward_signature(TRAIN)
    runner=OnPolicyRunner(env,json.loads((cp.parent/'agent.json').read_text()),log_dir=None,device=raw.device)
    runner.load(str(cp),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
    assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in saved['actor_state_dict'].items())
    policy=runner.get_inference_policy(raw.device);dt=float(raw.step_dt);assert dt==.02;frames=500
    writer=None;logger=None
    @torch.inference_mode()
    def reset_episode(speed):
        raw.configure_fixed_commands(torch.tensor([speed],device=raw.device),torch.zeros(1,device=raw.device))
        with torch.inference_mode():raw.reset()
        rng=np.random.default_rng(np.random.SeedSequence([7101,round(speed*100)]))
        q=raw.stand[None].clone()+torch.tensor(rng.uniform(-.008,.008,(1,23)),device=raw.device,dtype=raw.stand.dtype);qd=torch.tensor(rng.uniform(-.01,.01,(1,23)),device=raw.device,dtype=q.dtype)
        assert ((q>=raw.target_lo)&(q<=raw.target_hi)).all()
        raw.robot.write_joint_state_to_sim(q,qd);raw.targets[:]=q;raw.previous_targets[:]=q;raw.previous_vel[:]=qd
        raw.scene.update(raw.step_dt);raw.history.zero_();raw.critic_history.zero_();raw._command_tick=-1;raw._last_history_tick=-1
        return raw._get_observations(),dict(seed_sequence=[7101,round(speed*100)],q=q.cpu().tolist(),qd=qd.cpu().tolist())
    try:
        obs,_=reset_episode(0.)
        for _ in range(80):
            with torch.inference_mode():obs,_,_,_=env.step(policy(obs))
            camera(raw)
        meshes=mesh_inventory(raw)
        for _ in range(20):raw.render()
        preview=raw.render();assert preview.shape==(720,1280,3);cv2.imwrite(str(DEST/'neutral_preview.png'),cv2.cvtColor(preview,cv2.COLOR_RGB2BGR))
        save(DEST/'preview_ready.json',dict(usd=converter.usd_path,meshes=len(meshes),source_physics_equal=True));print('FULL_BODY_PREVIEW_READY',flush=True)
        deadline=time.monotonic()+600
        while not (OUT/'capture_approved.json').exists():
            assert time.monotonic()<deadline,'Preview review timeout; training stays stopped'
            verify_pause();launcher.app.update();time.sleep(.2)
        assert json.loads((OUT/'capture_approved.json').read_text())['full_body_preview_passed']
        manifest=dict(gui=True,source_run=str(RUN),visual_urdf=capture['visual_urdf'],visual_sha256=capture['visual_sha256'],training_urdf_sha256=capture['training_urdf_sha256'],mesh_count=140,physics_joints_collisions_inertias_equal=True,duration_per_clip_s=10.,frames_per_clip=500,fps=50.,resolution=[1280,720],joint_names=raw.robot.joint_names,body_names=raw.robot.body_names,control='Exact deterministic single preserved actor, native caps, gravity, free base, truthful inference',no_learning=True,clips=[])
        for clip in capture['clips']:
            verify_pause();speed=clip['speed'];label=f'locomotion_{round(speed*100):03d}';folder=DEST/label;folder.mkdir(exist_ok=False)
            if logger is not None:raw.actuator.compute=logger.original
            obs,initial=reset_episode(speed);save(folder/'initial_state.json',initial);logger=NativeCapture(raw,folder)
            falls=0
            for _ in range(100):
                with torch.inference_mode():obs,_,done,_=env.step(policy(obs))
                logger.done(done,torch.zeros_like(done));falls+=int(done[0]);camera(raw)
            raw_path=DEST/(label+'_mp4v.mp4');final_path=DEST/(label+'.mp4')
            assert not raw_path.exists() and not final_path.exists()
            writer=cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*'mp4v'),50.,(1280,720));assert writer.isOpened()
            rows=[];start=time.perf_counter();first=raw.robot.data.root_pos_w[0].detach().cpu().numpy().copy()
            for tick in range(frames):
                with torch.inference_mode():actions=policy(obs);obs,_,done,_=env.step(actions)
                logger.done(done,torch.zeros_like(done));falls+=int(done[0]);camera(raw);frame=raw.render();assert frame.shape==(720,1280,3) and frame.std()>2.
                m=raw.telemetry();root=raw.robot.data.root_state_w[0].detach().cpu().numpy().copy()
                bgr=cv2.cvtColor(np.ascontiguousarray(frame),cv2.COLOR_RGB2BGR)
                text=f'Command {speed:.2f} m/s | signed forward {root[7]:.2f} m/s | {(tick+1)*dt:05.2f} / 10.00 s'
                scope='Single preserved actor1064 | native caps | retention failures remain'
                for txt,y in [(text,35),(scope,65)]:
                    cv2.putText(bgr,txt,(24,y),cv2.FONT_HERSHEY_SIMPLEX,.63,(20,20,20),4,cv2.LINE_AA);cv2.putText(bgr,txt,(24,y),cv2.FONT_HERSHEY_SIMPLEX,.63,(245,245,245),2,cv2.LINE_AA)
                writer.write(bgr)
                rows.append(dict(frame=tick,video_time_s=tick*dt,post_step_sim_time_s=(tick+101)*dt,motor_first_sample=logger.count-4,motor_last_sample=logger.count-1,render_state_after_motor_interval_end_s=logger.count*.005,root=root.tolist(),q=raw.robot.data.joint_pos[0].cpu().tolist(),qd=raw.robot.data.joint_vel[0].cpu().tolist(),physical_target=raw.delayed_targets[0].cpu().tolist(),action=actions[0].cpu().tolist(),body_pos=raw.robot.data.body_pos_w[0].cpu().tolist(),body_quat=raw.robot.data.body_quat_w[0].cpu().tolist(),com=m['com'][0].cpu().tolist(),clearance=m['clearance'][0].cpu().tolist(),force=m['foot_force'][0].cpu().tolist(),command=raw.action_commands[0].cpu().tolist(),terminated=int(done[0]),alive=bool(logger.alive[0])))
                if tick in (0,124,249,374,499):cv2.imwrite(str(DEST/f'{label}_frame_{tick:03d}.png'),bgr)
                if tick%100==0:print(json.dumps(dict(clip=label,frames=tick+1,wall_seconds=time.perf_counter()-start)),flush=True)
            writer.release();writer=None;logger.save();assert logger.count==2400
            save(DEST/(label+'_telemetry.json'),rows);save(folder/'frame_alignment.json',rows)
            result=dict(**clip,command_m_s=speed,raw_video=str(raw_path),video=str(final_path),motor_npz=str(folder/'native_motor_200hz.npz'),alignment=str(folder/'frame_alignment.json'),frames=500,fps=50.,duration_s=10.,settling_s=2.,native_samples=2400,record_wall_seconds=time.perf_counter()-start,falls=falls,mean_signed_world_forward_speed_m_s=float(np.mean([r['root'][7] for r in rows])),net_forward_travel_m=float(rows[-1]['root'][0]-first[0]),maximum_command_difference=max(abs(r['command'][0]-speed) for r in rows),geometric_flight_fraction=float(np.mean([min(r['clearance'])>.001 and max(r['force'])<1. for r in rows])))
            manifest['clips'].append(result);save(DEST/'capture_progress.json',manifest);print(json.dumps(dict(clip_complete=result)),flush=True)
        save(DEST/'manifest.json',manifest)
    finally:
        if writer is not None:writer.release()
        env.close()
    print('FINAL_MOTOR_VIDEO_CAPTURE_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:launcher.app.close()
