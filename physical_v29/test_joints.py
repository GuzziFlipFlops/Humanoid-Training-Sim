"""Fixed-base, zero-gravity PhysX joint validation. Test drives are not hardware limits."""
import argparse,json,math,time
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--hold_seconds',type=float,default=0.)
parser.add_argument('--convert_only',action='store_true')
parser.add_argument('--force_convert',action='store_true',help='Regenerate USD from the current URDF before testing')
parser.add_argument('--hardware_actuators',action='store_true',help='Use user motor assumptions and CAD differential mapping, instead of the original fixture drives')
parser.add_argument('--validation_dir',default=None,help='Output directory; defaults to the current actuator profile or original fixture')
parser.add_argument('--visual_fast',action='store_true',help='Large smooth one-joint-at-a-time movements with GUI controls')
parser.add_argument('--seconds_per_joint',type=float,default=2.5,help='Desired preview duration per joint; extended when required by speed limits')
parser.add_argument('--repeat',action='store_true',help='Repeat the visual cycle until the window is closed')
parser.add_argument('--exit_after_cycle',action='store_true',help='Close the visual preview after one cycle')
parser.add_argument('--joint_count',type=int,default=23,help='Limit visual preview length for a startup check')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args);simulation_app=launcher.app

import numpy as np
import torch
torch.set_num_threads(1)
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation,ArticulationCfg
from scipy.spatial.transform import Rotation
from pxr import UsdPhysics

HERE=Path(__file__).resolve().parent;ASSET=HERE/'assets'
meta=json.loads((ASSET/'export_metadata.json').read_text())
profile=meta.get('hardware_validation_dir','validation') if args.hardware_actuators else 'validation'
OUT=HERE/(args.validation_dir or (profile+'_visual' if args.visual_fast else profile));OUT.mkdir(parents=True,exist_ok=True)
report={'scope':'23 anatomical output joints; wrist and fingers static','fixture':'fixed base, gravity off, no contact dynamics, 1kg/link proxy masses, test-only PD drives; hardware torque/velocity limits pending','native_joint_map_source':meta.get('native_joint_map_source'),'native_archive_sha256':meta.get('native_archive_sha256'),'started':time.time(),'joints':[]}
if args.hardware_actuators:
  report['fixture']='fixed base, gravity off, no contacts, 1kg/link proxy masses and test-only 0.05 kg m2 joint armature; September 30 user actuator assumptions with CAD-dependent coupled ankle/waist transmission. No real thermal/voltage/pack qualification.'
def save():
 (OUT/'joint_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def fk(q):
 poses={'pelvis':(np.array([0,0,meta['base_height_over_floor_m']]),np.eye(3))}
 pending=list(meta['joints'])
 while pending:
  progress=False
  for row in pending[:]:
   if row['parent'] not in poses:continue
   p,r=poses[row['parent']];axis=np.array(row['axis']);angle=q[row['name']]
   poses[row['child']]=(p+r@np.array(row['origin_xyz_m']),r@Rotation.from_rotvec(axis*angle).as_matrix());pending.remove(row);progress=True
  assert progress
 return poses

def main():
 cfg=UrdfConverterCfg(asset_path=str(ASSET/'humanoid_physical_v29.urdf'),usd_dir=str(ASSET/'usd'),usd_file_name='humanoid_physical_v29.usd',fix_base=True,root_link_name='pelvis',force_usd_conversion=args.force_convert or not (ASSET/'usd/humanoid_physical_v29.usd').exists(),make_instanceable=False,merge_fixed_joints=True,collision_from_visuals=False,self_collision=False,joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=800.,damping=40.)))
 converter=UrdfConverter(cfg);report['usd']=converter.usd_path
 report['runtime']={'isaaclab':'0.54.4','isaacsim':'5.1.0.0','device':args.device};save()
 print('V29 USD ready:',converter.usd_path,flush=True)
 if args.convert_only:return
 sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.01 if args.visual_fast else .005,device=args.device))
 sim.set_camera_view(eye=(2.0,-2.0,1.35),target=(0.,0.,.80))
 light=sim_utils.DomeLightCfg(intensity=100.,visible_in_primary_ray=False);light.func('/World/Light',light)
 actuators={'TEST_FIXTURE_ONLY':ImplicitActuatorCfg(joint_names_expr=['.*'],effort_limit_sim=1000.,velocity_limit_sim=10.,stiffness=800.,damping=40.)}
 if args.hardware_actuators:
  from hardware_actuators import make_actuators
  heavy=lambda name:name.startswith('waist_') or '_hip_' in name or '_knee_' in name
  actuators=make_actuators(stiffness={j['name']:120. if heavy(j['name']) else 35. for j in meta['joints']},damping={j['name']:(8. if args.visual_fast else 15.) if heavy(j['name']) else 4. for j in meta['joints']},armature=.05)
 robot_cfg=ArticulationCfg(prim_path='/World/Robot',spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path,rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,max_angular_velocity=100.),articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=64,solver_velocity_iteration_count=8)),init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,meta['base_height_over_floor_m']),joint_pos={'.*':0.},joint_vel={'.*':0.}),soft_joint_pos_limit_factor=1.,actuators=actuators)
 robot=Articulation(robot_cfg)
 # Conservative collision boxes are deliberately disabled for this axis test.
 stage=sim.stage;disabled=0
 for prim in stage.Traverse():
  if prim.GetPath().pathString.startswith('/World/Robot') and prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False);disabled+=1
 report['disabled_collision_prims']=disabled
 sim.reset();dt=sim.get_physics_dt()
 names=robot.joint_names;lookup={n:i for i,n in enumerate(names)};body_lookup={n:i for i,n in enumerate(robot.body_names)}
 expected={j['name'] for j in meta['joints']};assert set(names)==expected,(names,expected)
 assert len(robot.body_names)==24
 nominal=torch.zeros_like(robot.data.default_joint_pos);zeros=nominal.clone()
 robot.write_joint_state_to_sim(nominal,zeros);robot.reset();robot.set_joint_position_target(nominal)
 motor_peaks={};invalid_transmission_steps=0
 ticks=0
 def step(targets,n=1,velocity_targets=None):
  nonlocal invalid_transmission_steps
  nonlocal ticks
  robot.set_joint_position_target(targets)
  robot.set_joint_velocity_target(zeros if velocity_targets is None else velocity_targets)
  for _ in range(n):
   if not simulation_app.is_running():raise KeyboardInterrupt
   robot.write_data_to_sim()
   sim.step(render=not args.headless and ticks%(3 if args.visual_fast else 4)==0);robot.update(dt);ticks+=1
   if args.hardware_actuators:
    actuator=robot.actuators['cad_hardware']
    if actuator.invalid_transmission is not None:invalid_transmission_steps+=int(actuator.invalid_transmission.any())
    for key,state in actuator.last_motor_states.items():
     motor_peaks[key]=max(motor_peaks.get(key,0),float(state['motor_applied'].abs().max()))
 def measure():
  q=robot.data.joint_pos[0].cpu().numpy();pose=fk({n:float(q[i]) for i,n in enumerate(names)})
  positions=robot.data.body_link_pos_w[0].cpu().numpy();quats=robot.data.body_link_quat_w[0].cpu().numpy();errors=[]
  for name,(p,r) in pose.items():
   bi=body_lookup[name];qr=quats[bi];actual=Rotation.from_quat([qr[1],qr[2],qr[3],qr[0]]).as_matrix();angle=Rotation.from_matrix(r.T@actual).magnitude()
   errors.append({'link':name,'position_error_m':float(np.linalg.norm(positions[bi]-p)),'rotation_error_rad':float(angle)})
  return q,errors
 step(nominal,100);q0,initial_errors=measure()
 report['rest_fk_max_position_error_m']=max(e['position_error_m'] for e in initial_errors);report['rest_fk_max_rotation_error_rad']=max(e['rotation_error_rad'] for e in initial_errors)
 assert report['rest_fk_max_position_error_m']<.002 and report['rest_fk_max_rotation_error_rad']<.01,initial_errors
 if args.visual_fast:
  from visual_joint_test import run_visual_test
  run_visual_test(args,simulation_app,sim,robot,meta,report,save,step,measure,nominal,lookup,dt,motor_peaks)
  report['invalid_transmission_steps']=invalid_transmission_steps;save()
  return
 for count,row in enumerate(meta['joints'],1):
  name=row['name'];i=lookup[name];lo,hi=row['limits_rad'];commands=[]
  for sign in (-1,1):
   room=-lo if sign<0 else hi
   if room>.025:commands.append(sign*min(math.radians(12),room*.65))
  assert commands
  rec={'name':name,'native_name':row['native_name'],'axis':row['axis'],'limits_rad':[lo,hi],'samples':[]}
  for command in commands:
   targets=nominal.clone();targets[0,i]=command
   step(targets,450 if args.hardware_actuators else 180)
   measured,errors=measure();err=float(abs(measured[i]-command));other=float(np.max(np.abs(np.delete(measured,i))))
   maxpos=max(e['position_error_m'] for e in errors);maxrot=max(e['rotation_error_rad'] for e in errors)
   rec['samples'].append({'command_rad':command,'measured_rad':float(measured[i]),'tracking_error_rad':err,'other_joint_motion_rad':other,'fk_max_position_error_m':maxpos,'fk_max_rotation_error_rad':maxrot,'pass':err<.005 and other<.005 and maxpos<.002 and maxrot<.01})
  step(nominal,450 if args.hardware_actuators else 180);qreset,_=measure();rec['reset_error_rad']=float(np.max(np.abs(qreset)));rec['pass']=all(s['pass'] for s in rec['samples']) and rec['reset_error_rad']<.005
  report['joints'].append(rec);save();print(f"[{count:02d}/23] {name}: {'PASS' if rec['pass'] else 'FAIL'} tracking={max(s['tracking_error_rad'] for s in rec['samples']):.6f} reset={rec['reset_error_rad']:.6f}",flush=True)
 report['all_pass']=len(report['joints'])==23 and all(j['pass'] for j in report['joints']);report['finished']=time.time()
 if args.hardware_actuators:
  report['differential_applied_motor_peak_Nm']=motor_peaks
  report['invalid_transmission_steps']=invalid_transmission_steps
  report['all_pass']=report['all_pass'] and invalid_transmission_steps==0 and all(v<=(30 if k=='waist' else 45)+1e-4 for k,v in motor_peaks.items())
 save()
 assert report['all_pass'],'Joint validation failed; see JSON'
 print('PASS: all 23 joints moved, matched measured-frame FK, and reset.'+(' User actuator assumptions and coupled CAD mapping passed; physical qualification remains pending.' if args.hardware_actuators else ' Original test drives are not physical limits.'),flush=True)
 for _ in range(round(args.hold_seconds/dt)):step(nominal)

if __name__=='__main__':
 try:main()
 except Exception as e:report['error']=repr(e);save();raise
 finally:simulation_app.close(wait_for_replicator=False)
