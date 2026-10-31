"""Render free-base scripted reward probes with frame-synchronous telemetry.

Scripted diagnostics are NEVER classified as a learned gait or a passing gate.
Automatic resets are held so a fall remains visible; normal training/evaluation
uses the unchanged termination/reset rules. No root support or balance helper.
"""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE.parent))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
parser.add_argument('--duration',type=float,default=4.)
parser.add_argument('--checkpoint',default='')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.enable_cameras=True
app=AppLauncher(args)
import numpy as np,torch
from PIL import Image,ImageDraw,ImageOps
import cv2
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment_unitree import LocomotionEnv,LocomotionCfg,MOTION
from unitree_reward_port import UnitreeRewards
torch.set_num_threads(4)

class ProbeEnv(LocomotionEnv):
 hold_resets=False
 def _reset_idx(self,env_ids):
  if self.hold_resets:return
  super()._reset_idx(env_ids)

def main():
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 assert not(out/'result.json').exists(),'Preserve completed probe'
 converter=UrdfConverter(UrdfConverterCfg(asset_path=str(HERE/'assets/robot_training.urdf'),
  usd_dir=str(HERE/'assets/usd'),usd_file_name='robot_training.usd',fix_base=False,root_link_name='pelvis',
  merge_fixed_joints=True,make_instanceable=True,collision_from_visuals=False,self_collision=True,
  joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
 cfg=LocomotionCfg();cfg.scene.num_envs=1;cfg.evaluation=True;cfg.stage_speed=.35
 cfg.sim.device=args.device or 'cuda:0';cfg.robot_usd=converter.usd_path;cfg.episode_length_s=100.
 cfg.viewer.resolution=(720,720);cfg.sim.render_interval=20
 env=ProbeEnv(cfg,render_mode='rgb_array');env.initialize_randomization()
 light=sim_utils.DomeLightCfg(intensity=1000.);light.func('/World/ProbeLight',light)
 env.sim.set_camera_view((1.9,-2.3,1.6),(0.,0.,.8))
 h1=UnitreeRewards(env.robot.joint_names,env.robot.body_names,env.contacts.body_names,float(MOTION['root_height']),
                  env.reward_port.adaptations['foot_origin_standing_height_m'],'h1')
 (out/'reward_manifest.json').write_text(json.dumps(dict(g1=env.reward_port.manifest(),h1=h1.manifest()),indent=2))
 policy=None;runner=None
 if args.checkpoint:
  # Frozen source worker owns the compatible architecture/configuration.
  saved=torch.load(args.checkpoint,map_location=env.device,weights_only=False)
  assert saved['infos']['contract']=='physical_v29_history5_unitree_defaultoffset_v2'
  signature=hashlib.sha256(b''.join((HERE/n).read_bytes() for n in ['unitree_reward_port.py','environment_unitree.py','unitree_vendor/sources.json'])).hexdigest()
  assert saved['infos']['reward_signature']==signature,'Use matching frozen environment/reward source for policy recording'
  config=json.loads((Path(args.checkpoint).parent/'agent.json').read_text())
  wrapped=RslRlVecEnvWrapper(env,clip_actions=3.)
  runner=OnPolicyRunner(wrapped,config,log_dir=None,device=env.device)
  runner.load(args.checkpoint,load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
  policy=runner.get_inference_policy(env.device)
 cases=['stand','alternating_targets','in_place_foot_flick','slide_impulse','hop_impulse','roll_impulse','combined_pitch_roll']
 if policy:cases=['policy_stand','policy_walk']
 allrows=[];summary=[]
 for case in cases:
  env.hold_resets=False;env.reset();env.hold_resets=True
  env.commands[:]=0.;env.commands[:,0]=0. if case in ['stand','policy_stand'] else .35
  rows=[];frames=[];first_done=None
  for tick in range(round(args.duration/env.step_dt)):
   t=tick*env.step_dt;q=env.stand[None].clone();action=torch.zeros_like(q)
   if case in ['alternating_targets','in_place_foot_flick']:
    for s,offset in [('left',0.),('right',np.pi)]:
     phase=2*np.pi*t/.8+offset;swing=max(0.,np.sin(phase))
     if case=='alternating_targets':
      q[:,env.lookup[s+'_hip_pitch']]+=.22*np.cos(phase)
      q[:,env.lookup[s+'_knee_pitch']]+=.45*swing
      q[:,env.lookup[s+'_ankle_pitch']]-=.20*swing
     else:q[:,env.lookup[s+'_ankle_pitch']]+=.10*np.sin(phase)
   if case=='combined_pitch_roll':
    for s in ['left','right']:
     q[:,env.lookup[s+'_hip_pitch']]+=.12*np.sin(2*np.pi*t/1.2)
     q[:,env.lookup[s+'_hip_roll']]+=.07*np.sin(2*np.pi*t/1.2)
     q[:,env.lookup[s+'_ankle_pitch']]+=.07*np.sin(2*np.pi*t/1.2)
     q[:,env.lookup[s+'_ankle_roll']]+=.06*np.sin(2*np.pi*t/1.2)
   action=(q-env.stand)/env.scales
   if policy:action=policy({'policy':env._get_observations()['policy']})
   if tick==25 and case.endswith('impulse'):
    v=env.robot.data.root_vel_w.clone()
    if case=='slide_impulse':v[:,0]+=1.
    elif case=='hop_impulse':v[:,2]+=1.2
    elif case=='roll_impulse':v[:,3]+=2.
    env.robot.write_root_velocity_to_sim(v)
   _,reward,done,timeout,_=env.step(action)
   m=env.telemetry();d=env.robot.data
   g1rate={k:float(v[0]) for k,v in env.last_reward_terms.items()}
   htotal,hraw,hw=h1.evaluate(env.reward_proxy)
   if bool(done[0]) and first_done is None:first_done=t+env.step_dt
   row=dict(case=case,t=t+env.step_dt,reward_g1=float(reward[0]),rate_g1=sum(g1rate.values()),rate_h1=float(htotal[0]/env.step_dt),
    terms_g1=g1rate,terms_h1={k:float(v[0]) for k,v in hw.items()},done=bool(done[0]),
    failures=[k for k,v in env.termination_reasons.items() if bool(v[0])],
    root=d.root_state_w[0].cpu().tolist(),q=d.joint_pos[0].cpu().tolist(),target=env.targets[0].cpu().tolist(),
    foot_pos=m['foot_position'][0].cpu().tolist(),force=m['foot_force'][0].cpu().tolist(),
    contact=m['contact'][0].cpu().tolist(),clearance=m['clearance'][0].cpu().tolist(),slip=float(m['slip'][0]))
   rows.append(row)
   if tick%5==0:
    arr=env.render()
    if tick==0:
     for warm in range(4):arr=env.render()
    img=Image.fromarray(arr);draw=ImageDraw.Draw(img)
    draw.rectangle((0,0,720,66),fill='white')
    draw.text((8,5),f'{case}   t={row["t"]:.2f}s   G1 rate={row["rate_g1"]:.2f}   H1 rate={row["rate_h1"]:.2f}',fill='black')
    draw.text((8,25),f'contact={row["contact"]} clearance={[round(x,3) for x in row["clearance"]]} done={row["done"]}',fill='black')
    draw.text((8,45),'SCRIPTED FREE-BASE PROBE; not learned walking' if not policy else 'UNQUALIFIED LEARNED CANDIDATE; not a passing gait',fill='black')
    framefile=out/f'{case}_{tick:04d}.png';img.save(framefile);frames.append(img)
    row['frame']=framefile.name
  video=cv2.VideoWriter(str(out/(case+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),10.,(720,720))
  for frame in frames:video.write(cv2.cvtColor(np.array(frame),cv2.COLOR_RGB2BGR))
  video.release()
  for start in range(0,len(frames),12):
   sheet=Image.new('RGB',(1280,3*320),'white')
   for k,frame in enumerate(frames[start:start+12]):sheet.paste(frame.resize((320,320)),((k%4)*320,(k//4)*320))
   sheet.save(out/f'{case}_sheet_{start//12:02d}.jpg',quality=93)
  summary.append(dict(case=case,first_physical_failure_s=first_done,mean_reward_rate_g1=float(np.mean([r['rate_g1'] for r in rows])),
   mean_reward_rate_h1=float(np.mean([r['rate_h1'] for r in rows])),final_root=rows[-1]['root'],frames=len(frames),
   frame_sequence=case+'.mp4',qualification='None; scripted diagnostic' if not policy else 'None; independently gated elsewhere'))
  allrows+=rows
  print('[PROBE] '+json.dumps(summary[-1]),flush=True)
 (out/'telemetry.json').write_text(json.dumps(dict(joint_names=env.robot.joint_names,rows=allrows),indent=2))
 (out/'result.json').write_text(json.dumps(dict(completed=True,cases=summary,gravity=True,fixed_base=False,normal_actuator_limits=True,
  resets_held_for_visualization=True,checkpoint=args.checkpoint,scope='Diagnostic motion/reward frame inspection; does not qualify locomotion'),indent=2))
 env.close()

if __name__=='__main__':
 try:main()
 except BaseException:
  traceback.print_exc();Path(args.output).mkdir(parents=True,exist_ok=True)
  (Path(args.output)/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc())))
 finally:app.app.close()
