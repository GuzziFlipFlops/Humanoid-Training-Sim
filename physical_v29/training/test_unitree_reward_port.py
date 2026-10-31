"""CPU counterfactual reward tests, including adversarial motions.

These are controlled telemetry fixtures, not simulated or learned walking.
Expected values are analytic and test rankings/metamorphic properties rather
than reproducing the port's implementation. Live physics is tested separately.
"""
import argparse, copy, hashlib, json, math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
torch.set_num_threads(4)
from unitree_reward_port import UnitreeRewards, Scene, FUNCTIONS, VENDOR, gait_parameters

HERE=Path(__file__).parent
MOTION=np.load(HERE/'assets/motion_reference.npz')
META=json.loads((HERE.parent/'assets/export_metadata.json').read_text())
NAMES=MOTION['names'].tolist()
BODIES=list(json.loads((HERE/'mass_estimate.json').read_text())['links'])
FEET=[BODIES.index(s+'_foot') for s in ['left','right']]
ROOT=float(MOTION['root_height'])
FOOT=.06  # fixture link origin; live probe checks the actual origin separately

def fixture(speed=.35, ticks=0):
    q=torch.tensor(MOTION['stand'],dtype=torch.float32)[None]
    d=NS(root_pos_w=torch.tensor([[0.,0.,ROOT]]),root_quat_w=torch.tensor([[1.,0.,0.,0.]]),
         root_lin_vel_w=torch.tensor([[speed,0.,0.]]),root_lin_vel_b=torch.tensor([[speed,0.,0.]]),
         root_ang_vel_b=torch.zeros(1,3),projected_gravity_b=torch.tensor([[0.,0.,-1.]]),
         joint_pos=q.clone(),default_joint_pos=q.clone(),joint_vel=torch.zeros_like(q),
         joint_acc=torch.zeros_like(q),applied_torque=torch.zeros_like(q),
         soft_joint_pos_limits=torch.stack([q-1,q+1],dim=-1),
         body_pos_w=torch.zeros(1,len(BODIES),3),body_lin_vel_w=torch.zeros(1,len(BODIES),3))
    d.body_pos_w[:,FEET,2]=FOOT
    sd=NS(current_contact_time=torch.zeros(1,len(BODIES)),
          net_forces_w_history=torch.zeros(1,3,len(BODIES),3))
    sd.current_contact_time[:,FEET]=.1;sd.net_forces_w_history[:,:,FEET,2]=170.
    command=torch.tensor([[speed,0.,0.]])
    return NS(scene=Scene(NS(data=d),NS(data=sd)),device='cpu',num_envs=1,step_dt=.02,
              command_manager=NS(get_command=lambda name:command),command=command,
              action_manager=NS(action=torch.zeros_like(q),prev_action=torch.zeros_like(q)),
              termination_manager=NS(terminated=torch.tensor([False])),episode_length_buf=torch.tensor([ticks]))

def contacts(e, flags):
    sd=e.scene['contact_forces'].data
    sd.current_contact_time.zero_();sd.net_forces_w_history.zero_()
    for index,flag in zip(FEET,flags):
        if flag:sd.current_contact_time[0,index]=.1;sd.net_forces_w_history[0,:,index,2]=170.

def run(output):
    checks=[]
    def check(name, truth, evidence=None):
        checks.append(dict(name=name,passed=bool(truth),evidence=evidence))
        assert truth,(name,evidence)
    def near(name,actual,expected):
        actual=float(actual);expected=float(expected);check(name,math.isclose(actual,expected,rel_tol=1e-5,abs_tol=1e-6),dict(actual=actual,expected=expected))
    source=json.loads((VENDOR/'sources.json').read_text())
    for entry in source['files']:
        check('source_hash_'+entry['file'],hashlib.sha256((VENDOR/entry['file']).read_bytes()).hexdigest()==entry['sha256'])
    ports={n:UnitreeRewards(NAMES,BODIES,BODIES,ROOT,FOOT,n) for n in ['g1','h1']}
    for name,p in ports.items():
        e=fixture();total,raw,w=p.evaluate(e)
        near(name+'_velocity_match',raw['track_lin_vel_xy'],1.)
        e.scene['robot'].data.root_lin_vel_w[:,0]=0.
        near(name+'_velocity_miss',p.evaluate(e)[1]['track_lin_vel_xy'],math.exp(-.35**2/.25))
        e.scene['robot'].data.root_ang_vel_b[:,2]=.4
        near(name+'_yaw_miss',p.evaluate(e)[1]['track_ang_vel_z'],math.exp(-.4**2/.25))
        near(name+'_zero_action_rate',raw['action_rate'],0.)
        e.action_manager.action[:]=.2
        near(name+'_abrupt_action_cost',p.evaluate(e)[1]['action_rate'],23*.2**2)
        e.scene['robot'].data.joint_vel[:]=2.;e.scene['robot'].data.applied_torque[:]=3.
        if name=='g1':
            near('g1_energy',p.evaluate(e)[1]['energy'],23*6.)
            near('g1_joint_vel',p.evaluate(e)[1]['joint_vel'],23*4.)
        e.scene['robot'].data.joint_acc[:]=10.
        near(name+'_acceleration_cost',p.evaluate(e)[1]['joint_acc'],23*100.)
        e.scene['robot'].data.joint_pos[0,0]+=1.2
        near(name+'_joint_limit_cost',p.evaluate(e)[1]['dof_pos_limits'],.2)
        e.scene['robot'].data.root_pos_w[:,2]+= .12
        near(name+'_height_error',p.evaluate(e)[1]['base_height'],.12**2)
        e.scene['robot'].data.projected_gravity_b[:]=torch.tensor([.6,0.,-.8])
        near(name+'_tilt_cost',p.evaluate(e)[1]['flat_orientation_l2'],.36)
        e.scene['robot'].data.root_lin_vel_b[:,2]=.3
        near(name+'_vertical_motion',p.evaluate(e)[1]['base_linear_velocity'],.09)
        e.scene['robot'].data.root_ang_vel_b[:,:2]=torch.tensor([.2,.3])
        near(name+'_angular_motion',p.evaluate(e)[1]['base_angular_velocity'],.13)
        e=fixture();e.termination_manager.terminated[:]=True
        near(name+'_fallen_alive_removed',p.evaluate(e)[1]['alive'],0.)
        e=fixture();e.scene['contact_forces'].data.net_forces_w_history[:,:,BODIES.index('torso'),2]=20.
        near(name+'_torso_contact',p.evaluate(e)[1]['undesired_contacts'],1.)
        e=fixture();e.scene['robot'].data.body_lin_vel_w[:,FEET,0]=.2
        near(name+'_contact_slide',p.evaluate(e)[1]['feet_slide'],.4)
        contacts(e,[False,False])
        near(name+'_swing_not_slip',p.evaluate(e)[1]['feet_slide'],0.)
        target=p.terms['feet_clearance']['params']['target_height']
        e.scene['robot'].data.body_pos_w[:,FEET,2]=target
        near(name+'_raised_swing_clearance',p.evaluate(e)[1]['feet_clearance'],1.)
        e.scene['robot'].data.body_pos_w[:,FEET,2]=FOOT
        expected=math.exp(-2*(target-FOOT)**2*math.tanh(.4)/.05)
        near(name+'_moving_low_feet_clearance',p.evaluate(e)[1]['feet_clearance'],expected)
        e.scene['robot'].data.body_lin_vel_w.zero_()
        near(name+'_stationary_clearance_limitation',p.evaluate(e)[1]['feet_clearance'],1.)
        e=fixture();total=p.evaluate(e)[0]
        e.step_dt=.01;near(name+'_dt_invariance',p.evaluate(e)[0]*2,total)
        # A rigid 90 degree yaw rotation must preserve commanded forward speed.
        e=fixture();e.scene['robot'].data.root_quat_w[:]=torch.tensor([2**-.5,0.,0.,2**-.5])
        e.scene['robot'].data.root_lin_vel_w[:]=torch.tensor([0.,.35,0.])
        near(name+'_yaw_frame_invariance',p.evaluate(e)[1]['track_lin_vel_xy'],1.)
        e=fixture();groups=p.mapping
        for group in [key for key in groups if key.startswith('joint_deviation_')]:
            joint=groups[group][0];e.scene['robot'].data.joint_pos[0,NAMES.index(joint)]+=.17
            near(name+'_'+group+'_anatomy_mapping',p.evaluate(e)[1][group],.17)
            e.scene['robot'].data.joint_pos.copy_(e.scene['robot'].data.default_joint_pos)
    e=fixture();e.scene['contact_forces'].data.net_forces_w_history[:,:,FEET[0],2]=700.
    near('h1_force_excess',ports['h1'].evaluate(e)[1]['feet_contact_forces'],200.)
    for name,p in ports.items():
        e=fixture(0.,3);contacts(e,[True,False])
        near(name+'_zero_command_gait',p.evaluate(e)[1]['gait'],0. if name=='g1' else 2.)
    # Exhaustive contact combinations, phases and commands: original walk task.
    for speed in [0.,.15,.35,1.,1.5,2.,2.5]:
        e=fixture(speed);period,duty=gait_parameters(e.command)
        for ticks in range(100):
            e.episode_length_buf[:]=ticks
            phase=((ticks*.02)%float(period))/float(period)
            expected_contact=[phase<float(duty),(phase+.5)%1<float(duty)]
            for left,right in [(False,False),(True,False),(False,True),(True,True)]:
                contacts(e,[left,right]);actual=ports['g1'].evaluate(e)[1]['gait']
                expected=sum(a==b for a,b in zip([left,right],expected_contact)) if speed>.1 else 0
                # Boundary rounding can differ by one floating point ULP.
                if any(abs(((phase+off)%1)-float(duty))<1e-5 or min((phase+off)%1,1-(phase+off)%1)<1e-5 for off in [0,.5]):continue
                near(f'gait_{speed}_{ticks}_{int(left)}{int(right)}',actual,expected)
        if speed>=1.5:
            both_air=sum(not(a<float(duty) or (a+.5)%1<float(duty)) for a in np.arange(1000)/1000)
            check(f'run_{speed}_sampled_flight_fraction',abs(both_air/1000-.14)<=.0021,dict(actual=both_air/1000,expected=.14,discrete_sample_tolerance=.0021))
    # Whole cycles: adversarial bad motions may get positive reward; they must
    # rank below matched travel/alternation, and independent gates remain needed.
    traces={}
    cases=['matched_walk','stationary','in_place_alternation','ice_skating','bunny_hop','reverse_travel','sideways','tilted','fallen','rough_actions','foot_flick','one_foot_only']
    for name,p in ports.items():
        for case in cases:
            rows=[];period=.8 if name=='g1' else .6
            for tick in range(120):
                e=fixture(.35,tick);d=e.scene['robot'].data;phase=(tick*.02%period)/period
                flags=[phase<.55,(phase+.5)%1<.55];contacts(e,flags)
                target=p.terms['feet_clearance']['params']['target_height']
                for i,(b,on) in enumerate(zip(FEET,flags)):
                    if not on:d.body_pos_w[0,b,2]=target;d.body_lin_vel_w[0,b,0]=.7
                if case=='stationary':contacts(e,[True,True]);d.body_lin_vel_w.zero_();d.root_lin_vel_w.zero_()
                if case=='in_place_alternation':d.root_lin_vel_w.zero_()
                if case=='ice_skating':contacts(e,[True,True]);d.body_lin_vel_w[:,FEET,0]=.35;d.body_pos_w[:,FEET,2]=FOOT
                if case=='bunny_hop':contacts(e,[phase<.55]*2);d.root_lin_vel_b[:,2]=.6
                if case=='reverse_travel':d.root_lin_vel_w[:,0]=-.35
                if case=='sideways':d.root_lin_vel_w[:]=torch.tensor([0.,.35,0.])
                if case=='tilted':d.projected_gravity_b[:]=torch.tensor([.6,0.,-.8])
                if case=='fallen':
                    d.root_pos_w[:,2]=.35;d.projected_gravity_b[:]=torch.tensor([1.,0.,0.]);e.termination_manager.terminated[:]=True
                    e.scene['contact_forces'].data.net_forces_w_history[:,:,BODIES.index('torso'),2]=170.
                if case=='rough_actions':e.action_manager.action[:]=1. if tick%2 else -1.;e.action_manager.prev_action[:]=-e.action_manager.action
                if case=='foot_flick':d.root_lin_vel_w.zero_();d.body_lin_vel_w[:,FEET,0]=1.;contacts(e,[True,True]);d.joint_vel[:]=6.;d.applied_torque[:]=10.
                if case=='one_foot_only':contacts(e,[True,False])
                total,raw,w=p.evaluate(e)
                check(f'{name}_{case}_{tick}_finite',bool(torch.isfinite(total).all()))
                rows.append(dict(t=tick*.02,rate=float(total/e.step_dt),weighted={k:float(v) for k,v in w.items()}))
            traces[name+'_'+case]=dict(mean_rate=float(np.mean([r['rate'] for r in rows])),rows=rows)
        good=traces[name+'_matched_walk']['mean_rate']
        for case in cases[1:]:
            bad=traces[name+'_'+case]['mean_rate']
            check(name+'_ranks_'+case,good>bad,dict(matched=good,adversarial=bad))
    report=dict(passed=True,checks=len(checks),unitree_commit=source['unitree_commit'],port_sha256=hashlib.sha256((HERE/'unitree_reward_port.py').read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),profiles={n:p.manifest() for n,p in ports.items()},
                scope='Controlled CPU telemetry tests, not physics, learning or a gait qualification',
                limitations=['Stationary feet maximize upstream clearance reward','H1 rewards gait at zero command','In-place alternating feet can get gait reward','Instantaneous positive reward does not imply net travel or survival','Neither profile guarantees a human motion style'],
                comparisons={k:v['mean_rate'] for k,v in traces.items()},checks_detail=checks,traces=traces)
    output.mkdir(parents=True,exist_ok=True);(output/'counterfactual_rewards.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k in ['passed','checks','comparisons','limitations']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'diagnostics/unitree_port_20261002')
    run(parser.parse_args().output)
