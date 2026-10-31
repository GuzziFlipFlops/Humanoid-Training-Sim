"""Behavioral reward rankings, lifetime returns, support and transfer regressions.

CPU telemetry fixtures are not simulated/learned gait qualification. The
supervisor separately runs actual physics, PPO, reload and independent gates.
"""
import argparse, ast, copy, hashlib, json, math
from pathlib import Path
import torch
from test_unitree_reward_port import fixture, contacts, NAMES, BODIES, FEET, ROOT, FOOT
from stage_rewards import CurriculumRewards
from support_geometry import horizontal_support
from reward_warm_start import validate_actor_transfer

HERE=Path(__file__).resolve().parent
R22=HERE/'runs/locomotion_20261002_r22_unitree_nominal'
ACTOR=R22/'016_stand_train_001_part_09/candidate.pt'
torch.set_num_threads(1)


def telemetry(e,flags=(True,True),drift=0.,tilt=0.):
    contacts(e,flags)
    d=e.scene['robot'].data
    feet=torch.tensor([[[0.,-.1,0.],[0.,.1,0.]]])+d.root_pos_w[:,None,:]
    feet[:,:,2]=FOOT
    return dict(tilt=torch.tensor([tilt]),torso_tilt=torch.tensor([tilt]),
        height=d.root_pos_w[:,2],reference_height=torch.tensor([ROOT]),
        velocity=d.root_lin_vel_b,angular=d.root_ang_vel_b,
        contact=torch.tensor([flags]),foot_position=feet,
        com=d.root_pos_w+torch.tensor([[0.,0.,-.2]]),
        clearance=torch.zeros(1,2),drift=torch.tensor([drift]),heading_error_rad=torch.zeros(1))


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checks=[]
    def check(name,condition,**evidence):
        assert bool(condition),(name,evidence)
        checks.append(dict(name=name,passed=True,evidence=evidence))
    reward=CurriculumRewards(NAMES,BODIES,BODIES,ROOT,FOOT)
    def score(e,m=None):
        e.cad_metrics=telemetry(e) if m is None else m
        total,raw,w=reward.evaluate(e)
        return float(total/e.step_dt),{k:float(v) for k,v in w.items()}
    base,terms=score(fixture(0.))
    examples=[dict(case='quiet_stand',rate=base,terms=terms)]
    for label in ['slide','lean','one_loaded_foot','drift','upside_down']:
        e=fixture(0.);m=telemetry(e)
        if label=='slide':
            e.scene['robot'].data.root_lin_vel_b[0,1]=.15
            e.scene['robot'].data.body_lin_vel_w[:,FEET,1]=.15
        elif label=='lean':m['tilt'][:]=math.radians(12.5);m['torso_tilt'][:]=math.radians(12.5)
        elif label=='one_loaded_foot':m=telemetry(e,(True,False))
        elif label=='drift':m['drift'][:]=.3
        elif label=='upside_down':m['tilt'][:]=math.pi;m['torso_tilt'][:]=math.pi
        rate,w=score(e,m)
        check('quiet_beats_'+label,base>rate,quiet=base,alternative=rate)
        examples.append(dict(case=label,rate=rate,terms=w))
    check('stationary_feet_no_clearance_credit',terms['feet_clearance']==0.)
    check('standing_no_gait_or_flight',terms['gait']==terms['running_flight']==0.)
    # The old floor collapsed these distinctions under action-change noise.
    for std in [.0,.2,.35,.5,1.]:
        e=fixture(0.);e.action_manager.action[:]=math.sqrt(2)*std
        m=telemetry(e);quiet,_=score(e,m)
        m['tilt'][:]=math.radians(12.5);m['torso_tilt'][:]=math.radians(12.5)
        lean,_=score(e,m)
        check('posture_feedback_survives_noise_'+str(std),quiet>lean+1. and lean>0.,quiet=quiet,lean=lean)
    # Monotonic penalties remain distinguishable, without flattening the sum.
    noise_scores=[]
    for amount in [0.,.1,.2,.35,.5,1.,3.]:
        e=fixture(0.);e.action_manager.action[:]=amount
        noise_scores.append(score(e)[0])
    check('action_noise_monotonic',all(a>b for a,b in zip(noise_scores,noise_scores[1:])),rates=noise_scores)
    # Prefix-equivalent lifetime returns: exiting cannot avoid future costs.
    for rate in [0.,.0001,.2,2.,base]:
        full=sum(.99**i*rate*.02 for i in range(1000))
        for length in [1,4,50,300]:
            exit_return=sum(.99**i*rate*.02 for i in range(length))-.99**(length-1)*4.
            check(f'no_early_exit_advantage_{rate}_{length}',exit_return<full)
    for dt in [.005,.02,.04]:
        e=fixture(0.);e.step_dt=dt;e.termination_manager.terminated[:]=True
        rate,w=score(e)
        check('physical_fault_not_offset_'+str(dt),math.isclose(rate,-200.,abs_tol=1e-5) and all(v==0. for k,v in w.items() if k!='physical_termination'))
        check('terminal_event_units_'+str(dt),math.isclose(rate*dt,-200*dt,abs_tol=1e-5))
    for amount in [1.,10.,100.]:
        e=fixture(0.);e.action_manager.action[:]=amount
        rate,w=score(e)
        check('large_cost_keeps_regular_nonnegative_'+str(amount),rate>=0. and 'living_clip_adjustment' not in w)
    # Whole walking cycles, rather than selecting a favorable phase.
    cycles={}
    for name in ['matched_walk','in_place_steps','skating','walking_hop','matched_run','in_place_running_hop']:
        speed=1.5 if 'run' in name else .35
        rows=[]
        for tick in range(80):
            e=fixture(speed,tick);d=e.scene['robot'].data
            if name in ['in_place_steps','in_place_running_hop']:d.root_lin_vel_b.zero_();d.root_lin_vel_w.zero_()
            period=1/(.9+.5*speed) if speed>=1.5 else .8
            duty=.43 if speed>=1.5 else .55
            ph=tick*.02/period%1
            flags=(ph<duty,(ph+.5)%1<duty)
            if name=='skating':flags=(True,True);d.body_lin_vel_w[:,FEET,0]=speed
            if name in ['walking_hop','in_place_running_hop']:flags=(ph<.5,ph<.5)
            m=telemetry(e,flags);m['clearance'][0]=torch.tensor([0. if f else (.065 if speed>=1.5 else .04) for f in flags])
            rows.append(score(e,m)[0])
        cycles[name]=sum(rows)/len(rows)
    for name in ['in_place_steps','skating','walking_hop']:
        check('full_cycle_walk_beats_'+name,cycles['matched_walk']>cycles[name],matched=cycles['matched_walk'],alternative=cycles[name])
    check('running_beats_in_place_hop',cycles['matched_run']>cycles['in_place_running_hop'],cycles=cycles)
    e=fixture(1.5);d=e.scene['robot'].data;d.root_lin_vel_b.zero_();d.root_lin_vel_w.zero_()
    _,w=score(e,telemetry(e,(False,False)))
    check('in_place_flight_not_rewarded',w['running_flight']==w['feet_clearance']==w['gait']==0.)
    # Support must depend on gravity projection, not root roll/pitch or location.
    feet=torch.tensor([[[0.,-.1,-.9],[0.,.1,-.9]]]);com=torch.tensor([[0.,0.,-.35]])
    for pitch in [0.,math.radians(12.5),-.4]:
        q=torch.tensor([[math.cos(pitch/2),0.,math.sin(pitch/2),0.]])
        outside,_=horizontal_support(q,torch.zeros(1,3),feet,com,torch.tensor([[True,True]]))
        check('tilted_root_COM_over_support_'+str(pitch),not outside.item())
    for yaw in [0.,.4,1.7]:
        q=torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]])
        rotation=torch.tensor([[math.cos(yaw),-math.sin(yaw),0.],[math.sin(yaw),math.cos(yaw),0.],[0.,0.,1.]])
        shift=torch.tensor([[12.,-4.,.5]])
        outside,_=horizontal_support(q,shift,feet@rotation.T+shift[:,None],com@rotation.T+shift,torch.tensor([[True,True]]))
        check('yaw_translation_invariance_'+str(yaw),not outside.item())
    for comx in [-.2,.2]:
        outside,_=horizontal_support(torch.tensor([[1.,0.,0.,0.]]),torch.zeros(1,3),feet,
            torch.tensor([[comx,0.,-.35]]),torch.tensor([[True,True]]))
        check('genuine_COM_outside_'+str(comx),outside.item())
    outside,_=horizontal_support(torch.tensor([[1.,0.,0.,0.]]),torch.zeros(1,3),feet,com,torch.tensor([[True,False]]))
    check('missing_standing_support_fails',outside.item())
    # No changes to joint/controller/reset/observation contract or gate thresholds.
    old_source=R22/'source/physical_v29'
    protected=['training/environment.py','training/assets/robot_training.urdf','training/assets/motion_reference.npz',
        'training/mass_estimate.json','hardware_actuators.py','cad_differential.py',
        'assets/actuator_limits.json','assets/differential_geometry.json','assets/export_metadata.json']
    for rel in protected:
        current=HERE.parent/rel
        check('protected_'+rel,current.read_bytes()==(old_source/rel).read_bytes())
    def node(path,name):
        return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
    for method in ['_reference','_pre_physics_step','_get_observations','_reset_idx']:
        check('unchanged_'+method,ast.dump(node(HERE/'environment_unitree.py',method),include_attributes=False)==
            ast.dump(node(old_source/'training/environment_unitree.py',method),include_attributes=False))
    check('all_independent_gate_thresholds_unchanged',ast.dump(node(HERE/'evaluation.py','gate_case'),include_attributes=False)==
        ast.dump(node(old_source/'training/evaluation.py','gate_case'),include_attributes=False))
    saved=torch.load(ACTOR,map_location='cpu',weights_only=True)
    transfer=validate_actor_transfer(saved,HERE)
    check('selected_actor_compatible',transfer['actor_only'] and not transfer['qualification_reused'])
    bad=copy.deepcopy(saved);bad['infos']['actuator_sha256']='wrong'
    try:validate_actor_transfer(bad,HERE)
    except AssertionError:check('changed_motor_limits_rejected',True)
    else:raise AssertionError('Mismatched actuators accepted')
    bad=copy.deepcopy(saved);bad['infos']['contract']='old_22_joint_robot'
    try:validate_actor_transfer(bad,HERE)
    except AssertionError:check('old_robot_rejected',True)
    else:raise AssertionError('Old robot accepted')
    result=dict(passed=True,checks=checks,standing_examples=examples,cycle_rankings=cycles,
        selected_actor=str(ACTOR),selected_actor_sha256=hashlib.sha256(ACTOR.read_bytes()).hexdigest(),transfer=transfer,
        manifest=reward.manifest(),source_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()
            for n in ['stage_rewards.py','support_geometry.py','environment_unitree.py','worker_unitree.py','reward_warm_start.py','evaluation.py','supervisor_unitree.py']},
        scope='CPU adversarial fixtures and actual artifact provenance; not learned gait qualification')
    (out/'tests.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=True,checks=len(checks),quiet_rate=base,cycles=cycles,output=str(out/'tests.json')),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    run(p.parse_args().output)
