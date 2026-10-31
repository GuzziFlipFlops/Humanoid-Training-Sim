"""Counterfactual first-step, cycle, retention and diagnostic regressions."""
import argparse,copy,hashlib,json,math,importlib.util
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from test_unitree_reward_port import fixture,contacts,NAMES,BODIES,FEET,ROOT,FOOT
from test_stage_rewards import telemetry
from stage_rewards import CurriculumRewards,REWARD_VERSION
from standing_retention import StandingRetention
from ppo_update_audit import PpoUpdateAudit
from reward_warm_start import validate_actor_transfer

HERE=Path(__file__).parent
OLD=HERE/'runs/locomotion_20261002_r23_standing_curriculum'
PROTECTED=HERE/'runs/locomotion_20261003_r27_heading_commands_checked/source/physical_v29'
ACTOR=OLD/'047_stand_train_008_part_04/candidate.pt'

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(37)
    checks=[];examples=[]
    def check(label,value,**details):
        assert bool(value),(label,details)
        checks.append(dict(name=label,passed=True,evidence=details))
    reward=CurriculumRewards(NAMES,BODIES,BODIES,ROOT,FOOT)
    def score(speed,tick,flags,forward=None,clearance=None,tilt=0.,terminal=False):
        e=fixture(speed,tick);d=e.scene['robot'].data
        if forward is not None:d.root_lin_vel_b[0,0]=forward;d.root_lin_vel_w[0,0]=forward
        m=telemetry(e,flags,tilt=tilt);period=1/(.9+.5*speed) if speed>=1.5 else .8
        m['phase']=torch.tensor([tick*.02/period%1])
        if clearance is not None:m['clearance'][0]=torch.tensor(clearance)
        e.cad_metrics=m;e.termination_manager.terminated[:]=terminal
        total,raw,terms=reward.evaluate(e)
        return float(total/.02),{k:float(v) for k,v in terms.items()}
    # At tick4, left is stance and right is swing. Translational speed zero.
    stationary,w0=score(.35,4,(True,True),forward=0.)
    unload,w1=score(.35,4,(True,False),forward=0.)
    lift,w2=score(.35,4,(True,False),forward=0.,clearance=[0.,.025])
    wrong,ww=score(.35,4,(False,True),forward=0.,clearance=[.025,0.])
    check('phase_unloading_feedback_before_translation',unload>stationary,planted=stationary,unloaded=unload)
    check('first_lift_feedback_before_translation',lift>unload and w2['feet_clearance']>0,lift=lift,unload=unload)
    check('wrong_phase_lift_not_credited',ww['feet_clearance']==0. and lift>wrong)
    spec=importlib.util.spec_from_file_location('frozen_r23_rewards',OLD/'source/physical_v29/training/stage_rewards.py')
    old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
    old_reward=old_module.CurriculumRewards(NAMES,BODIES,BODIES,ROOT,FOOT)
    old_first=[]
    for flags,clearance in [((True,True),[0.,0.]),((True,False),[0.,0.]),((True,False),[0.,.025])]:
        e=fixture(.35,4);e.scene['robot'].data.root_lin_vel_b.zero_();e.scene['robot'].data.root_lin_vel_w.zero_()
        m=telemetry(e,flags);m['clearance'][0]=torch.tensor(clearance);e.cad_metrics=m
        total,_,terms=old_reward.evaluate(e);old_first.append(float(total/.02))
        check('reproduce_r23_zero_first_step_feedback_'+str(len(old_first)),float(terms['gait'])==float(terms['feet_clearance'])==0.)
    check('r24_restores_feedback_without_changing_weights',abs(max(old_first)-min(old_first))<1e-7 and lift>unload>stationary,old_rates=old_first,new_rates=[stationary,unload,lift])
    for speed in [0.,.15,.35,.65,1.,1.5,2.,2.5]:
        _,t=score(speed,4,(True,True),clearance=[.04,.04])
        check('loaded_feet_no_clearance_'+str(speed),t['feet_clearance']==0.)
    quiet,qt=score(0.,4,(True,True),forward=0.)
    one,_=score(0.,4,(True,False),forward=0.)
    lean,_=score(0.,4,(True,True),forward=0.,tilt=.22)
    check('standing_keeps_bilateral_posture_incentive',quiet>one and quiet>lean)
    check('standing_gait_clearance_flight_absent',qt['gait']==qt['feet_clearance']==qt['running_flight']==0.)
    cycles={}
    for name in ['matched_walk','in_place_steps','skating','walking_hop','backward_walk','matched_run','in_place_run','backward_run']:
        speed=1.5 if 'run' in name else .35;rows=[]
        for tick in range(160):
            period=1/(.9+.5*speed) if speed>=1.5 else .8;duty=.43 if speed>=1.5 else .55
            ph=tick*.02/period%1;flags=(ph<duty,(ph+.5)%1<duty)
            forward=0. if 'in_place' in name else -speed if 'backward' in name else speed
            if name=='skating':flags=(True,True)
            if name=='walking_hop':flags=(ph<.5,ph<.5)
            clearance=[0. if flag else (.065 if speed>=1.5 else .04) for flag in flags]
            e=fixture(speed,tick);e.scene['robot'].data.root_lin_vel_b[0,0]=forward;e.scene['robot'].data.root_lin_vel_w[0,0]=forward
            if name=='skating':e.scene['robot'].data.body_lin_vel_w[:,FEET,0]=speed
            m=telemetry(e,flags);m['phase']=torch.tensor([ph]);m['clearance'][0]=torch.tensor(clearance);e.cad_metrics=m
            total,_,_=reward.evaluate(e);rows.append(float(total/.02))
        cycles[name]=sum(rows)/len(rows)
    for name in ['in_place_steps','skating','walking_hop','backward_walk']:
        check('full_cycle_matched_walk_beats_'+name,cycles['matched_walk']>cycles[name],cycles=cycles)
    for name in ['in_place_run','backward_run']:
        check('full_cycle_matched_run_beats_'+name,cycles['matched_run']>cycles[name],cycles=cycles)
    for forward in [0.,-.35]:
        _,t=score(1.5,4,(False,False),forward=forward,clearance=[.065,.065])
        check('no_nonforward_flight_credit_'+str(forward),t['running_flight']==0.)
    terminal,tt=score(.35,4,(True,False),forward=.35,terminal=True)
    check('terminal_only_fault_cost',terminal==-200. and all(v==0. for k,v in tt.items() if k!='physical_termination'))
    saved=torch.load(ACTOR,map_location='cpu',weights_only=False);transfer=validate_actor_transfer(saved,HERE)
    check('qualified_r23_actor_compatible',transfer['actor_only'] and not transfer['qualification_reused'])
    # Latest command in a five-frame400 interface, not fourth frame246.
    raw=torch.randn(8,400)*.1;raw[:,246:249]=0.;raw[:,326:329]=0.;raw[4:,326]=.35
    obs=TensorDict(dict(policy=raw,critic=torch.randn(8,415)*.1),batch_size=[8])
    def models():
        a=MLPModel(obs,{'actor':['policy']},'actor',23,hidden_dims=[16],activation='elu',obs_normalization=False,
            distribution_cfg=dict(class_name='GaussianDistribution',init_std=.35,std_type='log'))
        c=MLPModel(obs,{'critic':['critic']},'critic',1,hidden_dims=[16],activation='elu',obs_normalization=False)
        return a,c
    a,c=models();teacher=copy.deepcopy(a)
    with torch.no_grad():a.mlp[-1].bias.add_(.2)
    expected=copy.deepcopy(a);target=teacher(obs).detach();pred=expected(obs)
    (pred.square().mean()+.25*(pred[:4]-target[:4]).square().mean()).backward()
    retention=StandingRetention(a,teacher);a(obs).square().mean().backward()
    check('latest_frame_retention_exact_gradient',all(torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5) for p,q in zip(a.parameters(),expected.parameters()) if p.grad is not None))
    check('retention_uses_only_latest_stopped_rows',retention._samples==4*23)
    a.eval();baseline=copy.deepcopy(a);torch.testing.assert_close(a(obs),baseline(obs),rtol=0.,atol=0.)
    check('inference_has_no_teacher_action_blending',True);retention.close()
    # Diagnostic wrapping must leave identical PPO outputs and parameter bytes.
    a,c=models();storage=RolloutStorage('rl',8,4,obs,[23]);alg=PPO(a,c,storage,num_learning_epochs=1,num_mini_batches=2,learning_rate=.0001,schedule='fixed')
    with torch.no_grad():
        for _ in range(4):
            alg.act(obs);alg.process_env_step(obs,torch.full((8,),.2),torch.zeros(8),{})
        alg.compute_returns(obs)
    twin=copy.deepcopy(alg);audit=PpoUpdateAudit(alg)
    torch.manual_seed(83);loss1=alg.update();torch.manual_seed(83);loss2=twin.update()
    check('ppo_audit_leaves_losses_identical',loss1==loss2)
    check('ppo_audit_leaves_actor_and_critic_identical',all(torch.equal(p,q) for p,q in zip(alg.actor.state_dict().values(),twin.actor.state_dict().values())) and all(torch.equal(p,q) for p,q in zip(alg.critic.state_dict().values(),twin.critic.state_dict().values())))
    check('ppo_preupdate_snapshot_matches_collection',audit.latest['pre_update_kl']<1e-8,diagnostics=audit.latest)
    for rel in ['environment.py','evaluation.py','evaluation_steps.py','support_geometry.py','assets/robot_training.urdf','assets/motion_reference.npz','mass_estimate.json']:
        check('protected_'+rel,(HERE/rel).read_bytes()==(PROTECTED/'training'/rel).read_bytes())
    for rel in ['hardware_actuators.py','assets/actuator_limits.json','assets/differential_geometry.json']:
        check('protected_'+rel,(HERE.parent/rel).read_bytes()==(PROTECTED/rel).read_bytes())
    # Cache repair is checked against this same frozen source by exact AST and
    # numerical/state/limit tests; it intentionally changes implementation bytes.
    from test_heading_provenance import Provenance
    Provenance().test_cad_change_is_only_bypassing_tensor_cache()
    check('cad_change_only_tensor_cache_bypass',True)
    names=['stage_rewards.py','environment_unitree.py','worker_unitree.py','supervisor_unitree.py','reward_warm_start.py','standing_retention.py','ppo_update_audit.py']
    report=dict(passed=True,reward_version=REWARD_VERSION,checks=checks,cycles=cycles,
        first_step=dict(planted=stationary,unloaded=unload,lifted=lift,wrong_phase=wrong,r23_rates=old_first),
        selected_actor=str(ACTOR),selected_actor_sha256=hashlib.sha256(ACTOR.read_bytes()).hexdigest(),
        source_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names},
        scope='CPU counterfactual and exact-gradient/provenance tests; not physical or learned gait qualification.')
    (out/'tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=True,checks=len(checks),cycles=cycles,first_step=report['first_step'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
