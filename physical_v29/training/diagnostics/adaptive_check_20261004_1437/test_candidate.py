"""Counterfactuals, prior-reward/gate parity and source-bound actor transfer."""
import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
d=json.loads((OUT/'drain.json').read_text());frozen=Path(d['run'])/'source/physical_v29/training'
candidate=OUT/'candidate_source/training';sys.path[:0]=[str(frozen),str(HERE)]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module('old_stage',frozen/'stage_rewards.py');new=module('new_stage',candidate/'stage_rewards.py')
warm=module('new_warm',candidate/'reward_warm_start.py')
from test_unitree_reward_port import fixture,contacts,NAMES,BODIES,FEET,ROOT,FOOT
from test_stage_rewards import telemetry
from heading_control import direction_checks
checks=[]
def check(name,value,**evidence):
    assert bool(value),(name,evidence);checks.append(dict(name=name,passed=True,evidence=evidence))
rewards=[m.CurriculumRewards(NAMES,BODIES,BODIES,ROOT,FOOT) for m in [old,new]]
def metrics(e,flags,phase):
    m=telemetry(e,flags);m['phase']=torch.tensor([phase]);m['foot_force']=torch.tensor([[new.TOTAL_WEIGHT_N if f else 0. for f in flags]])
    return m
def rates(e,m):
    e.cad_metrics=m;return [r.evaluate(e) for r in rewards]
for speed in [0.,.15,.35,.65,1.]:
    for tick in range(120):
        phase=tick*.02/.8%1;flags=(phase<.55,(phase+.5)%1<.55) if speed else (True,True)
        e=fixture(speed,tick);m=metrics(e,flags,phase)
        m['clearance']=torch.tensor([[0. if f else .04 for f in flags]])
        a,b=rates(e,m)
        assert torch.equal(a[0],b[0]),('prior reward changed',speed,tick)
        assert all(torch.equal(a[2][k],b[2][k]) for k in a[2])
        assert b[2]['running_unload'].item()==0.
    check('prior_reward_bitwise_'+str(speed),True,ticks=120)
forces=[]
for f in [338.,239.,180.,100.,40.,20.,18.]:
    e=fixture(1.5,14);m=metrics(e,(True,False),14*.02/(1/1.65)%1)
    contacts(e,(True,False));m['foot_force']=torch.tensor([[f,0.]])
    a,b=rates(e,m);forces.append(dict(force_N=f,old_rate=float(a[0]/.02),new_rate=float(b[0]/.02)))
check('partial_unload_legacy_flat',len({x['old_rate'] for x in forces})==1)
check('partial_unload_new_strictly_ordered',all(a['new_rate']<b['new_rate'] for a,b in zip(forces,forces[1:])),rows=forces)
force=torch.tensor([[100.,0.]],requires_grad=True)
quality=new.running_unload_quality(force,torch.tensor([[True,True]]));quality.sum().backward()
check('partial_force_derivative_negative_finite',torch.isfinite(force.grad).all() and force.grad[0,0]<0.,derivative=float(force.grad[0,0]))
for speed in [1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
    scores={}
    for name in ['matched_run','fast_walk','loaded_slide','in_place','backward','same_foot_hop']:
        score=0.
        for tick in range(150):
            ph=tick*.02*(.9+.5*speed)%1;duty=.5 if name=='fast_walk' else .43
            flags=(ph<duty,(ph+.5)%1<duty)
            if name=='loaded_slide':flags=(True,True)
            if name=='same_foot_hop':flags=(ph<.43,ph<.43)
            e=fixture(speed,tick);m=metrics(e,flags,ph)
            m['clearance']=torch.tensor([[0. if f else .065 for f in flags]])
            if name=='loaded_slide':e.scene['robot'].data.body_lin_vel_w[:,FEET,0]=speed
            if name in ['in_place','backward']:
                e.scene['robot'].data.root_lin_vel_b[0,0]=0. if name=='in_place' else -speed
                e.scene['robot'].data.root_lin_vel_w[0,0]=e.scene['robot'].data.root_lin_vel_b[0,0]
            score+=float(rates(e,m)[1][0]/.02)
        scores[name]=score/150
    check('run_beats_exploits_'+str(speed),all(scores['matched_run']>v for k,v in scores.items() if k!='matched_run'),scores=scores)
for force in [0.,.5,10.]:
    e=fixture(1.5,14);m=metrics(e,(False,False),.46);m['foot_force'].fill_(force)
    for clear in [0.,.00005,.0009]:
        m['clearance'].fill_(clear);check(f'grounded_no_flight_{force}_{clear}',rates(e,m)[1][2]['running_flight'].item()==0.)
    m['clearance'].fill_(.01);r=rates(e,m)[1][2]['running_flight'].item()
    check('unloaded_air_credit_'+str(force),(r>0.)==(force<1.))
e=fixture(1.5,14);m=metrics(e,(False,False),.46);m['clearance'].fill_(.1);e.termination_manager.terminated.fill_(True)
a,b=rates(e,m);check('terminal_fault_preserved',float(b[0]/.02)==-200. and all(v.item()==0. for k,v in b[2].items() if k!='physical_termination'))
for magnitude in [0.,3.,10.,100.]:
    e=fixture(1.5,14);e.action_manager.action.fill_(magnitude);m=metrics(e,(True,False),.46)
    check('bounded_cost_survival_'+str(magnitude),rates(e,m)[1][0]>=0.)
def gate(path):
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gate_case')
    mod=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(mod);scope={'direction_checks':direction_checks}
    exec(compile(mod,str(path),'exec'),scope);return scope['gate_case']
old_gate,new_gate=gate(frozen/'evaluation.py'),gate(candidate/'evaluation.py')
report=json.loads((Path(d['operation'])/'result.json').read_text())
for case in report['cases']:
    stats=copy.deepcopy(case['statistics']);speed=stats['command_m_s']
    if speed<=1.:
        for pushed in [False,True]:check(f'prior_classifier_unchanged_{speed}_{pushed}',old_gate(speed,stats,pushed)==new_gate(speed,stats,pushed))
    else:
        stats['flight_fraction']=.03;stats['geometric_flight_fraction']=0.
        check('threshold_only_grounded_flight_rejected',old_gate(speed,stats)['passed'] and not new_gate(speed,stats)['passed'])
        stats['geometric_flight_fraction']=.03
        check('actual_geometric_flight_can_pass',new_gate(speed,stats)['passed'])
saved=torch.load(d['checkpoint'],map_location='cpu',weights_only=False)
transfer=warm.validate_actor_transfer(saved,HERE)
check('audited_actor_exploration_preserved',transfer['preserve_trained_exploration'] and not transfer['qualification_reused'] and not transfer['critic_loaded'] and not transfer['optimizer_loaded'])
for key,badvalue in [('reward_signature','wrong'),('contract','old_22_joint_robot'),('actuator_sha256','wrong')]:
    broken=copy.deepcopy(saved);broken['infos'][key]=badvalue
    try:warm.validate_actor_transfer(broken,HERE)
    except AssertionError:check('reject_mismatch_'+key,True)
    else:raise AssertionError('Bad transfer accepted')
result=dict(passed=True,checks=checks,force_response=forces,transfer=transfer,
    unchanged_prior_reward_frames=600,physical_probe=str(OUT/'physical_response/result.json'),
    scope='CPU controlled telemetry and source-bound actual saved actor; no learned running qualification')
(OUT/'cpu_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(checks),'prior_frames':600,'force_response':forces},indent=2))
