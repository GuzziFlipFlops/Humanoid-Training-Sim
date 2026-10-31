"""Reproduce load-redistribution incentive and test an undeployed narrow repair."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import torch
torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent; HERE=OUT.parents[1]
RUN=Path(json.loads((HERE/'ACTIVE_RUN.json').read_text())['run'])
OLD=RUN/'source/physical_v29/training'
sys.path[:0]=[str(OLD),str(HERE)]
candidate=OUT/'candidate_source/training';candidate.mkdir(parents=True,exist_ok=True)
source=(OLD/'stage_rewards.py').read_text()
before='    score=torch.exp(-force.clamp_min(0.)/(total_weight_N*.25))\n    return (score*expected_swing).sum(1)/expected_swing.sum(1).clamp_min(1)'
after='    swing_load=(force.clamp_min(0.)*expected_swing).sum(1)\n    return torch.exp(-swing_load/(total_weight_N*.25))*expected_swing.any(1)'
assert before in source
source=source.replace(before,after).replace('cad_running_phase_unload_v7','cad_running_bilateral_unload_v8')
source=source.replace('Bounded positive swing-force feedback; quarter body-weight e-fold scale.',
    'Bounded simultaneous swing-load feedback; quarter body-weight e-fold scale.')
source=source.replace('This local adaptation follows periodic force composition. A partially',
    'Sum intended swing loads before exponentiation: transferring load between\n    two intended airborne feet gives no credit. Single-foot swing is unchanged.\n    This local adaptation follows periodic force composition. A partially')
(candidate/'stage_rewards.py').write_text(source)
# This module reads its own adjacent mass file. Stage it without changing it.
(candidate/'mass_estimate.json').write_bytes((OLD/'mass_estimate.json').read_bytes())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('frozen_rewards',OLD/'stage_rewards.py');new=load('candidate_rewards',candidate/'stage_rewards.py')
checks=[]
def check(name,condition):
    assert condition,name;checks.append({'name':name,'passed':True})
W=old.TOTAL_WEIGHT_N
rows=[]
for pair in [(W/2,W/2),(W,0.),(W*.75,W*.25),(W*.25,W*.25),(W*.1,0.),(0.,0.)]:
    force=torch.tensor([pair]);swing=torch.tensor([[True,True]])
    rows.append({'force_N':pair,'total_N':sum(pair),'old':float(old.running_unload_quality(force,swing)),
                 'candidate':float(new.running_unload_quality(force,swing))})
check('old_rewards_redistribution_at_constant_total_load',rows[1]['old']>rows[0]['old'])
check('candidate_constant_total_load_has_no_redistribution_credit',abs(rows[1]['candidate']-rows[0]['candidate'])<1e-7)
check('candidate_partial_bilateral_unloading_improves',rows[3]['candidate']>rows[0]['candidate'])
check('candidate_true_unloading_maximal',rows[-1]['candidate']==1.)
generator=torch.Generator().manual_seed(731)
force=torch.rand(4096,2,generator=generator)*W*2
for swing in [torch.tensor([[True,False]]).expand_as(force),torch.tensor([[False,True]]).expand_as(force)]:
    check('single_swing_bitwise_equal_'+str(swing[0].tolist()),torch.equal(old.running_unload_quality(force,swing),new.running_unload_quality(force,swing)))
check('no_swing_zero',not new.running_unload_quality(force,torch.zeros_like(force,dtype=torch.bool)).any())
check('bounded',bool(((new.running_unload_quality(force,torch.ones_like(force,dtype=torch.bool))>=0)&(new.running_unload_quality(force,torch.ones_like(force,dtype=torch.bool))<=1)).all()))
force=torch.tensor([[0.,W],[W/2,W/2],[W*.1,0.]],requires_grad=True)
value=new.running_unload_quality(force,torch.ones_like(force,dtype=torch.bool))
gradient=torch.autograd.grad(value.sum(),force)[0]
check('both_foot_force_derivatives_negative',bool((gradient<0).all()))
check('derivatives_equal_per_total_force',torch.equal(gradient[:,0],gradient[:,1]))
from test_unitree_reward_port import fixture,NAMES,BODIES,ROOT,FOOT
from test_stage_rewards import telemetry
rewards=[m.CurriculumRewards(NAMES,BODIES,BODIES,ROOT,FOOT) for m in [old,new]]
equal=0
for speed in [0.,.15,.35,.65,1.]:
    for tick in range(120):
        e=fixture(speed,tick);ph=tick*.02/.8%1
        flags=(True,True) if speed==0 else (ph<.55,(ph+.5)%1<.55)
        m=telemetry(e,flags);m['phase']=torch.tensor([ph]);m['foot_force']=torch.tensor([[W if x else 0. for x in flags]])
        e.cad_metrics=m
        a=rewards[0].evaluate(e);b=rewards[1].evaluate(e)
        assert torch.equal(a[0],b[0]) and all(torch.equal(a[2][k],b[2][k]) for k in a[2]);equal+=1
check('600_full_standing_walking_fixtures_bitwise_identical',equal==600)
result={'passed':True,'checks':checks,'load_redistribution':rows,'gradient_per_N':gradient.tolist(),
    'standing_walking_bitwise_fixtures':equal,'old_sha256':hashlib.sha256((OLD/'stage_rewards.py').read_bytes()).hexdigest(),
    'candidate_sha256':hashlib.sha256((candidate/'stage_rewards.py').read_bytes()).hexdigest(),
    'weights_changed':False,'deployed':False,'scope':'CPU counterfactual only; dynamics and qualification remain unproven'}
(OUT/'bilateral_counterfactuals.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'rows':rows,'equal_fixtures':equal},indent=2))
