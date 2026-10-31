"""Undeployed force-shaping hypothesis and frozen reward cycle counterexamples."""
import json
from pathlib import Path
import sys

import torch

torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent
HERE=OUT.parents[1]
RUN=Path(json.loads((HERE/'ACTIVE_RUN.json').read_text())['run'])
FROZEN=RUN/'source/physical_v29'
sys.path[:0]=[str(FROZEN/'training'),str(HERE)]
from unitree_reward_port import gait_parameters
from stage_rewards import CurriculumRewards
from test_unitree_reward_port import fixture, contacts, NAMES, BODIES, FEET, ROOT, FOOT
from test_stage_rewards import telemetry


reward=CurriculumRewards(NAMES,BODIES,BODIES,ROOT,FOOT)
checks=[]
def check(name, condition):
    assert condition, name
    checks.append(dict(name=name,passed=True))


force_rows=[]
for force in [338.,239.,180.,100.,40.,20.,18.,16.,1.,0.]:
    # At 1.5m/s tick14, both legs are in their desired .43-duty swing phases.
    e=fixture(1.5,14)
    sensor_flags=(force>1.,False)
    physical_flags=(force>34.46334144435157*9.81*.05,False)
    m=telemetry(e,physical_flags)
    # Fixture telemetry initializes sensor contact too. Restore the distinct
    # upstream touching-contact channel after constructing loaded-foot metrics.
    contacts(e,sensor_flags)
    m['phase']=torch.tensor([14*.02/(1/1.65)%1])
    m['foot_force']=torch.tensor([[force,0.]])
    m['clearance']=torch.zeros((1,2))
    e.cad_metrics=m
    total,_,terms=reward.evaluate(e)
    force_rows.append(dict(force_N=force,reward_rate=float(total/.02),gait_rate=float(terms['gait']),
                           flight_rate=float(terms['running_flight']),
                           undeployed_force_progress=.5*float(torch.exp(-torch.tensor(force)/(338.09*.5)))))
loaded=[r for r in force_rows if r['force_N']>=18.]
check('frozen_reward_is_flat_during_loaded_foot_unloading',max(r['reward_rate'] for r in loaded)==min(r['reward_rate'] for r in loaded))
check('nonzero_flight_credit_requires_crossing_contact_threshold',all(r['flight_rate']==0 for r in loaded))
check('undeployed_force_signal_orders_partial_unloading',all(a['undeployed_force_progress']<b['undeployed_force_progress'] for a,b in zip(force_rows,force_rows[1:])))

cycles=[]
for speed in [1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
    scores={}
    for name in ['matched_run','fast_walk','loaded_slide','in_place','backward','same_foot_hop']:
        rates=[]
        for tick in range(150):
            period=1/(.9+.5*speed)
            phase=tick*.02/period%1
            duty=.5 if name=='fast_walk' else .43
            flags=(phase<duty,(phase+.5)%1<duty)
            if name=='loaded_slide':flags=(True,True)
            if name=='same_foot_hop':flags=(phase<.43,phase<.43)
            e=fixture(speed,tick)
            forward=0. if name=='in_place' else -speed if name=='backward' else speed
            e.scene['robot'].data.root_lin_vel_b[0,0]=forward
            e.scene['robot'].data.root_lin_vel_w[0,0]=forward
            if name=='loaded_slide':e.scene['robot'].data.body_lin_vel_w[:,FEET,0]=speed
            m=telemetry(e,flags);m['phase']=torch.tensor([phase])
            m['clearance'][0]=torch.tensor([0. if f else .065 for f in flags])
            e.cad_metrics=m
            total,_,_=reward.evaluate(e);rates.append(float(total/.02))
        scores[name]=sum(rates)/len(rates)
    check(f'matched_run_beats_counterexamples_{speed}',all(scores['matched_run']>scores[n] for n in scores if n!='matched_run'))
    cycles.append(dict(command_m_s=speed,scores=scores,fast_walk_gap=scores['matched_run']-scores['fast_walk']))

result=dict(passed=True,checks=checks,force_response=force_rows,cycles=cycles,
            diagnosis='The current binary contact/flight terms distinguish an ideal run from fast walking but provide no response while one loaded foot is partially unloaded above the threshold. A continuous phase-conditioned force term is a hypothesis, not a tested physical repair.',
            hypothesis_deployed=False,
            scope='Controlled telemetry only, held posture/velocity/costs. No changed rewards, physics, policies or qualification. Cycle dominance does not prove lift-off feasibility or convergence.')
(OUT/'running_counterfactuals.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,checks=len(checks),force_response=force_rows,
                      running_fast_walk_gaps=[(x['command_m_s'],x['fast_walk_gap']) for x in cycles]),indent=2))
