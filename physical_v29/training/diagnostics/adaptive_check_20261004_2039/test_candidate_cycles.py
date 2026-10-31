"""Full-cycle reward counterexamples at every future running speed, CPU only."""
import json, sys, torch
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
sys.path[:0]=[str(OUT/'validation_source/physical_v29/training'),str(HERE)]
torch.set_num_threads(1)
from stage_rewards import CurriculumRewards,TOTAL_WEIGHT_N
from test_unitree_reward_port import fixture,NAMES,BODIES,ROOT,FOOT,FEET
from test_stage_rewards import telemetry
reward=CurriculumRewards(NAMES,BODIES,BODIES,ROOT,FOOT)
cycles=[];checks=[]
for speed in [1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
    scores={}
    for name in ['matched_run','fast_walk','loaded_slide','in_place','backward','same_foot_hop']:
        rates=[]
        for tick in range(150):
            phase=tick*.02/(1/(.9+.5*speed))%1
            duty=.5 if name=='fast_walk' else .43
            flags=(phase<duty,(phase+.5)%1<duty)
            if name=='loaded_slide':flags=(True,True)
            if name=='same_foot_hop':flags=(phase<.43,phase<.43)
            e=fixture(speed,tick);forward=0. if name=='in_place' else -speed if name=='backward' else speed
            e.scene['robot'].data.root_lin_vel_b[0,0]=forward
            e.scene['robot'].data.root_lin_vel_w[0,0]=forward
            if name=='loaded_slide':e.scene['robot'].data.body_lin_vel_w[:,FEET,0]=speed
            m=telemetry(e,flags);m['phase']=torch.tensor([phase]);m['clearance'][0]=torch.tensor([0. if f else .065 for f in flags])
            m['foot_force']=torch.tensor([[TOTAL_WEIGHT_N/max(sum(flags),1) if f else 0. for f in flags]])
            e.cad_metrics=m;total,_,terms=reward.evaluate(e);rates.append(float(total/.02))
            if not all(not f for f in flags):assert float(terms['running_flight'])==0.
        scores[name]=sum(rates)/len(rates)
    assert all(scores['matched_run']>scores[n] for n in scores if n!='matched_run'),scores
    checks.append({'name':'matched_run_beats_counterexamples_'+str(speed),'passed':True})
    cycles.append({'command_m_s':speed,'scores':scores})
r={'passed':True,'checks':checks,'cycles':cycles,'manifest':reward.manifest(),
   'scope':'Controlled full-cycle telemetry only; no learned gait or contact-fidelity qualification'}
(OUT/'candidate_cycle_tests.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(checks),'speed_gap':[(c['command_m_s'],c['scores']['matched_run']-c['scores']['fast_walk']) for c in cycles]}))
