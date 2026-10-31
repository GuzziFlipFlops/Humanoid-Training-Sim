"""Return-level regression for the actual upright differential-reset exploit."""
from pathlib import Path
import ast, json, math
import torch
from test_unitree_reward_port import fixture, NAMES, BODIES, ROOT, FOOT
from unitree_reward_port import UnitreeRewards

HERE=Path(__file__).parent
baseline=UnitreeRewards(NAMES,BODIES,BODIES,ROOT,FOOT)
repaired=UnitreeRewards(NAMES,BODIES,BODIES,ROOT,FOOT,termination_weight=-200.,positive_living=True)
checks=[]
def check(name,condition,**evidence):
    assert condition,(name,evidence)
    checks.append(dict(name=name,passed=True,evidence=evidence))

check('unchanged_g1_core',all(v==repaired.terms[k] for k,v in baseline.terms.items()))
# Check the published coefficients/order against pinned primary sources,
# rather than accepting a prose description as implementation provenance.
vendor=HERE/'unitree_vendor'
h2=ast.parse((vendor/'h2_velocity_cfg.py').read_text(encoding='utf-8'))
term=next(n for n in ast.walk(h2) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='is_terminated' for k in n.keys))
call=next(v for k,v in zip(term.keys,term.values) if isinstance(k,ast.Constant) and k.value=='is_terminated')
check('h2_pinned_fault_cost',ast.literal_eval(next(k.value for k in call.keywords if k.arg=='weight'))==-200.)
gym=ast.parse((vendor/'rlgym_legged_robot.py').read_text(encoding='utf-8'))
compute=next(n for n in ast.walk(gym) if isinstance(n,ast.FunctionDef) and n.name=='compute_reward')
clip=next(n for n in compute.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self.cfg.rewards.only_positive_rewards')
terminal=next(n for n in compute.body if isinstance(n,ast.If) and ast.unparse(n.test)=="'termination' in self.reward_scales")
clip_call=next(n for n in ast.walk(clip) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.clip')
check('rlgym_pinned_clip_before_terminal',clip.lineno<terminal.lineno and ast.literal_eval(next(k.value for k in clip_call.keywords if k.arg=='min'))==0.)
gymcfg=ast.parse((vendor/'rlgym_legged_robot_config.py').read_text(encoding='utf-8'))
check('rlgym_pinned_positive_living_default',any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='only_positive_rewards' for t in n.targets) and ast.literal_eval(n.value) is True for n in ast.walk(gymcfg)))
for terminated in [False,True]:
    for dt in [.005,.01,.02,.04]:
        e=fixture();e.step_dt=dt;e.termination_manager.terminated[:]=terminated
        before=baseline.evaluate(e)[0];after,raw,w=repaired.evaluate(e)
        expected=float(before.clamp_min(0.))-(200*dt if terminated else 0.)
        check(f'terminal_delta_{terminated}_{dt}',math.isclose(float(after),expected,abs_tol=1e-6),actual=float(after),expected=expected)
        check(f'terminal_indicator_{terminated}_{dt}',float(raw['physical_termination'])==float(terminated))
e=fixture();e.termination_manager.time_outs=torch.tensor([True])
check('timeout_has_no_penalty',float(repaired.evaluate(e)[2]['physical_termination'])==0.)
# A very negative physical pose/action reward must not hide the fault cost.
e=fixture();e.action_manager.action[:]=3.;e.action_manager.prev_action[:]=-3.
e.termination_manager.terminated[:]=True
check('terminal_added_after_clipping',math.isclose(float(repaired.evaluate(e)[0]),-4.,abs_tol=1e-6))

run=HERE/'runs/locomotion_20261002_r20_unitree_g1'
records=[]
for folder in ['009_stand_train_001','010_stand_train_001_part_02','011_stand_train_001_part_03','012_stand_train_001_part_04']:
    records += [json.loads(line) for line in (run/folder/'progress.jsonl').read_text().splitlines()]
rates=[sum(v for k,v in r['behavior'].items() if k.startswith('reward_')) for r in records]
gamma=.99;dt=.02;table=[]
# Prefix-equivalent exits: a fault cannot improve return just by avoiding
# nonpositive future costs, including actual observed negative-rate extremes.
for rate in sorted(set([min(rates),-.5,-1.,-1.85,-2.,-10.,-100.,0.,2.357])):
    old_alive=sum(gamma**i*rate*dt for i in range(1000))
    new_alive=sum(gamma**i*max(rate,0.)*dt for i in range(1000))
    for length in [1,2,4,10,50,100,300]:
        old_exit=sum(gamma**i*rate*dt for i in range(length))
        new_exit=sum(gamma**i*max(rate,0.)*dt for i in range(length))-gamma**(length-1)*200*dt
        if rate<0:check(f'baseline_exit_profitable_{rate}_{length}',old_exit>old_alive)
        check(f'repaired_exit_not_profitable_{rate}_{length}',new_exit<new_alive,alive=new_alive,early_exit=new_exit)
        table.append(dict(rate=rate,length=length,baseline_alive_return=old_alive,baseline_exit_return=old_exit,
                          repaired_alive_return=new_alive,repaired_exit_return=new_exit))
# Core positive gait rewards still distinguish the intended walk from skating.
e=fixture(.35);good=float(repaired.evaluate(e)[0]);e.scene['robot'].data.root_lin_vel_w.zero_()
check('forward_tracking_gradient_retained',float(repaired.evaluate(e)[0])<good)
e=fixture(0.);e.termination_manager.terminated[:]=True
check('upright_motor_fault_cost',float(repaired.evaluate(e)[2]['physical_termination'])==-200.)
out=HERE/'diagnostics/unitree_termination_20261002';out.mkdir(parents=True,exist_ok=True)
# Independently generated whole-cycle adversarial traces from the exact raw
# G1 functions. Flooring cannot make any of these outrank matching travel.
core=json.loads((out/'counterfactual_rewards.json').read_text())
rankings={}
for case,trace in core['traces'].items():
    if not case.startswith('g1_'):continue
    fault=case=='g1_fallen'
    rankings[case]=sum(max(r['rate'],0.)-200.*fault for r in trace['rows'])/len(trace['rows'])
for case,rate in rankings.items():
    if case=='g1_matched_walk':continue
    check('active_v2_ranks_'+case,rankings['g1_matched_walk']>rate,matched=rankings['g1_matched_walk'],adversarial=rate)
# Re-score recorded physical failures only up to their first termination;
# post-fault visualization deliberately holds resets and is not an episode.
live=json.loads((HERE/'diagnostics/unitree_port_20261002/r20_policy_failure_visuals/telemetry.json').read_text())
replays=[]
for case in ['policy_stand','policy_walk']:
    rows=[]
    for r in [r for r in live['rows'] if r['case']==case]:
        rows.append(r)
        if r['done']:break
    old_return=sum(gamma**i*r['reward_g1'] for i,r in enumerate(rows))
    new_rewards=[max(r['rate_g1'],0.)*dt-200.*dt*r['done'] for r in rows]
    new_return=sum(gamma**i*x for i,x in enumerate(new_rewards))
    check('recorded_fault_penalized_'+case,new_rewards[-1]<=-3.9 and new_return<0.,terminal_reward=new_rewards[-1],episode_return=new_return)
    replays.append(dict(case=case,first_failure_s=rows[-1]['t'],cause=rows[-1]['failures'],baseline_return=old_return,repaired_return=new_return,
        scope='Off-line reward re-scoring of old-policy physical trajectory; not a newly trained repaired policy'))
result=dict(passed=True,checks=checks,discounted_returns=table,observed_r20_reward_rate_range=[min(rates),max(rates)],
    active_v2_rankings=rankings,recorded_failure_replays=replays,
    scope='Prefix-equivalent episode-return tests; original core arithmetic unchanged; not convergence or learned gait proof',
    limitation='Rewards do not certify natural style or physical behavior. In-place positive reward remains possible; independent net-travel/step gates remain mandatory.')
(out/'lifetime_tests.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(passed=True,checks=len(checks),observed_rate_range=result['observed_r20_reward_rate_range']),indent=2))
