"""Meaningful finite/gradient/parity/objective counterexamples on CPU."""
import json,sys
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent
TRAIN=OUT.parents[1]/'runs/locomotion_20261007_r40_goal_budget_full_state/source/physical_v29/training'
sys.path.insert(0,str(TRAIN));torch.set_num_threads(4)
from slow_tracking_kernel import score,tracking_width,VERSION
from unitree_reward_port import FUNCTIONS,profile
from types import SimpleNamespace
original=FUNCTIONS['track_lin_vel_xy_yaw_frame_exp']
assert profile('g1')['track_lin_vel_xy']['params']['std']==.5
checks=[];examples=[]
for speed in [0.,.1,.15,.35,.65,1.,1.00001,1.2,1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
    c=torch.tensor([[speed,0.,0.]],dtype=torch.float64)
    tol=max(.1,.12*speed)
    for error in [-.3,-tol,-.03,0.,.03,tol,.3]:
        velocity=torch.tensor([[speed+error,0.,0.]],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
        env=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(root_quat_w=q,root_lin_vel_w=velocity))},
            command_manager=SimpleNamespace(get_command=lambda name:c))
        before=original(env,std=.5,command_name='base_velocity');after=score(c,velocity)
        assert torch.isfinite(after).all() and 0<=after.item()<=1
        grad=torch.autograd.grad(after.sum(),velocity)[0]
        assert torch.isfinite(grad).all() and (grad[0,0].item()*error<=1e-14)
        if not(.1<speed<=1.):assert torch.equal(before,after)
        if error==0:assert after.item()==1.
        if .1<speed<=1. and abs(abs(error)-tol)<1e-10:assert abs(after.item()-torch.exp(torch.tensor(-1.,dtype=torch.float64)).item())<1e-12
        examples.append(dict(command=speed,error=error,old_rate=4*before.item(),new_rate=4*after.item(),width=tracking_width(c).item()))
        checks.append(True)
# All velocity directions, signs and raw vertical velocity; no bias or new frame.
c=torch.tensor([[.35,.02,0.],[3.5,0.,0.]],dtype=torch.float64)
v=torch.tensor([[.39,.05,8.],[3.4,.1,-8.]],dtype=torch.float64)
assert torch.equal(score(c,v),score(c,torch.cat([v[:,:2],torch.zeros(2,1)],1)))
assert torch.allclose(score(-c,-v),score(c,v),atol=0.,rtol=0.)
# Positive kernel cannot pay stopping, backward sliding or diagonal motion more
# than exact tracking at fixed healthy state. Full physical return remains untested.
for speed in [.15,.35,.65,1.]:
    c=torch.tensor([[speed,0.,0.]],dtype=torch.float64)
    wrong=torch.tensor([[0.,0.,0.],[-speed,0.,0.],[speed,.14,0.]],dtype=torch.float64)
    assert (score(c.repeat(3,1),wrong)<1.).all()
    assert (score(c.repeat(3,1),wrong)<torch.exp(-((wrong[:,:2]-c[:,:2]).square().sum(1))/.5**2)).all()
result=dict(passed=True,version=VERSION,numerical_gradient_parity_cases=len(checks),extra_categories=6,
    original_std=.5,unchanged_weights=True,unchanged_frame=True,unchanged_stopped_and_speed_above_1=True,
    complete_physical_return_not_established=True,examples=examples,
    limitations=['Instantaneous velocity contains gait oscillations; physical and learning tests must measure harm.',
    'The width boundary at command1.0 is deliberate and current prior commands are fixed. Transitions and independent states require testing.',
    'No direct 1.5 flight reward or high-speed reward change. This candidate alone is not presumed to qualify the stage.'])
(OUT/'slow_reward_CPU_tests.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))
