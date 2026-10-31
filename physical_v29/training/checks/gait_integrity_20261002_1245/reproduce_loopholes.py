"""Replay identical CPU trajectories through frozen v5/v2 and repaired code."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE))
PRIOR=HERE/'runs/locomotion_20261002_r15_motion_discovery'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def discovery(rewards):
    bank=rewards.StepDiscovery(1,'cpu');tracker=rewards.StepTracker(1,'cpu')
    root=torch.zeros(1,2);feet=torch.tensor([[[0.,-.15],[0.,.15]]]);heading=torch.tensor([[1.,0.]])
    command=torch.tensor([.15]);total=0.;steps=0.
    def tick(force=(169.,169.),clearance=(0.,0.)):
        nonlocal total,steps
        f=torch.tensor([force]);h=torch.tensor([clearance]);eligible=torch.tensor([True])
        events=tracker.update(f>16.9,feet,h,root,heading,command,eligible,.02)
        total+=float(bank.update(f,feet,h,root,heading,command,eligible,torch.ones(1),events,338.,.02))*.02
        steps+=float(events.sum())
    for _ in range(5):tick()
    for side in [0,1]*8:
        force=(0.,338.) if side==0 else (338.,0.)
        height=(.04,0.) if side==0 else (0.,.04)
        tick(force,height);root[:,0]+=.01;feet[:,side,0]+=.06
        for _ in range(5):tick(force,height)
        for _ in range(20):tick()
    return dict(physical_accepted_steps=steps,integrated_discovery_credit=total)


def overlapping_hops(evaluator,legacy=False):
    counter=evaluator.IndependentLandings(1,'cpu')
    feet=torch.tensor([[[0.,-.12],[0.,.12]]]);root=torch.zeros(1,2);height=torch.zeros(1,2)
    command=torch.tensor([1.5]);hop=0.;touches=0.;steps=0.
    def tick(c):
        nonlocal hop,touches,steps
        e,_,h,t=counter.update(torch.tensor([c]),feet,height,root,torch.tensor([[1.,0.]]),command,torch.tensor([True]),.02)
        steps+=float(e.sum());hop+=float(h);touches+=float(t)
    for _ in range(5):tick([True,True])
    height[:]=.04
    for _ in range(8):tick([False,False])
    feet[:,:,0]=.15;root[:,0]=.1;height[:]=0.
    for _ in range(3):tick([True,False])
    for _ in range(4):tick([False,True])
    tick([True,True])
    return dict(raw_touches=touches,hop_fraction=(2 if legacy else 1)*hop/max(touches,1.),credited_steps=steps)


if __name__=='__main__':
    old_rewards=module('frozen_v5',PRIOR/'source/physical_v29/training/walking_rewards.py')
    new_rewards=module('repaired_v6',HERE/'walking_rewards.py')
    old_gate=module('frozen_v2',PRIOR/'source/physical_v29/training/evaluation_steps.py')
    new_gate=module('repaired_v3',HERE/'evaluation_steps.py')
    report=dict(prior_run=str(PRIOR),discovery=dict(old=discovery(old_rewards),new=discovery(new_rewards)),
                overlapping_hops=dict(old=overlapping_hops(old_gate,True),new=overlapping_hops(new_gate)))
    assert report['discovery']['old']['physical_accepted_steps']==16
    assert report['discovery']['new']['physical_accepted_steps']==16
    assert report['discovery']['old']['integrated_discovery_credit']>15.9
    assert abs(report['discovery']['new']['integrated_discovery_credit']-2.)<1e-6
    assert report['overlapping_hops']['old']['hop_fraction']>1.
    assert report['overlapping_hops']['new']['hop_fraction']==1.
    from supervisor import verify_reward_migration,verify_gate_upgrade
    verify_reward_migration(PRIOR,HERE.parent,True)
    report['gate_upgrade']=verify_gate_upgrade(PRIOR/'source/physical_v29/training/evaluation.py',HERE/'evaluation.py')
    report['source_sha256']={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest()
        for p in ['walking_rewards.py','environment.py','evaluation.py','evaluation_steps.py','supervisor.py']}
    output=Path(__file__).with_name('counterexamples.json');output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
