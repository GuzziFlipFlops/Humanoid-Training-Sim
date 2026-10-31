"""CPU counterexample audit; does not change or interrupt the live policy."""
import importlib.util, hashlib, json, math
from pathlib import Path
import torch

ROOT = Path('E:/Aditya/robot_sim/physical_v29/training')
run = Path(json.loads((ROOT/'ACTIVE_RUN.json').read_text())['run'])
source = run/'source/physical_v29/training/walking_rewards.py'
spec = importlib.util.spec_from_file_location('frozen_reward_audit', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
dt = .02
count = 1000
t = torch.arange(count, dtype=torch.float64)*dt
command = torch.full((count,), .15, dtype=torch.float64)
phase = torch.remainder(t*.975, 1.)
expected = torch.stack([phase, torch.remainder(phase+.5,1.)],1) < .62
contact = torch.ones((count,2), dtype=torch.bool)
clearance = torch.zeros((count,2), dtype=torch.float64)
feet = torch.tensor([[[0.,-.15,-.90],[0.,.15,-.90]]],dtype=torch.float64).expand(count,-1,-1).clone()

def calculate(oscillate):
    vel = torch.zeros((count,3), dtype=torch.float64)
    root_x = torch.zeros(count, dtype=torch.float64)
    if oscillate:
        vel[:,0] = .15*torch.cos(2*math.pi*t)
        root_x = .15/(2*math.pi)*torch.sin(2*math.pi*t)
    relfeet = feet.clone()
    relfeet[:,:,0] -= root_x[:,None]
    terms = module.walking_terms(command,vel,torch.ones_like(command),
        torch.zeros_like(command), contact,expected,clearance,relfeet,feet,
        torch.full_like(command,.04),torch.zeros_like(clearance),dt)
    rates = {key:float(terms[key].mean()) for key in
             ['track','progress','reference','gait','missing_swing','completed_step']}
    return dict(mean_forward_velocity_m_s=float(vel[:,0].mean()),
        integrated_displacement_m=float(vel[:,0].sum()*dt),
        root_rocking_amplitude_m=.15/(2*math.pi) if oscillate else 0.,
        completed_steps=0., both_feet_planted=True, foot_slip_m_s=0.,
        reward_rates=rates, selected_rate_sum=sum(rates.values()))

record = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    stationary=calculate(False), zero_net_rocking=calculate(True),
    scope='Frozen reward-function counterexample, not a physics rollout or proof of the live actor trajectory. Posture and actuator costs are excluded from the selected rate sum. Exact real velocity/posture constraints may differ.',
    conclusion='Rectified instantaneous tracking can pay periodic zero-net movement even with no foot lifts; signed progress cancels but positive tracking does not.')
assert abs(record['zero_net_rocking']['integrated_displacement_m']) < 1e-9
assert record['zero_net_rocking']['reward_rates']['track'] > .25
path = Path(__file__).with_name('zero_net_motion_audit.json')
path.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
