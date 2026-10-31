"""Record the old rollout failure and replay only observable planted-foot credit."""
import importlib.util, json, hashlib
from pathlib import Path
import numpy as np
import torch
from walking_rewards import walking_terms, REWARD_VERSION

here = Path(__file__).resolve().parent
prior = here/'runs/locomotion_20261001_r10_auto_vram_duration'
out = here/'checks/net_travel_20261001'
out.mkdir(exist_ok=True)
state = json.loads((prior/'state.json').read_text())
gate_folders = sorted(prior.glob('*gate*nominal'))
history = []
for folder in gate_folders:
    if not (folder/'evaluation.json').exists(): continue
    gate = json.loads((folder/'evaluation.json').read_text())
    history.append(dict(folder=folder.name, cases=[dict(command=c['statistics']['command_m_s'],
        survival=c['statistics']['survival'], speed=c['statistics']['forward_speed_m_s'],
        left_steps=c['statistics']['steps_left'], right_steps=c['statistics']['steps_right'],
        failures=c['failures']) for c in gate['cases']]))
sources = []
for folder in sorted(prior.glob('*train*')):
    if not (folder/'command.json').exists(): continue
    argv = json.loads((folder/'command.json').read_text())['argv']
    sources.append(dict(folder=folder.name, checkpoint=argv[argv.index('--checkpoint')+1]))
latest = gate_folders[-1]
data = np.load(latest/'telemetry.npz')
selected = data['case_labels']==list(data['case_speeds']).index(.15)
m = data['metrics'][::5, selected]
contact = data['foot_contact'][:, selected]
valid = (m[:,:,0]>.5) & contact.all(-1) & (m[:,:,10]==0) & (m[:,:,11]==0) & (m[:,:,9]==0)
health = np.exp(-6*m[:,:,6]**2-5*m[:,:,7]**2-((m[:,:,1]-m[:,:,2])/.06)**2)
velocity = torch.tensor(np.stack([m[:,:,3][valid],m[:,:,4][valid],np.zeros(valid.sum())],-1),dtype=torch.float32)
count = int(valid.sum())
zero_feet = torch.zeros((count,2,3))
arguments = (torch.full((count,),.15), velocity, torch.tensor(health[valid]), torch.tensor(m[:,:,8][valid]),
    torch.ones((count,2),dtype=torch.bool), torch.tensor([[True,False]]).expand(count,-1),
    torch.zeros((count,2)), zero_feet, zero_feet, torch.zeros(count), torch.zeros((count,2)), .02)
spec = importlib.util.spec_from_file_location('old_reward',prior/'source/physical_v29/training/walking_rewards.py')
old = importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
oldterms = old.walking_terms(*arguments)
newterms = walking_terms(*arguments)
credit = lambda terms: {k:float(terms[k].clamp_min(0).mean()) for k in ['track','progress','reference','gait','completed_step']}
names = data['joint_names'].tolist();q = data['q'][:, selected]
ranges = {n:float(np.rad2deg(np.ptp(q[:,:,i][m[:,:,0]>.5]))) for i,n in enumerate(names) if '_hip_' in n or '_knee_' in n or '_ankle_' in n}
report = dict(reward_version=REWARD_VERSION, old_run=str(prior), latest_rollout=str(latest),
    total_updates_this_run=state['total_training_iterations'], gate_history=history, training_sources=sources,
    replay=dict(samples=count, old_positive_rates=credit(oldterms), new_positive_rates=credit(newterms),
        scope='Recorded both-loaded, before-first-fall, no crossing/invalid/nonfoot samples. Reference error zero is an explicit upper-bound proxy; tracking/progress use recorded velocities and slip. No missing foot event is fabricated.'),
    observed_leg_joint_range_degrees=ranges,
    conclusion='Walking did not pass. Repeated blocks loaded the same standing prior. Positive rectified velocity credit existed with planted feet; v3 removes it on these recorded samples.',
    limits='This does not prove every old reward gain was deliberate exploitation and does not demonstrate learned walking after the repair.',
    sources={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ['walking_rewards.py','environment.py','worker.py','supervisor.py','test_net_travel.py']})
assert sum(credit(newterms).values())==0
(out/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(gates=len(history),latest=history[-1],replay=report['replay'],source_counts={p:sum(x['checkpoint']==p for x in sources) for p in set(x['checkpoint'] for x in sources)}),indent=2))
