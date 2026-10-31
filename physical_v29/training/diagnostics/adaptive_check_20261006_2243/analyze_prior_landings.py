"""Replay the unchanged verifier on preserved diagnostic traces; no simulator."""
import json,sys,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
out=Path(__file__).resolve().parent;old=out.parent/'adaptive_check_20261006_1743'
sys.path.insert(0,str(out/'control_source/physical_v29/training'))
from evaluation_steps import IndependentLandings
rows=[]
for branch in ['control','candidate']:
 for updates in [64,128]:
  for test in ['nominal','push']:
   folder=old/'matched_direction'/f'{branch}_{updates:03d}_{test}'
   with np.load(folder/'gait_diagnostics_50hz.npz') as f:z={k:f[k] for k in f.files}
   n=len(z['commands']);verifier=IndependentLandings(n,'cpu');alive=torch.ones(n,dtype=torch.bool)
   hops=[];touches=[];events=[]
   for tick,t in enumerate(z['time']):
    alive&=~torch.tensor(z['done'][tick]).bool()
    root=z['root_state'][tick];quat=root[:,3:7]
    forward=Rotation.from_quat(quat[:,[1,2,3,0]]).apply(np.tile([1.,0.,0.],(n,1)))[:,:2]
    forward/=np.maximum(np.linalg.norm(forward,axis=1,keepdims=True),.1)
    _,_,h,tc=verifier.update(torch.tensor(z['contact'][tick]).bool(),torch.tensor(z['foot_position'][tick,:,:,:2]),
      torch.tensor(z['clearance'][tick]),torch.tensor(root[:,:2]),torch.tensor(forward,dtype=torch.float32),
      torch.tensor(z['commands']),alive,.02)
    assert torch.equal(h,torch.tensor(z['hop_touches'][tick])),(branch,updates,test,t)
    hops.append(h.numpy());touches.append(tc.numpy())
    for robot in np.flatnonzero((h.numpy()>0)&np.isclose(z['commands'],2.5)):
     events.append(dict(robot=int(robot),time_s=float(t),hop_impacts=float(h[robot]),
       contact=z['contact'][tick,robot].tolist(),forces_N=z['foot_force'][tick,robot].tolist(),
       clearance_m=z['clearance'][tick,robot].tolist(),root_vertical_speed_m_s=float(root[robot,9]),
       previous_touch_times_s=verifier.touch_time[robot].tolist(),
       during_initial_two_seconds=bool(t<=2.)))
   selection=np.isclose(z['commands'],2.5);h=np.asarray(hops)[:,selection];tc=np.asarray(touches)[:,selection]
   ev=json.loads((folder/'evaluation.json').read_text());case=next(c for c in ev['cases'] if c['statistics']['command_m_s']==2.5)
   cp=Path(json.loads((folder/'run_config.json').read_text())['arguments']['checkpoint'])
   rows.append(dict(branch=branch,updates=updates,test=test,checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
    full_population_hop_fraction=case['statistics']['hop_landing_fraction'],full_population_pass=case['passed'],
    representative_robots=int(selection.sum()),representative_hop_impacts=float(h.sum()),representative_air_qualified_touches=float(tc.sum()),
    representative_hop_fraction=float(h.sum()/max(tc.sum(),1)),events=events))
record=dict(completed=True,rows=rows,unchanged_verifier_reproduces_every_saved_50hz_hop_event=True,
 scope='16 representative robots per preserved gate, 50Hz over complete20s including startup. Full-population results reported separately. Diagnostic reward override remains undeployed and original signature does not certify it. No gate relaxation or causal policy repair inferred.')
(out/'prior_landing_event_analysis.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps([dict(branch=r['branch'],updates=r['updates'],test=r['test'],population_hop_fraction=r['full_population_hop_fraction'],events=r['events']) for r in rows],indent=2))
