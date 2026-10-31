"""Re-score only changed term on saved actual force/phase/posture trajectories."""
import json, numpy as np, torch
from pathlib import Path
OUT=Path(__file__).resolve().parent
data=np.load(OUT/'physical_response/telemetry.npz',allow_pickle=False)
report=json.loads((OUT/'physical_response/result.json').read_text())
W=338.0853795690889
period=1/(.9+.5*torch.tensor(1.5))
# The saved phase is action-time. Reward runs after the episode counter advances.
# Reconstruct the exact float32 post-step reward phase; validate against saved term.
phase=np.broadcast_to(torch.remainder(torch.arange(1,len(data['time'])+1).float()*.02/period,1.).numpy()[:,None],data['phase'].shape)
force=np.maximum(data['force'],0)
legs=np.remainder(phase[...,None]+np.array([0.,.5],dtype=np.float32),1.)
swing=legs>=.43
old=(np.exp(-force/(W*.25))*swing).sum(-1)/np.maximum(swing.sum(-1),1)
new=np.exp(-(force*swing).sum(-1)/(W*.25))*swing.any(-1)
healthy=(data['term_upright']/3.5)*(data['term_torso_upright']/.75)*(data['term_height']/2.5)
key='term_running_unload'
assert key in data,data.files
expected=.5*old*healthy
# The final group is walking: this term must be exactly zero there.
per=int(data['robots_per_group']);expected[:,-per:]=0
error=float(np.abs(expected-data[key]).max());assert error<1e-4,error
candidate=data['reward_rate']+(.5*new*healthy-data[key]);candidate[:,-per:]=data['reward_rate'][:,-per:]
window=data['time']>=3.;rows=[]
for i,name in enumerate(data['groups']):
    ids=slice(i*per,(i+1)*per);mask=swing[window,ids].all(-1)
    original=data['reward_rate'][window,ids];alternative=candidate[window,ids]
    f=force[window,ids]
    rows.append({'name':str(name),'old_reward_rate':float(original.mean()),'candidate_reward_rate':float(alternative.mean()),
        'old_unloading_rate':float(data[key][window,ids].mean()),
        'candidate_unloading_rate':float((.5*new*healthy)[window,ids].mean()) if i<len(report['groups'])-1 else 0.,
        'both_swing_samples':int(mask.sum()),
        'lower_foot_force_in_both_swing_N':float(f.min(-1)[mask].mean()) if mask.any() else None,
        'larger_foot_force_in_both_swing_N':float(f.max(-1)[mask].mean()) if mask.any() else None,
        'both_swing_old_quality':float(old[window,ids][mask].mean()) if mask.any() else None,
        'both_swing_candidate_quality':float(new[window,ids][mask].mean()) if mask.any() else None,
        'flight_fraction':report['groups'][i]['unloaded_geometric_flight_fraction']})
r={'passed':True,'frozen_term_reconstruction_max_error':error,'groups':rows,
   'scope':'Re-scored recorded trajectories only. Not a learned candidate rollout or causal convergence proof. No helper action is deployed.'}
(OUT/'physical_objective_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
