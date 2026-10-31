"""CPU completed-cycle coverage and noncancelling measured shoulder bias."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter,sosfiltfilt
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_sequence'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
record=[]
for branch in ['control','candidate']:
    for updates in [64,128]:
        folder=ROOT/f'{branch}_{updates:03d}_train'
        if not (folder/'result.json').exists() or not read(folder/'result.json').get('completed'):continue
        saved=torch.load(folder/'actual_collection_cycles.pt',map_location='cpu',weights_only=False)
        checkpoint=torch.load(folder/'candidate.pt',map_location='cpu',weights_only=False)
        weights=checkpoint['actor_state_dict']
        dimensions=[400,512,256,128,23];layers=[]
        for i in range(4):
            layer=torch.nn.Linear(dimensions[i],dimensions[i+1]);layer.weight.data.copy_(weights[f'mlp.{2*i}.weight']);layer.bias.data.copy_(weights[f'mlp.{2*i}.bias'])
            layers.append(layer)
            if i<3:layers.append(torch.nn.ELU())
        model=torch.nn.Sequential(*layers)
        names=np.load(ROOT/f'{branch}_{updates:03d}_nominal/telemetry.npz')['joint_names'].tolist()
        ids=[names.index(side+'_shoulder_pitch') for side in ['left','right']]
        with torch.no_grad():raw=model(saved['observations'].flatten(0,1))[:,ids].reshape(-1,8,2)
        means=raw.clamp(-3,3).mean(1)*.25*180/math.pi
        commands=np.array(saved['commands']);groups={}
        for command in np.unique(commands):
            mask=commands==command;m=means[mask]
            groups[str(float(command))]=dict(cycles=int(mask.sum()),robots=len(set(np.array(saved['env_ids'])[mask].tolist())),
                effective_cycle_mean_deg=m.mean(0).tolist(),noncancelling_cycle_mean_RMS_deg=m.square().mean(0).sqrt().tolist())
        physical=[]
        for test in ['nominal','push']:
            path=ROOT/f'{branch}_{updates:03d}_{test}/telemetry.npz'
            with np.load(path) as z:
                q=np.rad2deg(z['q'][:,:,ids]).astype(float);labels=z['case_labels'];speeds=z['case_speeds'];alive=z['metrics'][::5,:,0]>.5
            for case,speed in enumerate(speeds):
                robots=np.flatnonzero(labels==case);terminal=np.flatnonzero(~alive[:,robots].all(1));end=int(terminal[0]) if len(terminal) else len(q)
                if end<80:continue
                slow=sosfiltfilt(butter(4,.5,fs=10.,output='sos'),q[:end,robots],axis=0)
                w=slow[30:-30]
                physical.append(dict(test=test,command=float(speed),population_robots=len(robots),
                    slow_RMS_deg=np.sqrt((w*w).mean((0,1))).tolist(),
                    slow_abs_mean_deg=np.abs(w).mean((0,1)).tolist(),
                    slow_signed_mean_deg=w.mean((0,1)).tolist(),
                    slow_range_median_deg=np.median(np.quantile(w,.95,axis=0)-np.quantile(w,.05,axis=0),axis=0).tolist()))
        record.append(dict(branch=branch,updates=updates,checkpoint=str(folder/'candidate.pt'),sha256=hashlib.sha256((folder/'candidate.pt').read_bytes()).hexdigest(),
            cycles=len(commands),robots=len(set(saved['env_ids'])),actual_command_coverage=groups,
            snapshot_tick_range=[min(saved['start_ticks']),max(saved['end_ticks'])],
            executed_cycle_samples_truthful=saved['truthful'],snapshot_is_latest_update_only=True,physical=physical))
result=dict(records=record,
    scope='Latest completed training-cycle snapshot versus independent deterministic nominal/pushed physical motion. Exact frozen nonnormalized400-input ELU actor replay; physical targets are pre-action and are not mean actions. Population0.5Hz slow RMS is noncancelling;3s filter edges and resets excluded. Different training/evaluation states and exploration remain,so a difference is not a numerical bug.',
    human_or_Unitree_motion_certified=False,diagnostic_actors_deployed=False)
(OUT/'sequence_coverage_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([dict(branch=r['branch'],updates=r['updates'],cycles=r['cycles'],robots=r['robots'],
    fast_physical=[v for v in r['physical'] if v['command']==3.5]) for r in record]))
