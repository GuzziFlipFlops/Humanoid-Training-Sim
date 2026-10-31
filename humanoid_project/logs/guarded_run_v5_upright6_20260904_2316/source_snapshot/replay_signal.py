"""Read-only calibration against archived contacts; NOT a policy-performance test."""
import json
from pathlib import Path
import numpy as np
import torch
from contact_state import CompletedStances, SmoothedStances

ROOT=Path(__file__).resolve().parents[2]
run=ROOT/'logs/guarded_run_v2_20260902_2103'
results=[]
for block in ('walking_seed_eval_47','01_balance_1_0200_eval_47',
              '01_balance_1_0400_eval_47','01_balance_1_0600_eval_47'):
    report=json.loads((run/block/'evaluation.json').read_text())
    data=np.load(run/block/'telemetry.npz')
    a,b=report['fields']['fz']; n=report['replicas']
    contacts=data['samples'][:,3*n:4*n,a:b]>5.
    old,new=CompletedStances(n,'cpu'),SmoothedStances(n,'cpu')
    costs=[]
    for i,frame in enumerate(contacts):
        old.update(torch.from_numpy(frame)); new.update(torch.from_numpy(frame))
        if i>=150: costs.append([float(old.penalty().mean())*.2,float(new.penalty().mean())*.6])
    results.append(dict(block=block,stance_asymmetry=report['rows']['candidate_1.0']['stance_asymmetry'],
                        weighted_cost_magnitudes=np.mean(costs,axis=0).tolist()))
print(json.dumps(results,indent=2))
