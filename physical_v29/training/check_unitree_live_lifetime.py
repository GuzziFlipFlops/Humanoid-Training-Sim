"""Audit actual new Isaac/PPO reward logs without launching another simulator."""
from pathlib import Path
import argparse,json,math
import numpy as np
HERE=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=HERE/'runs/locomotion_20261002_r22_unitree_nominal')
parser.add_argument('--output',type=Path,default=HERE/'diagnostics/unitree_termination_20261002/live_smoke_audit_r22.json')
args=parser.parse_args();run=args.run
progress=[json.loads(s) for s in (run/'002_ppo_smoke/progress.jsonl').read_text().splitlines()]
checks=[]
for row in progress:
    b=row['behavior'];core=sum(v for k,v in b.items() if k.startswith('reward_') and k not in ['reward_physical_termination','reward_living_clip_adjustment'])
    clipped=core+b['reward_living_clip_adjustment']
    terminal=b['reward_physical_termination'];expected=-4.*b['falls_per_s']
    assert clipped>=-1e-5,(clipped,row['iteration'])
    assert math.isclose(terminal,expected,rel_tol=1e-5,abs_tol=1e-5),(terminal,expected)
    assert all(math.isfinite(v) for v in b.values())
    nominal=row['policy'].get('nominal_pose_max_abs_drift_rad')
    if run.name=='locomotion_20261002_r22_unitree_nominal':assert nominal==0.,nominal
    checks.append(dict(iteration=row['iteration'],regular_aggregate_rate=core,clipped_regular_rate=clipped,
        terminal_rate=terminal,expected_terminal_rate=expected,
        invalid_differential_resets_per_s=b['termination_invalid_differential_per_s'],nominal_pose_max_abs_drift_rad=nominal))
data=np.load(run/'003_checkpoint_reload/gait_diagnostics_50hz.npz',allow_pickle=False)
cases=[]
for label,index in [('stand',0),('walk015',1)]:
    failures=np.flatnonzero(data['done'][:,index])
    first=int(failures[0]) if len(failures) else None
    cases.append(dict(case=label,first_failure_s=float(data['time'][first]) if first is not None else None,
        first_causes=data['failure_names'][data['failure_causes'][first,index].astype(bool)].tolist() if first is not None else [],
        first_pose_scope='Automatic-reset done pose excluded from reconstruction'))
evaluation=json.loads((run/'003_checkpoint_reload/evaluation.json').read_text())
result=dict(passed=True,run=str(run.resolve()),scope='Actual new four-update Isaac/PPO reward aggregation and checkpoint reload; not qualification',
    checks=checks,first_failures=cases,independent_gate_passed=evaluation['passed'],
    limitation='Four updates are an execution smoke; failed behavior does not qualify walking or establish eventual convergence.')
out=args.output
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
