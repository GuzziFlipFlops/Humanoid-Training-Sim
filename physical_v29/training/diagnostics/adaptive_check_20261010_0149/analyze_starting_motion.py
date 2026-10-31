"""Compare actually replayed starting states with unchanged physics and gates."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_full_state'
read=lambda p:json.loads(Path(p).read_text())
rows=[]
for name in ['current_nominal_replay','earlier_baseline_nominal','earlier_baseline_push','earlier_baseline_diverse_nominal','earlier_baseline_diverse_push']:
    folder=ROOT/name
    if not (folder/'operation_receipt.json').exists():continue
    cfg=read(folder/'run_config.json')['arguments'];ev=read(folder/'evaluation.json');result=read(folder/'result.json')
    with np.load(folder/'gait_diagnostics_50hz.npz') as z:
        window=z['time']>=2.;commands=z['commands'];cases=[]
        for c in ev['cases']:
            s=c['statistics'];speed=s['command_m_s'];ids=np.flatnonzero(np.isclose(commands,speed))
            q=z['q'][window][:,ids];raw=z['raw_action'][window][:,ids];target=z['target'][window][:,ids]
            cases.append(dict(command=speed,passed=c['passed'],failures=c['failures'],statistics=s,
                representative_robots=len(ids),actual_joint_mean_rad=q.mean((0,1)).tolist(),physical_target_mean_rad=target.mean((0,1)).tolist(),
                raw_action_output_outside_clamp_fraction=float((abs(raw)>3).mean()),
                scope='Actual first-episode50Hz representative arrays; original full-population statistics retained above. No physical cause inferred from averages.'))
        mismatch=float(abs(z['observed_command']-z['action_command']).max());finite=all(np.isfinite(z[k]).all() for k in ['q','qd','target','raw_action','com','foot_force'])
    assert mismatch==0 and finite
    cp=Path(cfg['checkpoint'])
    rows.append(dict(folder=str(folder),name=name,checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
        passed=ev['passed'],cases=cases,result=result,command_mismatch=mismatch,finite_actual_arrays=finite))
data=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),all_five_baselines_complete=len(rows)==5,records=rows,
    scope='Whole earlier learned state versus current full state under identical original runtime. Actor, critic, Adam and exploration differ at initialization. This is not an isolated actor, mechanical-coupling or reward-causality experiment.',
    actuator_velocity_ceiling_proven=False,production_change=False)
(OUT/'starting_motion_comparison.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(complete=len(rows)==5,records=[dict(name=r['name'],focal=[dict(command=c['command'],passed=c['passed'],failures=c['failures'],signed=c['statistics']['reset_frame_forward_speed_m_s'],lateral=c['statistics']['reset_frame_lateral_speed_m_s'],flight=c['statistics']['geometric_flight_fraction']) for c in r['cases'] if c['command'] in [.15,1.5,3.5]]) for r in rows]),indent=2))
