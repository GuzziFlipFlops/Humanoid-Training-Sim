"""CPU-only immutable status and per-command trend capture."""
import json, shutil
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent; HERE=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json')
run=Path(live['active_run']['run'])
(OUT/'initial_owned_observation.json').write_text(json.dumps(live,indent=2)+'\n')
shutil.copy2(run/'state.json',OUT/'state_initial.json')
records=[]
for p in sorted(run.glob('*_gate_*/evaluation.json')):
    gate=read(p); rows=[]
    for case in gate['cases']:
        s=case['statistics']
        rows.append(dict(command=s['command_m_s'],passed=case['passed'],failures=case['failures'],
          signed_speed=s['reset_frame_forward_speed_m_s'],lateral=s['reset_frame_lateral_speed_m_s'],
          heading=s['final_heading_abs_mean_deg'],survival=s['survival'],flight=s['geometric_flight_fraction']))
    records.append(dict(folder=p.parent.name,rows=rows))
latest=records[-2:]
for record in latest:
    data=np.load(run/record['folder']/'telemetry.npz');names=list(data['joint_names']);ids=[names.index('head_yaw'),names.index('head_pitch')]
    for i,row in enumerate(record['rows']):
        head=data['q'][:,data['case_labels']==i][:,:,ids]
        row['head_RMS_deg']=float(np.rad2deg(np.sqrt((head**2).sum(-1).mean())))
result=dict(run=str(run),completed_main_updates=live['operations']['main_training']['completed_updates'] if 'main_training' in live['operations'] else None,
  latest=latest,records=records)
(OUT/'gate_trends.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(state_checkpoint=live['state']['checkpoint'],latest=latest),indent=2))
