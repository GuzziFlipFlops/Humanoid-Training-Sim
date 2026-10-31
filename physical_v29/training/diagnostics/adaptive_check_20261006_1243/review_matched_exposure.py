import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_exposure'
read=lambda p:json.loads(Path(p).read_text())
rows=[];pairs=[]
for branch in ['control','candidate']:
 for updates in [64,128]:
  train=ROOT/f'{branch}_{updates:03d}_train';nom=ROOT/f'{branch}_{updates:03d}_nominal';push=ROOT/f'{branch}_{updates:03d}_push'
  if not all((f/'result.json').exists() and read(f/'result.json').get('completed') for f in [train,nom,push]):continue
  cp=train/'candidate.pt';pair=dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(nom),pushed=str(push))
  row=dict(branch=branch,updates=updates,checkpoint=str(cp),cases={})
  for test,folder in [('nominal',nom),('pushed',push)]:
   assert Path(read(folder/'run_config.json')['arguments']['checkpoint']).resolve()==cp.resolve()
   result=read(folder/'evaluation.json');z=np.load(folder/'telemetry.npz');names=z['joint_names'].tolist();hid=[names.index(n) for n in ['head_yaw','head_pitch']]
   cases=[]
   for i,c in enumerate(result['cases']):
    s=c['statistics'];q=z['q'][:,z['case_labels']==i][:,:,hid]
    cases.append(dict(command=s['command_m_s'],passed=c['passed'],failures=c['failures'],survival=s['survival'],trials=s['trials'],signed_speed=s['reset_frame_forward_speed_m_s'],lateral_speed=s['reset_frame_lateral_speed_m_s'],flight=s['geometric_flight_fraction'],head_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(q*q,-1))))),statistics=s))
   pair[test+'_cases']=cases;pair[test+'_all_pass']=result['passed'];row['cases'][test]=cases
  rows.append(row);pairs.append(pair)
  review=dict(run=read(OUT/'diagnostic_pause.json')['run'],latest=pair,promotions=[])
  (ROOT/f'{branch}_{updates:03d}_gate_review.json').write_text(json.dumps(review,indent=2))
complete=len(rows)==4
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),complete=complete,rows=rows,diagnostic_actors_deployed=False,source_reward_physics_limits_unchanged=True,maximum_independent_trials_per_command=32)
(OUT/'matched_exposure_review.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(complete=complete,rows=[dict(branch=r['branch'],updates=r['updates'],cases={t:[{k:c[k] for k in ['command','passed','failures','signed_speed','lateral_speed','flight','head_RMS_deg']} for c in cases] for t,cases in r['cases'].items()}) for r in rows]),indent=2))
