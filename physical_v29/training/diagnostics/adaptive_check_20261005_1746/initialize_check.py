import json,shutil,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json');run=Path(live['active_run']['run'])
assert run.name=='locomotion_20261005_r37_offline_ground_full_state'
assert live['state']['status']=='running' and all(live['frozen_source_integrity'].values())
shutil.copy2(HERE/'diagnostics/walk_regression_20261003/live_handoff.json',OUT/'initial_owned_observation.json')
pairs=[]
for nom in sorted(run.glob('*_run_150_gate_*_nominal')):
 if not (nom/'evaluation.json').exists():continue
 suffix=nom.name.split('_',1)[1].replace('_nominal','_push');push=next(iter(run.glob('*_'+suffix)),None)
 if not push or not (push/'evaluation.json').exists():continue
 cp=Path(read(nom/'run_config.json')['arguments']['checkpoint'])
 assert cp.resolve()==Path(read(push/'run_config.json')['arguments']['checkpoint']).resolve()
 cases={mode:[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],
  signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],flight=c['statistics']['geometric_flight_fraction'],
  survival=c['statistics']['survival']) for c in read(folder/'evaluation.json')['cases']] for mode,folder in [('nominal',nom),('pushed',push)]}
 pairs.append(dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(nom),pushed=str(push),cases=cases))
last=pairs[-1];heads={}
for mode in ['nominal','pushed']:
 d=np.load(Path(last[mode])/'telemetry.npz');q=d['q'];labels=d['case_labels'];names=list(d['joint_names']);ids=[names.index('head_yaw'),names.index('head_pitch')]
 heads[mode]=[dict(command=c['command'],head_combined_RMS_deg=float(np.sqrt(np.mean(np.sum(q[:,labels==i][:,:,ids]**2,-1)))*180/np.pi)) for i,c in enumerate(last['cases'][mode])]
record=dict(recorded_utc=live['recorded_utc'],run=str(run),state=live['state'],hashes=55,workers=live['actual_worker_pids'],
 gate_trends=pairs,latest_head=heads,scope='Matching exact checkpoint nominal/pushed pairs only; fresh live learner is separately unqualified')
(OUT/'initial_evidence.json').write_text(json.dumps(record,indent=2)+'\n')
pause=(HERE/'diagnostics/adaptive_check_20261005_1142/pause_for_diagnosis.py').read_text()
pause=pause.replace("run.name=='locomotion_20261005_r36_posture_conditioned_priors'","run.name=='locomotion_20261005_r37_offline_ground_full_state'")
pause=pause.replace('Bounded normal-limit take-off timing and full-objective diagnosis; resume same supervisor','Bounded normal-limit full-return trajectory search; resume same supervisor')
(OUT/'pause_for_diagnosis.py').write_text(pause)
print(json.dumps(dict(saved_main=live['state']['total_training_iterations'],latest_pair=last,head=heads),indent=2))
