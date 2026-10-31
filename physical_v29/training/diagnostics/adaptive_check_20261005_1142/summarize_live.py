import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json');run=Path(live['active_run']['run']);s=live['state']
write(OUT/'initial_owned_observation.json',live)
gates=[]
for folder in sorted(run.glob('*_gate_*')):
 if not (folder/'evaluation.json').exists():continue
 e=read(folder/'evaluation.json');cfg=read(folder/'run_config.json');z=np.load(folder/'telemetry.npz');names=list(z['joint_names'])
 ids=[names.index('head_yaw'),names.index('head_pitch')];cases=[]
 for i,c in enumerate(e['cases']):
  t=c['statistics'];head=z['q'][:,z['case_labels']==i][:,:,ids]
  cases.append(dict(command=t['command_m_s'],passed=c['passed'],failures=c['failures'],survival=t['survival'],
   speed=t['reset_frame_forward_speed_m_s'],lateral=t['reset_frame_lateral_speed_m_s'],heading=t['final_heading_abs_mean_deg'],
   efficiency=t['forward_path_efficiency'],flight=t['geometric_flight_fraction'],force_flight=t['flight_fraction'],
   steps=[t['steps_left'],t['steps_right']],cadence=t['cycle_hz'],clearance=t['swing_clearance_m'],
   head_RMS_deg=float(np.rad2deg(np.sqrt((head**2).sum(-1).mean())))))
 gates.append(dict(folder=str(folder),checkpoint=cfg['arguments']['checkpoint'],perturb=e['perturb'],cases=cases,passed=e['passed']))
pairs=[]
for cp in dict.fromkeys(g['checkpoint'] for g in gates):
 a=[g for g in gates if g['checkpoint']==cp and not g['perturb']];b=[g for g in gates if g['checkpoint']==cp and g['perturb']]
 if a and b:pairs.append(dict(checkpoint=cp,nominal=a[-1],pushed=b[-1]))
write(OUT/'gate_trends.json',dict(run=str(run),completed_main=s['total_training_iterations'],pairs=pairs))
main=[]
for a in s['attempts']:
 f=Path(a['folder'])
 if a['operation']!='train' or '_run_150_train_' not in f.name or not a.get('completed'):continue
 rows=[json.loads(r) for r in (f/'progress.jsonl').read_text().splitlines()]
 assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 main.append(dict(folder=str(f),updates=len(rows),environments=rows[-1]['num_envs'],samples=sum(r['num_envs']*24 for r in rows),
   active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows),worker_seconds=a['seconds'],
   minGPU=min(r['gpu']['free_mb'] for r in rows),minhost=min(r['gpu']['ram_available_mb'] for r in rows),
   private_growth=rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb'],
   raw_clip=rows[-1]['ppo']['rollout_action_clip_fraction'],last_retention=rows[-1]['locomotion_retention']))
samples=sum(x['samples'] for x in main)
throughput=dict(completed_samples=samples,completed_main=sum(x['updates'] for x in main),
 active_samples_s=samples/sum(x['active_seconds'] for x in main),worker_effective_samples_s=samples/sum(x['worker_seconds'] for x in main),
 whole_chain_samples_s=samples/(live['recorded_unix']-s['started_unix']),minGPU=min(x['minGPU'] for x in main),
 minhost=min(x['minhost'] for x in main),all_actual_updates_KL_and_commands_zero=True,blocks=main,
 scope='R36 main updates only; whole chain includes initialization/capacity/evaluations. No diagnostic pause in R36 yet.')
write(OUT/'throughput.json',throughput)
write(OUT/'baseline.json',dict(run=str(run),main=s['total_training_iterations'],checkpoint=s['checkpoint'],
 checkpoint_sha256=hashlib.sha256(Path(s['checkpoint']).read_bytes()).hexdigest(),latest_pair=pairs[-1],
 actual_workers=live['actual_worker_pids'],verified_hashes=len(live['frozen_source_integrity']),
 memory=live['system_memory'],gpu=live['gpu'],throughput={k:v for k,v in throughput.items() if k!='blocks'}))
print(json.dumps(dict(main=s['total_training_iterations'],pairs=len(pairs),recent=[dict(folder=Path(p['nominal']['folder']).name,
   nominal=[{k:c[k] for k in ['command','passed','speed','flight','head_RMS_deg']} for c in p['nominal']['cases']],
   pushed=[{k:c[k] for k in ['command','passed','speed','flight','head_RMS_deg']} for c in p['pushed']['cases']]) for p in pairs[-3:]],
 throughput={k:v for k,v in throughput.items() if k!='blocks'}),indent=2))
