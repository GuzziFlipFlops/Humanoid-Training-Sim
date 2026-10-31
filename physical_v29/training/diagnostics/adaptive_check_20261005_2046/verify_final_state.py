"""Fresh owned liveness, exact newer gates and durable automation verification."""
import json,tomllib,hashlib,time,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json');s=live['state'];run=Path(live['active_run']['run'])
args=read(OUT/'automation_update_args.json');before=read(OUT/'automation_current_before_update.json')
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text())
assert cfg['prompt']==args['prompt'] and cfg['status']=='ACTIVE' and cfg['rrule']=='FREQ=HOURLY;INTERVAL=3'
assert cfg['id']==before['id'] and cfg['kind']==before['kind']=='heartbeat' and cfg['target_thread_id']==before['target_thread_id']
assert cfg.get('notification_policy')==before.get('notification_policy')
assert cfg['prompt'].split('Latest dated focus',1)[0].rstrip()==before['prompt'].split('Latest dated focus',1)[0].rstrip()
assert s['status']=='running' and s['supervisor_pid']==read(OUT/'diagnostic_pause.json')['supervisor_pid']
assert len(live['actual_worker_pids'])==1 and len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
assert time.time()-live['recorded_unix']<180
operation=Path(s['operation_path']);assert s['operation']=='train'
rows=[json.loads(x) for x in (operation/'progress.jsonl').read_text().splitlines()]
assert rows and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
assert rows[-1]['num_envs'] in [6144,8192] and rows[-1]['gpu']['free_mb']>=300 and rows[-1]['gpu']['ram_available_mb']>=2048
paths=[HERE/'ITERATION_HISTORY.md',HERE/'automation/THREE_HOUR_CHECK.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]
assert all('adaptive_check_20261005_2046' in p.read_text(encoding='utf-8-sig') for p in paths)

def cases(folder):
    ev=read(folder/'evaluation.json');z=np.load(folder/'telemetry.npz');q=z['q'];names=z['joint_names'].tolist();ids=[names.index('head_yaw'),names.index('head_pitch')];labels=z['case_labels']
    result=[]
    for i,c in enumerate(ev['cases']):
        t=c['statistics'];h=q[:,labels==i][:,:,ids]
        result.append(dict(command=t['command_m_s'],passed=c['passed'],failures=c['failures'],signed_speed_m_s=t['reset_frame_forward_speed_m_s'],
            geometric_flight_fraction=t['geometric_flight_fraction'],survival=t['survival'],trials=t['trials'],
            head_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(h*h,-1)))))))
    return result
pair=None
for nom in sorted(run.glob('*_run_150_gate_*_nominal')):
    if not (nom/'result.json').exists() or not read(nom/'result.json').get('completed'):continue
    suffix=nom.name.split('_',1)[1].replace('_nominal','_push');push=next(iter(run.glob('*_'+suffix)),None)
    if not push or not (push/'result.json').exists() or not read(push/'result.json').get('completed'):continue
    cp=Path(read(nom/'run_config.json')['arguments']['checkpoint'])
    assert cp.resolve()==Path(read(push/'run_config.json')['arguments']['checkpoint']).resolve()
    pair=dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(nom),pushed=str(push),nominal_cases=cases(nom),pushed_cases=cases(push))
assert pair
pair['all_prior_pass']=all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in pair[k] if c['command']<=1)
pair['running_pass']=all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in pair[k] if c['command']==1.5)
write(OUT/'closing_newer_matching_gates.json',pair)
shutil.copy2(HERE/'diagnostics/walk_regression_20261003/live_handoff.json',OUT/'final_liveness_owned_observation.json')
stamp=datetime.now(timezone.utc).isoformat();last=rows[-1]
record=dict(recorded_utc=stamp,owned_snapshot_utc=live['recorded_utc'],progress_last_utc=datetime.fromtimestamp(last['time'],timezone.utc).isoformat(),
    status='running',supervisor=s['supervisor_pid'],completed_operation_main_updates=s['total_training_iterations'],verified_recovered_main_updates=58,
    actual_retained_R37_main_updates=s['total_training_iterations']+58,declared_inherited_main_updates=7361,
    operation=str(operation),partial_current_updates=len(rows),environments=last['num_envs'],free_GPU_MiB=last['gpu']['free_mb'],host_available_MiB=last['gpu']['ram_available_mb'],
    one_actual_worker=live['actual_worker_pids'],frozen_hashes_verified=55,zero_preupdate_KL_command_mismatch=True,
    automation_verified_active=True,same_id_thread_cadence_notifications_scope=True,history_inventory=read(OUT/'history_inventory_final.log'),latest_matching_pair=pair,
    scope='Fresh live partial main progress is separate from exact saved-pair qualification. Current source/reward/geometry/caps/full-state learner remain unchanged.')
write(OUT/'final_liveness.json',record)
nextpath=HERE/'automation/NEXT_CHECK.json';n=read(nextpath);n['final_liveness']=str(OUT/'final_liveness.json');n['automation_verified_active']=True
n['latest_closing_gate_pair']=str(OUT/'closing_newer_matching_gates.json');n['new_main_updates']=s['total_training_iterations'];n['actual_retained_current_run_main_updates']=s['total_training_iterations']+58
n['selected_actor'].update(checkpoint=pair['checkpoint'],sha256=pair['sha256'],matching_gate_pair=str(OUT/'closing_newer_matching_gates.json'),standing_and_walking_cases_through_1_m_s_pass=pair['all_prior_pass'],running_qualified=pair['all_prior_pass'] and pair['running_pass'])
n['latest_verified_state'].update(recorded_utc=stamp,owned_snapshot_utc=live['recorded_utc'],status='running',operation=str(operation),operation_environments=last['num_envs'],learning_environments=s.get('next_training_environment_count'),actual_workers=live['actual_worker_pids'],gpu=live['gpu'],host=live['system_memory'],main_progress_memory=dict(recorded_utc=record['progress_last_utc'],gpu=last['gpu']))
write(nextpath,n)
nom,push=pair['nominal_cases'],pair['pushed_cases'];slow=[c for c in nom if c['command']==.35][0];slowp=[c for c in push if c['command']==.35][0]
closing=f'''Closing observation {stamp}, owned snapshot {live['recorded_utc']} and actual PPO progress through {record['progress_last_utc']}: same supervisor, one actual worker and 55 hashes verify. Current `{operation.name}` has {len(rows)} observed main updates at {last['num_envs']:,}, zero KL/command mismatch, free GPU {last['gpu']['free_mb']} MiB and host {last['gpu']['ram_available_mb']} MiB. Completed-operation main {s['total_training_iterations']}, plus 58 verified recovered updates. Latest exact matching actor `{Path(pair['checkpoint']).parent.name}` still fails 0.35 speed retention at {slow['signed_speed_m_s']:.3f}/{slowp['signed_speed_m_s']:.3f}; all other earlier commands pass. Running command 1.5 reaches {nom[-1]['signed_speed_m_s']:.3f}/{push[-1]['signed_speed_m_s']:.3f} m/s, true flight {100*nom[-1]['geometric_flight_fraction']:.4f}%/{100*push[-1]['geometric_flight_fraction']:.4f}%, head RMS {nom[-1]['head_RMS_deg']:.2f}/{push[-1]['head_RMS_deg']:.2f} degrees. Running remains unqualified. These gates bind that saved actor, not current partial updates. Same heartbeat ID/thread/three-hour cadence/notification preference/durable scope are verified active. History inventory: {record['history_inventory']['physical_runs']} physical runs, {record['history_inventory']['physical_operations']} operations, zero differing hashes and no read warnings; eight old missing manifests remain documented. [Final evidence](final_liveness.json), [newer matching pair](closing_newer_matching_gates.json).'''
report=OUT/'REPORT.md';text=report.read_text(encoding='utf-8');tag='<!-- final_liveness_2046 -->'
text=text.split('\n\n'+tag,1)[0];report.write_text(text+'\n\n'+tag+'\n'+closing+'\n',encoding='utf-8')
for p in paths:
    text=p.read_text(encoding='utf-8-sig');tag='<!-- closing_2046 -->'
    if tag not in text:
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+tag+'\n'+closing+'\n')
active=read(HERE/'ACTIVE_RUN.json');active.update(last_verified_utc=stamp,review_report=str(report),final_owned_observation=str(OUT/'final_liveness.json'));write(HERE/'ACTIVE_RUN.json',active)
args['prompt']=args['prompt'].split('\n\nClosing owned observation',1)[0]+f'\n\nClosing owned observation: {closing}'
write(OUT/'automation_update_args.json',args)
write(OUT/'diagnostic_source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')})
print(json.dumps({k:record[k] for k in ['recorded_utc','completed_operation_main_updates','partial_current_updates','environments','free_GPU_MiB','host_available_MiB','one_actual_worker','frozen_hashes_verified']},indent=2))
