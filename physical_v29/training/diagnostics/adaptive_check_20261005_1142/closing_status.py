"""Append a newer observation without overwriting this check's saved baseline."""
import hashlib
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json');s=live['state'];run=Path(live['active_run']['run'])
assert s['status']=='running' and s['supervisor_pid']==19116 and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==49 and all(live['frozen_source_integrity'].values())
write(OUT/'closing_owned_observation.json',live)
folders=[Path(a['folder']) for a in s['attempts'] if a.get('completed') and '_gate_' in Path(a['folder']).name and (Path(a['folder'])/'evaluation.json').exists()]
bound={}
for f in folders:
    cfg=read(f/'run_config.json');evaluation=read(f/'evaluation.json')
    key=cfg['arguments']['checkpoint'];bound.setdefault(key,{})[evaluation['perturb']]=f
cp,pair_folders=next((key,v) for key,v in reversed(list(bound.items())) if False in v and True in v)
nominal,push=pair_folders[False],pair_folders[True]
cases=[]
for a,b in zip(read(nominal/'evaluation.json')['cases'],read(push/'evaluation.json')['cases']):
    x,y=a['statistics'],b['statistics']
    assert x['trials']>=32 and y['trials']>=32
    cases.append(dict(command=x['command_m_s'],nominal_pass=a['passed'],pushed_pass=b['passed'],
                      nominal_speed=x['reset_frame_forward_speed_m_s'],pushed_speed=y['reset_frame_forward_speed_m_s'],
                      nominal_flight=x['geometric_flight_fraction'],pushed_flight=y['geometric_flight_fraction'],
                      nominal_survival=x['survival'],pushed_survival=y['survival'],failures=sorted(set(a['failures']+b['failures']))))
heads=[]
for f in [nominal,push]:
    z=np.load(f/'telemetry.npz');names=list(z['joint_names']);i=5
    q=z['q'][:,z['case_labels']==i][:,:,[names.index('head_yaw'),names.index('head_pitch')]]
    heads.append(float(np.rad2deg(np.sqrt((q**2).sum(-1).mean()))))
pair=dict(checkpoint=cp,checkpoint_sha256=hashlib.sha256(Path(cp).read_bytes()).hexdigest(),
          nominal=str(nominal),pushed=str(push),cases=cases,running_head_RMS_deg=heads,
          running_qualified=False,scope='Fresh matching 192-robot20-second nominal/pushed evaluations,32 trials per command; no gates changed.')
write(OUT/'closing_matching_gates.json',pair)
current=Path(s['operation_path']);p=live['worker_progress']
progress_scope='Current captured PPO progress'
if p is None:
    prior_main=next(Path(a['folder']) for a in reversed(s['attempts']) if a.get('completed') and '_run_150_train_' in Path(a['folder']).name)
    p=read(prior_main/'progress.json')
    progress_scope=f'Latest completed PPO block {prior_main.name}; current operation is evaluation and has no PPO progress file.'
free_gpu=int(live['gpu'].split(',')[-2].strip().split()[0])
free_host=live['system_memory']['available_mb']
inventory=read(OUT/'history_inventory_final.log');auto=read(OUT/'automation_verified.json')
assert inventory['differing_manifest_hashes']==0 and not inventory['read_warnings'] and auto['verified']
record=dict(recorded_utc=live['recorded_utc'],progress_recorded_unix=p['time'],run=str(run),completed_main=s['total_training_iterations'],
            operation=str(current),environments=s['environment_count'],learning_environments=p['num_envs'],observed_updates=p['iteration'],progress_scope=progress_scope,
            free_GPU_MiB=free_gpu,host_available_MiB=free_host,
            current_pre_update_KL=p['ppo']['pre_update_kl'],current_command_difference=p['behavior']['action_command_difference'],
            workers=live['actual_worker_pids'],hashes_verified=49,latest_matching=pair,
            history_inventory=inventory,automation_verification=auto)
if (OUT/'closing_metadata_verification.json').exists() and not (OUT/'closing_metadata_before_timeclarification.json').exists():
    write(OUT/'closing_metadata_before_timeclarification.json',read(OUT/'closing_metadata_verification.json'))
write(OUT/'closing_metadata_verification.json',record)
n=read(HERE/'automation/NEXT_CHECK.json')
n.update(recorded_utc=live['recorded_utc'],new_main_updates=s['total_training_iterations'],latest_observation=str(OUT/'closing_owned_observation.json'),
         latest_gate_trends=str(OUT/'closing_matching_gates.json'))
n['selected_actor'].update(checkpoint=cp,sha256=pair['checkpoint_sha256'],matching_gate_pair=str(OUT/'closing_matching_gates.json'))
n['latest_verified_state'].update(recorded_utc=live['recorded_utc'],operation=str(current),environments=s['environment_count'],learning_environments=p['num_envs'],progress_scope=progress_scope,
                                  completed_main_updates=s['total_training_iterations'],observed_current_operation_updates=p['iteration'],
                                  actual_workers=live['actual_worker_pids'],gpu=live['gpu'],host=live['system_memory'])
write(HERE/'automation/NEXT_CHECK.json',n)
run_case=cases[-1]
text=f'''\n\n## Closing owned observation\n\nAt {live['recorded_utc']}, the same R36 learner has {s['total_training_iterations']:,} completed new main updates, with operation `{current.name}` at {s['environment_count']:,} environments. PPO progress scope: {progress_scope}; its saved iteration is {p['iteration']} at {p['num_envs']:,} environments. GPU free is {free_gpu} MiB and host available {free_host} MiB. Saved PPO pre-update KL and command mismatch are zero. New matching `{nominal.name}` / `{push.name}` tests on `{Path(cp).parent.name}` still pass stopped and all earlier walking commands through 1.0. At 1.5, signed speed is {run_case['nominal_speed']:.3f}/{run_case['pushed_speed']:.3f} m/s but true flight remains {100*run_case['nominal_flight']:.4f}%/{100*run_case['pushed_flight']:.4f}%. Head RMS is {heads[0]:.2f}/{heads[1]:.2f} degrees. Slow speeds are {cases[1]['nominal_speed']:.3f}/{cases[1]['pushed_speed']:.3f} at 0.15 and {cases[2]['nominal_speed']:.3f}/{cases[2]['pushed_speed']:.3f} at 0.35, slightly down from the 960-update values but still above command. Tracking is nonmonotonic. One actual worker and all 49 hashes verify. No runtime change, qualification or endpoint is inferred. [Newer exact gates](closing_matching_gates.json), [owned state](closing_owned_observation.json), [closing metadata](closing_metadata_verification.json). History inventory has {inventory['physical_runs']} physical runs, {inventory['physical_operations']} operations, zero differing hashes and zero read warnings; eight historical missing manifests remain recorded. Same automation is active and verified; computer stays on.\n'''
report=(OUT/'REPORT.md').read_text(encoding='utf-8').split('\n\n## Closing owned observation',1)[0]
(OUT/'REPORT.md').write_text(report+text,encoding='utf-8')
for f in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with f.open('a',encoding='utf-8') as stream:
        stream.write('\n\nTimestamp clarification: the earlier closing note combined the owned snapshot with a later direct progress read. The following note uses the progress captured within the same CPU-owned observation; previous raw metadata is preserved in closing_metadata_before_timeclarification.json.\n'+text)
args=read(OUT/'automation_update_args.json')
args['prompt']=args['prompt'].split('\n\nClosing owned observation',1)[0]
args['prompt']+=f'''\n\nClosing owned observation {live['recorded_utc']}: same R36 learner, {s['total_training_iterations']} completed main updates, current operation {current.name} at {s['environment_count']} environments. Progress scope: {progress_scope}; saved PPO iteration {p['iteration']} at {p['num_envs']} environments. GPU free {free_gpu} MiB and host available {free_host} MiB. Read closing_matching_gates.json and closing_owned_observation.json. New checkpoint {Path(cp).parent.name} matching {nominal.name}/{push.name} retains every stopped/prior case. Running signed speed is {run_case['nominal_speed']:.3f}/{run_case['pushed_speed']:.3f}, true flight {100*run_case['nominal_flight']:.4f}%/{100*run_case['pushed_flight']:.4f}%, head RMS {heads[0]:.2f}/{heads[1]:.2f} degrees; running remains unqualified. Slow commands are {cases[1]['nominal_speed']:.3f}/{cases[1]['pushed_speed']:.3f} at 0.15 and {cases[2]['nominal_speed']:.3f}/{cases[2]['pushed_speed']:.3f} at 0.35, slightly down from 960 but still overspeeding. Tracking is nonmonotonic. Current KL/command mismatch zero, one worker and 49 hashes verified. Preserve the focused matched-control diagnosis and same three-hour cadence; no endpoint is met.'''
write(OUT/'automation_update_args.json',args)
print(json.dumps({k:v for k,v in record.items() if k not in ['latest_matching','history_inventory','automation_verification']},indent=2))
