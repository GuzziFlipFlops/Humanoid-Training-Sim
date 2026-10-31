"""Bind real replacement PPO/load/memory/progress; exclude discarded capacity."""
import hashlib,importlib.util,json,time,sys
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
sys.path.insert(0,str(HERE))
read=lambda p:json.loads(Path(p).read_text())
p=HERE/'record_owned_observation_handoff.py';spec=importlib.util.spec_from_file_location('owned_recorder',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.OUT=OUT/'closing_cpu';module.main()
live=read(OUT/'closing_cpu/live_handoff.json');run=Path(live['active_run']['run']);s=live['state']
assert s['status']=='running' and s['stage']=='run_350' and s.get('prior_training_iterations')==16936
assert read(run/'termination_requalification.json')['all_prior_gates_fresh']
blocks=[];samples=0;active_seconds=worker_seconds=0.;partial=[]
first_reported_main_time=None
for f in sorted(run.glob('*_run_*_train_*')):
 if not (f/'progress.jsonl').exists():continue
 rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines() if x.strip()]
 if not rows:continue
 first_reported_main_time=min(first_reported_main_time or rows[0]['time'],rows[0]['time'])
 proof=read(f/'full_load_proof.json');assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
 assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 result=read(f/'result.json') if (f/'result.json').exists() else None
 count=sum(r['num_envs']*24 for r in rows);seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows)
 item=dict(folder=str(f),full_load_proof=proof,observed_updates=len(rows),environments=rows[-1]['num_envs'],
  genuine_main=True,completed=bool(result and result.get('completed')),samples=count,active_collection_PPO_samples_s=count/seconds,
  zero_max_preupdate_KL_and_command_mismatch=True,
  minimum_free_gpu_MiB=min(r['gpu']['free_mb'] for r in rows),minimum_host_available_MiB=min(r['gpu']['ram_available_mb'] for r in rows),
  private_memory_growth_MiB=rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb'],
  torch_allocation_growth_MiB=rows[-1]['gpu']['torch_allocated_mb']-rows[0]['gpu']['torch_allocated_mb'],
  latest_gpu=rows[-1]['gpu'],prior_targets_active=rows[-1].get('locomotion_retention'),upper_refinement=rows[-1].get('upper_refinement'))
 if item['completed']:
  assert result['iterations']==len(rows);samples+=count;active_seconds+=seconds
  attempt=next(a for a in s['attempts'] if Path(a['folder']).resolve()==f.resolve());worker_seconds+=attempt['seconds'];blocks.append(item)
 else:partial.append(item)
assert blocks or partial,'No genuine main PPO progress observed'
active=live['active_run'];active.update(status='running',supervisor_pid=s['supervisor_pid'],
 launcher_pid=read(OUT/'certificate_retry_deployment.json')['new_supervisor']['pid'],
 last_verified_utc=live['recorded_utc'],final_owned_observation=str(OUT/'closing_cpu/live_handoff.json'),
 review_report=str(OUT/'REPORT.md'))
from status_io import atomic_json
atomic_json(HERE/'ACTIVE_RUN.json',active)
live['active_run']=active;atomic_json(OUT/'closing_cpu/live_handoff.json',live)
completed_updates=sum(x['observed_updates'] for x in blocks)
assert completed_updates<=s['total_training_iterations']
rates=dict(completed_main_active_collection_PPO_samples_s=samples/active_seconds if samples else None,
 completed_main_worker_effective_samples_s=samples/worker_seconds if samples else None,
 completed_main_whole_chain_samples_s=samples/(live['recorded_unix']-s['started_unix']) if samples else None,
 scope='R39 completed main only; startup/capacity/fresh gates included in whole-chain clock, never in main samples. Partial active rate is separate. Earlier R37 pause accounting remains separately dated.')
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=True,run=str(run),supervisor_pid=s['supervisor_pid'],
 one_actual_worker=live['actual_worker_pids'],frozen_hashes_verified=len(live['frozen_source_integrity']),
 new_completed_main_updates=s['total_training_iterations'],inherited_actual_main_updates=16936,
 completed_blocks=blocks,partial_main_blocks=partial,rates=rates,learning_actors_are_unqualified=True,
 full_checkpoint_source_is_original459=True,smoke_actors_deployed=False)
(OUT/'new_main_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
boundary=read(OUT/'throughput_and_loads.json')['measurement_endpoint_unix']
(OUT/'continuation_timing.json').write_text(json.dumps(dict(saved_boundary_suspension_unix=boundary,
 R38_start_and_retirement_receipt=str(OUT/'handoff_completion.json'),
 active_R39_supervisor_start_unix=s['started_unix'],saved_boundary_to_R39_supervisor_seconds=s['started_unix']-boundary,
 first_reported_R39_main_PPO_unix=first_reported_main_time,
 saved_boundary_to_first_reported_R39_PPO_seconds=first_reported_main_time-boundary,
 scope='Includes bounded diagnosis, failedR38 startup and repeated R39 capacity/requalification. First PPO report bounds completion of its first real update, not exact launch or continuous learner stall.'),indent=2)+'\n')
print(json.dumps({'rates':rates,'new_completed_main_updates':s['total_training_iterations'],'observed_partial_updates':[x['observed_updates'] for x in partial]}))
