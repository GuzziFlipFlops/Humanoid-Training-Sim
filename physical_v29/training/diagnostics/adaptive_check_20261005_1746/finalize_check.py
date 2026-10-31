"""CPU-owned checkpoint/gate/throughput consolidation after the bounded pause."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json');run=Path(live['active_run']['run']);s=live['state']
pause=read(OUT/'diagnostic_pause.json');resume=read(OUT/'diagnostic_resume.json');analysis=read(OUT/'trajectory_analysis.json')
assert s['status']=='running' and len(live['actual_worker_pids'])<=1
assert len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
assert s['supervisor_pid']==pause['supervisor_pid'] and s['prior_training_iterations']==7361
assert not analysis['deploy_supported'] and not resume['training_source_changed']
shutil.copy2(HERE/'diagnostics/walk_regression_20261003/live_handoff.json',OUT/'final_owned_observation.json')
blocks=[];total_samples=active_seconds=worker_seconds=0.
for f in sorted(run.glob('*_run_150_train_*')):
 if not (f/'result.json').exists() or not read(f/'result.json').get('completed'):continue
 rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()];load=read(f/'full_load_proof.json')
 assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
 assert rows and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 samples=sum(r['num_envs']*24 for r in rows);seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows)
 attempt=next(a for a in s['attempts'] if Path(a['folder']).resolve()==f.resolve())
 total_samples+=samples;active_seconds+=seconds;worker_seconds+=attempt['seconds'];gpu=[r['gpu'] for r in rows]
 blocks.append(dict(folder=str(f),updates=len(rows),environments=rows[0]['num_envs'],checkpoint=str(f/'candidate.pt'),
  checkpoint_sha256=sha(f/'candidate.pt'),full_load=load,zero_KL_command_mismatch=True,
  min_GPU_MiB=min(g['free_mb'] for g in gpu),min_host_MiB=min(g['ram_available_mb'] for g in gpu),
  private_growth_MiB=gpu[-1]['process_private_mb']-gpu[0]['process_private_mb'],
  Torch_allocation_growth_MiB=gpu[-1]['torch_allocated_mb']-gpu[0]['torch_allocated_mb'],active_samples_s=samples/seconds,
  worker_effective_samples_s=samples/attempt['seconds']))
resumed=[b for b in blocks if b['full_load'].get('source_sha256')==pause['checkpoint_sha256']]
assert resumed and sum(b['updates'] for b in resumed)>=64,'Wait for genuine resumed block; startup is not learning'
rates=dict(completed_main_updates=sum(b['updates'] for b in blocks),completed_main_samples=int(total_samples),
 active_collection_PPO_samples_s=total_samples/active_seconds,main_worker_effective_samples_s=total_samples/worker_seconds,
 whole_chain_samples_s=total_samples/(live['recorded_unix']-s['started_unix']-resume['pause_seconds']),
 excluded_diagnostic_pause_seconds=resume['pause_seconds'],blocks=blocks,resumed_blocks=resumed,
 scope='R37 completed main samples only. Active excludes startup/exit/gates; worker-effective includes main startup/exit. Whole-chain includes R37 capacity/gates and excludes this diagnostic pause. Earlier R36/R35 pauses/failure precede this run. Paused nominal061 wrapper includes this pause; do not use it as evaluation throughput.')
write(OUT/'resumed_learning_verification.json',rates)
def cases(folder):
 ev=read(folder/'evaluation.json');trace=np.load(folder/'telemetry.npz');q=trace['q'];names=list(trace['joint_names']);head=[names.index('head_yaw'),names.index('head_pitch')];labels=trace['case_labels'];rows=[]
 for i,c in enumerate(ev['cases']):
  st=c['statistics'];z=q[:,labels==i][:,:,head]
  rows.append(dict(command=st['command_m_s'],passed=c['passed'],failures=c['failures'],signed_speed_m_s=st['reset_frame_forward_speed_m_s'],
   geometric_flight_fraction=st['geometric_flight_fraction'],survival=st['survival'],trials=st['trials'],
   head_combined_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(z*z,-1)))))))
 return rows
pairs=[]
for nom in sorted(run.glob('*_run_150_gate_*_nominal')):
 if not (nom/'result.json').exists() or not read(nom/'result.json').get('completed'):continue
 suffix=nom.name.split('_',1)[1].replace('_nominal','_push');push=next(iter(run.glob('*_'+suffix)),None)
 if not push or not (push/'result.json').exists() or not read(push/'result.json').get('completed'):continue
 cp=Path(read(nom/'run_config.json')['arguments']['checkpoint']);assert cp.resolve()==Path(read(push/'run_config.json')['arguments']['checkpoint']).resolve()
 pair=dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),nominal=str(nom),pushed=str(push),nominal_cases=cases(nom),pushed_cases=cases(push))
 pair['all_prior_pass']=all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in pair[k] if c['command']<=1)
 pair['running_pass']=all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in pair[k] if c['command']==1.5)
 pairs.append(pair)
assert pairs
latest=pairs[-1];latest['scope']='Qualification binds exact checkpoint and matching pair, never newer learner'
write(OUT/'closing_matching_gates.json',latest);write(OUT/'gate_trends.json',pairs)
write(OUT/'last_all_prior_passing_actor.json',next(p for p in reversed(pairs) if p['all_prior_pass']))
priority=[
 'Resolve the measured physical timing and objective tradeoff before another running repair. Independent validation gives 0.915% flight, all just after the intended window, and loses 5.31 complete-return units; tracking dominates the loss. Saved flight half-cycle phases are 0.005-0.014 after wrap, versus the intended 0.43-0.50 window. Test earlier feasible leg timing and smoothed periodic force/velocity feedback using truthful recorded states, normal-limit physics and complete-return counterexamples for hopping, sliding and early exit. A one-tick clock distinction is not proof that the reward clock is wrong or that a fixed latency shift is optimal. Require a justified narrow candidate, unchanged current full-state control at 64 and 128 updates, and all nominal, pushed, prior, head and direction checks before deployment.',
 'Compare matching stopped, 0.15, 0.35, 0.65, 1.0 and 1.5 m/s gates, head/arm telemetry and per-command conditioned-teacher coverage. Prior retention regressed at 1,024 and recovered at 1,152 updates; preserve passing actors and current posture rather than claiming monotonic recovery. Diagnose recurring speed or lateral drift. Clipping-equivalence, running-row transfer and 44% exposure remain rejected absent new evidence. Keep production observations truthful; do not deploy pulses, hybrids or a tilted-head rollback.',
 'Verify the same full-state learner, local ground, 55 frozen hashes, one actual worker, moving 8,192/6,144 capacity, 300 MiB reserve, host guards and 180 continuous healthy training seconds before ramps. Exclude the documented diagnostic pause from throughput. Keep native CAD caps: the earlier 200 Hz probe had no driving-speed clips through 2.0 m/s, and fallen 2.5 m/s spikes do not establish a velocity ceiling. Preserve unsaved Fusion work and completed videos.'
]
decision=dict(recorded_utc=live['recorded_utc'],run=str(run),action='Reject local diagnostic trajectory transfer; resume exact same R37 full-state learner',
 paused_checkpoint=pause['checkpoint'],paused_checkpoint_sha256=pause['checkpoint_sha256'],
 actor_critic_Adam_exploration_retained=True,training_source_reward_physics_caps_unchanged=True,diagnostic_offsets_or_actors_deployed=False,
 running_qualified=latest['all_prior_pass'] and latest['running_pass'],velocity_ceiling_proven=False,next_tests=priority,
 limitation=analysis['limitation'])
write(OUT/'decision.json',decision)
e=dict(recorded_utc=live['recorded_utc'],initial=read(OUT/'initial_evidence.json'),decision=decision,trajectory_analysis=analysis,
 pause=pause,resume=resume,current_main=s['total_training_iterations'],inherited_main=7361,rates=rates,gates=latest,
 current_operation=s['operation_path'],operation_type=s['operation'],operation_environments=s['environment_count'],
 next_learning_environments=s.get('next_training_environment_count'),workers=live['actual_worker_pids'],frozen_hash_count=55,gpu=live['gpu'],host=live['system_memory'])
write(OUT/'evidence.json',e)
active=read(HERE/'ACTIVE_RUN.json');active.update(last_verified_utc=live['recorded_utc'],review_report=str(OUT/'REPORT.md'))
temp=HERE/'ACTIVE_RUN.tmp';write(temp,active);temp.replace(HERE/'ACTIVE_RUN.json')
print(json.dumps(dict(main=s['total_training_iterations'],resumed_blocks=[Path(b['folder']).name for b in resumed],
 rates={k:v for k,v in rates.items() if k not in ['blocks','resumed_blocks']},latest_gate=Path(latest['checkpoint']).parent.name,
 prior_pass=latest['all_prior_pass'],running_pass=latest['running_pass'],workers=live['actual_worker_pids']),indent=2))
