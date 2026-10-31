"""Read-only evidence consolidation; no Isaac import or live source changes."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json')
run=Path(live['active_run']['run']);state=live['state']
assert run.name=='locomotion_20261005_r37_offline_ground_full_state'
assert state['status']=='running' and len(live['actual_worker_pids'])<=1
assert len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
assert state.get('prior_training_iterations')==7361,'Wait for main continuation initialization'
shutil.copy2(HERE/'diagnostics/walk_regression_20261003/live_handoff.json',OUT/'final_owned_observation.json')
deployment=read(OUT/'deployment.json');blocks=[];samples=active_seconds=worker_seconds=0.
for f in sorted(run.glob('*_run_150_train_*')):
 if not (f/'result.json').exists():continue
 result=read(f/'result.json')
 if not result.get('completed'):continue
 rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
 load=read(f/'full_load_proof.json')
 assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
 assert rows and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 size=sum(r['num_envs']*24 for r in rows);seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows)
 attempt=next(a for a in state['attempts'] if Path(a['folder']).resolve()==f.resolve())
 samples+=size;active_seconds+=seconds;worker_seconds+=attempt['seconds']
 gpu=[r['gpu'] for r in rows]
 blocks.append(dict(folder=str(f),updates=len(rows),environments=rows[0]['num_envs'],
  checkpoint=str(f/'candidate.pt'),checkpoint_sha256=sha(f/'candidate.pt'),full_load=load,
  zero_KL_command_mismatch=True,min_GPU_MiB=min(g['free_mb'] for g in gpu),
  min_host_MiB=min(g['ram_available_mb'] for g in gpu),private_growth_MiB=gpu[-1]['process_private_mb']-gpu[0]['process_private_mb'],
  Torch_allocation_growth_MiB=gpu[-1]['torch_allocated_mb']-gpu[0]['torch_allocated_mb'],
  active_samples_s=size/seconds,worker_effective_samples_s=size/attempt['seconds']))
assert blocks,'Wait for one genuine main block; benchmarks and smoke do not count'
rates=dict(completed_main_updates=sum(b['updates'] for b in blocks),completed_main_samples=int(samples),
 active_collection_PPO_samples_s=samples/active_seconds,main_worker_effective_samples_s=samples/worker_seconds,
 whole_chain_samples_s=samples/(live['recorded_unix']-state['started_unix']),
 scope='R37 completed genuine main samples only. Active excludes startup/exit/gates; worker-effective includes main worker startup/exit; whole-chain includes R37 preflight, capacity and gates. Earlier R36 failure/downtime/diagnostics precede R37 and are not counted as R37 training time.',blocks=blocks)
(OUT/'resumed_learning_verification.json').write_text(json.dumps(rates,indent=2)+'\n')
def case_rows(folder):
 result=read(folder/'evaluation.json');trace=np.load(folder/'telemetry.npz');q=trace['q'];labels=trace['case_labels'];names=list(trace['joint_names']);head=[names.index('head_yaw'),names.index('head_pitch')];rows=[]
 for i,case in enumerate(result['cases']):
  st=case['statistics'];v=q[:,labels==i][:,:,head]
  rows.append(dict(command=st['command_m_s'],passed=case['passed'],failures=case['failures'],
   signed_speed_m_s=st['reset_frame_forward_speed_m_s'],geometric_flight_fraction=st['geometric_flight_fraction'],
   survival=st['survival'],head_combined_RMS_deg=float(np.sqrt(np.mean(np.sum(v*v,-1)))*180/np.pi)))
 return rows
pairs=[]
for nom in sorted(run.glob('*_run_150_gate_*_nominal')):
 if not (nom/'evaluation.json').exists():continue
 suffix=nom.name.split('_',1)[1].replace('_nominal','_push')
 push=next(iter(run.glob('*_'+suffix)),None)
 if not push or not (push/'evaluation.json').exists():continue
 cp=Path(read(nom/'run_config.json')['arguments']['checkpoint'])
 assert cp.resolve()==Path(read(push/'run_config.json')['arguments']['checkpoint']).resolve()
 pairs.append(dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),nominal=str(nom),pushed=str(push),
  nominal_cases=case_rows(nom),pushed_cases=case_rows(push)))
assert pairs,'Wait for a full matching independent pair'
latest=pairs[-1];latest['all_prior_pass']=all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in latest[k] if c['command']<=1.)
latest['running_pass']=all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in latest[k] if c['command']==1.5)
latest['qualification_scope']='Exact stated checkpoint only; never attribute its gates to a newer learner'
(OUT/'closing_matching_gates.json').write_text(json.dumps(latest,indent=2)+'\n')
initial=read(OUT/'initial_evidence.json');matched=read(OUT/'matched_comparison.json');probe=read(OUT/'physical_speed_response/result.json');motor=read(OUT/'physical_speed_response/substep_motor_audit.json')
assert all(not d['deploy_supported'] for d in matched['decisions'].values())
priority=[
 'Running still plateaus with retained neutral head and walking. Perform a bounded constraint-aware search or trajectory optimization around truthful current-state leg targets, resolving pre-action/reward phase and actual sole-force/target delay. Optimize complete return, not flight alone; validate any feasible normal-limit take-off trajectory in separate seeds and all prior/head cases before a 64/128-update matched-control continuation. Smoothed periodic force/velocity composition is a research lead; do not change arbitrary weights or deploy pulses, observation rewrites or hybrids.',
 'Compare fresh exact-checkpoint nominal/pushed stopped,0.15,0.35,0.65,1.0 and1.5 cases, head/arm motion and true flight. Preserve independently passing actors. The teacher-clipping-equivalence candidate failed to improve flight at64 and128 and is rejected; running-row transfer and44% exposure remain rejected. Diagnose the full running objective/phase/action timing, not repeated reward-identical waiting or raw-action loss alone.',
 'Verify local-ground provenance, exact actor/critic/Adam/exploration recovery, one worker/55 frozen hashes, moving8192/6144 memory,300MiB reserve and180 healthy training seconds before ramps. Use actual5ms CAD motor speed and driving-clip attribution; cap spikes during extrapolated falls never establish the endpoint. Retain all higher-speed/lower-speed gates and leave the computer on.'
]
decision=dict(recorded_utc=live['recorded_utc'],run=str(run),action='Exact full-state continuation with tested startup-only offline ground maintenance; reject training-gradient candidate',
 source_checkpoint=deployment['exact_continued_checkpoint'],source_sha256=deployment['checkpoint_sha256'],
 actor_critic_Adam_exploration_retained=True,reward_weights_physics_caps_PPO_unchanged=True,
 offline_ground_physical_parity=True,training_gradient_candidate_deployed=False,
 running_qualified=latest['running_pass'] and latest['all_prior_pass'],velocity_ceiling_proven=False,next_tests=priority)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
evidence=dict(recorded_utc=live['recorded_utc'],initial=initial,deployment=deployment,decision=decision,
 current_main=state['total_training_iterations'],inherited_main=7361,rates=rates,gates=latest,
 current_operation=state['operation_path'],learning_environments=state.get('next_training_environment_count'),
 frozen_hash_count=55,actual_workers=live['actual_worker_pids'],gpu=live['gpu'],host=live['system_memory'],
 matched_decisions=matched['decisions'],probe=probe,motor_summary=motor)
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
active=read(HERE/'ACTIVE_RUN.json');active.update(status='running',last_verified_utc=live['recorded_utc'],review_report=str(OUT/'REPORT.md'));temp=HERE/'ACTIVE_RUN.tmp';temp.write_text(json.dumps(active,indent=2)+'\n');temp.replace(HERE/'ACTIVE_RUN.json')
print(json.dumps(dict(main=state['total_training_iterations'],completed_blocks=len(blocks),latest_gate=latest['checkpoint'],all_prior_pass=latest['all_prior_pass'],running_pass=latest['running_pass'],rates={k:v for k,v in rates.items() if k!='blocks'},workers=live['actual_worker_pids']),indent=2))
