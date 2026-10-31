"""Bind fresh owned state, resumed PPO and exact matching independent gates."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json')
s=live['state'];run=Path(live['active_run']['run']);pause=read(OUT/'diagnostic_pause.json');resume=read(OUT/'diagnostic_resume.json')
assert str(run)==pause['run'] and s['status']=='running'
assert len(live['actual_worker_pids'])==1 and s['supervisor_pid']==pause['supervisor_pid']==19116
assert len(live['frozen_source_integrity'])==49 and all(live['frozen_source_integrity'].values())
write(OUT/'final_owned_observation.json',live)
block=run/'053_run_150_train_015';proof=read(block/'full_load_proof.json')
assert proof['source_sha256']==pause['checkpoint_sha256']
assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
rows=[json.loads(r) for r in (block/'progress.jsonl').read_text().splitlines()]
assert len(rows)==64 and read(block/'result.json')['completed']
assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
wrapper=next(a for a in s['attempts'] if Path(a['folder']).resolve()==block.resolve())
samples=sum(r['num_envs']*24 for r in rows)
resumed=dict(recorded_utc=live['recorded_utc'],same_supervisor=True,supervisor_pid=19116,
            block=str(block),completed_main_updates=len(rows),environments=rows[-1]['num_envs'],
            full_load=proof,maximum_pre_update_KL=0,maximum_command_difference=0,
            minimum_free_GPU_MiB=min(r['gpu']['free_mb'] for r in rows),
            minimum_host_MiB=min(r['gpu']['ram_available_mb'] for r in rows),
            private_memory_growth_MiB=rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb'],
            torch_allocation_growth_MiB=rows[-1]['gpu']['torch_allocated_mb']-rows[0]['gpu']['torch_allocated_mb'],
            active_samples_s=samples/sum(r['collect_seconds']+r['learn_seconds'] for r in rows),
            main_worker_effective_samples_s=samples/wrapper['seconds'],
            source_change=False,diagnostic_actor_deployed=False)
write(OUT/'resumed_learning_verification.json',resumed)

allg=[]
for folder in sorted(run.glob('*_gate_*')):
    if not (folder/'evaluation.json').exists():continue
    e=read(folder/'evaluation.json');cfg=read(folder/'run_config.json')
    allg.append(dict(folder=str(folder),checkpoint=cfg['arguments']['checkpoint'],perturb=e['perturb'],
                     cases=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],
                                 statistics=c['statistics']) for c in e['cases']],passed=e['passed']))
pairs=[]
for cp in dict.fromkeys(g['checkpoint'] for g in allg):
    a=[g for g in allg if g['checkpoint']==cp and not g['perturb']]
    b=[g for g in allg if g['checkpoint']==cp and g['perturb']]
    if a and b:pairs.append(dict(checkpoint=cp,nominal=a[-1],pushed=b[-1]))
latest=pairs[-1]
assert Path(latest['checkpoint']).resolve()==(block/'candidate.pt').resolve()
upper=read(OUT/'upper_audit.json')['cases']
case_records=[]
for a,b in zip(latest['nominal']['cases'],latest['pushed']['cases']):
    x,y=a['statistics'],b['statistics']
    assert x['trials']>=32 and y['trials']>=32
    u=[v for v in upper if Path(v['checkpoint']).resolve()==Path(latest['checkpoint']).resolve() and v['command_m_s']==a['command']]
    case_records.append(dict(command=a['command'],nominal_pass=a['passed'],pushed_pass=b['passed'],
       failures=sorted(set(a['failures']+b['failures'])),nominal_speed=x['reset_frame_forward_speed_m_s'],
       pushed_speed=y['reset_frame_forward_speed_m_s'],nominal_flight=x['geometric_flight_fraction'],
       pushed_flight=y['geometric_flight_fraction'],nominal_survival=x['survival'],pushed_survival=y['survival'],
       nominal_heading_deg=x['final_heading_abs_mean_deg'],pushed_heading_deg=y['final_heading_abs_mean_deg'],
       upper_telemetry=u))
matched=dict(checkpoint=latest['checkpoint'],checkpoint_sha256=hashlib.sha256(Path(latest['checkpoint']).read_bytes()).hexdigest(),
             nominal=latest['nominal']['folder'],pushed=latest['pushed']['folder'],cases=case_records,
             running_stage_pass=False,trials_per_command=32,seconds_per_trial=20)
write(OUT/'latest_matching_gates.json',matched)
write(OUT/'final_gate_trends.json',dict(run=str(run),pairs=pairs))

main=[]
for a in s['attempts']:
    f=Path(a['folder'])
    if a['operation']!='train' or '_run_150_train_' not in f.name or not a.get('completed'):continue
    rows=[json.loads(r) for r in (f/'progress.jsonl').read_text().splitlines()]
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    main.append(dict(folder=str(f),updates=len(rows),samples=sum(r['num_envs']*24 for r in rows),
                     active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows),worker_seconds=a['seconds'],
                     minGPU=min(r['gpu']['free_mb'] for r in rows),minhost=min(r['gpu']['ram_available_mb'] for r in rows),
                     private_growth=rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb']))
samples=sum(r['samples'] for r in main)
elapsed=live['recorded_unix']-s['started_unix']-resume['pause_seconds']
throughput=dict(completed_main_samples=samples,completed_main_updates=sum(r['updates'] for r in main),
                active_collection_PPO_samples_s=samples/sum(r['active_seconds'] for r in main),
                main_worker_effective_samples_s=samples/sum(r['worker_seconds'] for r in main),
                whole_chain_samples_s_excluding_pause=samples/elapsed,
                excluded_pause_seconds=resume['pause_seconds'],minGPU=min(r['minGPU'] for r in main),
                minhost=min(r['minhost'] for r in main),blocks=main,
                scope='R36 completed genuine main samples only. Whole chain includes initialization, capacity and evaluations; excludes current owned diagnostic pause. Eval051 wrapper includes that pause and is not physics timing.')
write(OUT/'final_throughput.json',throughput)
decision=dict(recorded_utc=live['recorded_utc'],run=str(run),action='Preserve and resume the same full-state R36 learner; no production repair deployed this check.',
              reason='Walking retention and neutral head hold across repeated independent gates. Bounded normal-limit take-off grid does not produce genuine flight. Exact target replay finds action/joint/slew dead zones, but no tested closed-loop training change yet improves running while retaining the prior skills.',
              training_sources_changed=False,weights_changed=False,physics_limits_changed=False,
              actor_critic_Adam_exploration_retained=True,diagnostic_pulses_deployed=False,
              running_qualified=False,velocity_ceiling_proven=False,
              next_tests=[
                'Audit running-only effective ankle/knee target exploration and local dead zones on current truthful neutral-head states. Compare a bounded training-only constraint-aware candidate against unchanged full-state control at 64 and 128 updates, with all nominal/pushed prior/head gates, before any deployment. No production action/observation rewrite or raised caps.',
                'Resolve pre-action versus reward phase and measured target-to-joint/sole response delay; test feasible normal-limit timing and complete objective return. The tested bilateral pulses were all grounded and lower reward. Do not treat their failure as proof of a mechanical impossibility or arbitrarily increase a flight weight.',
                'Compare fresh matched flight, stopped/prior speed and upper telemetry; preserve improved actors. Slow tracking is drifting upward despite still passing: 0.15 rose from 0.189 after 64 to 0.229 after 960 updates; 0.35 from 0.391 to 0.441. Inspect exact per-command prior target/gradient coverage and broad tracking-kernel sensitivity if this continues. Verify 8192 memory, full-state fallback, 180-second healthy ramp and throughput; keep rejected running-row/exposure variants rejected without new evidence.'
              ])
write(OUT/'decision.json',decision)
evidence=dict(recorded_utc=live['recorded_utc'],current_main=s['total_training_iterations'],inherited_main=5825,
              latest_matching=matched,resumed_learning=resumed,decision=decision,throughput=throughput,
              physical_response=read(OUT/'physical_response/result.json'),
              reward_audit=read(OUT/'reward_phase_audit.json'),
              target_replay=read(OUT/'physical_objective_analysis.json')['projection_replay'],
              actual_worker_pids=live['actual_worker_pids'],frozen_hashes_verified=49,
              files=['physical_response/telemetry.npz','physical_objective_analysis.json','motor_speed_audit.json','phase_objective.png','phase_objective.pdf','upper_audit.json'],
              rejected_diagnostic_attempt='Initial CPU analysis repeatedly decompressed the same NPZ arrays and was stopped by its exact process identity. Its logs are preserved; bounded array caching fixed the diagnostic only. Immediate log move encountered a file lock; raw physical evidence was untouched.',
              legacy_helper_erratum='Only term_running_unload represents deployed v8; result mean_unload_score is legacy diagnostic averaging and must not guide reward conclusions.')
write(OUT/'evidence.json',evidence)
print(json.dumps(dict(utc=live['recorded_utc'],main=s['total_training_iterations'],latest=matched['checkpoint'],
  cases=[{k:v for k,v in c.items() if k!='upper_telemetry'} for c in case_records],
  throughput={k:v for k,v in throughput.items() if k!='blocks'},resumed=resumed),indent=2))
