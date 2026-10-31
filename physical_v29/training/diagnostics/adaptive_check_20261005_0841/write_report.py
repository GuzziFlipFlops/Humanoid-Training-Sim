"""CPU-only chronological evidence and deployment report."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
pause=read(OUT/'diagnostic_pause.json');deployment=read(OUT/'deployment.json');old=Path(pause['run']);state=read(old/'state.json')
comparison=read(OUT/'all_prior_comparison.json');running=read(OUT/'matched_comparison.json');extended=read(OUT/'matched_extension_comparison.json')
partial=[]
for suffix in ['nominal','push']:
 e=read(OUT/('matched_prior_context_'+suffix)/'evaluation.json')
 partial.append(dict(condition=suffix,cases=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],
   signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],flight=c['statistics']['geometric_flight_fraction'],failures=c['failures']) for c in e['cases']]))
write(OUT/'partial_prior_comparison.json',dict(deployed=False,results=partial,reason='Unconditioned0.15 overspeeds; narrower all-prior correction performs better.'))
decision=dict(selected='posture_conditioned_walking_prior_v1',deployed=True,run=deployment['run'],
 exact_checkpoint=pause['checkpoint'],checkpoint_sha256=pause['checkpoint_sha256'],
 actor_critic_Adam_exploration_retained=True,diagnostic_learned_actors_deployed=False,
 running_imitation_deployed=False,student_observations_truthful=True,
 reward_geometry_caps_commands_unchanged=True,prior_gradient_formula_unchanged=True,prior_teacher_targets_changed=True,
 why='Reproduced tilted-head teacher feedback dependence; all-prior-only correction passes all walking/standing cases in matched nominal/pushed test and improves head posture. Running-only imitation loses its initial flight gain by128updates.',
 running_qualified=False,velocity_ceiling_proven=False)
write(OUT/'decision.json',decision)
main=[]
for f in sorted(old.glob('*_train_*')):
 if not (f/'result.json').is_file() or not read(f/'result.json').get('completed'):continue
 rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
 wrapper=next((a for a in state['attempts'] if Path(a['folder']).resolve()==f.resolve()),None)
 if wrapper:main.append((rows,wrapper['seconds']))
samples=sum(sum(r['num_envs']*24 for r in rows) for rows,_ in main)
active=sum(sum(r['collect_seconds']+r['learn_seconds'] for r in rows) for rows,_ in main)
worker=sum(seconds for _,seconds in main)
previous_pause=read(HERE/'diagnostics/adaptive_check_20261005_0240/diagnostic_resume.json')['pause_seconds']+read(HERE/'diagnostics/adaptive_check_20261005_0540/diagnostic_resume.json')['pause_seconds']
elapsed=pause['paused_unix']-state['started_unix']-previous_pause
throughput=dict(completed_main_samples=samples,completed_main_blocks=len(main),completed_main_updates=sum(len(rows) for rows,_ in main),
 active_collection_PPO_samples_s=samples/active,main_worker_effective_samples_s=samples/worker,
 full_chain_samples_s_before_this_pause_excluding_previous_pauses=samples/elapsed,
 previous_excluded_pause_seconds=previous_pause,this_diagnostic_pause_seconds=deployment['training_pause_seconds'],
 scope='R35 completed main samples only. Active is collection+PPO; worker-effective includes main-worker startup/exit; full chain includes gates/capacity before pause. Diagnostic A/B updates excluded. Eval214 wrapper includes this pause and must not be used as physics timing.')
write(OUT/'throughput.json',throughput)
evidence=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),decision=decision,deployment=deployment,
 previous_run_total_new_main=4352,previous_inherited_main=1473,
 latest_complete_old_pair=read(OUT/'gate_trends.json')['latest'],
 old_pair_main_updates=4288,drained_full_checkpoint_main_updates=4352,
 all_prior_matched_comparison=comparison,partial_prior_comparison=partial,
 running_only_64_comparison=running,running_only_128_comparison=extended,
 source_inventory=read(OUT/'all_prior_source_inventory.json'),cpu_checks=27,migration_checks=12,
 physical_parity=read(OUT/'all_prior_physical_parity.json'),throughput=throughput,
 velocity_ceiling_proven=False,running_qualified=False,gui_request_already_fulfilled=True)
write(OUT/'evidence.json',evidence)
report=f'''# October 5, 04:41 EDT check: posture-aware walking retention

R36 is a full-state continuation of R35 at its saved 4,352-update checkpoint, with another 1,473 inherited R34 main updates preserved. A reproduced head-feedback mismatch in the old walking teacher is corrected only in its training targets. Actor, critic, Adam, exploration and adaptive learning rate are retained; no old tilted-head actor or diagnostic learned actor is deployed. **Running remains unqualified.** [Decision](decision.json), [deployment](deployment.json), [complete evidence](evidence.json).

## Production behavior before repair

Fresh initial status at08:42 UTC showed one actual Isaac/CUDA worker, all47 frozen hashes and6144 environments. The final complete pre-pause pair211/212 belongs to checkpoint210 at4288 new R35 updates, not to the later drained checkpoint213. All192 robots survived each20-second test, with32 trials per command. Stopped,0.15 and1.0 cases passed both;0.35 and0.65 oversped. At1.5command the signed speed was1.426/1.429m/s with zero genuine flight in both, below the unchanged1.5% minimum. Running head RMS was2.958/2.954degrees, compared with51.71 before upper refinement. Earlier moving commands retained larger head error; natural arm motion is not certified. [Exact trends](gate_trends.json), [upper audit](upper_audit.json).

Supervisor45868 was suspended at an owned saved boundary, worker214 drained, and checkpoint213 SHA `{pause['checkpoint_sha256']}` preserved. The diagnostic pause lasted{deployment['training_pause_seconds']:.1f}seconds, including controlled physical tests and several matched PPO comparisons. R35 sources, checkpoints and logs remain intact. Eval214's supervisor wrapper includes this pause; its worker result is the appropriate physics timing record. [Pause receipt](diagnostic_pause.json).

## Reproduced teacher dependence

Five cohorts per command compared current R35, original teacher, a physically neutral head with actual feedback, neutral head with original simultaneous head feedback, and a compact context. Each cohort had32 robots, normal gravity/free base/self collision and unchanged CAD motor limits. At0.35/0.65/1.0, neutral physical head with actual feedback oversped to0.815/1.254/1.657m/s; original head histories restored0.436/0.733/1.053. At1.5, neutral head plus actual feedback turned70.10degrees with1.405m/s absolute lateral travel. Original head feedback restored0.876degrees and0.109m/s lateral while actual head remained near1.8degrees, with1.814% flight after the transient. This demonstrates learned feedback dependence, not absence of mechanical coupling. All observation rewrites/head overrides are diagnostic and absent from production. [Causal physics](prior_causal_response/result.json).

Constant context overspeeds lower commands and is rejected. A64-bin periodic head context derived from original teacher trajectories is interpolated using each history frame's saved phase. Neutral-head physical teacher replay produces signed0.434/0.729/1.051/1.485m/s at0.35/0.65/1.0/1.5, with1.781% genuine flight at1.5. The separate0.15 experiment reproduces0.454m/s with actual neutral-head feedback versus0.226 with original history; periodic context restores0.225m/s with0.95degrees actual head RMS. These validate diagnostic teacher dynamics, not a new inference controller. [Phase test](phase_causal_response/result.json), [0.15 phase test](015_phase_causal_response/result.json), [visually reviewed comparison](phase_causal_comparison.png).

Teacher queries on truthful CURRENT R35 states reconstruct current actor actions within6.2e-6 and show substantial lower-target differences from teacher head conditioning. Nominal joint-target error is before CAD projection, slew and torque dynamics. Labels on every future learner state are not certified by a successful teacher replay. [Target/gradient audit](teacher_target_audit.json).

## Three tested learning variants

All variants start from the exact213 full checkpoint,4096 environments, seed71, the existing learning-rate ceiling and identical reward/physics/PPO settings. Their independent nominal/pushed tests use192 robots for20seconds. No diagnostic learned checkpoint is deployed.

1. **Running-only final-body-row imitation:**19 CPU checks, same-actor physical parity and actual PPO/load tests pass. After64updates it increases flight from control0/0.0069% to0.490/0.576%, still below1.5%. After128updates it returns to0flight and0.65 still overspeeds. Reject this fragile running transfer. [64-update comparison](matched_comparison.json), [128-update comparison](matched_extension_comparison.json).
2. **Running imitation plus three conditioned prior commands:**0.35/0.65/1.0 improve but unconditioned0.15 overspeeds near0.268m/s. Running still has no qualifying flight. Reject this broader variant. [Partial-prior result](partial_prior_comparison.json).
3. **All four walking prior targets conditioned, no new running imitation:** the existing0.25 MSE strength,23-output denominator, stopped teacher and head/arm auxiliary are retained. Teacher-only head context changes targets on exact0.15/0.35/0.65/1.0 cohorts with five matching command frames, valid phase and latest actual head within10degrees. Student observations remain truthful. Shared-trunk effects of the original prior gradient and ordinary PPO remain possible; all gates stay mandatory. This variant passes stopped and every prior walking case in both tests, improves head error and is selected for exact full-state continuation.

| Command | Control nominal / pushed m/s | Corrected nominal / pushed m/s | Corrected pair |
|---|---|---|---|
'''
for a,b,c,d in zip(comparison['arms']['control']['nominal'],comparison['arms']['control']['pushed'],comparison['arms']['all_prior']['nominal'],comparison['arms']['all_prior']['pushed']):
 report+=f"| {a['command']:.2f} | {a['signed_speed']:.3f} / {b['signed_speed']:.3f} | {c['signed_speed']:.3f} / {d['signed_speed']:.3f} | {'Pass' if c['passed'] and d['passed'] else 'Flight fails'} |\n"
report+='''
All robots survive. Corrected moving head RMS is about3.1–3.5degrees versus control4.4–5.5degrees at prior speeds. Running geometric flight is0/0.0174%, still below1.5%; a walking-retention repair is not learned running. One64-update comparison does not establish long-term stability. [Matched source-bound comparison](all_prior_comparison.json).

## Validation and exact continuation

27 CPU checks cover input ownership, unchanged inference mean, exact MSE formula, no direct upper-row teacher copying, mixed-history/invalid-phase/nonneutral exclusions and unchanged stopped/running derivatives.12 migration checks reject bad checkpoint/source/signature/evidence/state/qualification declarations. Actual64-env4-update reload is exact for actor/critic/Adam/exploration, with zero pre-update KL and command mismatch. The selected-source, same-actor192-robot20-second replay is bitwise equal in all39 saved arrays and every case. [CPU checks](cpu_all_prior_tests.json), [migration checks](migration_cpu_tests.json), [full-state smoke](all_prior_full_state_smoke.json), [parity](all_prior_physical_parity.json).

Three frozen runtime files change: upper_body_refinement.py and new teacher_head_context.py/json.44 protected authored files, including worker, rewards, classifier, geometry/mass/contact, CAD Jacobians/motor caps, commands/history/actions and iteration4 execution, are byte-identical. The supervisor gains strict source/checkpoint/evidence-pinned migration and inventory coverage. The prior's derivative formula is unchanged, but its numerical targets/gradients are intentionally changed; inherited header wording must not be read as numeric gradient parity. The old teacher's qualification does not qualify conditioned labels or the current learner.

R36 starts at the original213 checkpoint, not at a diagnostic A/B checkpoint. Fresh capacity/PPO/load checks and old teacher requalification precede new main learning. All current and retention gates are repeated. [Protected inventory](all_prior_source_inventory.json), [strict continuation proof](teacher_context_continuation_proof.json).

## Resources, research and preserved failures

'''
report+=f"R35 completed-main throughput is{throughput['active_collection_PPO_samples_s']:.0f} active collection-plus-PPO,{throughput['main_worker_effective_samples_s']:.0f} main-worker-effective and{throughput['full_chain_samples_s_before_this_pause_excluding_previous_pauses']:.0f} full-chain samples/s before this pause, excluding earlier owned pauses. These denominators differ; diagnostic A/B samples are excluded. [Timing evidence](throughput.json).\n\n"
report+='''Pre-pause6144 PPO left about1.7GiB GPU free. The old ramp required2211MiB to add2048 environments and correctly excluded eval memory from its hold. New capacity is measured afresh. Target8192,300MiB reserve, host fallback and180 continuous healthy live-training seconds remain. Exact CAD50Hz audit finds current ankle p99 below75% of cap with isolated peaks; diagnostic teacher has rare sampled right-ankle crossings. No sustained locomotion-relevant ceiling is established; substep attribution is untested. [Motor audit](motor_speed_audit.json).

[Ross, Gordon and Bagnell's primary DAgger paper](https://proceedings.mlr.press/v15/ross11a.html) describes the observation-distribution problem in sequential imitation and motivates testing labels under learner-induced states. The selected repair is a local training-target adaptation, not a complete DAgger implementation or a convergence guarantee.

Failed preparation/launch attempts remain saved: phase script was initially launched before preparation finished and no simulator started; an older trace lacked a requested time array; a completed probe PID was stale; one patch context and one inline quoting attempt failed before runtime. The raw0.15 causal result retains a cloned scope annotation listing earlier commands; its group command fields, script and telemetry all show0.15. This is a metadata erratum, not altered telemetry. An extra poll of an already completed tool session returned unknown session; completion artifacts were verified. No failed diagnostic actor enters production.

Full-CAD video requests were already fulfilled; no repeat or new GUI capture is claimed here. Preserve unrelated unsaved Fusion work. No5m/s or causally proven actuator-ceiling endpoint is reached. The same adaptive3h heartbeat remains active and the computer stays on. Next compare full matched prior retention, posture and flight; reject fragile transfer instead of changing arbitrary reward weights. Fresh main-learning verification and automation receipts follow below.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'REPORT.md'),decision=decision['selected'],throughput=throughput),indent=2))
