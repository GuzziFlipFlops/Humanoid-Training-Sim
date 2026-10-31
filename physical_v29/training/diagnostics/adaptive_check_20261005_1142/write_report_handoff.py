"""Save this check's evidence-backed report, chronological history and focus."""
import json
import tomllib
from pathlib import Path
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
e=read(OUT/'evidence.json');d=e['decision'];live=read(OUT/'final_owned_observation.json');s=live['state']
g=e['latest_matching'];t=e['throughput'];r=e['resumed_learning']
a=read(OUT/'physical_objective_analysis.json');motor=read(OUT/'motor_speed_audit.json')
run=Path(d['run']);base=a['groups'][0]
running=g['cases'][-1]
head=[u['head_combined_rms_deg'] for u in running['upper_telemetry']]
report=f'''# October 5, 07:42 EDT: retained walking and constrained take-off diagnosis

**R36 remains the same full-state learner; running is not qualified.** At the saved {e['current_main']}-update observation, all stopped and prior walking commands through 1.0 m/s pass matching independent nominal/pushed tests. Head RMS at the running command is {head[0]:.2f}/{head[1]:.2f} degrees. The 1.5 command travels at {running['nominal_speed']:.3f}/{running['pushed_speed']:.3f} m/s, but genuine flight is {100*running['nominal_flight']:.4f}%/{100*running['pushed_flight']:.4f}%, below the unchanged 1.5% minimum. No whole running-stage, 5 m/s or actuator-ceiling endpoint is reached. [Decision](decision.json), [evidence](evidence.json), [owned state](final_owned_observation.json).

## Exact independent behavior

The latest complete pair is `{Path(g['nominal']).name}` / `{Path(g['pushed']).name}`, bound to `{Path(g['checkpoint']).parent.name}/candidate.pt`, SHA `{g['checkpoint_sha256']}`. Each test has 192 robots, 20 seconds and 32 independent trials per command. All survive. A prior-case pass does not promote the running stage.

| Command m/s | Nominal / pushed signed speed | Nominal / pushed true flight | Case result |
|---|---|---|---|
'''
for c in g['cases']:
    report+=f"| {c['command']:.2f} | {c['nominal_speed']:.3f} / {c['pushed_speed']:.3f} | {100*c['nominal_flight']:.4f}% / {100*c['pushed_flight']:.4f}% | {'Pass' if c['nominal_pass'] and c['pushed_pass'] else ', '.join(c['failures'])} |\n"
report+='''
Slow tracking is still drifting upward: 0.15 moved from 0.189 m/s after the first 64 updates to 0.229 after 960; 0.35 moved from 0.391 to 0.441. They still pass, but the passes should not be presented as perfect command tracking or indefinite stability. Earlier 832-update values were about 0.207 and 0.431. Running flight remains nearly zero across repeated complete pairs. [Latest exact gates](latest_matching_gates.json), [all gate trends](final_gate_trends.json), [per-command upper telemetry](upper_audit.json).

## Bounded physical intervention

The owned supervisor was suspended at its saved 896-update checkpoint 050 and its worker drained before a separate simulator started. Twelve cohorts, 24 robots each, ran for 20 simulated seconds. Nine experimental cohorts used smooth bilateral knee/hip/ankle shortening, varied lead time, or preceding stance extension. The other cohorts were running, walking and stopped controls. Interventions began after 3 seconds; statistics start at 5 seconds. Observations remain truthful, base free, gravity and self-collision active, and all native action, joint, slew, CAD projection and torque/speed constraints remain. There are no root impulses, pose writes, support forces, phase-input changes, trained diagnostic actors or deployed pulses.

All 288 robots survived without resets, but **no cohort produced genuine elevated-and-unloaded flight**. Some force-threshold gaps were grounded gaps. The best sampled minimum two-sole clearance was only 0.188 mm, below 1 mm. Retracting alone reduced speed to roughly 1.31–1.36 m/s; adding extension reached 1.46 m/s but remained grounded. Every running intervention had a lower complete mean reward than the unmodified running control. This grid rules out these particular interventions as a deployable repair; it does not prove take-off physically impossible or every natural running trajectory lower reward. The older preserved actor has previously demonstrated physical flight under the same caps.

[Physical cohorts and complete reward breakdown](physical_response/result.json), [synchronized 50 Hz trace](physical_response/telemetry.npz), [phase plot](phase_objective.png), [exportable plot](phase_objective.pdf).

## Effective target and reward diagnosis

'''
report+=f'''A CPU replay reconstructs the actual action clamp, default-offset targets, soft joint limits, slew and CAD feasibility projection on {a['projection_replay']['samples']:,} recorded states. Maximum difference from the deployed target is {a['projection_replay']['maximum_error_rad']:.3g} rad; no sampled target is rejected by the coupled CAD feasibility projection. Local finite differences hold each previous target fixed. Small ankle-action changes produce zero target derivative in {100*base['joints']['left_ankle_pitch']['local_target_derivative_zero_fraction']:.1f}%/{100*base['joints']['right_ankle_pitch']['local_target_derivative_zero_fraction']:.1f}% of the running-control samples; knee values are {100*base['joints']['left_knee_pitch']['local_target_derivative_zero_fraction']:.1f}%/{100*base['joints']['right_knee_pitch']['local_target_derivative_zero_fraction']:.1f}%. These are action/joint/slew dead zones, not measured policy gradients or a motor velocity ceiling. Walking also shows ankle dead zones, so their presence alone is insufficient to explain running failure.

During intended double swing, the control still has about {base['mean_total_force_in_double_swing_N']:.1f} N of total foot load. Larger extension/retraction reduces that mean to about 157 N in one cohort but never yields true flight. Phase plots show target-to-joint response lag and timing differences; the pre-action phase and post-step reward phase are saved separately. Their one-control-tick difference must be respected in any future timing experiment.

Sixteen CPU reward checks pass: the actual v8 summed-load term reconstructs the saved term; full term sums reproduce reward rates; progressive fixed-state simultaneous unloading earns monotonically more credit; redistributing unchanged load between two intended swing feet earns no extra credit; walking and stopping have no running-unload credit; and flight credit cannot appear without physical clearance and unloading. This establishes formula consistency, not favorable long-horizon return or convergence. The diagnostic's old `mean_unload_score` helper still averages foot exponentials. **Use `term_running_unload`, not that helper, for all v8 objective conclusions.** [Target/phase analysis](physical_objective_analysis.json), [reward counterexamples](reward_phase_audit.json).

Exact CAD Jacobian/reduction motor mapping and directional finite-difference unit checks pass on the saved 50 Hz q/qd trace. In the unmodified running cohort, sampled ankle p99 speeds remain well below their caps and there are no sampled cap crossings. Diagnostic interventions can change these statistics, but no sustained locomotion-relevant ceiling is established. Five-millisecond substep attribution remains untested. [Motor audit](motor_speed_audit.json).

The primary-source lead is [Siekmann et al., Periodic Reward Composition](https://arxiv.org/html/2011.01387v2). It combines phase-dependent foot-force and foot-velocity terms and smooths interval boundaries to support learning. Its Cassie controller and training contract differ from this robot. Smoothed timing or constraint-aware exploration is a hypothesis to test on the current truthful-state controller, not a reason to copy numerical weights or claim guaranteed convergence.

## Resume, resources and preservation

The same owned supervisor PID 19116 resumed after {t['excluded_pause_seconds']:.1f} seconds. The first resumed main block 053 completed 64 actual PPO updates at {r['environments']:,} environments, with exact actor, critic, Adam and exploration equality to the saved 050 full checkpoint and zero maximum pre-update KL/observed-action command mismatch. No source, reward, observation/action semantics, geometry, contact, motor cap, optimizer state or actor was reset. This actual completed block verifies continuation; no extra changed-source PPO smoke is needed because no runtime change was deployed. [Pause receipt](diagnostic_pause.json), [resume receipt](diagnostic_resume.json), [resumed learning](resumed_learning_verification.json).

Completed-main throughput is {t['active_collection_PPO_samples_s']/1000:.2f}k active collection-plus-PPO, {t['main_worker_effective_samples_s']/1000:.2f}k main-worker-effective and {t['whole_chain_samples_s_excluding_pause']/1000:.2f}k whole-chain samples/s after excluding this pause. The denominators differ. Main memory minima are GPU {t['minGPU']} MiB and host available {t['minhost']} MiB. The resumed block has minimum GPU {r['minimum_free_GPU_MiB']} MiB and host {r['minimum_host_MiB']} MiB, private growth {r['private_memory_growth_MiB']} MiB and no Torch allocation growth. Eval051 wrapper time includes the diagnostic pause and is not physics throughput. One actual CUDA worker and all 49 frozen authored hashes verify. Target 8,192, 300 MiB reserve, host fallback and 180 healthy-training-second upward ramps remain. [Rates and block evidence](final_throughput.json).

R36 preserves another 5,825 inherited main updates. Head-forward progress and walking teachers are retained; arm telemetry is recorded, but natural arm motion is not certified. Full-body recordings are already fulfilled and no new recording is pending. The initial CPU analysis repeatedly decompressed the NPZ and was stopped using its exact process identity; its initial logs and the file-lock move error are recorded. Caching bounded arrays corrected only this diagnostic. Raw physical evidence and production training were untouched.

## Decision and next check

Keep this same learner running while preparing a justified bounded comparison. Do not raise caps or deploy the unsuccessful pulses. Do not repeat rejected running-only row imitation or the 44% running-exposure change without new evidence. The next check must measure a candidate's actual flight, direction, all prior-speed tracking and head/arm behavior against unchanged full-state control; a local dead-zone statistic does not establish a successful repair.

'''
for i,test in enumerate(d['next_tests'],1):report+=f'{i}. {test}\n'
report+='\nUpdate this same adaptive three-hour heartbeat, preserving cadence, thread, notification preference, durable qualification scope and computer-on pause-and-brief endpoint. No endpoint is proven.\n'
(OUT/'REPORT.md').write_text(report,encoding='utf-8')

marker='adaptive_check_20261005_1142'
brief=f'''<!-- {marker}_current -->
## Latest R36 take-off and target diagnosis: October 5, 07:42 EDT

Read {str(OUT/'REPORT.md').replace(chr(92),'/')} and fresh ACTIVE_RUN/state. Same R36 learner and supervisor resumed after a 431.7-second bounded physical diagnostic pause; no production source, reward, policy/optimizer state or cap change. Latest saved observation has 960 new and 5,825 inherited main updates. Matching 054/055 tests retain stopped and all walking cases through 1.0, head about {head[0]:.2f}/{head[1]:.2f} degrees at the running command. Slow commands drift upward despite passing: 0.15 about 0.229, 0.35 about 0.441. Running 1.5 travels 1.371/1.375 with 0%/0.0035% true flight and remains unqualified.

Twelve 24-robot normal-limit cohorts, 20 seconds, all survive but none flies. Exact 14,400-state CPU target replay differs by only 2.98e-8 rad; ankle local target derivatives are zero in about 79–88% of running samples due to action/joint/slew constraints, with no sampled CAD projection rejection. This is not a policy gradient or motor ceiling. Sixteen reward checks reproduce v8 summed-force credit and reject load redistribution; only term_running_unload is authoritative, not the legacy mean_unload_score helper. Exact 50 Hz CAD speeds do not establish a ceiling; substeps remain unaudited. First resumed 64-update 8,192 block loads exact actor/critic/Adam/exploration, zero KL/command difference. 49 hashes and one worker verify. Rates about 46.68k active / 40.39k worker-effective / 25.88k whole-chain excluding pause.

Next bounded running-only constraint-aware target/exploration comparison and phase-response/full-objective diagnosis, with matching control, 64/128 learning checks and all prior/head gates before deployment. Also watch slow tracking drift and exact prior coverage. Preserve rejected variants, current posture progress, 8,192 target/300 MiB reserve/180-second healthy ramp and same adaptive three-hour computer-on endpoint. No new GUI request.

'''
for p in [HERE/'automation/THREE_HOUR_CHECK.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
    old=p.read_text(encoding='utf-8-sig')
    if f'<!-- {marker}_current -->' not in old:p.write_text(brief+old,encoding='utf-8')
entry=f'''

<!-- {marker} -->
## October 5, 07:42 EDT — R36 constrained take-off diagnosis, same full-state continuation

[Report]({(OUT/'REPORT.md').as_posix()}). Latest saved observation: 960 R36 main updates plus 5,825 inherited; exact 053 checkpoint and matching 054/055 nominal/pushed tests preserve stopped and walking through 1.0. All 192 robots survive, 32 trials per command for 20 seconds. Running command 1.5 travels 1.371/1.375 m/s, but only 0%/0.0035% geometric flight; no running or 5 m/s qualification. Head about {head[0]:.2f}/{head[1]:.2f} degrees; natural arms not certified. Slow tracking drifts despite passing (0.15: 0.189→0.229; 0.35: 0.391→0.441).

Paused exact supervisor 19116 at 896-update full checkpoint 050, SHA {e['physical_response']['checkpoint_sha256']}, drained worker and ran 12 cohorts × 24 robots × 20 seconds with truthful observations, free base, gravity and normal motor/CAD/joint/action constraints. Nine smooth retraction/timing/stance-extension variants all survive but none produces true flight; every intervention lowers full reward. Grounded force gaps are not flight. No pulse, root support, observation rewrite, diagnostic actor or changed reward is deployed. Exact 14,400-state target replay passes at 2.98e-8 rad maximum difference; ankle local target derivatives are zero in 79–88% of control samples from existing action/joint/slew constraints, no sampled CAD pair rejection. Sixteen CPU reward counterexamples pass, including actual v8 summed-force term reconstruction and redistribution invariance. Legacy mean_unload_score is explicitly corrected in interpretation only. Exact CAD 50 Hz motor mapping/unit checks pass, no relevant ceiling or substep attribution proven. Primary research lead: https://arxiv.org/html/2011.01387v2, periodic smoothed force/velocity composition; no research weights copied this check.

Same learner resumes after 431.7 seconds; actual 64-update block 053 preserves exact actor, critic, Adam and exploration, zero KL/command difference, at 8,192. No runtime source/weight/physics/cap/reset; 49 hashes and one actual CUDA worker. Rates 46.68k active / 40.39k worker-effective / 25.88k whole-chain excluding pause; 479 MiB minimum free GPU, 5,457 MiB minimum host available. Eval051 wrapper includes the pause. Initial slow CPU NPZ analysis and file-lock log-move attempt are preserved; diagnostic array caching repairs it, training untouched. Next bounded constraint-aware candidate versus unchanged control, phase/target latency and full-objective tradeoff, exact prior coverage/drift and matched independent gates before deployment. Same adaptive three-hour automation, 8,192 target/300 MiB reserve/180-second healthy ramp, endpoint save/pause/brief with computer on. All earlier evidence retained.
'''
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    if f'<!-- {marker} -->' not in p.read_text(encoding='utf-8-sig'):
        with p.open('a',encoding='utf-8') as f:f.write(entry)

nextpath=HERE/'automation/NEXT_CHECK.json';old=read(nextpath)
if not (OUT/'next_check_before.json').exists():write(OUT/'next_check_before.json',old)
nextcheck=dict(recorded_utc=live['recorded_utc'],automation_id=old['automation_id'],active_run=str(run),
               report=str(OUT/'REPORT.md'),latest_observation=str(OUT/'final_owned_observation.json'),
               selected_actor=dict(checkpoint=g['checkpoint'],sha256=g['checkpoint_sha256'],
                                   matching_gate_pair=str(OUT/'latest_matching_gates.json'),
                                   full_actor_critic_Adam_exploration=True,
                                   standing_and_walking_cases_through_1_m_s_pass=True,running_qualified=False),
               new_main_updates=e['current_main'],inherited_main_updates_preserved=5825,
               running_qualified=False,velocity_ceiling_proven=False,focused_tests=d['next_tests'],
               latest_repair_decision=d,throughput={k:v for k,v in t.items() if k!='blocks'},
               latest_verified_state=dict(recorded_utc=live['recorded_utc'],status=s['status'],stage=s['stage'],
                                          operation=s['operation_path'],actual_workers=live['actual_worker_pids'],
                                          frozen_hashes_verified=49,gpu=live['gpu'],host=live['system_memory']),
               diagnostic_pause_resume=str(OUT/'diagnostic_resume.json'),
               resumed_learning_verification=str(OUT/'resumed_learning_verification.json'),
               next_candidate_status='Undeployed hypothesis; require bounded matched control and independent gates.',
               endpoint=old['endpoint'],recording_request=old['recording_request'])
write(nextpath,nextcheck)
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text(encoding='utf-8'))
write(OUT/'automation_current_before_update.json',cfg)
assert cfg['id']==old['automation_id'] and cfg['kind']=='heartbeat' and cfg['rrule']=='FREQ=HOURLY;INTERVAL=3'
prefix=cfg['prompt'].split('Latest dated focus',1)[0].rstrip()
focus=f'''Latest dated focus, October 5, {live['recorded_utc']}: R36_posture_conditioned_priors remains the same full-state learner and supervisor PID 19116. Read diagnostics/adaptive_check_20261005_1142/REPORT.md, evidence.json, decision.json, latest_matching_gates.json, physical_objective_analysis.json, reward_phase_audit.json, motor_speed_audit.json, phase_objective.png and resumed_learning_verification.json. Read fresh ACTIVE_RUN/state rather than assuming this dated observation persists.

At 960 R36 main updates plus 5,825 inherited, exact checkpoint 053 matching nominal 054/pushed 055 tests pass stopped and all prior walking commands through 1.0. All 192 robots survive, 32 trials per command, 20 seconds. Running at the 1.5 command achieves 1.371/1.375 signed m/s but true flight is 0%/0.0035%, below the unchanged 1.5% minimum. Head RMS is {head[0]:.2f}/{head[1]:.2f} degrees; natural arms are not certified. Slow command tracking drifts upward despite passing: 0.15 from 0.189 after 64 to 0.229 after 960; 0.35 from 0.391 to 0.441. Retention is not perfect tracking or long-term stability. Preserve current progress and the separate qualified walking teacher and earlier 038 flight actor.

This check suspended the owned supervisor at its exact 896-update checkpoint 050, drained its worker and ran twelve cohorts of 24 robots for 20 seconds. Nine smooth bilateral shortening, timing and preceding stance-extension variants all survived but none produced genuine geometric flight; every running intervention lowered complete mean reward. Some force gaps remained grounded. Observations were truthful, free base/gravity/self-collision and native constraints remained; no pulse, root impulse/support, pose rewrite, actor hybrid or diagnostic actor entered production. These failures do not prove mechanical take-off impossible or all natural running lower return.

Exact CPU replay of 14,400 saved targets passes at 2.98e-8 rad maximum error. Local ankle-action target derivatives are zero in 79–88% of running-control samples from existing action/joint/slew constraints, with no sampled coupled-CAD projection rejection. These are local derivatives that hold previous targets fixed, not PPO gradients or a motor ceiling. Walking also has ankle dead zones. Sixteen reward counterexamples reproduce actual v8 summed-load credit and full term sums, monotonic fixed-state unloading and redistribution invariance. Use term_running_unload: the old diagnostic mean_unload_score helper is legacy averaging and must not guide objective conclusions. Phase plots save pre-action and reward phases separately and show target/joint response timing. Exact 50 Hz CAD motor Jacobian/reduction and unit checks pass; no sustained relevant ceiling is established and 5 ms substep attribution is untested. Primary lead is Siekmann et al. Periodic Reward Composition, https://arxiv.org/html/2011.01387v2; smoothed timing is a research hypothesis, not deployed weights or a guarantee.

Same supervisor resumed after 431.7 seconds with no source/reward/cap or actor/critic/Adam/exploration reset. Resumed block 053 completed 64 genuine PPO updates at 8,192 with exact full checkpoint load and zero KL/command mismatch. Minimum GPU 540 MiB, host 5,584 MiB, private memory growth 9 MiB, Torch allocation growth zero. One actual worker and 49 hashes verify. Completed-main rates are about 46.68k active collection-plus-PPO, 40.39k worker-effective and 25.88k whole-chain samples/s excluding this pause; denominators differ. Eval 051 wrapper includes the pause and is not physics timing. Target 8,192, reserve 300 MiB, host fallback and the 180 healthy-training-second ramp remain. Preserve full-body recordings; no pending repeat.

Next focused tests: (1) {d['next_tests'][0]} (2) {d['next_tests'][1]} (3) {d['next_tests'][2]} Do not spend the next check merely waiting or deploy a candidate just because local derivatives improve. Keep the same adaptive three-hour heartbeat active until verified actual 5 m/s or a causally established relevant ceiling, then save, pause and brief with the computer on.'''
args=dict(mode='update',id=cfg['id'],kind='heartbeat',name=cfg['name'],status='ACTIVE',rrule=cfg['rrule'],targetThreadId=cfg['target_thread_id'],prompt=prefix+'\n\n'+focus)
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
write(OUT/'automation_update_args.json',args)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),current_main=e['current_main'],same_run=str(run),automation_args=str(OUT/'automation_update_args.json')),indent=2))
