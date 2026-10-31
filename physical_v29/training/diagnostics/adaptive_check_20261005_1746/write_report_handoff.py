"""Durable check records and native heartbeat update arguments; no runtime edits."""
import json,tomllib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
e=read(OUT/'evidence.json');live=read(OUT/'final_owned_observation.json');d=e['decision'];g=e['gates'];rates=e['rates'];a=e['trajectory_analysis']
s=live['state'];now=e['recorded_utc'];run=Path(d['run']);head=[g[k][-1]['head_combined_RMS_deg'] for k in ['nominal_cases','pushed_cases']]
report=f'''# October 5, 17:46 UTC adaptive check — measured trajectory tradeoff; original learner resumed

At {now}, **R37_offline_ground_full_state** is running with {s['total_training_iterations']} saved main updates and **7,361 inherited updates**. The same supervisor PID {s['supervisor_pid']} resumes from its full saved actor, critic, Adam and exploration. All 55 frozen authored hashes verify. No training source, reward, motor cap, physics, teacher, optimizer or observation change was deployed. Running and the 5 m/s endpoint remain unqualified.

## Actual independent progress

The 1,024-update pair present at the beginning of this check had zero genuine flight and failed pushed 0.35/0.65 speed retention. The 1,152-update checkpoint060 subsequently recovered every stopped/prior command through1.0 in nominal061/pushed062, but its1.5 case failed flight and world lateral speed. Retention is nonmonotonic. Passing records are preserved and never attributed to a newer actor.

Latest completed matching pair is bound to `{Path(g['checkpoint']).parent.name}`, SHA `{g['checkpoint_sha256']}`. Each test contains192 robots,20 seconds,32 independent trials per command, with unchanged physical, contact, direction, gait, nominal/pushed and stopped/prior classifiers.

| Command m/s | Nominal signed speed | Pushed signed speed | Nominal/pushed pass | True flight % nominal/pushed |
|---:|---:|---:|:---:|---:|
'''
for x,y in zip(g['nominal_cases'],g['pushed_cases']):
 report+=f"| {x['command']:.2f} | {x['signed_speed_m_s']:.3f} | {y['signed_speed_m_s']:.3f} | {x['passed']}/{y['passed']} | {100*x['geometric_flight_fraction']:.4f}/{100*y['geometric_flight_fraction']:.4f} |\n"
report+=f'''
Stopped/prior cases all pass both tests: **{g['all_prior_pass']}**. Running case passes: **{g['running_pass']}**. At1.5 the head's combined yaw/pitch RMS is {head[0]:.2f}/{head[1]:.2f} degrees, preserving its forward-posture improvement. Natural arm motion is not certified. This check reruns the CPU-only upper-body audit of saved full-population positions/velocities; the5-degree reporting band remains diagnostic. [Exact gates](closing_matching_gates.json), [all checkpoint-bound trends](gate_trends.json), [last prior-passing actor](last_all_prior_passing_actor.json), [upper-body measurements](upper_audit.json).

## New bounded physical timing search

At checkpoint060, SHA `{d['paused_checkpoint_sha256']}`, the owned supervisor was suspended at an evaluation boundary; its worker was allowed to finish and drained before diagnostics. The search used the exact saved actor, truthful400-input history and executed-action feedback, native CAD differential/joint/motor/slew/action constraints, free base, gravity and self collision. It modified only diagnostic hip-pitch, knee-pitch and ankle-pitch actions. No head override, root support, pose write, phase write, fabricated observation or new learned actor was used.

Ten bounded parameters describe two smooth raised-cosine offset windows, their centers/widths and hip/knee/ankle amplitudes. Windows use the **observed pre-action phase**; reward phase is recorded separately. Offsets fade in after3s. They pass through the existing action clamp, soft joint limits, slew limiter and CAD pair feasibility checks, rather than bypassing them. Zero-offset controls execute the same clamped actions as the actor. Thirteen CPU formula/selection/terminal-return checks pass. These fixtures do not prove physical gait quality. [Parameter bounds](trajectory_parameters.py), [CPU checks](trajectory_cpu_tests.json).

Four successive populations at search seed71 evaluate32 trajectory cohorts with8 robots each for12s:128 cohorts including unchanged controls and elite means. Elites are selected by **complete return**, including the first terminal penalty and excluding later reset rewards. All search robots survive. Target, actual joints, forces, geometric clearances, raw/effective actions, command snapshots and every reward term are saved at50Hz. Observed/action command difference is zero. All generated offsets remain diagnostic and are absent from production.

The search's highest-return trajectories remain grounded; its maximum real geometric-flight fraction is **{100*a['maximum_search_flight_fraction']:.3f}%**, below the existing1.5% minimum. This is useful negative evidence within these bounds, not a proof that the physical robot cannot run. [Full search populations](trajectory_search/result.json), [objective/flight plot](trajectory_return_flight.png).

## Independent-seed validation and full objective costs

The best complete-return and best-flight trajectories are frozen, then retested against unchanged control with seed83:32 robots per cohort,20 seconds. All96 survive. This is a diagnostic feasibility test of offsets, not policy or running-stage qualification. Robots within a population are not exact paired-state counterfactuals.

| Diagnostic cohort | Signed speed m/s | True flight % | Complete20s return delta | Head RMS degrees |
|---|---:|---:|---:|---:|
'''
for row in a['validation']:
 report+=f"| {row['label']} | {row['signed_forward_m_s']:.3f} | {100*row['geometric_flight_fraction']:.4f} | {row['return_delta']:+.3f} | {row['head_RMS_deg']:.2f} |\n"
report+='''
The highest-flight trajectory reproduces about0.915% flight, with mean minimum sole clearance about1.48mm in those samples. Its longest observed run is one50Hz frame; this is not substep-certified20ms sustained flight. **All its genuine-flight samples occur outside the desired double-swing reward window**, while99.94% of intended double-swing samples retain load. Its complete return is5.31 lower than control. Average rate changes are dominated by linear tracking−0.2069, heading−0.0317, gait−0.0267 and angular tracking−0.0249, versus flight+0.00205. Joint-acceleration cost changes by only−0.00782. This observed trajectory has a timing/tracking tradeoff; it does not prove every natural running trajectory has lower reward.

The highest-return validation cohort gains1.95 complete-return units and tracks1.469m/s, but its true flight is only0.0083%. None satisfies flight, full-return and direction/head feasibility together. Every recorded full reward exactly reconstructs from its saved term sum. Action/reward phase is separated by0.033 cycles at1.65Hz, one20ms control tick. A descriptive target lag from the previous check is not a justified universal phase shift.

**Decision: reject trajectory transfer for deployment.** No candidate reaches the prerequisite physical behavior, so injecting these labels into a64/128-update learner would not be justified. No diagnostic learned actor is deployed and no new learning experiment is falsely reported as passed. Preserve the128 populations, independent validation, bounds, seeds, all telemetry and full checkpoint. [Selection](validation_selection.json), [independent physical validation](trajectory_validation/result.json), [term/timing analysis](trajectory_analysis.json), [decision](decision.json).

The next research lead is [Periodic Reward Composition](https://arxiv.org/html/2011.01387v2): phase-dependent force/velocity signals with smoothed interval boundaries. The measured phase mismatch supports a controlled timing test; the paper supplies neither a guarantee nor robot-specific numerical weights. Any new signal must pass fixed-state counterexamples, feasible normal-limit physics and a matched current-full-state64/128-update control with every retained/head/direction gate before production.

## Exact resume, memory and throughput

'''
for b in rates['resumed_blocks']:
 report+=f"Resumed `{Path(b['folder']).name}` completes{b['updates']} genuine updates at{b['environments']:,}; actor/critic/Adam/exploration load equality all pass, maximum pre-update KL and command mismatch are zero. Minimum free GPU{b['min_GPU_MiB']}MiB, host{b['min_host_MiB']}MiB, private growth{b['private_growth_MiB']}MiB, Torch allocation growth{b['Torch_allocation_growth_MiB']}MiB.\n\n"
report+=f'''The same supervisor resumes after **{e['resume']['pause_seconds']:.1f}s** after worker drain. Nominal061's supervisor wrapper includes this pause; it must not be treated as physical evaluation speed. Completed-main rates are **{rates['active_collection_PPO_samples_s']/1000:.2f}k active collection-plus-PPO**, **{rates['main_worker_effective_samples_s']/1000:.2f}k worker-effective** and **{rates['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s**, excluding this diagnostic pause from whole-chain time. Whole chain includes R37 startup, capacity and gates; prior R36/R35 failures/pauses precede R37 and are not in its clock. Capacity/smoke/diagnostic frames are not main training progress. [Pause](diagnostic_pause.json), [resume](diagnostic_resume.json), [exact resumed learning](resumed_learning_verification.json), [owned observation](final_owned_observation.json).

Current operation `{Path(s['operation_path']).name}` is `{s['operation']}` at{s['environment_count']} environments; an evaluation count is not PPO capacity. Next selected training count is{s.get('next_training_environment_count')}. One actual worker,55 frozen hashes,8,192 target,300MiB reserve,host fallback and180 continuous healthy training-second upward ramp remain. The earlier passive200Hz motor audit found no driving-speed clipping through2.0 for that actor/experiment; isolated clips around fallen2.5 extrapolation do not establish a locomotion-relevant ceiling. This trajectory search does not add a new200Hz ceiling certification. Native limits remain unchanged.

## Records and next check

One attempted PowerShell/Python inline resume-script construction failed from quoting before creating or executing the script. It is preserved in tool evidence; the saved standalone resume script then succeeds on the exact verified supervisor identity. A read during main startup correctly found no progress file yet; subsequent actual progress is used, never startup memory as steady capacity. Earlier reports and all failed/rejected experiments remain intact. No full-body video request is pending; completed videos and unrelated unsaved Fusion work are preserved.

'''
for i,p in enumerate(d['next_tests'],1):report+=f'{i}. {p}\n\n'
report+='Keep the same adaptive three-hour heartbeat active. At verified actual5m/s or a causally established relevant velocity ceiling, save,pause training and this automation,brief the user and leave the computer on.\n'
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
marker='adaptive_check_20261005_1746'
brief=f'''<!-- {marker}_current -->
## Latest trajectory search and exact resume: October5,{now}

Read {(OUT/'REPORT.md').as_posix()} and fresh ACTIVE_RUN/state. Same R37 full-state learner/supervisor,{s['total_training_iterations']} saved main+7,361 inherited,55 hashes. Latest exact prior pass:{g['all_prior_pass']}; running case pass:{g['running_pass']}. Head at1.5 {head[0]:.2f}/{head[1]:.2f}deg; natural arms uncertified. Retention regressed1024 and recovered1152; do not infer monotonic recovery or saved gates on newer actors.

New128-cohort normal-limit smooth hip/knee/ankle timing search and independent96-robot20s validation reproduce a tradeoff. Best-flight validation gives0.915% flight,below1.5%,all outside intended double swing,and−5.31 complete return; tracking dominates loss. Best-return gains1.95 but stays effectively grounded0.0083%. Thirteen CPU checks and exact reward-term reconstruction pass. Reject offsets/labels; no production source/reward/physics/cap/optimizer/actor reset. Same supervisor resumes after{e['resume']['pause_seconds']:.1f}s; real64-update full-state block,zero KL/command mismatch. Rates{rates['active_collection_PPO_samples_s']/1000:.1f}k active/{rates['main_worker_effective_samples_s']/1000:.1f}k worker-effective/{rates['whole_chain_samples_s']/1000:.1f}k whole-chain excluding pause. Next controlled phase/objective signal,matched full-state comparison and all gates; no arbitrary weights or claim of impossible running. Preserve posture/walking teachers/038 and rejected variants. Same3h automation,target8192/reserve300/180healthy ramp and computer-on endpoint remain.

'''
for p in [HERE/'automation/THREE_HOUR_CHECK.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
 old=p.read_text(encoding='utf-8-sig')
 if f'<!-- {marker}_current -->' not in old:p.write_text(brief+old,encoding='utf-8')
entry=f'''\n\n<!-- {marker} -->
## October5,{now} — bounded physical trajectory search rejected; exact R37 resumed

[Report]({(OUT/'REPORT.md').as_posix()}). Same full-state learner paused at060 SHA{d['paused_checkpoint_sha256']},worker drained.128 smooth ten-parameter leg timing cohorts,8 robots/cohort,12s,followed by separate seed83,96 robots/20s. All survive;13 CPU checks; saved full term sums exactly reconstruct reward; observed/action commands match. Best-flight validation0.915% below1.5%,flight outside double swing,return−5.31,tracking loss−0.2069rate dominatesflight+0.00205. Best-return+1.95 remains0.0083%flight. Rejected for transfer,no diagnostic actor/offset/observation rewrite/root support/hybrid/head override deployed. Bounds are not proof of impossible running or actuator ceiling. Primary timing lead [Periodic Reward Composition](https://arxiv.org/html/2011.01387v2); no copied numerical weights.

Same supervisor resumes after{e['resume']['pause_seconds']:.1f}s; genuine64-update main full actor/critic/Adam/exploration equality,zero KL/command mismatch.55 frozen hashes,one actual worker,no runtime/reward/cap/optimizer change. Latest exact pair prior pass{g['all_prior_pass']},running pass{g['running_pass']},head{head[0]:.2f}/{head[1]:.2f}deg; natural arms uncertified. R37{s['total_training_iterations']} main+7,361 inherited. Rates{rates['active_collection_PPO_samples_s']/1000:.2f}k active/{rates['main_worker_effective_samples_s']/1000:.2f}k worker-effective/{rates['whole_chain_samples_s']/1000:.2f}k whole-chain excluding pause. Preserve initial1024 slow-speed regression,1152 recovery and checkpoint-bound newer tests; no monotonic claim. Failed inline quoting and startup progress read preserved. Next physical phase/objective timing,per-command retention,matched64/128 before any change. Same adaptive3h,target8192/reserve300/180healthy ramp,save/pause/brief computer-on endpoint; full-body videos fulfilled and unsaved Fusion retained.
'''
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
 if f'<!-- {marker} -->' not in p.read_text(encoding='utf-8-sig'):
  with p.open('a',encoding='utf-8') as f:f.write(entry)
nextpath=HERE/'automation/NEXT_CHECK.json';old=read(nextpath)
if not (OUT/'next_check_before.json').exists():write(OUT/'next_check_before.json',old)
new=dict(recorded_utc=now,automation_id=old['automation_id'],active_run=str(run),report=str(OUT/'REPORT.md'),latest_observation=str(OUT/'final_owned_observation.json'),
 new_main_updates=s['total_training_iterations'],inherited_main_updates_preserved=7361,
 selected_actor=dict(checkpoint=g['checkpoint'],sha256=g['checkpoint_sha256'],matching_gate_pair=str(OUT/'closing_matching_gates.json'),
  full_actor_critic_Adam_exploration=True,standing_and_walking_cases_through_1_m_s_pass=g['all_prior_pass'],running_qualified=d['running_qualified']),
 running_qualified=d['running_qualified'],velocity_ceiling_proven=False,focused_tests=d['next_tests'],latest_repair_decision=d,
 throughput={k:v for k,v in rates.items() if k not in ['blocks','resumed_blocks']},resumed_learning_verification=str(OUT/'resumed_learning_verification.json'),
 latest_verified_state=dict(recorded_utc=now,status=s['status'],stage=s['stage'],operation=s['operation_path'],actual_workers=live['actual_worker_pids'],
  frozen_hashes_verified=55,gpu=live['gpu'],host=live['system_memory'],operation_environments=s['environment_count'],learning_environments=s.get('next_training_environment_count')),
 endpoint=old['endpoint'],recording_request=old['recording_request'],latest_gate_trends=str(OUT/'gate_trends.json'),trajectory_analysis=str(OUT/'trajectory_analysis.json'))
write(nextpath,new)
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text(encoding='utf-8'))
assert cfg['id']==old['automation_id'] and cfg['kind']=='heartbeat' and cfg['rrule']=='FREQ=HOURLY;INTERVAL=3'
write(OUT/'automation_current_before_update.json',cfg);prefix=cfg['prompt'].split('Latest dated focus',1)[0].rstrip()
focus=f'''Latest dated focus, October 5, {now}: The same R37_offline_ground_full_state learner and supervisor PID {s['supervisor_pid']} have {s['total_training_iterations']} saved main updates plus 7,361 inherited updates. Read diagnostics/adaptive_check_20261005_1746/REPORT.md, evidence.json, trajectory_analysis.json, validation_selection.json, trajectory_search/result.json, trajectory_validation/result.json, flight_event.json, decision.json, closing_matching_gates.json, gate_trends.json and resumed_learning_verification.json. Read fresh ACTIVE_RUN/state before assuming this dated observation persists. No production source, reward, physics, CAD motor cap, optimizer or actor reset was deployed. All 55 hashes verify. The original local ground, truthful 400-input/23-output inference, teacher context, head module, PPO and iteration 4 execution remain.

Prior speed retention regressed at 1,024 and recovered at 1,152 updates; it is nonmonotonic. Latest exact pair on {Path(g['checkpoint']).parent.name}: all stopped/prior cases pass both tests: {g['all_prior_pass']}; running case passes: {g['running_pass']}. Running signed speeds are {g['nominal_cases'][-1]['signed_speed_m_s']:.3f}/{g['pushed_cases'][-1]['signed_speed_m_s']:.3f} m/s, genuine flight {100*g['nominal_cases'][-1]['geometric_flight_fraction']:.4f}%/{100*g['pushed_cases'][-1]['geometric_flight_fraction']:.4f}%, and head RMS {head[0]:.2f}/{head[1]:.2f} degrees. Natural arms remain uncertified. Saved gates bind that exact actor, never newer updates. No full running stage, 5 m/s or motor-ceiling endpoint is established.

The bounded physical search evaluates 128 smooth hip/knee/ankle timing cohorts, eight robots per cohort for 12 seconds, with four successive resets at seed 71. Independent seed 83 validates control, best return and best flight with 32 robots each for 20 seconds. All survive; 13 CPU checks pass, full reward term sums reconstruct exactly, and commands match. Best-flight validation gives 0.915% genuine flight, below 1.5%, with mean minimum sole clearance 1.48 mm and isolated 50 Hz samples. All flight is just after the intended double-swing window: reward half-cycle phases 0.005-0.014 after wrap, versus intended 0.43-0.50. Complete return loses 5.31 versus control. Tracking rate loses 0.2069, compared with flight gain 0.00205 and acceleration loss 0.00782. Best return gains 1.95 but flight is only 0.0083%. No trajectory meets flight, return and direction feasibility. Diagnostic offset transfer is REJECTED; no 64/128-update candidate was deployed or falsely reported as tested. Native limits, free base, gravity and truthful feedback remain; no root support, pose/phase/input rewrite, head override or hybrid enters production. These bounded failures do not prove running impossible or all natural running lower reward. The timing distinction is not proof of a wrong reward clock or an optimal fixed latency correction.

The same supervisor resumes after a {e['resume']['pause_seconds']:.1f}-second diagnostic pause after worker drain. A genuine 64-update main block completes with exact actor, critic, Adam and exploration loading and zero pre-update KL or command mismatch. Completed-main throughput is {rates['active_collection_PPO_samples_s']/1000:.2f}k active, {rates['main_worker_effective_samples_s']/1000:.2f}k worker-effective and {rates['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s, with this pause excluded from whole-chain time. Paused nominal 061's wrapper includes the pause. Current operation {Path(s['operation_path']).name} is {s['operation']} at {s['environment_count']} environments; evaluation count is not PPO capacity. One actual worker and 55 hashes verify. Target 8,192, 300 MiB reserve, host guards, fallback and the 180-second healthy-training ramp remain. The earlier 200 Hz motor probe found no driving-speed clips through 2.0 m/s; 2.5 m/s falls and brief clips do not prove a ceiling. This search adds no new substep ceiling certification. Full-body videos remain fulfilled and unsaved Fusion work must be preserved.

Next focused tests:
1. {d['next_tests'][0]}
2. {d['next_tests'][1]}
3. {d['next_tests'][2]}

Keep this same adaptive three-hour heartbeat active until verified actual 5 m/s or a causally established relevant actuator ceiling. At the endpoint, save, pause training and this automation, and brief the user with the computer on. Preserve all learned neutral-head, leg and walking state and independently passing actors. Do not spend the next check solely waiting or repeatedly optimizing a grounded local return.'''
args=dict(mode='update',id=cfg['id'],kind='heartbeat',name=cfg['name'],status='ACTIVE',rrule=cfg['rrule'],targetThreadId=cfg['target_thread_id'],prompt=prefix+'\n\n'+focus)
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
write(OUT/'automation_update_args.json',args)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),new_main=s['total_training_iterations'],prior_pass=g['all_prior_pass'],running_pass=g['running_pass'],automation_args=str(OUT/'automation_update_args.json')),indent=2))
