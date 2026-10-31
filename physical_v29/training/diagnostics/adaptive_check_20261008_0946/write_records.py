"""Chronological records, exact gate lineage and same-heartbeat arguments."""
import json,tomllib,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
live=read(OUT/'liveness_cpu/live_handoff.json');v=read(OUT/'liveness_learning_verification.json')
g=read(OUT/'closing_gate_review.json');pair=g['latest'];cycle=read(OUT/'individual_cycle_analysis.json')
ctx=read(OUT/'state_context_sensitivity.json');validation=read(OUT/'diagnostic_validation.json')
resume=read(OUT/'diagnostic_resume.json');failed=read(OUT/'attempt01_initial_import_failure/diagnostic_resume.json')
upper=read(OUT/'final_upper/upper_joint_time_audit.json')
u=next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
assert live['state']['status']=='running' and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
assert validation['passed'] and Path(pair['checkpoint'])==Path(cycle['checkpoint']) and u['checkpoint']==pair['checkpoint'] if 'checkpoint' in u else validation['passed']
first_resumed=next(b for b in v['blocks'] if Path(b['folder']).name=='608_run_350_train_200')
assert first_resumed['full_load_proof']['source_sha256']==pair['sha256'] and first_resumed['updates']==64
a,b=pair['nominal_cases'][-1],pair['pushed_cases'][-1]
next_tests=[
    'Preserve exact605,whose3.5 running case passes both tests,while completing current and all earlier retention on newer exact actors.605 still fails0.15 net travel,1.5 flight and2.5 slip. Compare those actual per-command advantages,teacher targets and landing/force events before a narrow retention continuation. Current-speed success alone does not promote the stage; preserve267 and separately certified349.',
    'Required upper timelines plus per-robot completed cycles remain primary. Same exact605 actor has3.5 phase-balanced shoulder outputs(-16.04,-28.52deg) on actual stable-command training states versus(-28.15,+9.90deg) on deterministic evaluation states. Measure truthful sequential training-cycle coverage,command transitions/exploration and hidden-feature/arm-history response; stage an individual-cycle or state-conditioned two-final-row derivative only after reset-safe CPU fixtures,physical response and exact-full-state64/128 controls with all gates/independent seeds. The earlier ensemble mean remains rejected. No observation substitution,hybrid,imposed human curve,cap/gain raise or whole-policy reset.',
    'Verify same39000/one worker55 hashes/local ground/full-state teachers and accounting:27507 inherited plus current completed and50 recoveries;32 unsaved R39 excluded. Exclude110.1s failed-import pause and197.7s successful capture when before the timing endpoint. Keep8192 target,reserve300,host fallback and180 healthy TRAINING seconds;do not force a ramp from evaluation memory. New motor audit correctly binds interleaved case_labels;605/3.5 right ankle motor2 is at/above cap on1.96% of10Hz samples. Measure current-actor5ms requested/applied torque and sustained driving clips if a plateau appears;these samples are not a ceiling. Only qualified5 or qualified4 plus causal native limits blocking5 ends the same5h heartbeat.'
]
decision=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),decision='PRESERVE_IMPROVING_ORIGINAL_FULL_LEARNER',
    explanation='605 improves direction enough to pass both current3.5 cases,while retention and upper motion remain incomplete. Passive truthful-state and actual PPO captures establish a state-distribution/arm-feedback lead. Four gradient batches have both signs of alignment,not universal opposition. No new learning candidate is justified or deployed by these captures alone. Existing upper/both priors continue20batches/update. The earlier ensemble-mean candidate stays rejected.',
    production_change=False,diagnostic_actors_deployed=False,new_candidate_learning_tested=False,next_tests=next_tests)
write(OUT/'decision.json',decision)
failures={label:[dict(command=c['command'],failures=c['failures']) for c in pair[label+'_cases'][:-1] if not c['passed']] for label in ['nominal','pushed']}
summary=(f"Same R40 full learner/supervisor39000 is running at {v['owned_snapshot_utc']},one actual worker55 hashes. "
 f"{v['completed_main_updates']} completed+{v['hash_proven_recovered_main_updates']} recovered={v['actual_retained_main_updates']} retained new main updates,+27507 inherited;{v['observed_partial_main_updates']} partial separate. "
 f"Reviewed exact605 nominal606/pushed607 passes BOTH3.5 running cases:signed{a['signed_speed']:.3f}/{b['signed_speed']:.3f},lateral{a['lateral_speed']:.3f}/{b['lateral_speed']:.3f}m/s,trueflight{a['flight']*100:.2f}/{b['flight']*100:.2f}%. "
 f"Whole stage remains unqualified;prior failures{failures}.Gates never qualify newer partial actors. "
 f"No production source/reward/gain/cap/optimizer/actor reset. Existing upper/both priors20batches continue. "
 f"Mandatory605 upper angle/target/velocity/bias plots inspected:headRMS{u['head_rms_deg']:.2f}deg;held shoulder bias/asymmetry remain,not natural/Unitree/optical certification. "
 f"256 actual representative completed cycles and4 passive4096-env PPO batches reveal same-actor training/evaluation shoulder-state differences. "
 f"The unchanged320-robot20s passive replay matches every case and51 physical arrays bitwise;all4 full-load PPO KL/command checks zero. "
 f"Same supervisor resumed after110.1s failed diagnostic import and197.7s successful capture;failed logs preserved. "
 f"First resumed608 completes64 real6144 updates with exact605 actor/critic/Adam/exploration,zeroKL/mismatch. "
 f"Rates{v['active_PPO_samples_s']/1000:.2f}kactive/{v['worker_effective_samples_s']/1000:.2f}kworker/{v['whole_chain_samples_s']/1000:.2f}kchain,distinct denominators. "
 "CPU motor audit corrected interleaved-case attribution;no runtime defect or ceiling proved. Same ACTIVE5h verified5/conditional4 endpoint;computer on.")
write(OUT/'evidence.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),summary=summary,owned=str(OUT/'liveness_cpu/live_handoff.json'),
    learning=str(OUT/'liveness_learning_verification.json'),reviewed_exact_pair=str(OUT/'closing_gate_review.json'),upper=str(OUT/'final_upper'),
    cycles=str(OUT/'individual_cycle_analysis.json'),state_response=str(OUT/'state_context_sensitivity.json'),
    actual_gradients=str(OUT/'current_gradient_analysis.json'),validation=str(OUT/'diagnostic_validation.json'),
    motor_audit=str(OUT/'motor_speed_audit.json'),decision=decision,production_change=False))
table=['|Command|Nominal signed/lateral m/s|Pushed signed/lateral m/s|Failures|','|---|---|---|---|']
for n,p in zip(pair['nominal_cases'],pair['pushed_cases']):
    table.append(f"|{n['command']:g}|{n['signed_speed']:.3f}/{n['lateral_speed']:.3f}|{p['signed_speed']:.3f}/{p['lateral_speed']:.3f}|{'; '.join(sorted(set(n['failures']+p['failures']))) or 'pass both'}|")
contexts=['|Command|Training effective pitch means(deg)|Evaluation effective pitch means(deg)|','|---|---|---|']
for r in ctx['rows']:contexts.append(f"|{r['command']:g}|{r['training_phase_balanced_effective_mean_deg'][0]:.2f}/{r['training_phase_balanced_effective_mean_deg'][1]:.2f}|{r['evaluation_phase_balanced_effective_mean_deg'][0]:.2f}/{r['evaluation_phase_balanced_effective_mean_deg'][1]:.2f}|")
report=f'''# Five-hour check — October8 09:46 UTC

{summary}

## Decision and qualification

{decision['explanation']} No learner reset,diagnostic actor,reward/physical/action/history change. Preserve605 SHA `{pair['sha256']}` and its matching606/607 gates,267's all-prior pair and original349's separate3m certification. No5m/s,conditional4m/s or native-limit endpoint established. Latest owned operation `{Path(v['operation_path']).name}` is separately recorded; evaluation robot count is not PPO capacity.

The reviewed pair has320 robots,20 seconds and32 trials percommand,100% survival. Nominal clones do not establish final independently seeded initial-state diversity.

{chr(10).join(table)}

Relative to earlier524,direction improves from0.184/0.163m/s lateral to0.117/0.120.3m direction recovers;0.15 path efficiency and1.5 flight remain failed,and2.5 slip regresses after602 briefly passed it. Preserve these nonmonotonic results. Current-speed passes are useful,but all earlier gates must also pass before promotion.

## Upper joint motion and truthful state coverage

Actual head yaw/pitch,shoulder pitch/roll,elbow angle/target/velocity,bilateral coordination,ipsilateral hip phase,smoothness,saturation and descriptive0.5Hz slow/faster decomposition were plotted and visually inspected for602 and605.32 robots/cmd at10Hz are distinguished from16 total50Hz representatives,one or two percommand. Resets are segmented;10Hz cannot certify fast jitter.605 at3.5 has headRMS{u['head_rms_deg']:.3f}deg,shoulder pitch ranges{u['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{u['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}deg and arm/ipsilateral-hip phases169.7/-174.0deg at2.67Hz. Phase alone does not certify natural motion.602's slow2.5/3m reversals become more nearly held offsets in605; physical targets carry those offsets. Roll asymmetry,elbow coordination and gait-linked head oscillation persist.

A passive320-robot20s replay captures truthful400-input five-frame histories at10Hz for all robots and50Hz for16 representatives. It returns the same policy actions. All cases and51 physical arrays match original606 bitwise. CPU mean replay across the full input capture differs by at most{cycle['CPU_action_replay_max_error']:.3g} action units from the actual GPU outputs. No modified observations were used in physical inference.

The [cycle plot](individual_cycle_means.png) was inspected.256 complete8-bin first-episode cycles between5 and17s quantify individual held effective pitch offsets,not an average that can cancel robots or slow reversals. At3.5,the representative policy cycle-mean RMS is28.14/9.99deg and actual joint cycle-mean RMS21.81/11.11deg. These are one robot's31 cycles,not32 independent robots. The5deg band is diagnostic only and not a new gate. Three counterexamples reproduce quiet/zero-mean swing and cancellation; an undeployed detached-feature two-row temporal VJP is quantified. No production temporal loss or64/128 candidate learning was tested this check.

The first actual4096-env PPO minibatch mean-network tensors are bitwise equal to605. Restricting both datasets to exact command in all five frames and balancing8 phase bins still gives:

{chr(10).join(contexts)}

CPU-only phase-matched history substitutions measure policy sensitivity while holding other inputs fixed. At3.5,borrowing actual training arm q/qd/previous-action histories changes shoulder outputs1.708RMS action units and lower outputs0.439RMS; head-history substitution changes shoulders only0.038RMS. These artificial fixed-state combinations are not physically self-consistent trajectories,proof mechanical effects absent,or deployable observations. The large lower response rejects any immediate history-spoof/hybrid approach. See `state_context_sensitivity.json` for each speed and feature scope. The discrepancy is a reproduced state-distribution lead,not an observation bug or causal motor explanation.

Four passive actual PPO minibatches have shoulder final-row PPO norms4.29–15.74 versus existing upper auxiliary2.50–2.92. Alignment against the proposed evaluation-cycle gradient ranges-0.915 to+0.924; no universal opposition is established. Per-command actual advantages/ratios,teacher coverage,effective targets and pre-auxiliary parameter VJPs remain saved. The existing256-row auxiliary has only1–8 exact2.5 samples in these particular batches. This is measured coverage,not proof a bigger weight/sample count repairs behavior. CPU stopped-hook subtraction is validated against directly captured pre-standing PPO derivatives; no production hook fault inferred.

The [primary human arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) uses wrist/lower-back IMUs and supports comparing amplitude,asymmetry and smoothness byspeed; it supplies no morphology-matched shoulder encoder targets. Our smaller fast pitch swing and held biases do not establish equivalent human motion. The [official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab) was revisited; no mapped,speed-matched G1/H1 encoder curve was obtained. No reference curve is fabricated. Native head is torso-relative; known Isaac/FK parity does not calibrate the original CAD face/camera optical axis or certify world gaze.

## Synchronized physics and motor accounting

`closing_com_contact_audit.json` verifies605's50Hz representative COM,joints,physical targets,actions,forces,contact and command synchronization: finite,no resets,zero observed/action command difference. Existing full-population support/contact gates stay unchanged; moving COM offsets and simplified collision envelopes are not pressure certification.

The older0741 CPU motor helper assumed contiguous cohorts,but actual evaluation saves interleaved `case_labels`. The new audit stably sorts q/qd by those saved labels before unchanged exact CAD Jacobian/reduction mapping; a synthetic fixture and directional finite-difference checks pass. This is a diagnostic accounting correction only. Preserve the older artifact and treat its per-command motor attribution as unreliable. Aggregate observed samples and native caps were not changed; no historical ceiling was established from it.

New605/3.5 right ankle motor2 p99 is1.301 times its400RPM cap,with1.962% of10Hz samples at/abovecap; leftmotor2 p99 is0.853. These are measured pose-derived motor speeds,not native5ms driving-clipping/torque attribution or a sustained high-speed ceiling. Requested/applied torque and phase-specific substep evidence are still required. Native caps stay unchanged; no4m/s fallback success inferred.

## Failure preservation,safe continuation and resources

The initial passive wrapper imported Isaac math before AppLauncher and failed with missingpxr. Its finally cleanup resumes the original supervisor after{failed['pause_seconds']:.1f}s including{failed['worker_drain_seconds']:.1f}s drain. Logs and receipts are preserved under `attempt01_initial_import_failure`. Only diagnostic import ordering changed; the retry finishes the passive replay plus4 actual4096-env PPO updates. All initial actor/critic/Adam/exploration comparisons,KL0 and command0 checks pass; smoke actor discarded. The same original supervisor resumes after{resume['pause_seconds']:.1f}s including{resume['worker_drain_seconds']:.1f}s drain. The small CPU analysis syntax failure is corrected separately,without simulator impact. This check's complete operational harness succeeds; waiting/drain has a separate budget from diagnostic execution.

First resumed original608 finishes64 genuine6144 updates,exact605 full checkpoint load,zero KL/command mismatch and both teachers/upper20 batches/update. Its minimum freeGPU{first_resumed['minimum_free_gpu_MiB']}MiB,host{first_resumed['minimum_host_available_MiB']}MiB,private growth{first_resumed['private_memory_growth_MiB']}MiB and Torch growth{first_resumed['torch_allocation_growth_MiB']}MiB. Current owned status/frozen55/one actual worker is verified. {v['completed_main_updates']} completed+50 recovered={v['actual_retained_main_updates']} retained current-run updates,+27507 inherited;32 unsaved R39 remain excluded. Partial{v['observed_partial_main_updates']} is separate.

Retained samples/time:{v['active_PPO_samples_s']:.0f} active collection+PPO,{v['worker_effective_samples_s']:.0f} worker-effective,{v['whole_chain_samples_s']:.0f} whole-chain samples/s. Whole-chain excludes{v['documented_diagnostic_pause_seconds_excluded']:.1f}s documented suspensions through its endpoint. Capacity/smoke/diagnostic samples are excluded; denominators differ. Maintain8192 target,reserve300MiB,host fallback and180 healthy TRAINING seconds before ramps.6144 steady~1.9GiB free cannot force the added tier; evaluation memory is not capacity. Fusion stays closed and queued work/full videos preserved;computer remains on.

## Next focused tests

{chr(10).join(f'{i+1}. {s}' for i,s in enumerate(next_tests))}

Same active5h thread/notification preference/authority. Finish only after5m/s actual≥4.95 with all matching nominal/pushed/stopped/prior gates and32 independently seeded states/cmd,or4m/s≥3.96 plus all gates and controlled native limits proving5 blocked. At verified endpoint save,pause training and automation,brief,and leave computer on.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
now=datetime.now(timezone.utc).isoformat();entry=f'\n## Five-hour check October8 09:46 UTC — {now}\n\n{summary}\n\n{decision["explanation"]}\n\nReport: {OUT/"REPORT.md"}. Passive physical/load/gradient evidence,import failure/cleanup,CPU state-response/cycle data and corrected motor-case attribution preserved. No production change. Next: '+ ' '.join(next_tests)+'\n'
for p in [TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with p.open('a',encoding='utf-8') as f:f.write(entry)
marker='adaptive_check_20261008_0946_record'
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    text=p.read_text(encoding='utf-8');assert marker not in text
    p.write_text(f'<!-- {marker} -->\n{summary} Read {OUT/"REPORT.md"},then freshACTIVE_RUN/state. Earlier entries preserved.\n<!-- /{marker} -->\n\n'+text,encoding='utf-8')
old=read(OUT/'previous_NEXT_CHECK.json');old.update(schema='adaptive_humanoid_check_v2',recorded_utc=now,report=str(OUT/'REPORT.md'),
    latest_observation=v['owned_snapshot_utc'],latest_liveness_utc=v['owned_snapshot_utc'],active_run=v['run'],current_stage=live['state']['stage'],
    completed_main_updates=v['completed_main_updates'],new_main_updates=v['completed_main_updates'],recovered_main_updates=50,
    actual_retained_current_run_main_updates=v['actual_retained_main_updates'],partial_observed_main_updates=v['observed_partial_main_updates'],
    current_owned_operation=v['operation_path'],selected_actor=pair['checkpoint'],selected_actor_sha256=pair['sha256'],latest_matching_pair=pair,
    current_speed_case_passes_both=True,current_stage_qualified=False,running_qualified=False,velocity_ceiling_proven=False,
    summary=summary,focused_tests=next_tests,next_tests=next_tests,focus=next_tests,decision=str(OUT/'decision.json'),
    latest_liveness=str(OUT/'liveness_cpu/live_handoff.json'),resumed_learning_evidence=str(OUT/'liveness_learning_verification.json'),
    upper_joint_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'),mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),
    mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'),upper_target_bias_audit=str(OUT/'final_upper/upper_target_bias_audit.json'),
    individual_upper_cycles=str(OUT/'individual_cycle_analysis.json'),upper_state_context_sensitivity=str(OUT/'state_context_sensitivity.json'),
    synchronized_com_contact_audit=str(OUT/'closing_com_contact_audit.json'),motor_speed_audit=str(OUT/'motor_speed_audit.json'),
    diagnostic_resume=str(OUT/'diagnostic_resume.json'),previous_full_handoff_snapshot=str(OUT/'previous_NEXT_CHECK.json'),
    retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
    production_change=False,computer_on=True,history_inventory=str(OUT/'inventory.log'))
write(TRAIN/'automation/NEXT_CHECK.json',old)
before=tomllib.loads((OUT/'previous_automation.toml').read_text(encoding='utf-8'));prefix=before['prompt'].split('Latest dated five-hour check',1)[0]
prompt=prefix+'Latest dated five-hour check October8 09:46 UTC; '+summary+'\nRead diagnostics/adaptive_check_20261008_0946/REPORT.md,decision.json,individual_cycle_analysis.json,state_context_sensitivity.json,diagnostic_validation.json,motor_speed_audit.json,final_upper and liveness_learning_verification.json,then freshACTIVE_RUN/state.\n\nNext focused tests:\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests))+'\n\nPreserve the improving exact full learner and all rejected/failed evidence. Same ACTIVE5h heartbeat/thread/notifications/durable verified5 or qualified4+causal native limits endpoint;save,pause both,brief,computer on. Fusion closed,queued work preserved;videos fulfilled.'
args=dict(id='humanoid-training-research-and-repair-every-four-hours',mode='update',kind='heartbeat',name=before['name'],prompt=prompt,rrule=before['rrule'],status='ACTIVE',targetThreadId=before['target_thread_id'])
if 'notification_policy' in before:args['notificationPolicy']=before['notification_policy']
write(OUT/'automation_update_args.json',args)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),summary=summary),indent=2))
