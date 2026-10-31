"""Preserve this check, append chronological records, prepare native automation update."""
import hashlib,json,tomllib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
marker='adaptive_check_20261009_0547_record';now=datetime.now(timezone.utc).isoformat()
v=read(OUT/'final_learning_verification.json');live=read(OUT/'final_cpu/live_handoff.json')
pair=read(OUT/'closing_gate_review.json')['latest'];final_pair=read(OUT/'final_gate_review.json')['latest']
decision=read(OUT/'decision.json');validation=read(OUT/'physical_validation.json')
physical=read(OUT/'teacher_compatibility/result.json');factor=read(OUT/'head_arm_factorial/teacher_compatibility/result.json')
analyses=[read(OUT/'teacher_compatibility_analysis.json'),read(OUT/'head_arm_factorial/teacher_compatibility_analysis.json')]
gradient=read(OUT/'current_gradient_analysis.json')['batches']
upper=read(OUT/'final_upper/upper_joint_time_audit.json');bias=read(OUT/'final_upper/upper_bias_audit.json')
u=next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
ub=next(r for r in bias['records'] if r['test']=='nominal' and r['speed_m_s']==3.5)
resumes=[read(OUT/p) for p in ['attempt01_signature_metadata/diagnostic_resume.json','diagnostic_resume.json','head_arm_factorial/diagnostic_resume.json']]
failed=[read(OUT/p) for p in ['attempt01_signature_metadata/RECORD.json','attempt02_preboundary_lookup/RECORD.json']]
receivers=[next(b for b in v['blocks'] if b['full_load_proof']['source_sha256']==r['checkpoint_sha256']) for r in resumes]
before=tomllib.loads((OUT/'previous_automation.toml').read_text(encoding='utf-8'))
native=Path('C:/Users/rshan/.codex/automations')/before['id']/'automation.toml'
assert tomllib.loads(native.read_text(encoding='utf-8'))['prompt']==before['prompt']
assert validation['passed'] and not decision['deploy'] and u['sha256']==pair['sha256']
assert read(OUT/'final_upper/audit_set.json')['all_plots_visually_inspected']
old=read(OUT.parent/'adaptive_check_20261009_0046/final_learning_verification.json')
totals=lambda d:{k:sum(b[k] for b in d['blocks']) for k in ['updates','samples','active_seconds','worker_seconds']}
total=totals(v);previous=totals(old);delta={k:total[k]-previous[k] for k in total}
delta['chain_seconds_excluding_documented_pauses']=v['whole_chain_measurement_endpoint_unix']-old['whole_chain_measurement_endpoint_unix']-v['documented_diagnostic_pause_seconds_excluded']+old['documented_diagnostic_pause_seconds_excluded']
for name,den in [('active_samples_s','active_seconds'),('worker_effective_samples_s','worker_seconds'),('chain_samples_s','chain_seconds_excluding_documented_pauses')]:
 assert delta[den]>0;delta[name]=delta['samples']/delta[den]
last=v['blocks'][-1];a,b=pair['nominal_cases'][-1],pair['pushed_cases'][-1]
actor=Path(pair['checkpoint']).parent.name
prior_failures={k:[dict(command=c['command'],failures=c['failures']) for c in pair[k+'_cases'][:-1] if not c['passed']] for k in ['nominal','pushed']}
summary=(f"The same R40 full learner and supervisor 39000 are running at {v['owned_snapshot_utc']}, with one actual worker, 55 verified frozen hashes and 53 protected authored worker files. "
 f"It retains {v['completed_main_updates']:,} completed + 50 hash-proven recovered = {v['actual_retained_main_updates']:,} current-run main updates, plus 27,507 inherited; {v['observed_partial_main_updates']} partial updates are separate. "
 f"Reviewed exact actor {actor} reaches {a['signed_speed']:.3f}/{b['signed_speed']:.3f} m/s signed at command 3.5, with {a['lateral_speed']:.3f}/{b['lateral_speed']:.3f} m/s lateral drift; both direction gates fail. Prior failures: {prior_failures}. "
 "Actors 908/917 separately passed both current 3.5 cases but not every prior; the stage remains unqualified. Saved gates never qualify newer partial actors. "
 +decision['diagnostic_summary']+' '
 +f"Actual upper angle, target, velocity and bias timelines were inspected at 908 and 920, both tests and all commands. Actor 920 head RMS is {u['head_rms_deg']:.2f} degrees; shoulder pitch ranges are 11.74/10.25 degrees. Held biases, unequal roll/elbows and slow reversals remain. Human/Unitree encoder likeness and optical gaze remain uncertified. "
 f"The same supervisor resumes after 126.955-second failed-metadata, 476.254-second main compatibility and 371.378-second component-test pauses; the preboundary lookup attempt never paused training. "
 "Resumed blocks 920/923 each finish 64 genuine updates at 8192, loading exact 917/920 actor, critic, Adam and exploration, with zero KL/command mismatch and all three auxiliaries at 20 batches/update. "
 f"Last actual main minimum free GPU/host memory is {last['minimum_free_gpu_MiB']}/{last['minimum_host_available_MiB']} MiB, private growth {last['private_memory_growth_MiB']} MiB and Torch allocation growth {last['torch_allocation_growth_MiB']} MiB. "
 f"Retained rates are {v['active_PPO_samples_s']/1000:.2f}k active / {v['worker_effective_samples_s']/1000:.2f}k worker / {v['whole_chain_samples_s']/1000:.2f}k chain samples/s; documented suspensions are excluded through each endpoint. "
 "No production source, reward, gain, cap, actor or optimizer reset; no diagnostic controller or learned actor deployed. Existing upper and both teacher auxiliaries at 20 batches/update, 35/1 gains, truthful 400/23 inference, native CAD caps and iteration 4 remain. Same active five-hour endpoint; computer on.")
gate_table=['|Command m/s|Nominal signed|Pushed signed|Geometric flight nominal/pushed %|Nominal/pushed failures|','|---:|---:|---:|---:|---|']
for x,y in zip(pair['nominal_cases'],pair['pushed_cases']):
 gate_table.append(f"|{x['command']:g}|{x['signed_speed']:.4f}|{y['signed_speed']:.4f}|{100*x['flight']:.3f}/{100*y['flight']:.3f}|{', '.join(x['failures']) or 'pass'} / {', '.join(y['failures']) or 'pass'}|")
def physical_table(d):
 rows=['|Seed|Command|Controller composition|Signed m/s|Abs lateral m/s|True flight %|Complete reward rate|','|---:|---:|---|---:|---:|---:|---:|']
 for g in d['groups']:
  if g['transition'] or g['command']==3.5:continue
  rows.append(f"|{g['seed']}|{g['command']:g}|{g['mode']}|{g['signed_forward_m_s']:.4f}|{g['absolute_lateral_m_s']:.4f}|{100*g['genuine_flight_fraction']:.4f}|{g['mean_reward_rate']:.4f}|")
 return '\n'.join(rows)
report=f'''# Five-hour training check — October 9, 2026, 05:47 UTC

The same full R40 learner continues. It reaches about 3.48 m/s at the 3.5 command, but the reviewed checkpoint still fails direction and several retained skills. Arm fine-tuning remains active. The new controlled experiments identify a closed-loop arm/leg coordination tradeoff; they do not supply a safe production repair or establish an actuator ceiling.

## Exact gates and decision

{decision['explanation']}

Fully reviewed actor: `{pair['checkpoint']}`, SHA256 `{pair['sha256']}`; matching nominal921/pushed922 have320 robots,20 seconds and32 trials per command. All survive. Existing classifiers are unchanged. Nominal clones do not establish the independently generated initial-state diversity required at the final endpoint.

{chr(10).join(gate_table)}

Initial908 and later917 passed both3.5 current-speed cases;920 regresses direction. Earlier-speed retention remains nonmonotonic. Preserve those useful exact pairs,605 current-case evidence,267's earlier all-prior pair and separately certified349. Current-speed success alone cannot promote the whole stage. The owned snapshot has completed newer923 but its matching pair is incomplete at that observation; no923 gate or upper-motion qualification is inferred.

[Exact gate trends](gate_trends.json) compare current behavior with605/794/797/833/839/908/914/917/920. [Synchronized COM/contact/action audit](closing_com_contact_audit.json) verifies finite50Hz representative COM, foot forces, joints, physical targets and actions, zero command mismatch and no resets on reviewed920. Detailed mass-weighted COM coordinates are representative, while saved population metrics retain the existing full-population support classifier.

## Qualified teacher body and current upper controller

Physical actor remains exact917, SHA256 `{physical['checkpoint_sha256']}`, across both experiments. Qualified349 is exact SHA256 `{physical['teacher_sha256']}` and freshly certified under the current runtime through3 m/s. The newer920 full learner is preserved for resume; pinning917 for controlled diagnosis is not a production rollback.

The first experiment has1152 robots over20 seconds,36 cohorts, seeds4401/4411 and32 individually seeded small q/qd states per cohort, paired between controller branches. Its72-robot2-second smoke and13 CPU ownership/selector/projection checks pass first. Each controller reads its own actual400-input feedback/history. Diagnostic compositions split the23 outputs into body and upper rows; production inference never receives a composite. Context-conditioned variants modify only the explicitly identified diagnostic teacher input, while the current controller receives truthful observations. Teacher is never invoked above its3 m/s certificate; the3.5 sentinel uses current only.

All robots survive, commands match, effective-target reconstruction is exact and complete reward terms reconstruct within9.54e-7. Free base, gravity, self-collision,35/1 gains, geometry and native motor caps remain. There is no runtime root support, pose override, donor body-action replay or production observation substitution. Truthful3→0.15/1.5 transitions at6 seconds reproduce the same unfavorable tradeoffs; histories are not reset or fabricated at switches.

{physical_table(physical)}

Qualified349 still produces about1.81% genuine1.5 flight. Teacher body with current whole upper drops to about0.94% and loses complete return; at0.15 it overspeeds to about0.285m/s. Existing head-context conditioning does not make that whole-upper transfer safe. The inverse current body/teacher whole upper produces only about0.92% flight. These observations reject direct deployment of these tested compositions, not all future transfer approaches.

![Actual compatibility under own-state feedback](teacher_compatibility.png)

[Population and paired-state analysis](teacher_compatibility_analysis.json) saves per-robot differences,5th/95th percentiles, both seeds and transition windows. [Teacher effective target/history analysis](teacher_target_history_analysis.json) projects through the unchanged action clamp, joint bounds, slew and CAD map, with16 pre-action phase bins and next20ms sole response. Those instantaneous differences are not proof of an optimal delay, a safe gradient or a production numerical fault.

## Head versus arms: controlled component split

The second experiment has704 robots,22 cohorts,20 seconds, seeds4421/4431 and32 individually seeded paired states per cohort. A44-robot2-second smoke and9 CPU checks pass. Physical actor917 remains pinned, but the original supervisor resumes newer920 exactly afterward. All704 survive with exact target/command/reward checks.

`teacher_current_head` keeps teacher body and arms, selecting only current head yaw/pitch. `teacher_current_arms` keeps teacher body and head, selecting only the six current shoulder/elbow rows. `teacher_current_upper` selects both. Every branch remains in its own closed loop with actual feedback.

{physical_table(factor)}

Current arms on teacher body/head explain most0.15 overspeed and a larger1.5 flight loss. Current head alone preserves slow speed but gives1.478/1.502% flight, so it does not pass both seeds even as a diagnostic flight test. Arms-only current gives1.065/1.086%, below1.5%. Head effects are not absent. These controlled component changes include both physical dynamics and the lower controller's response to changed arm/head feedback; they do not isolate a purely mechanical cause. Student body with teacher arms-only while retaining current head has not yet been tested. No diagnostic composition enters production.

![Separated head and arm controller effects](head_arm_factorial/teacher_compatibility.png)

[Split population analysis](head_arm_factorial/teacher_compatibility_analysis.json) and [split effective targets](head_arm_factorial/teacher_target_history_analysis.json) preserve both seeds, actual angles, targets, velocities, root/contact/COM data and reset identities. This check claims no new64/128 candidate learning comparison.

## Motor limits and actual PPO derivatives

[917 native motor attribution](current_substep_motor_attribution.json) records4000 actual5ms calls at200Hz for the explicitly captured current, teacher and whole-upper compatibility cohorts. CAD Jacobians/reductions, requested/applied motor torque and directional available torque are preserved. All recorded driving-clip fractions and maximum durations are zero at0.15/1.5/3.5. Ankles can exceed the speed cap while braking; this is not sustained driving saturation. Available torque follows the frozen user-assumed flat-current/hard-speed BLDC and linear servo envelopes, not measured hardware voltage/current/thermal curves. The head-only/arms-only modes do not inherit uncaptured full-population motor attribution.

Reviewed920 sparse10Hz [nominal](closing_motor_nominal.json) and [pushed](closing_motor_pushed.json) audits use the actual interleaved case labels. At3.5 right ankle motor2 is at/above cap in2.22/2.12% of samples, with p99 about1.39/1.40 times cap. These different-actor sparse samples cannot inherit917's native driving/braking attribution. No persistent relevant ceiling or4/5 endpoint is established; no cap is raised.

Four genuine4096-environment diagnostic PPO updates load exact917 actor, critic, Adam and trained exploration, with zero pre-update KL/command mismatch and all three auxiliaries20 batches/update. The diagnostic learned actor is excluded from main progress and never deployed. [Actual gradient analysis](current_gradient_analysis.json) captures incoming derivatives before standing/moving hooks: prior norms{min(x['prior_parameter_norm'] for x in gradient):.3f}–{max(x['prior_parameter_norm'] for x in gradient):.3f}, full PPO{min(x['all_PPO_parameter_norm'] for x in gradient):.3f}–{max(x['all_PPO_parameter_norm'] for x in gradient):.3f}, upper{min(x['upper_auxiliary_parameter_norm'] for x in gradient):.3f}–{max(x['upper_auxiliary_parameter_norm'] for x in gradient):.3f}. Running/prior and upper/PPO cosines have both signs. Standing subtraction reproduces within5.71e-10 and moving mean-score cosine is approximately1. These are pre-global-clipping/Adam contributions, not universal opposition or causal repair evidence.

## Required upper angles, targets and velocities over time

All14 initial908 and14 reviewed920 plots were visually inspected, with hashes and notes. The [full audit](final_upper/upper_joint_time_audit.json) and CSV data include every command and both tests: head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets, velocities, bilateral coordination, ipsilateral arm/leg phase, cadence, smoothness and raw-action clamp diagnostics. Population sampling is32 robots per command at10Hz; high-rate traces are16 representatives per gate at50Hz. Each command's high-rate trace may contain fewer representatives;3.5 uses one. Resets/terminal rows are segmented;10Hz population statistics cannot certify fast jitter.

Reviewed920 nominal3.5 head RMS{u['head_rms_deg']:.3f}deg,p95{u['head_p95_deg']:.3f}deg. Shoulder pitch median per-robot p95-p05 ranges{u['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{u['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}deg; slow means{ub['joints']['left_shoulder_pitch']['slow_bias_mean_deg']:.2f}/{ub['joints']['right_shoulder_pitch']['slow_bias_mean_deg']:.2f}deg. Hip cycle{u['coordination']['hip_cycle_hz']:.3f}Hz, ipsilateral shoulder phases{u['coordination']['arms'][0]['phase_at_hip_cycle_deg']:.1f}/{u['coordination']['arms'][1]['phase_at_hip_cycle_deg']:.1f}deg. Representative upper physical target/joint RMS{u['representative_50hz']['target_joint_rms_error_deg']:.2f}deg; raw upper outputs outside clamp{100*u['representative_50hz']['raw_upper_action_outside_clamp_fraction']:.3f}%. Neither the5deg reporting band nor clamp statistics replace locomotion gates.

![Actual angles and physical targets](final_upper/upper_joint_angles.png)

![Actual velocities](final_upper/upper_joint_velocities.png)

![Slow bias versus faster residual](final_upper/upper_bias_decomposition.png)

![Commanded and measured slow bias](final_upper/upper_target_bias.png)

Opposite held fast shoulder biases, unequal roll/elbow motion and gait-locked neck oscillation remain. At2.5/3m/s physical target biases reverse slowly; total range is not periodic arm swing. The0.5Hz split excludes3-second edges, uses post-warmup time2.02s and is descriptive. Existing upper-only row refinement remains active20 batches/update; it has not certified natural arm motion.

Primary [Collins/Adamczyk/Kuo2009](https://pubmed.ncbi.nlm.nih.gov/19640879/) Figure6 contains published human shoulder/elbow angle and torque time plots. Its reported whole-body mechanical effects motivate component experiments; they do not prove this robot's cause. Exact speed, axes, rest poses and morphology mapping, and usable numerical reference curves remain unverified. [The human IMU study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) measures wrist/back motion, not mapped shoulder encoders. [Official Unitree code](https://github.com/unitreerobotics/unitree_rl_mjlab) has not supplied verified speed-matched encoder curves. Retrieval failures are preserved in [reference review](primary_motion_reference_review.json); no curve is fabricated. Comparison remains feature-based, and no human/Unitree likeness or calibrated optical gaze is claimed. Native head zero is torso-relative; historical Isaac/FK parity does not calibrate the CAD face/camera axis.

## Failed external attempts, safe resume and measured progress

The first external smoke stopped before rollout because it incorrectly required the original349 embedded reward signature to equal the newer corrected travel-guard runtime signature. The correction verifies exact349 SHA, matching asset/observation contract and complete fresh current-runtime certificate/gates, while keeping the current student's strict embedded signature. There is no generic signature bypass. Original executed source and failure logs remain in attempt01_signature_metadata; same supervisor resumed after126.955s. A stale smoke-result path was caught and canceled before the next saved-boundary pause; attempt02_preboundary_lookup never stopped the learner. A later CPU plot-receipt helper first used the wrong image location, then corrected the receipt path; the inspected plots were unchanged. These are diagnostic bookkeeping failures, not inferred production faults.

The compatibility pause lasts476.254s including108.801s worker drain. The component pause lasts371.378s including106.870s drain. One actual CUDA worker runs at a time. Resumed920 loads917 exactly and resumed923 loads920 exactly; both finish64 real main updates at8192. Both teachers and upper auxiliaries remain20 batches/update; every pre-update KL and observed/action command mismatch is zero. The original supervisor39000 is running; no silent pause or experimental controller remains. [Physical and recovery validation](physical_validation.json) passes{validation['count']} cross-evidence checks, plus153/102 detailed physical-analysis checks,13/9 CPU fixtures and the physical/PPO smokes.

Since the previous check's recorded endpoint: {delta['updates']:,} retained main updates and{delta['samples']:,} samples over{delta['active_seconds']:.1f}s active,{delta['worker_seconds']:.1f}s worker and{delta['chain_seconds_excluding_documented_pauses']:.1f}s chain excluding documented suspensions. Interval rates{delta['active_samples_s']/1000:.2f}k/{delta['worker_effective_samples_s']/1000:.2f}k/{delta['chain_samples_s']/1000:.2f}k samples/s. Current-run totals{total['samples']:,} samples,{v['actual_retained_main_updates']:,} retained updates,plus27507 inherited. Full lineage has{v['actual_retained_main_updates']+27507:,} actual main updates. Capacity/smoke/diagnostic learning and32 unsaved interruptedR39 updates are excluded; partial progress is separate.

Current-run rates{v['active_PPO_samples_s']/1000:.2f}k active/{v['worker_effective_samples_s']/1000:.2f}k worker-effective/{v['whole_chain_samples_s']/1000:.2f}k chain; denominators differ. {v['documented_diagnostic_pause_seconds_excluded']:.1f}s documented pauses are excluded only through each timing endpoint. An older untimestamped short bookkeeping pause remains conservatively included. Last genuine main923 at8192 keeps at least{last['minimum_free_gpu_MiB']}MiB GPU and{last['minimum_host_available_MiB']}MiB host free,private+{last['private_memory_growth_MiB']}MiB,Torch allocation+{last['torch_allocation_growth_MiB']}MiB. All55 frozen hashes and53 protected authored worker files verify. Target8192,reserve300,host guards/fallback and180 continuous healthy TRAINING seconds before ramps remain. Evaluation memory is not training capacity. Fusion stays closed with queued work preserved; videos are fulfilled.

## Next focused tests and endpoint

{chr(10).join(f'{i+1}. {s}' for i,s in enumerate(decision['next_tests']))}

Same ACTIVE five-hour heartbeat/thread/notification preference. Only actual≥4.95m/s with every unchanged nominal/pushed/stopped/prior gate and32 independently generated initial-state trials per command, or actual≥3.96m/s under those same requirements plus causal native limits proving5 blocked, ends training. At that endpoint save exact full state/hashes/gates/telemetry, pause owned training/prevent restart, pause this automation and brief with the computer on.
'''
evidence=dict(recorded_utc=now,summary=summary,decision=decision,fully_reviewed_pair=pair,final_snapshot_pair=final_pair,
 owned_snapshot=str(OUT/'final_cpu/live_handoff.json'),current_saved_checkpoint=live['state']['checkpoint'],
 physical_actor=physical['checkpoint'],physical_sha256=physical['checkpoint_sha256'],diagnostic_resumes=resumes,
 first_resumed_main_blocks=receivers,failed_attempts=failed,physical_validation=validation,
 mandatory_upper_actor=pair['checkpoint'],mandatory_upper_folder=str(OUT/'final_upper'),
 retained_main_totals=total,measured_since_previous=delta,production_change=False,computer_on=True)
histories=[TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]
refs=[Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
assert not (OUT/'REPORT.md').exists() and all(marker not in p.read_text(encoding='utf-8') for p in histories+refs)
write(OUT/'evidence.json',evidence);write(OUT/'failed_attempts.json',failed);(OUT/'REPORT.md').write_text(report,encoding='utf-8')
write(OUT/'resumed_learning_verification.json',dict(resumes=resumes,first_retained_receivers=receivers,all_later_main_loads_exact=True,diagnostic_samples_excluded=True))
entry=f'\n<!-- {marker} -->\n## Five-hour check — October 9,05:47 UTC — {now}\n\n{summary}\n\nReport: {OUT/"REPORT.md"}.\n\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(decision['next_tests']))+f'\n<!-- /{marker} -->\n'
for p in histories:
 with p.open('a',encoding='utf-8') as f:f.write(entry)
for p in refs:
 p.write_text(f'<!-- {marker} -->\n{summary}\nRead {OUT/"REPORT.md"}, then fresh ACTIVE_RUN/state. Earlier entries preserved.\n<!-- /{marker} -->\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
h=read(TRAIN/'automation/NEXT_CHECK.json')
historical={k:h.pop(k) for k in ['arm_coupling_analysis','teacher_effective_target_analysis','replay_sham_resume','replay_sham_analysis','failed_diagnostic_attempts','historical_diagnostic_paths','final_owned_observation','final_verification','automation_verification','inventory_verification'] if k in h}
h.update(recorded_utc=now,report=str(OUT/'REPORT.md'),latest_observation=v['owned_snapshot_utc'],latest_liveness_utc=v['owned_snapshot_utc'],active_run=v['run'],current_stage=v['stage'],
 completed_main_updates=v['completed_main_updates'],new_main_updates=v['completed_main_updates'],recovered_main_updates=50,
 actual_retained_current_run_main_updates=v['actual_retained_main_updates'],partial_observed_main_updates=v['observed_partial_main_updates'],
 current_owned_operation=v['operation_path'],current_checkpoint=live['state']['checkpoint'],current_checkpoint_sha256=hashlib.sha256(Path(live['state']['checkpoint']).read_bytes()).hexdigest(),
 selected_actor=pair['checkpoint'],selected_actor_sha256=pair['sha256'],latest_matching_pair=pair,
 reviewed_speed_case_passes_both=a['passed'] and b['passed'],current_stage_qualified=False,running_qualified=False,velocity_ceiling_proven=False,
 summary=summary,focused_tests=decision['next_tests'],next_tests=decision['next_tests'],focus=decision['next_tests'],decision=str(OUT/'decision.json'),
 latest_liveness=str(OUT/'final_cpu/live_handoff.json'),resumed_learning_evidence=str(OUT/'final_learning_verification.json'),
 mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),upper_joint_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'),
 mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'),upper_target_bias_audit=str(OUT/'final_upper/upper_target_bias_audit.json'),
 mandatory_all_speed_traces=str(OUT/'final_upper'),mandatory_all_speed_trace_checkpoint=pair['checkpoint'],mandatory_all_speed_trace_sha256=pair['sha256'],
 teacher_compatibility_analysis=str(OUT/'teacher_compatibility_analysis.json'),teacher_target_history_analysis=str(OUT/'teacher_target_history_analysis.json'),
 head_arm_factorial_analysis=str(OUT/'head_arm_factorial/teacher_compatibility_analysis.json'),
 current_gradient_analysis=str(OUT/'current_gradient_analysis.json'),synchronized_com_contact_audit=str(OUT/'closing_com_contact_audit.json'),
 motor_speed_audit=str(OUT/'closing_motor_nominal.json'),pushed_motor_speed_audit=str(OUT/'closing_motor_pushed.json'),motor_audit_checkpoint=pair['checkpoint'],
 motor_substep_attribution=str(OUT/'current_substep_motor_attribution.json'),motor_substep_checkpoint=physical['checkpoint'],
 physical_coupling_checkpoint=physical['checkpoint'],physical_coupling_sha256=physical['checkpoint_sha256'],
 diagnostic_resume=str(OUT/'diagnostic_resume.json'),head_arm_factorial_resume=str(OUT/'head_arm_factorial/diagnostic_resume.json'),
 failed_diagnostic_attempts=str(OUT/'failed_attempts.json'),historical_diagnostic_paths=historical,previous_full_handoff_snapshot=str(OUT/'previous_NEXT_CHECK.json'),
 retained_main_totals=total,measured_since_previous=delta,gate_trends=str(OUT/'gate_trends.json'),
 retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
 resources=dict(last_completed_main=last,target_environments=8192,reserve_MiB=300,healthy_training_ramp_seconds=180,evaluation_memory_is_not_capacity=True),
 production_change=False,computer_on=True,history_inventory=str(OUT/'inventory.log'),automation_verified_active=False)
h.pop('closing_summary',None);write(TRAIN/'automation/NEXT_CHECK.json',h)
prefix=before['prompt'].split('Latest dated five-hour check',1)[0]
assert hashlib.sha256(prefix.encode()).hexdigest()=='46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
prompt=prefix+'Latest dated five-hour check October9,05:47UTC: '+summary+'\nRead diagnostics/adaptive_check_20261009_0547/REPORT.md,decision.json,physical_validation.json,teacher_compatibility_analysis.json,head_arm_factorial/teacher_compatibility_analysis.json,teacher_target_history_analysis.json,current_substep_motor_attribution.json,current_gradient_analysis.json,final_upper plots/data and closing_gate_review.json,then fresh ACTIVE_RUN/state.\n\nNext focused tests:\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(decision['next_tests']))+'\n\nKeep the same ACTIVE five-hour heartbeat/thread/notification preference and verified5 or qualified4+causal native-limit endpoint. Preserve full learned state,forward head,closed Fusion queued work and completed videos. Save,pause both,prevent restart and brief with computer on only at the verified endpoint.'
arguments=dict(id=before['id'],mode='update',kind='heartbeat',name=before['name'],prompt=prompt,rrule=before['rrule'],status='ACTIVE',targetThreadId=before['target_thread_id'])
if 'notification_policy' in before:arguments['notificationPolicy']=before['notification_policy']
write(OUT/'automation_update_args.json',arguments)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),reviewed_actor=actor,retained_updates=v['actual_retained_main_updates'],interval=delta),indent=2))
