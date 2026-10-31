"""Write evidence and one chronological entry; prepare native heartbeat update."""
import hashlib,json,tomllib
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
marker='adaptive_check_20261008_1946_record'
live=read(OUT/'closing_cpu/live_handoff.json');v=read(OUT/'closing_learning_verification.json')
pair=read(OUT/'closing_gate_review.json')['latest'];all_trace_pair=read(OUT/'final_gate_review.json')['latest']
decision=read(OUT/'decision.json');comparison=read(OUT/'matched_focused_review.json');validation=read(OUT/'experiment_validation.json')
upper=read(OUT/'final_upper/upper_joint_time_audit.json');bias=read(OUT/'final_upper/upper_bias_audit.json')
physical=read(OUT/'motor_retention_summary.json');motors=read(OUT/'motor_substep_attribution.json')
resume=read(OUT/'diagnostic_resume.json');previous=read(OUT/'previous_NEXT_CHECK.json')
before=tomllib.loads((OUT/'previous_automation.toml').read_text(encoding='utf-8'))
current_toml=Path('C:/Users/rshan/.codex/automations')/before['id']/'automation.toml'
assert tomllib.loads(current_toml.read_text(encoding='utf-8'))['prompt']==before['prompt']
assert live['state']['status']=='running' and live['state']['supervisor_pid']==39000
assert len(live['actual_worker_pids'])==1 and len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
assert comparison['completed'] and validation['passed'] and not decision['deploy']
assert v['inherited_actual_main_updates']==27507 and v['hash_proven_recovered_main_updates']==50
assert read(OUT/'final_upper/audit_set.json')['all_plots_visually_inspected']
assert pair['sha256']==all_trace_pair['sha256']
assert physical['completed'] and motors['passed'] and physical['checkpoint_sha256']==resume['checkpoint_sha256']
for name in ['final_motor_speed_audit.json','final_pushed_motor_speed_audit.json']:
 assert read(OUT/name)['passed'] and read(OUT/name)['source_checkpoint_sha256']==pair['sha256']
u=next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
ub=next(r for r in bias['records'] if r['test']=='nominal' and r['speed_m_s']==3.5)
assert u['sha256']==pair['sha256']
prior_candidates=[]
for key in ['resumed_learning_evidence','final_verification']:
 path=previous.get(key)
 if isinstance(path,str) and Path(path).is_file():
  record=read(path)
  if 'blocks' in record:prior_candidates.append((path,record))
prior_path=OUT.parent/'adaptive_check_20261008_1446/closing_owned_learning_verification.json'
if prior_path.is_file():prior_candidates.append((str(prior_path),read(prior_path)))
old_path,old_learning=max(prior_candidates,key=lambda r:r[1]['owned_snapshot_utc'])
totals=lambda d:{k:sum(b[k] for b in d['blocks']) for k in ['updates','samples','active_seconds','worker_seconds']}
total=totals(v);old_total=totals(old_learning);delta={k:total[k]-old_total[k] for k in total}
delta['chain_seconds_excluding_documented_pauses']=v['whole_chain_measurement_endpoint_unix']-old_learning['whole_chain_measurement_endpoint_unix']-v['documented_diagnostic_pause_seconds_excluded']+old_learning['documented_diagnostic_pause_seconds_excluded']
for key,denominator in [('active_samples_s','active_seconds'),('worker_effective_samples_s','worker_seconds'),('chain_samples_s','chain_seconds_excluding_documented_pauses')]:
 assert delta[denominator]>0
 delta[key]=delta['samples']/delta[denominator]
delta['previous_learning_evidence']=old_path
resumed=next(b for b in v['blocks'] if b['full_load_proof']['source_sha256']==resume['checkpoint_sha256'])
assert resumed['updates']==64 and resumed['all_preupdate_KL_and_command_mismatch_zero']
write(OUT/'resumed_learning_verification.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),resume=resume,first_resumed_main_block=resumed,
 original_supervisor_and_sources_retained=True,all_later_loads_exact=True,diagnostic_samples_excluded=True))
last=v['blocks'][-1];now=datetime.now(timezone.utc).isoformat();a,b=pair['nominal_cases'][-1],pair['pushed_cases'][-1]
prior_failures={k:[dict(command=c['command'],failures=c['failures']) for c in pair[k+'_cases'][:-1] if not c['passed']] for k in ['nominal','pushed']}
next_tests=decision['next_tests'];assert len(next_tests)==3
summary=(f"The same R40 learner and supervisor39000 are running at {v['owned_snapshot_utc']}, with one actual worker and55 verified frozen hashes. "
 f"It retains {v['completed_main_updates']:,} completed+50 hash-proven recovered={v['actual_retained_main_updates']:,} current-run main updates,+27507 inherited; {v['observed_partial_main_updates']} partial updates remain separate. "
 f"Exact {Path(pair['checkpoint']).parent.name} nominal/pushed command3.5 reaches {a['signed_speed']:.3f}/{b['signed_speed']:.3f} signed and {a['lateral_speed']:.3f}/{b['lateral_speed']:.3f} lateral m/s, failing {a['failures']}/{b['failures']}. "
 f"Prior failures: {prior_failures}. All survive. Saved gates never qualify newer partial learning. "
 "Focused prior-bucket rehearsal is REJECTED after two learning seeds157/167, exact64/128 controls and32 per-robot seeded initial states. "
 "At128 it lowers1.5 genuine flight in both seeds and adds1m support failures;0.15 recovery is seed-dependent. No diagnostic actor/sampler enters production. "
 "Exact773 normal-limit448-robot20s current/qualified349 comparison gives teacher1.83-1.86% true1.5 flight and higher complete mean reward versus current0%. "
 "4000 actual5ms motor requests/applications yield zero driving-speed clips in every cohort; ankle overspeed is braking. This is773-specific, not a5m or4m endpoint. "
 f"Actual upper angle/target/velocity and bias timelines were inspected for767, all four128-update diagnostic actors and794 at every command. Current3.5 head RMS{u['head_rms_deg']:.2f}deg; held pitch, unequal amplitude/elbows/roll and gait-linked head oscillation remain. "
 "Same-command/phase-bin CPU arm-history counterfactuals change effective lower outputs0.324RMS at3.5, showing learned feedback dependence but not identified physical coupling. No natural/human/Unitree encoder or calibrated optical-gaze certification. "
 f"The same supervisor resumed after {resume['pause_seconds']:.1f}s; first resumed776 completes64 genuine8192 updates with exact actor/critic/Adam/exploration,zero KL/command mismatch and all three auxiliaries20/update. "
 f"Main rates{v['active_PPO_samples_s']/1000:.2f}k active/{v['worker_effective_samples_s']/1000:.2f}k worker-effective/{v['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s; documented suspensions excluded through each endpoint. "
 f"Last main minimumGPU{last['minimum_free_gpu_MiB']:.0f}MiB/host{last['minimum_host_available_MiB']:.0f}MiB,private+{last['private_memory_growth_MiB']:.0f}MiB,Torch allocation+{last['torch_allocation_growth_MiB']:.0f}MiB. "
 "No production reward,gain,cap,source,actor or optimizer reset. Original35/1,teacher context,upper20,truthful400/23 inference and iteration4 remain. Same ACTIVE5h goal; computer on.")
gate_table=['|Command m/s|Nominal signed|Pushed signed|Nominal / pushed failures|','|---:|---:|---:|---|']
for x,y in zip(pair['nominal_cases'],pair['pushed_cases']):
 gate_table.append(f"|{x['command']:g}|{x['signed_speed']:.4f}|{y['signed_speed']:.4f}|{', '.join(x['failures']) or 'pass'} / {', '.join(y['failures']) or 'pass'}|")
comparison_table=['|Learning seed|Branch|Updates|1.5 true flight N/P %|3.5 lateral N/P m/s|Earlier failures N/P|','|---:|---|---:|---:|---:|---|']
for r in comparison['rows']:
 get=lambda lab,cmd:next(c for c in r[lab+'_cases'] if c['command']==cmd)
 n,p=get('nominal',1.5),get('pushed',1.5);nf,pf=get('nominal',3.5),get('pushed',3.5)
 fail=lambda lab:'; '.join(f"{c['command']:g}:{','.join(c['failures'])}" for c in r[lab+'_cases'][:-1] if not c['passed']) or 'pass'
 comparison_table.append(f"|{r['seed']}|{r['branch']}|{r['updates']}|{100*n['flight']:.3f}/{100*p['flight']:.3f}|{nf['lateral_speed']:.3f}/{pf['lateral_speed']:.3f}|{fail('nominal')} / {fail('pushed')}|")
physical_table=['|Seed|Actor|Command|Signed m/s|Lateral m/s|True flight %|Full mean reward rate|','|---:|---|---:|---:|---:|---:|---:|']
for r in physical['records']:
 physical_table.append(f"|{r['name'].split('_')[0][4:]}|{r['actor']}|{r['command']:g}|{r['signed_speed']:.4f}|{r['lateral_speed']:.4f}|{100*r['flight']:.4f}|{r['reward_rate']:.4f}|")
grad=read(OUT/'passive_gradient_summary.json')
oppo=[g['upper_PPO_cosine'] for g in grad];gcos=[g['running_teacher_cosine'] for g in grad]
simple_failures=dict(setup_attempt01=read(OUT/'setup_attempt01.json'),final_upper_argument_attempt=dict(
 log=str(OUT/'final_upper_attempt01_argument.log'),error='Unrecognized --all-speeds CPU audit option; corrected to documented --all-traces.',
 simulator_started=False,training_paused=False,production_fault=False,repaired=True),matched_harness_exit_code=0,
 work_budget_seconds=6000,work_seconds=read(OUT/'matched_focused_prior/completion.json')['work_seconds'],safe_resume_confirmed=True)
write(OUT/'failed_attempts.json',simple_failures)
initial=read(OUT/'initial_gate_review.json')['latest']
trends=dict(previous_exact_pair=previous['latest_matching_pair'],initial_exact_pair=initial,paused_exact_checkpoint=resume,
 closing_exact_pair=pair,qualification_nonmonotonic=True,
 note='794 recovers3m direction in both tests and some1.5 flight, but remains below1.5% and fails0.15 travel/current3.5 direction. Historical605 passes its current-speed case only;267 all-prior and349 complete3m certificates stay separate.')
write(OUT/'gate_trends.json',trends)
report=f'''# Five-hour humanoid training check — October 8, 19:46 UTC

Recorded {now}. Read current ACTIVE_RUN/state before reusing any observation.

{summary}

## Decision and current independent behavior

{decision['explanation']}

The current complete stage is unqualified. These exact320-robot20-second nominal/pushed tests have32 trials per command, full-population contact/direction/COM checks and100% survival. Ordinary nominal resets are clones; they do not certify final independently seeded initial-state diversity. The additional diagnostic tests explicitly generate32 unique per-robot seeds per command. No4m,5m or actuator-limited completion is claimed.

Checkpoint: `{pair['checkpoint']}`. SHA256: `{pair['sha256']}`.

{chr(10).join(gate_table)}

Exact7943.5 genuine flight is{100*a['flight']:.3f}/{100*b['flight']:.3f}%; retained1.5 flight is only{100*next(c for c in pair['nominal_cases'] if c['command']==1.5)['flight']:.3f}/{100*next(c for c in pair['pushed_cases'] if c['command']==1.5)['flight']:.3f}%. Force gaps, current-command tracking and survival alone do not qualify the stage. Compared with767, direction improves,3m direction recovers and1.5 gains some flight;0.15 remains failed. Improvement is nonmonotonic. Preserve605's separately passing3.5 case,267's all-prior pair and original349's separate3m certification.

## Bounded rehearsal experiment

The existing source already assigns exact prior commands. Expected standing/prior/frontier fractions are32/24/44%; each of eight prior anchors gets3% of all robots. Duplicating0.15 and1.5 only inside the existing prior list gives each4.8%, and each other prior2.4%. All random draws, standing/frontier membership and other frontiers are bitwise preserved. Rewards, teacher formulas, upper auxiliary, truthful history, actions, physics, gains and caps are unchanged. This is a tested allocation hypothesis, not a missing-command bug or arbitrary weight repair.

Both branches start the exact773 actor,critic,Adam and trained exploration, using4096 environments and the existing restart-LR ceiling for64 then128 updates at learning seeds157/167. Each saved actor has matching nominal/pushed tests;128 actors also receive4231 nominal-physics and4251 pushed tests with `SeedSequence([seed,environment_id])`,32 unique smallq/qd states per command and paired initial-state hashes across branches. Original classifiers stay unchanged. All8 learning/evaluation combinations and8 diverse tests completed. Diagnostic actors are kept outside production; original signatures do not certify the changed sampler.

{chr(10).join(comparison_table)}

Seed157 candidate128 adds both1m support failures and fails0.15 where control passes. Seed167 recovers0.15 and3m but fails both1m support tests and has less1.5 flight. Diverse tests reproduce those failures. Head and arm changes are also mixed, despite an unchanged upper auxiliary. The execution validation passed; behavioral deployment criteria did not.

Validation:27 meaningful CPU sampler fixtures,53 protected authored-source equalities,all51 unchanged-actor physical arrays and every case bitwise equal,eight64-environment PPO smoke updates,four full-state reload updates,four passively observed4096-environment PPO updates and all matched main-trial loads. All pre-update KL and action-command mismatches are zero; existing auxiliaries remain20 batches/update. See [experiment validation](experiment_validation.json), [matched review](matched_focused_review.json) and [CPU fixtures](focused_prior_cpu_tests.json).

## Native physics, contact and motor attribution

The separate448-robot20-second probe uses two seeds,32 independent small initial joint states per cohort,current773 at0.15/1.5/3/3.5 and qualified349 at0.15/1.5/3. Free base,gravity,self-collision,truthful observations,original35/1 gains and native CAD limits remain. All survive without resets; command mismatch is zero and full reward terms reconstruct. Statistics below use the post5s window and are a physical comparison, not substituted qualification gates.

{chr(10).join(physical_table)}

Teacher1.5 reproduces1.831/1.856% true flight and higher complete mean reward than current773's0%. Current1.5 has force gaps but no bilateral sole-clearance flight. The same limits admit this retained gait; this does not prove one particular training repair will recover it.

All4000 native5ms actuator calls save requested/applied torque,motor velocity and cap ratios. Independent masks reconstructed from saved arrays exactly match driving-clip flags. Every tested cohort has zero driving-speed clips. Current3.5 ankles overspeed in about1.2% of samples, but their requests are braking or zero; they are not velocity-limited driving. Ankle requests peak around21Nm below the modeled45Nm cap. Sparse teacher elbow torque clipping is not a locomotion ceiling. These are declared simulation motor limits, not hardware/thermal identification.

![Current and qualified prior physical response](motor_retention_response.png)

See [raw motor attribution](motor_substep_attribution.json), [contact/objective analysis](contact_objective_analysis.json), [qualified teacher receipt](motor_retention_response/qualified_teacher_receipt.json) and saved200Hz NPZ. Exact794's separate label-corrected10Hz audits have sparse ankle overspeed but cannot inherit773's5ms torque attribution. They do not establish a later-actor ceiling. Full nominal/pushed COM/contact/action arrays are finite,synchronized and command-consistent; no resets occur.

## Actual PPO derivatives and arm feedback

Four real passively captured minibatches reproduce incoming moving/standing derivatives. Standing subtraction error is below4.5e-10. Total PPO mean-network norms are28.4-59.6,prior3.13-3.46 and upper2.25-2.45; running/prior cosine ranges{min(gcos):.3f} to{max(gcos):.3f},upper/PPO{min(oppo):.3f} to{max(oppo):.3f}. Per-command teacher contributions are smaller and have mixed alignment. These are pre-global-clipping/Adam mean-network contributions, not full optimizer steps or evidence of universal interference. Positive normalized advantage does not prove global return optimality. The diagnostic learner is discarded.

The first captured actor's CPU same-command/phase-bin donor armq,qd and previous-action histories change effective clipped lower outputs by about0.324RMS at3.5. These donors were truthful saved histories, but the recombined whole-body state need not be dynamically realizable. This reproduces learned input dependence, not identified mechanical coupling, a safe trajectory or justification for production observation substitution. See [gradient summary](passive_gradient_summary.json) and [arm sensitivity](arm_feedback_sensitivity.json).

## Upper-body motion over time

All fourteen767 plots,all four plots for each128-update diagnostic actor,and all fourteen794 nominal/pushed plots were visually inspected, with saved SHA256 receipts. CSV/JSON/SVG/PNG contain actual head yaw/pitch,shoulder pitch/roll,elbow angles,physical targets and velocities by every command. Full-population32/cmd sampling is10Hz;16 total selected representatives per gate are50Hz,reset/terminal rows segmented. Sparse population sampling cannot certify fast jitter.

Exact794 at3.5: head RMS{u['head_rms_deg']:.3f}deg,p95{u['head_p95_deg']:.3f}deg. Median shoulder pitch spans{u['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{u['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}deg; slow means{ub['joints']['left_shoulder_pitch']['slow_bias_mean_deg']:.2f}/{ub['joints']['right_shoulder_pitch']['slow_bias_mean_deg']:.2f}deg. Hip cadence{u['coordination']['hip_cycle_hz']:.2f}Hz; ipsilateral shoulder/hip phases{u['coordination']['arms'][0]['phase_at_hip_cycle_deg']:.1f}/{u['coordination']['arms'][1]['phase_at_hip_cycle_deg']:.1f}deg. Representative target/joint RMS error{u['representative_50hz']['target_joint_rms_error_deg']:.2f}deg and selected raw upper clamp fraction0%. The5deg head band is diagnostic only.

Held shoulder biases,unequal amplitude/elbows/roll and gait-linked head oscillation persist. The794 nominal3m representative slowly reverses bias;767's shown2.5/3m robots instead held bias. Do not transfer that description between actors. Training seed and changed lower exposure can indirectly alter upper behavior. The0.5Hz slow/residual split uses2.02s actual time origin,excludes3s filter edges and does not certify periodic natural motion.

![794 actual angles and physical targets](final_upper/upper_joint_angles.png)

![794 actual joint velocities](final_upper/upper_joint_velocities.png)

![794 slow bias and faster residual](final_upper/upper_bias_decomposition.png)

The primary [human arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) uses wrist/trunk IMUs,not mapped shoulder encoders; comparison is feature-based and morphology-limited. The [official Unitree runtime](https://github.com/unitreerobotics/unitree_rl_mjlab) is not a verified speed-matched encoder trajectory for this CAD. No reference curves are fabricated and no human/Unitree likeness is certified. Native-zero head is torso-relative; historical Isaac/FK parity does not calibrate the original face/camera optical axis. Existing upper20-batch refinement continues; no new gain or imposed arm curve is deployed.

## Owned pause,resume and retained samples

Supervisor39000 is suspended at exact773 saved boundary20:04:48.637UTC; worker drain takes{resume['worker_drain_seconds']:.1f}s. Work ends within the6000s budget in5371.7s; the harness exits0 and resumes the same supervisor21:34:54.017UTC. Total suspension{resume['pause_seconds']:.1f}s is timestamped and excluded through each rate endpoint. No silent pause remains. First resumed776 loads exact773 actor,critic,Adam and exploration and completes64 real8192 updates; all later loads and all KL/command checks pass.

Since the previous saved learning evidence: {delta['updates']:,} retained main updates,{delta['samples']:,} samples,{delta['active_seconds']:.1f}s active collection+PPO,{delta['worker_seconds']:.1f}s worker-effective and{delta['chain_seconds_excluding_documented_pauses']:.1f}s whole-chain excluding newly documented pauses. Interval rates are{delta['active_samples_s']/1000:.2f}k/{delta['worker_effective_samples_s']/1000:.2f}k/{delta['chain_samples_s']/1000:.2f}k samples/s. Current-run totals:{total['samples']:,} main samples,{v['actual_retained_main_updates']:,} retained updates,plus27507 inherited; current partial updates are separately unqualified. Capacity,smoke and diagnostic training never count as retained main progress. The32 unsaved interruptedR39 updates remain excluded.

Run rates are{v['active_PPO_samples_s']/1000:.2f}k active/{v['worker_effective_samples_s']/1000:.2f}k worker-effective/{v['whole_chain_samples_s']/1000:.2f}k whole-chain. Whole-chain includes startup/capacity/gates,excludes{v['documented_diagnostic_pause_seconds_excluded']:.1f}s documented suspensions and conservatively includes an old untimestamped brief bookkeeping pause. These denominators differ. Last actual8192 block minimumGPU{last['minimum_free_gpu_MiB']:.0f}MiB and host{last['minimum_host_available_MiB']:.0f}MiB,private growth{last['private_memory_growth_MiB']:.0f}MiB,Torch allocation growth{last['torch_allocation_growth_MiB']:.0f}MiB. One worker and55 frozen hashes verify.8192target,300MiB reserve,host fallback and180 continuous healthy TRAINING-second ramps remain. Fusion is closed; evaluation/startup free memory is not training capacity.

Preserved failed attempts: initialization quoting before any pause/CUDA,and an unrecognized CPU audit option after resume. Both corrected locally; logs retained,neither a production fault. No training source,reward,cap,gain,optimizer or actor reset is deployed. No unsaved user/Fusion work is altered.

## Next focused tests and endpoint

{chr(10).join(f'{i+1}. {s}' for i,s in enumerate(next_tests))}

Keep the same ACTIVE five-hour heartbeat,thread and notification preference. Endpoint remains all-gate qualified actual5m/s at≥4.95,or all-gate qualified4m/s at≥3.96 with controlled causal native limits blocking5,and32 independently seeded initial states per command. At that endpoint save,pause owned training/prevent restart,pause this automation,brief and leave the computer on. Fulfilled full-body videos remain complete.
'''
evidence=dict(recorded_utc=now,summary=summary,owned_snapshot=str(OUT/'closing_cpu/live_handoff.json'),latest_exact_pair=pair,
 decision=decision,comparison=str(OUT/'matched_focused_review.json'),validation=validation,upper_comparison=str(OUT/'upper_matched_comparison.json'),
 mandatory_all_command_upper=str(OUT/'final_upper'),mandatory_all_command_upper_checkpoint=pair,
 physical_motor_checkpoint=resume['checkpoint'],physical_motor_SHA256=resume['checkpoint_sha256'],physical_motor_summary=physical,
 retained_main_totals=total,measured_since_previous=delta,diagnostic_resume=resume,first_resumed_main=resumed,failed_attempts=simple_failures,
 production_change=False,computer_on=True)
history_paths=[TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]
ref_paths=[Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
assert not (OUT/'REPORT.md').exists(),'Preserve earlier records; do not append twice.'
assert all(p.is_file() and marker not in p.read_text(encoding='utf-8') for p in history_paths+ref_paths)
write(OUT/'evidence.json',evidence);(OUT/'REPORT.md').write_text(report,encoding='utf-8')
entry=f'\n<!-- {marker} -->\n## Five-hour check — October8,19:46UTC — {now}\n\n{summary}\n\n{decision["explanation"]}\n\nReport: {OUT/"REPORT.md"}.\n\nNext focused tests:\n\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests))+f'\n<!-- /{marker} -->\n'
for path in history_paths:
 with path.open('a',encoding='utf-8') as f:f.write(entry)
for path in ref_paths:
 text=path.read_text(encoding='utf-8')
 path.write_text(f'<!-- {marker} -->\n{summary}\nRead {OUT/"REPORT.md"},then freshACTIVE_RUN/state. Earlier entries preserved.\n<!-- /{marker} -->\n\n'+text,encoding='utf-8')
old=read(TRAIN/'automation/NEXT_CHECK.json')
historical={k:old.pop(k) for k in ['upper_sequence_comparison','sequence_coverage_analysis','historical_diagnostic_paths','failed_diagnostic_attempts','previous_actor_motor_speed_audit','final_owned_observation','final_verification','automation_verification','inventory_verification'] if k in old}
old.update(recorded_utc=now,report=str(OUT/'REPORT.md'),latest_observation=v['owned_snapshot_utc'],latest_liveness_utc=v['owned_snapshot_utc'],
 active_run=v['run'],current_stage=live['state']['stage'],completed_main_updates=v['completed_main_updates'],new_main_updates=v['completed_main_updates'],
 recovered_main_updates=50,actual_retained_current_run_main_updates=v['actual_retained_main_updates'],partial_observed_main_updates=v['observed_partial_main_updates'],
 current_owned_operation=v['operation_path'],selected_actor=pair['checkpoint'],selected_actor_sha256=pair['sha256'],latest_matching_pair=pair,
 reviewed_speed_case_passes_both=a['passed'] and b['passed'],current_stage_qualified=False,running_qualified=False,velocity_ceiling_proven=False,
 summary=summary,focused_tests=next_tests,next_tests=next_tests,focus=next_tests,decision=str(OUT/'decision.json'),latest_liveness=str(OUT/'closing_cpu/live_handoff.json'),
 resumed_learning_evidence=str(OUT/'closing_learning_verification.json'),upper_joint_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'),
 mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'),upper_target_bias_audit=str(OUT/'final_upper/upper_target_bias_audit.json'),
 mandatory_all_speed_traces=str(OUT/'final_upper'),mandatory_all_speed_trace_checkpoint=pair['checkpoint'],mandatory_all_speed_trace_sha256=pair['sha256'],
 upper_comparison=str(OUT/'upper_matched_comparison.json'),arm_feedback_sensitivity=str(OUT/'arm_feedback_sensitivity.json'),current_gradient_analysis=str(OUT/'current_gradient_analysis.json'),
 matched_retention_comparison=str(OUT/'matched_focused_review.json'),synchronized_com_contact_audit=str(OUT/'final_com_contact_audit.json'),
 motor_speed_audit=str(OUT/'final_motor_speed_audit.json'),pushed_motor_speed_audit=str(OUT/'final_pushed_motor_speed_audit.json'),motor_audit_checkpoint=pair['checkpoint'],
 motor_substep_attribution=str(OUT/'motor_substep_attribution.json'),motor_substep_checkpoint=resume['checkpoint'],physical_retention_comparison=str(OUT/'motor_retention_summary.json'),
 historical_diagnostic_paths=historical,diagnostic_resume=str(OUT/'diagnostic_resume.json'),failed_diagnostic_attempts=str(OUT/'failed_attempts.json'),previous_full_handoff_snapshot=str(OUT/'previous_NEXT_CHECK.json'),
 retained_main_totals=total,measured_since_previous=delta,gate_trends=str(OUT/'gate_trends.json'),
 retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
 resources=dict(last_completed_main=last,target_environments=8192,reserve_MiB=300,healthy_training_ramp_seconds=180,evaluation_memory_is_not_capacity=True),
 production_change=False,computer_on=True,history_inventory=str(OUT/'inventory.log'),automation_verified_active=False)
old.pop('closing_summary',None);write(TRAIN/'automation/NEXT_CHECK.json',old)
prefix=before['prompt'].split('Latest dated five-hour check',1)[0]
assert hashlib.sha256(prefix.encode()).hexdigest()=='46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
prompt=prefix+'Latest dated five-hour check October8,19:46UTC: '+summary+'\nRead diagnostics/adaptive_check_20261008_1946/REPORT.md,decision.json,matched_focused_review.json,experiment_validation.json,motor_retention_summary.json,motor_substep_attribution.json,passive_gradient_summary.json,arm_feedback_sensitivity.json,upper_matched_comparison.json,final_upper(all commands,exact794),closing_gate_review.json and resumed_learning_verification.json,then freshACTIVE_RUN/state.\n\nNext focused tests:\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests))+'\n\nKeep same ACTIVEfive-hour heartbeat/thread/notifications,verified5 or qualified4+causal native limits endpoint. Save,pause both,prevent restart,brief,computeron only at that verified endpoint. Preserve full learned state,forward head,fulfilled videos and closedFusion queued work.'
arguments=dict(id=before['id'],mode='update',kind='heartbeat',name=before['name'],prompt=prompt,rrule=before['rrule'],status='ACTIVE',targetThreadId=before['target_thread_id'])
if 'notification_policy' in before:arguments['notificationPolicy']=before['notification_policy']
write(OUT/'automation_update_args.json',arguments)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),retained_updates=v['actual_retained_main_updates'],main_samples=total['samples'],interval=delta,reviewed_actor=Path(pair['checkpoint']).parent.name,decision=decision['verdict']),indent=2))
