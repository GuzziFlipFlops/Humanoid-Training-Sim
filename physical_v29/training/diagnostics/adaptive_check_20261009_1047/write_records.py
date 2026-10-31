"""Append this check only after experiments, main recovery and visual QA finish."""
import hashlib,json,tomllib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat();marker='adaptive_check_20261009_1047_record'
v=read(OUT/'final_learning_verification.json');live=read(OUT/'final_cpu/live_handoff.json')
pair=read(OUT/'closing_gate_review.json')['latest'];initial=read(OUT/'initial_gate_review.json')['latest']
decision=read(OUT/'decision.json');validation=read(OUT/'experiment_validation.json')
matched=read(OUT/'matched_reward_review.json');physical=read(OUT/'motor_retention_response/result.json')
high=read(OUT/'high_speed_extrapolation/analysis.json');gradient=read(OUT/'passive_gradient_summary.json')
objective_capture=read(OUT/'objective_advantage_comparison.json');takeoff=read(OUT/'takeoff_phase_comparison.json')
inverse_arms=read(OUT/'inverse_arm_response_analysis.json')
assert objective_capture['completed'] and objective_capture['diagnostic_updates_excluded_from_main']==8
assert inverse_arms['completed'] and not inverse_arms['production_composite_deployed']
upper=read(OUT/'final_upper/upper_joint_time_audit.json');bias=read(OUT/'final_upper/upper_bias_audit.json')
u=next(x for x in upper['records'] if x['actor_tag']=='latest' and x['test']=='nominal' and x['command_m_s']==3.5)
ub=next(x for x in bias['records'] if x['test']=='nominal' and x['speed_m_s']==3.5)
assert not decision['deploy'] and validation['passed'] and matched['completed']
assert not high['endpoint_proven'] and not high['causal_velocity_ceiling_proven']
assert read(OUT/'final_upper/audit_set.json')['all_plots_visually_inspected'] and u['sha256']==pair['sha256']
assert all(read(OUT/f'seed{s}_{b}_upper/visual_QA.json')['visually_inspected'] for s in [179,191] for b in ['control','candidate'])
assert all(read(OUT/f'high_speed_extrapolation/upper_{t}/visual_QA.json')['visually_inspected'] for t in ['nominal','pushed'])
assert read(OUT/'cpu_numeric_order_repair.json')['passed']
resumes=[read(OUT/'diagnostic_resume.json'),read(OUT/'high_speed_extrapolation/diagnostic_resume.json')]
receivers=[next(b for b in v['blocks'] if b['full_load_proof']['source_sha256']==r['checkpoint_sha256']) for r in resumes]
old=read(OUT.parent/'adaptive_check_20261009_0547/liveness_learning_verification.json')
totals=lambda d:{k:sum(b[k] for b in d['blocks']) for k in ['updates','samples','active_seconds','worker_seconds']}
total=totals(v);prior_total=totals(old);delta={k:total[k]-prior_total[k] for k in total}
delta['chain_seconds_excluding_documented_pauses']=v['whole_chain_measurement_endpoint_unix']-old['whole_chain_measurement_endpoint_unix']-v['documented_diagnostic_pause_seconds_excluded']+old['documented_diagnostic_pause_seconds_excluded']
for name,den in [('active_samples_s','active_seconds'),('worker_effective_samples_s','worker_seconds'),('chain_samples_s','chain_seconds_excluding_documented_pauses')]:
    assert delta[den]>0;delta[name]=delta['samples']/delta[den]
last=v['blocks'][-1];a,b=pair['nominal_cases'][-1],pair['pushed_cases'][-1]
actor=Path(pair['checkpoint']).parent.name
prior_failures={k:[dict(command=c['command'],failures=c['failures']) for c in pair[k+'_cases'][:-1] if not c['passed']] for k in ['nominal','pushed']}
before=tomllib.loads((OUT/'previous_automation.toml').read_text(encoding='utf-8'))
native=Path('C:/Users/rshan/.codex/automations')/before['id']/'automation.toml'
assert tomllib.loads(native.read_text(encoding='utf-8'))['prompt']==before['prompt']
summary=(f"Same R40 full learner/supervisor39000 is running at {v['owned_snapshot_utc']}, one actual worker and55 verified frozen hashes/53 protected authored files. "
    f"{v['completed_main_updates']:,} completed+50 hash-proven recovered={v['actual_retained_main_updates']:,} current-run main updates, plus27507 inherited; {v['observed_partial_main_updates']} observed partial updates remain separate. "
    f"Reviewed exact{actor}:3.5 signed{a['signed_speed']:.3f}/{b['signed_speed']:.3f}, lateral{a['lateral_speed']:.3f}/{b['lateral_speed']:.3f}m/s, current failures{a['failures']}/{b['failures']}. Prior failures{prior_failures}. Saved gates never qualify newer partial actors. "
    +decision['diagnostic_summary']+' '
    +f"All required actual upper angle/target/velocity and noncancelling bias timelines inspected at992, all four matched128 actors and final{actor}. Final head RMS{u['head_rms_deg']:.2f}deg; held bias, unequal swing/roll/elbows remain, no human/Unitree encoder or optical-gaze certification. "
    +f"Same original supervisor resumed after{resumes[0]['pause_seconds']:.1f}s reward comparison and{resumes[1]['pause_seconds']:.1f}s higher-speed tests, with exact full-state receivers and all auxiliaries20batches/update. "
    +f"Retained rates{v['active_PPO_samples_s']/1000:.2f}kactive/{v['worker_effective_samples_s']/1000:.2f}kworker/{v['whole_chain_samples_s']/1000:.2f}kchain; denominators differ and documented pauses are excluded through each endpoint. Last real main block minimumGPU/host{last['minimum_free_gpu_MiB']}/{last['minimum_host_available_MiB']}MiB, private+{last['private_memory_growth_MiB']}MiB and Torch growth{last['torch_allocation_growth_MiB']}MiB. "
    +"No production reward,source,gain,cap,actor or optimizer reset; experimental actors/rewards remain outside production. Original35/1 gains,truthful400/23 inference,both teachers/upper20 and iteration4 remain. Same ACTIVE5h endpoint,computer on.")

def gate_table(p):
    rows=['|Command m/s|Nominal signed|Pushed signed|Genuine geometric flight N/P %|Nominal / pushed failures|','|---:|---:|---:|---:|---|']
    for x,y in zip(p['nominal_cases'],p['pushed_cases']):
        rows.append(f"|{x['command']:g}|{x['signed_speed']:.4f}|{y['signed_speed']:.4f}|{100*x['flight']:.3f}/{100*y['flight']:.3f}|{', '.join(x['failures']) or 'pass'} / {', '.join(y['failures']) or 'pass'}|")
    return '\n'.join(rows)

experiment_table=['|Learning seed|Branch|Updates|3.5 lateral N/P m/s|1.5 true flight N/P %|Prior failures N/P|','|---:|---|---:|---:|---:|---|']
for r in matched['rows']:
    x,y=r['nominal_cases'][-1],r['pushed_cases'][-1]
    xn=next(c for c in r['nominal_cases'] if c['command']==1.5);yn=next(c for c in r['pushed_cases'] if c['command']==1.5)
    failed={k:[str(c['command'])+':'+','.join(c['failures']) for c in r[k+'_cases'][:-1] if not c['passed']] for k in ['nominal','pushed']}
    experiment_table.append(f"|{r['seed']}|{r['branch']}|{r['updates']}|{x['lateral_speed']:.4f}/{y['lateral_speed']:.4f}|{100*xn['flight']:.3f}/{100*yn['flight']:.3f}|{'; '.join(failed['nominal']) or 'none'} / {'; '.join(failed['pushed']) or 'none'}|")

high_table=['|Test|Command|Signed m/s|Lateral m/s|Survival|Failures|','|---|---:|---:|---:|---:|---|']
for r in high['gates']:
    for c in r['cases']:
        if c['command']<3.5:continue
        s=c['statistics'];signed=f"{c['measured_signed_forward_m_s']:.4f}" if c['measured_signed_forward_m_s'] is not None else 'unavailable'
        lateral=f"{c['measured_absolute_lateral_m_s']:.4f}" if c['measured_absolute_lateral_m_s'] is not None else 'unavailable'
        high_table.append(f"|{r['label']}|{c['command']:g}|{signed}|{lateral}|{s['survival']:.3f}|{', '.join(c['failures']) or 'pass'}|")

high_clip_count=sum(m['first_episode_alive_driving_clip_count'] for r in high['motor_records'] for m in r['startup_inclusive_first_episode_motors'])
high_first_failures=sum(r['first_episode_failed_robots'] for r in high['first_episode_failures'])
arm_table=['|Seed/controller|Command m/s|Signed m/s|Lateral m/s|Genuine flight %|Survival|Head RMS degrees|',
           '|---|---:|---:|---:|---:|---:|---:|']
number=lambda n,factor=1:'unavailable' if n is None else f'{factor*n:.4f}'
for r in inverse_arms['records']:
    arm_table.append(f"|{r['name']}|{r['command']:g}|{number(r['signed_forward_m_s'])}|{number(r['absolute_lateral_m_s'])}|"
                     f"{number(r['genuine_flight_fraction'],100)}|{number(r['survival'])}|{number(r['head_RMS_deg'])}|")
first_objective=objective_capture['batches'][0]
frontier_original=next(c for c in first_objective['signals']['original']['commands'] if c['command']=='frontier_above3')
frontier_narrow=next(c for c in first_objective['signals']['narrow']['commands'] if c['command']=='frontier_above3')
frontier_gradient=next(c for c in first_objective['command_gradient_comparisons'] if c['command']=='frontier_above3')

report=f'''# Five-hour humanoid training check — October 9, 2026, 10:47 UTC

The user correctly identified the prolonged failure to qualify beyond3.5m/s. This check compares earlier qualified learning, tests a new narrow reward change against exact full-state controls and explicitly evaluates higher commands through5m/s. It preserves the existing learner and every gate instead of treating more elapsed training as success.

{decision['explanation']}

## Exact current learner and gates

Initial reviewed992 passed both current3.5 cases, signed3.477/3.484m/s and lateral0.107/0.107. It failed retained0.15/0.35/0.65 tracking and1.5 geometric flight. Newer998 was pinned for the matched reward experiment; its direction had already regressed. These facts distinguish current-speed capability from whole-stage qualification.

Final reviewed actor `{pair['checkpoint']}`, SHA256 `{pair['sha256']}`. Both matching tests have320 robots,20seconds and32 trials per command. All classifiers and initial/startup events remain unchanged. Saved gates belong to this exact actor, not the snapshot's newer partial learning. Ordinary nominal clones remain distinct from the separately generated independent initial states.

{gate_table(pair)}

Preserve992's useful passing current case,605/908/917's separate passing evidence,267's earlier all-prior pair and freshly qualified349 through3m/s. Current-speed success alone does not promote the stage. [Gate trends](gate_trends.json), [synchronized COM/contact/action analysis](closing_com_contact_audit.json), and exact nominal/pushed motor audits retain actor scope.

## Earlier checkpoint and full physical objective

The controlled native-physics comparison pins current998, SHA256 `{physical['checkpoint_sha256']}`, against exact freshly qualified349, SHA256 `{physical['actor_sha256']['qualified_prior']}`. There are832 robots, two initial-state seeds4491/4511,32 paired independent q/qd realizations per command/controller and20seconds. Every robot survives; actual input/action commands match and all physical reward terms reconstruct within{physical['maximum_reward_term_sum_error']:.3g}. Free base,gravity,self collision,35/1 gains,local ground and native CAD motor caps remain. Controllers use their own truthful state/history; no hybrid,pose support,action pulse or learned diagnostic actor enters production.

At1.5m/s the current controller gives0.141/0.133% genuine unloaded geometric flight. The earlier349 gives1.835/1.818% and about0.132/0.133 higher complete mean reward rate. This reproduces retained-skill loss, not an actuator ceiling or a claim that all natural running is discouraged. [Physical evidence](motor_retention_response/result.json) includes synchronized actual joints/targets/actions/root/contact/COM and native5ms requests/applications.

The [phase-conditioned take-off comparison](takeoff_phase_comparison.json) resolves the force/clearance difference on those same physical states. Both seeds reproduce the earlier policy's narrow unloading/clearance event around reward phase0.46; the current policy's much weaker event occurs in the following bin. Current physical target/joint RMS is smaller for several leg joints despite worse flight. This is a closed-loop trajectory difference, not same-state teacher-label error, a proven clock defect or justification for repeating the rejected fixed timing advances. The200Hz target/joint harmonic response describes one representative per cohort and does not identify open-loop hardware dynamics.

![Physical force and genuine flight by phase](takeoff_phase_comparison.png)

## New reward hypothesis and controlled learning

The original yaw-frame exponential tracker has0.5m/s width at every speed. For0.1<command<=1.0, the candidate substitutes `max(0.10,0.12*command)` from the existing independent speed tolerance. It changes no weights,frame,stopped or faster reward, sampler,teacher,upper auxiliary,PPO,physics or cap. This is a tested hypothesis about retained slow tracking, not a proved numerical fault or a universal optimal width.

[112 numeric/gradient/parity fixtures plus6 counterexample categories](slow_reward_CPU_tests.json) pass. The same-actor physical replay has all51 arrays and all gate cases bitwise identical. Eight real smoke PPO updates, four reload updates and four passive original PPO captures load full actor/critic/Adam/exploration exactly with zero pre-update KL or command mismatch. Experimental reward receipts explicitly state that the original source signature does **not** certify the executed override; no experimental checkpoint is silently admitted to production.

The [recorded-state full-objective counterfactual](physical_reward_counterfactual.json) also reproduces an important risk: narrower instantaneous width penalizes normal stride variation. It changes the ranking of qualified slow motion at several commands. It stays finite and bounded, but those execution facts do not establish improvement. [Bias/variation decomposition](tracking_variation_analysis.json) verifies the second-moment identity and distinguishes mean error from temporal/population variation; no mean-only temporal reward is implemented.

![Tracking width hypothesis](tracking_width_candidate.png)

![Recorded tracking variation](tracking_variation.png)

The matched experiment starts exact current998 full state in every branch,4096 environments, independent learning seeds179/191,64 and128 real updates. It keeps both priors and upper auxiliary20batches/update. Each checkpoint has unchanged320-robot20second nominal/pushed gates;128 also has32 independently generated initial joint states per command at seeds4531/4551, paired between branches. All diagnostic learning is excluded from main progress. Work-only budget6600s excludes the saved-boundary wait/drain. Execution and selection are separate: [independent experiment validation](experiment_validation.json) passes{validation['count']} checks, while the candidate decision is `{decision['candidate_status']}`.

{chr(10).join(experiment_table)}

[Exact checkpoint and diverse-state review](matched_reward_review.json) and [required upper comparisons](upper_matched_comparison.json) preserve all branches, seeds, failures and actor hashes. No diagnostic actor,reward override or optimizer reset is deployed.

## Explicit higher-speed tests

After safe main resume, the next fresh saved actor `{high['checkpoint']}`, SHA256 `{high['checkpoint_sha256']}`, is tested at commands through5m/s. Original nominal/pushed416-robot20second tests contain13 commands and32 trials per command. Additional nominal-physics/pushed tests generate32 distinct initial states per command at seeds4591/4611. These are untrained extrapolation tests; they do not bypass the retained-skill contract or assert that command5 is achieved.

{chr(10).join(high_table)}

A separate256-robot20second physical probe records3.5/4/4.5/5 under each of two seeds with native200Hz motor velocity, requested/applied torque and passive joint/target/phase data. [First-episode attribution](high_speed_extrapolation/analysis.json) excludes the terminal control tick and every later reset/post-fall motor sample. Failure reasons and times remain explicit. Falls,turning,untrained extrapolation or isolated cap spikes cannot prove a persistent locomotion-relevant velocity ceiling. No4/5 completion or fallback is claimed.

In this exact physical probe there are {high_first_failures} first-episode failures and {high_clip_count} native driving-speed clip samples across all eight cohorts, including startup. The higher-command failures therefore do not establish an actuator velocity ceiling. This does not prove that a future qualified 5 m/s gait is physically feasible, nor isolate torque limits from policy coordination.

Actual higher-command head/shoulder/elbow angles, physical targets and velocities were also plotted and visually inspected in [nominal](high_speed_extrapolation/upper_nominal/visual_QA.json) and [pushed](high_speed_extrapolation/upper_pushed/visual_QA.json) tests. One representative per plotted command shows positive shoulder-bias shifts and greater native neck-pitch bias at 4.5/5. Full-population distributions remain separate; the 2-8 second plotted interval does not represent the entire trial or certify natural motion.

![Higher-speed current-policy results](high_speed_extrapolation/higher_speed_results.png)

## Actual motor attribution and PPO contributions

The998 native motor probe records4000 actual5ms calls per cohort. Independent saved-data reconstruction matches the controller driving-clip masks exactly; all tested current and prior cohorts through3.5 have zero driving-speed clips. Ankle overspeed is braking demand. Elbow torque limits can be active, but those events do not establish a locomotion ceiling. Native output units and CAD differential Jacobians/reductions are preserved; declared flat-current/hard-speed BLDC and servo envelopes are assumptions, not measured voltage/current/thermal hardware identification. Later sparse10Hz actor audits cannot inherit998's substep attribution.

Four actual4096-environment PPO minibatches capture incoming gradients before standing/moving hooks. [Summary](passive_gradient_summary.json) gives prior parameter norms2.85–3.37, full mean-network PPO24.31–52.06 and upper1.86–2.22; alignment has both signs. Standing subtraction reproduces within6e-10 and moving mean-score cosine is1. Contributions are before global clipping/Adam and omit exploration parameters. Norm size, normalized advantages or mixed alignment do not prove globally opposed gradients or a sampling defect.

After the higher-speed physical tests, two additional4-update4096-environment passive captures start the same fresh exact full state, seed211, under the original and narrow objectives. Actor,critic,Adam and exploration match; all auxiliaries remain20batches/update and all KL/command checks pass. These8 diagnostic updates are discarded and excluded from main progress. [Actual paired advantage/gradient evidence](objective_advantage_comparison.json) records first-rollout observation/action/model equality rather than assuming it. Its first-minibatch paired attribution flag is `{objective_capture['first_minibatch_paired_attribution_valid']}`; normalized-advantage RMS difference is{objective_capture['batches'][0]['advantages']['RMS']:.6g}. The installed PPO normalizes advantages across the rollout, so changing only slow rewards can still change other commands' relative PPO signals. Later minibatches include earlier optimizer updates. Local derivative differences do not prove the cause of the long-term gait regressions or validate the rejected candidate.

In the exact first minibatch, the fast-command returns and retained critic values are identical. Global rollout advantage standard deviation changes from {first_objective['signals']['original']['global_rollout_value_statistics']['unnormalized_advantage_std']:.6f} to {first_objective['signals']['narrow']['global_rollout_value_statistics']['unnormalized_advantage_std']:.6f}. The frontier's mean normalized advantage changes from {frontier_original['normalized_advantage_mean']:.6f} to {frontier_narrow['normalized_advantage_mean']:.6f}; its actual pre-hook mean-network parameter gradient norm changes from {frontier_gradient['original_norm']:.6f} to {frontier_gradient['narrow_norm']:.6f}, with cosine {frontier_gradient['PPO_parameter_cosine']:.6f}. This directly reproduces a local cross-command learning-signal change from the objective intervention; it does not establish a biased policy-gradient estimator or a complete causal explanation of the 128-update direction loss.

## Current-body/current-head inverse arm diagnosis

The previously untested inverse arms-only comparison is completed on the same fresh saved actor as the higher-speed tests:384 robots,20seconds,two initial-state seeds4591/4611 and32 paired states per controller/command. It compares current,qualified349,and current head/body with only the six shoulder/elbow action rows supplied by349. Commands are limited to0.15/1.5 within the teacher certificate. Six CPU row-selection/rejection checks pass; every physical tick asserts exact current head/non-arm action preservation before the physical feedback response. Both policies receive truthful physical observations/history. The composite remains strictly diagnostic and outside original actor/signature certification.

[Inverse-arm results](inverse_arm_response_analysis.json) preserve speed,flight,survival,head/full reward and first-episode native motor attribution by seed. Its bounded physical-promising flag is `{inverse_arms['bounded_physical_candidate_promising']}`; this flag is not full qualification or a learning-tested repair. Physical and learned feedback effects remain coupled. No teacher label above3m/s,no production hybrid and no arm/controller rollback is deployed.

{chr(10).join(arm_table)}

![Inverse arms-only physical response](inverse_arm_response.png)

## Mandatory upper joint angles over time

All14 initial992 plots, four plots for each matched128 actor, and all14 final actor plots were visually inspected with hash-backed receipts. [Final data/plots](final_upper/upper_joint_time_audit.json) cover actual head yaw/pitch,shoulder pitch/roll and elbow angles, physical targets, velocities,bilateral amplitude,arm/ipsilateral-leg phase,cadence,smoothness and action saturation. Population32/cmd sampling is10Hz;16 representatives per gate are50Hz, with fewer representatives for some commands. First episodes and terminal/reset rows are segmented;10Hz sampling cannot certify faster jitter.

Final nominal3.5 head RMS{u['head_rms_deg']:.3f}deg,p95{u['head_p95_deg']:.3f}deg; shoulder pitch ranges{u['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{u['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}deg. Slow means{ub['joints']['left_shoulder_pitch']['slow_bias_mean_deg']:.2f}/{ub['joints']['right_shoulder_pitch']['slow_bias_mean_deg']:.2f}deg; hip cadence{u['coordination']['hip_cycle_hz']:.3f}Hz and shoulder phases{u['coordination']['arms'][0]['phase_at_hip_cycle_deg']:.1f}/{u['coordination']['arms'][1]['phase_at_hip_cycle_deg']:.1f}deg. Phase is interpreted with amplitude, not as natural-motion certification. Held bias,unequal roll/elbow motion and neck oscillation remain. The0.5Hz slow split excludes3second edges and is descriptive; slow reversals are not periodic arm swing.

![Actual upper angles and physical targets](final_upper/upper_joint_angles.png)

![Actual velocities](final_upper/upper_joint_velocities.png)

![Actual slow bias and faster residual](final_upper/upper_bias_decomposition.png)

![Physical target and measured bias](final_upper/upper_target_bias.png)

Primary [Collins/Adamczyk/Kuo2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/) Figure6 has human shoulder/elbow time plots. Its indexed primary text verifies10 adult subjects and1.25m/s walking; usable numerical curves,axes,rest poses and morphology mapping remain unverified. The official [Unitree motion assets](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/main/src/assets/motions) reviewed at commit1425b15f contain dance CSVs rather than matched locomotion encoders. Neither source supplies a verified numerical comparison here. No curve is fabricated, no human/Unitree likeness or calibrated optical gaze is claimed. Native neck zero is torso-relative; known Isaac/FK parity does not calibrate the original CAD face/camera axis. [Updated source metadata and limitations](updated_motion_reference_review.json) and [reward sources](research_sources.json) are preserved.

## Safe recovery and measured main progress

The same owned supervisor resumes after{resumes[0]['pause_seconds']:.1f}s reward comparison and{resumes[1]['pause_seconds']:.1f}s higher tests, including worker drains{resumes[0]['worker_drain_seconds']:.1f}/{resumes[1]['worker_drain_seconds']:.1f}s. Exact identity and all55 frozen files verify. No diagnostic worker remains. Each first main receiver loads its paused actor,critic,Adam and trained exploration exactly, with zero pre-update KL/mismatch and all three auxiliaries20batches/update. The original full learner,35/1 gains,teachers,upper module,400/23 truthful inference and iteration4 execution remain.

Relative to the previous check's actual06:54 endpoint: {delta['updates']:,} retained main updates,{delta['samples']:,} samples,{delta['active_seconds']:.1f}s active,{delta['worker_seconds']:.1f}s worker and{delta['chain_seconds_excluding_documented_pauses']:.1f}s whole-chain excluding documented pauses. Interval rates{delta['active_samples_s']/1000:.2f}k/{delta['worker_effective_samples_s']/1000:.2f}k/{delta['chain_samples_s']/1000:.2f}k samples/s. This check's diagnostic learning/smoke/capacity samples never count as main progress;32 unsaved interruptedR39 updates remain excluded.

Current R40 has{v['completed_main_updates']:,} completed+50 hash-proven received recovery updates={v['actual_retained_main_updates']:,}, plus27507 inherited. Total main samples{total['samples']:,}. Rates{v['active_PPO_samples_s']/1000:.2f}kactive/{v['worker_effective_samples_s']/1000:.2f}kworker/{v['whole_chain_samples_s']/1000:.2f}kchain, with{v['documented_diagnostic_pause_seconds_excluded']:.1f}s documented suspensions excluded only through each timing endpoint. Startup/capacity/gates remain in chain time; an older untimestamped brief bookkeeping pause remains conservatively included. Denominators differ.

Last real main block at{last['environments']} environments keeps minimumGPU{last['minimum_free_gpu_MiB']}MiB,host{last['minimum_host_available_MiB']}MiB; private+{last['private_memory_growth_MiB']}MiB,Torch allocation+{last['torch_allocation_growth_MiB']}MiB. Target8192,reserve300MiB,host guards/fallback and180 continuous healthy TRAINING seconds for upward ramps remain. Evaluation memory is not training capacity. Fusion stays closed; queued work and completed videos are preserved. Minor diagnostic plot-path/receipt-schema/monitor-order corrections do not change production and their failed outputs are preserved.

Crossing operation999 also exposed lexical ordering in CPU audit tools: an older999 pair could sort after1001. The [numeric-order repair](cpu_numeric_order_repair.json) preserves the three before-source files, reproduces the failure and passes five CPU boundary/actual-pair checks. Only the ownership recorder, history inventory and this check's reviewer changed. The original supervisor selects checkpoints from saved state/results; all55 frozen runtime hashes and53 authored worker files remain identical. This is a reporting repair, not an explanation for the gait plateau.

## Next focused tests

{chr(10).join(f'{i+1}. {s}' for i,s in enumerate(decision['next_tests']))}

Same ACTIVE five-hour heartbeat/thread/notification preference. Only actual≥4.95m/s with every unchanged nominal/pushed/stopped/prior gate and32 independently generated initial-state trials per command, or actual≥3.96m/s under those same requirements plus controlled causal native limits proving5 blocked, ends training. At that endpoint save exact full state/hashes/gates/telemetry,pause owned training/prevent restart,pause this automation and brief with the computer on.
'''
evidence=dict(recorded_utc=now,summary=summary,decision=decision,fully_reviewed_pair=pair,initial_reviewed_pair=initial,
    owned_snapshot=str(OUT/'final_cpu/live_handoff.json'),current_saved_checkpoint=live['state']['checkpoint'],
    reward_experiment_checkpoint=physical['checkpoint'],reward_experiment_sha256=physical['checkpoint_sha256'],
    higher_speed_checkpoint=high['checkpoint'],higher_speed_sha256=high['checkpoint_sha256'],
    experiment_validation=validation,diagnostic_resumes=resumes,first_main_receivers=receivers,
    mandatory_upper_actor=pair['checkpoint'],mandatory_upper_folder=str(OUT/'final_upper'),
    CPU_numeric_order_repair=str(OUT/'cpu_numeric_order_repair.json'),
    retained_main_totals=total,measured_since_previous=delta,production_change=False,computer_on=True)
evidence.update(objective_advantage_comparison=str(OUT/'objective_advantage_comparison.json'),takeoff_phase_comparison=str(OUT/'takeoff_phase_comparison.json'),
    inverse_arm_response_analysis=str(OUT/'inverse_arm_response_analysis.json'),
    higher_command_upper_visual_QA={t:str(OUT/f'high_speed_extrapolation/upper_{t}/visual_QA.json') for t in ['nominal','pushed']},
    total_diagnostic_learning_updates=validation['diagnostic_learning_updates']+8)
histories=[TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]
refs=[Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
assert not (OUT/'REPORT.md').exists() and all(marker not in p.read_text(encoding='utf-8') for p in histories+refs)
write(OUT/'evidence.json',evidence)
write(OUT/'resumed_learning_verification.json',dict(resumes=resumes,first_retained_receivers=receivers,
    all_later_main_loads_exact=True,diagnostic_samples_excluded=True))
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
entry=f'\n<!-- {marker} -->\n## Five-hour check — October9,10:47UTC — {now}\n\n{summary}\n\nReport: {OUT/"REPORT.md"}.\n\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(decision['next_tests']))+f'\n<!-- /{marker} -->\n'
for p in histories:
    with p.open('a',encoding='utf-8') as f:f.write(entry)
for p in refs:
    p.write_text(f'<!-- {marker} -->\n{summary}\nRead {OUT/"REPORT.md"}, then fresh ACTIVE_RUN/state. Earlier entries preserved.\n<!-- /{marker} -->\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
h=read(TRAIN/'automation/NEXT_CHECK.json')
historical={k:h.pop(k) for k in ['teacher_compatibility_analysis','teacher_target_history_analysis','head_arm_factorial_analysis',
    'head_arm_factorial_resume','failed_diagnostic_attempts','final_owned_observation','final_verification','automation_verification',
    'inventory_verification','historical_diagnostic_paths','physical_coupling_checkpoint','physical_coupling_sha256'] if k in h}
h.update(recorded_utc=now,report=str(OUT/'REPORT.md'),latest_observation=v['owned_snapshot_utc'],latest_liveness_utc=v['owned_snapshot_utc'],active_run=v['run'],current_stage=v['stage'],
    completed_main_updates=v['completed_main_updates'],new_main_updates=v['completed_main_updates'],recovered_main_updates=50,
    actual_retained_current_run_main_updates=v['actual_retained_main_updates'],partial_observed_main_updates=v['observed_partial_main_updates'],
    current_owned_operation=v['operation_path'],current_checkpoint=live['state']['checkpoint'],current_checkpoint_sha256=sha(live['state']['checkpoint']),
    selected_actor=pair['checkpoint'],selected_actor_sha256=pair['sha256'],latest_matching_pair=pair,
    reviewed_speed_case_passes_both=a['passed'] and b['passed'],current_stage_qualified=False,running_qualified=False,velocity_ceiling_proven=False,
    summary=summary,focused_tests=decision['next_tests'],next_tests=decision['next_tests'],focus=decision['next_tests'],decision=str(OUT/'decision.json'),
    latest_liveness=str(OUT/'final_cpu/live_handoff.json'),resumed_learning_evidence=str(OUT/'final_learning_verification.json'),
    mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),upper_joint_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'),
    mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'),upper_target_bias_audit=str(OUT/'final_upper/upper_target_bias_audit.json'),
    mandatory_all_speed_traces=str(OUT/'final_upper'),mandatory_all_speed_trace_checkpoint=pair['checkpoint'],mandatory_all_speed_trace_sha256=pair['sha256'],
    current_gradient_analysis=str(OUT/'current_gradient_analysis.json'),synchronized_com_contact_audit=str(OUT/'closing_com_contact_audit.json'),
    motor_speed_audit=str(OUT/'closing_motor_nominal.json'),pushed_motor_speed_audit=str(OUT/'closing_motor_pushed.json'),motor_audit_checkpoint=pair['checkpoint'],
    motor_substep_attribution=str(OUT/'motor_substep_attribution.json'),motor_substep_checkpoint=physical['checkpoint'],
    matched_reward_review=str(OUT/'matched_reward_review.json'),new_reward_experiment=str(OUT/'matched_slow_reward/experiment_manifest.json'),
    tracking_objective_counterfactual=str(OUT/'physical_reward_counterfactual.json'),tracking_variation_analysis=str(OUT/'tracking_variation_analysis.json'),
    objective_advantage_comparison=str(OUT/'objective_advantage_comparison.json'),takeoff_phase_comparison=str(OUT/'takeoff_phase_comparison.json'),
    inverse_arm_response_analysis=str(OUT/'inverse_arm_response_analysis.json'),inverse_arm_response_checkpoint=high['checkpoint'],
    upper_matched_comparison=str(OUT/'upper_matched_comparison.json'),higher_speed_diagnostic=str(OUT/'high_speed_extrapolation/analysis.json'),
    higher_speed_diagnostic_checkpoint=high['checkpoint'],higher_speed_diagnostic_sha256=high['checkpoint_sha256'],
    diagnostic_resume=str(OUT/'diagnostic_resume.json'),higher_speed_resume=str(OUT/'high_speed_extrapolation/diagnostic_resume.json'),
    historical_diagnostic_paths=historical,previous_full_handoff_snapshot=str(OUT/'previous_NEXT_CHECK.json'),
    retained_main_totals=total,measured_since_previous=delta,gate_trends=str(OUT/'gate_trends.json'),
    retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
    resources=dict(last_completed_main=last,target_environments=8192,reserve_MiB=300,healthy_training_ramp_seconds=180,evaluation_memory_is_not_capacity=True),
    production_change=False,computer_on=True,history_inventory=str(OUT/'inventory.log'),automation_verified_active=False)
h.pop('closing_summary',None);write(TRAIN/'automation/NEXT_CHECK.json',h)
prefix=before['prompt'].split('Latest dated five-hour check',1)[0]
assert hashlib.sha256(prefix.encode()).hexdigest()=='46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
prompt=prefix+'Latest dated five-hour check October9,10:47UTC: '+summary+'\nRead diagnostics/adaptive_check_20261009_1047/REPORT.md,decision.json,matched_reward_review.json,experiment_validation.json,physical_reward_counterfactual.json,tracking_variation_analysis.json,high_speed_extrapolation/analysis.json,passive_gradient_summary.json,final_upper plots/data and resumed_learning_verification.json,then fresh ACTIVE_RUN/state.\n\nNext focused tests:\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(decision['next_tests']))+'\n\nPreserve same ACTIVEfive-hour heartbeat/thread/notification preference,full neutral-head learner and all failed/rejected evidence. Only verified5 or verified4+causal native limits blocking5 ends training. Save,pause both,prevent restart and brief with computer on at that endpoint. Fusion stays closed,queued work/videos preserved.'
arguments=dict(id=before['id'],mode='update',kind='heartbeat',name=before['name'],prompt=prompt,rrule=before['rrule'],status='ACTIVE',targetThreadId=before['target_thread_id'])
if 'notification_policy' in before:arguments['notificationPolicy']=before['notification_policy']
write(OUT/'automation_update_args.json',arguments)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),reviewed_actor=actor,retained_updates=v['actual_retained_main_updates'],interval=delta),indent=2))
