"""Close this check only after exact comparisons, safe resume and visual QA."""
import argparse, hashlib, json, tomllib
from pathlib import Path
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parents[1]
MARKER = 'adaptive_check_20261009_1548_record'
PREFIX_SHA = '46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
AUTOMATION = Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: Path(p).write_text(json.dumps(d, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def record_paths():
    return [TRAIN / 'ITERATION_HISTORY.md', ROOT / 'docs/robot/TRAINING_JOURNAL_20261001.md',
        Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
        Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),
        Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]

def case(pair, test, command):
    return next(c for c in pair[test + '_cases'] if abs(c['command'] - command) < 1e-6)

def matched_case(row, test, command):
    return next(c for c in row['tests'][test]['cases'] if abs(c['command'] - command) < 1e-6)

def prepare():
    stamp = datetime.now(timezone.utc).isoformat()
    v = read(OUT / 'final_learning_verification.json')
    live = read(OUT / 'final_cpu/live_handoff.json')
    pair = read(OUT / 'closing_gate_review.json')['latest']
    matched = read(OUT / 'matched_coupled_teacher_review.json')
    validation = read(OUT / 'experiment_validation.json')
    certificate = read(OUT / 'neutral_teacher_certificate.json')
    physical = read(OUT / 'neutral_teacher_response_analysis.json')
    upper = read(OUT / 'final_upper/upper_joint_time_audit.json')
    bias = read(OUT / 'final_upper/upper_bias_audit.json')
    upper_matched = read(OUT / 'upper_matched_comparison.json')
    repair = read(OUT / 'joint_order_review_repair.json')
    assert matched['all_expected_components_complete'] and validation['passed']
    assert not matched['all_candidate_gates_pass'], 'Successful candidate needs separate versioned deployment review'
    assert upper_matched['all_four128_actors_complete']
    assert read(OUT / 'final_upper/audit_set.json')['all_plots_visually_inspected']
    assert all(all(x['visually_inspected'] for x in read(OUT / f'seed{s}_{b}_upper/visual_inspection.json')['plots'])
        for s in [229, 241] for b in ['control', 'candidate'])
    assert repair['experimental_head_mapping_correct'] and not repair['production_defect']
    assert v['supervisor_pid'] == 39000 and len(v['actual_worker_pids']) == 1
    assert v['frozen_files_verified'] == 55 and v['protected_authored_worker_files_equal'] == 53
    assert v['inherited_actual_main_updates'] == 27507 and v['hash_proven_recovered_main_updates'] == 50
    assert sha(pair['checkpoint']) == pair['sha256']
    assert all(c['trials'] == 32 for t in ['nominal', 'pushed'] for c in pair[t + '_cases'])
    u = next(r for r in upper['records'] if r['test'] == 'nominal' and abs(r['command_m_s'] - 3.5) < 1e-6)
    ub = next(r for r in bias['records'] if r['test'] == 'nominal' and abs(r['speed_m_s'] - 3.5) < 1e-6)
    assert u['sha256'] == pair['sha256']
    resumes = [read(OUT / p) for p in ['diagnostic_resume.json', 'secondary_analysis/diagnostic_resume.json',
        'coupled_training_trial/diagnostic_resume.json', 'coupled_training_trial/remainder_pause/diagnostic_resume.json']]
    assert all(r['same_supervisor'] for r in resumes)
    receivers = []
    for resume in resumes:
        blocks = [b for b in v['blocks'] if b['full_load_proof']['source_sha256'] == resume['checkpoint_sha256']]
        assert blocks, 'Require a genuine completed full-state main receiver after each pause'
        receivers.append(blocks[0])
    previous = read(OUT.parent / 'adaptive_check_20261009_1047/closeout_learning_verification.json')
    totals = lambda x: {k: sum(b[k] for b in x['blocks']) for k in ['updates', 'samples', 'active_seconds', 'worker_seconds']}
    total, oldtotal = totals(v), totals(previous)
    delta = {k: total[k] - oldtotal[k] for k in total}
    delta['chain_seconds_excluding_documented_pauses'] = (v['whole_chain_measurement_endpoint_unix'] - previous['whole_chain_measurement_endpoint_unix']
        - v['documented_diagnostic_pause_seconds_excluded'] + previous['documented_diagnostic_pause_seconds_excluded'])
    for name, denominator in [('active_samples_s', 'active_seconds'), ('worker_effective_samples_s', 'worker_seconds'),
        ('chain_samples_s', 'chain_seconds_excluding_documented_pauses')]:
        assert delta[denominator] > 0
        delta[name] = delta['samples'] / delta[denominator]
    gains = []
    for seed in [229, 241]:
        controls = next(r for r in matched['records'] if r['seed'] == seed and r['branch'] == 'control' and r['updates'] == 128)
        candidate = next(r for r in matched['records'] if r['seed'] == seed and r['branch'] == 'candidate' and r['updates'] == 128)
        for test in ['nominal', 'push', 'diverse_nominal', 'diverse_push']:
            c, a = matched_case(controls, test, 1.5), matched_case(candidate, test, 1.5)
            cp, ap = matched_case(controls, test, .15), matched_case(candidate, test, .15)
            gains.append(dict(seed=seed, test=test, control_flight=c['true_flight'], candidate_flight=a['true_flight'],
                control_path=cp['statistics']['forward_path_efficiency'], candidate_path=ap['statistics']['forward_path_efficiency'],
                control_current_lateral=matched_case(controls, test, 3.5)['lateral_speed'], candidate_current_lateral=matched_case(candidate, test, 3.5)['lateral_speed']))
    replicated = all(g['candidate_flight'] > g['control_flight'] and g['candidate_path'] > g['control_path'] for g in gains)
    decision = dict(recorded_utc=stamp, deploy=False, endpoint=False, production_reward_or_target_changed=False,
        tested_candidate='neutral267_coupled_exact_command_candidate_v1', status='Rejected for immediate deployment; preserve partial learned progress and all evidence',
        two_seed_targeted_behavioral_gain_replicated=replicated,
        genuine_flight_gain_replicated=all(g['candidate_flight'] > g['control_flight'] for g in gains),
        slow_path_gain_replicated=all(g['candidate_path'] > g['control_path'] for g in gains), matched_gains=gains,
        exact_full_state_controls_and_independent_initial_states_complete=True,
        explanation='The coupled target improves 1.5 m/s flight in both learning seeds but remains below the unchanged 1.5% minimum. Slow path improvement does not replicate: seed 241 worsens it at 128 updates. Seed 241 also loses the control’s passing 3.5 m/s direction at 64 updates and adds 3 m/s direction failures at 128. Neck RMS rises in both seeds. Reject this version for immediate deployment and do not extend the identical target solely for flight. No experimental actor or training-target override enters production; the exact original full learner resumes.',
        shared_target_head_route='Zero direct head derivative does not protect shared hidden features. Four fixed-state target-only SGD probes reproduce small head-output changes; this is not proof of the cause of the observed 128-update neck change.',
        reviewer_failure='The URDF metadata order was incorrectly used by the CPU reviewer. The experimental hook used the correct observed Isaac order. The unnecessary aborted evaluation and safe original resume remain preserved.',
        candidate_not_permanently_disproved=True, limits_not_raised=True, diagnostic_composites_deployed=False)
    write(OUT / 'decision.json', decision)
    a, b = case(pair, 'nominal', 3.5), case(pair, 'pushed', 3.5)
    prior = {t: [dict(command=c['command'], failures=c['failures']) for c in pair[t + '_cases'][:-1] if not c['passed']] for t in ['nominal', 'pushed']}
    block = v['blocks'][-1]
    summary = (f"Closing owned observation {v['owned_snapshot_utc']}: the same R40 full learner and supervisor 39000 are running, with one actual worker, 55 frozen hashes and 53 protected authored files. "
        f"It retains {v['completed_main_updates']:,} completed plus 50 hash-proven recovered = {v['actual_retained_main_updates']:,} current-run main updates, plus 27,507 inherited; {v['observed_partial_main_updates']} partial updates remain separate. "
        f"Exact actor {Path(pair['checkpoint']).parent.name}, SHA {pair['sha256']}, reaches {a['signed_speed']:.3f}/{b['signed_speed']:.3f} signed and {a['lateral_speed']:.3f}/{b['lateral_speed']:.3f} lateral m/s at command 3.5. Current failures {a['failures']}/{b['failures']}; prior failures {prior}. Saved gates apply only to this actor; no complete 3.5, 4, 5 m/s or causal ceiling endpoint is established. "
        "Earlier neutral-head actor 267 is freshly requalified through 3 m/s under current physics in both ordinary and 32-individually-seeded-state tests. Truthful two-seed physical diagnostics support coupled body/arm targets with the current head at 0.15/1.5; the whole-upper composite loses flight. Composites are diagnostic only. "
        "The new settled-exact-command coupled target is learning-tested at 64/128 with unchanged full-state controls, two seeds and independent initial states. It remains rejected for immediate deployment because the full gates fail; partial target gains and head/arm tradeoffs are preserved. No production source, reward, gain, cap, actor or optimizer reset. "
        f"Required all-command head/shoulder/elbow angle/target/velocity and slow-bias timelines are inspected for this exact closing actor and every matched 128 actor. Closing native neck RMS at 3.5 is {u['head_rms_deg']:.2f} degrees; held bias, unequal swing/roll/elbows remain. Human/Unitree encoder likeness and optical gaze are uncertified. "
        "The false-positive URDF-order CPU review and safely resumed aborted evaluation are corrected and preserved; the experimental runtime head mapping was correct. "
        f"Main rates are {v['active_PPO_samples_s']/1000:.2f}k active / {v['worker_effective_samples_s']/1000:.2f}k worker-effective / {v['whole_chain_samples_s']/1000:.2f}k whole-chain, with documented pauses excluded through each endpoint. Last genuine {block['environments']}-environment main block minimum GPU/host memory is {block['minimum_free_gpu_MiB']}/{block['minimum_host_available_MiB']} MiB; exact full loads, zero KL/command mismatch and all three 20-batch auxiliaries pass. Same active five-hour check, computer on, Fusion closed, queued work/videos preserved.")
    summary += (f" The latest completed original main block is {Path(block['folder']).name}; its full-state proof is separate from the fully plotted 1061 actor. Any newer incomplete or complete gate pair must be read separately before attributing qualification or posture.")
    focused = [
        'Preserve the current full learner, separately passing current-speed actors 1058/1061, the fresh neutral-head 267 all-prior certificate and certified 349. Compare fresh exact 3.5 direction, 0.15 path efficiency, 1.5 genuine flight and every retained gate. The coupled 267 target improves flight in both seeds but remains unqualified; seed 241 loses 3.5 direction at 64 and worsens slow path/adds 3 m/s direction failures at 128. Do not deploy those actors or extend the identical target solely for flight. Measure whether a narrow final-output-row derivative protects useful current-speed outputs, then compare it with the shared-feature route and unchanged control. Fixed-state head preservation does not guarantee gait retention. Any continuation needs exact full-state 64/128 controls, two learning seeds, independent initial states, every gate and versioned provenance. Keep existing weights and truthful inputs; never use 267/349 labels above their freshly verified 3 m/s certificate.',
        'Inspect actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets and velocities at every speed, including bilateral amplitude, arm-leg phase and noncancelling slow bias. This coupled target has zero direct head derivative but changes shared features; an isolated 1e-4 SGD probe slightly changes fixed-state head means, whereas final-row-only derivatives leave them unchanged. Neither result proves the cause of physical neck regression or certifies a repair. Measure per-command effective targets, shared-feature/PPO/upper gradients and free-base response before another continuation, with current full-state controls, two learning seeds and all independent gates. Preserve forward head, gait, both teachers, upper refinement at 20 batches/update and 35/1 gains. Human/Unitree encoder likeness and calibrated optical gaze remain unavailable; no production hybrid, observation substitution, imposed curve or cap raise.',
        'Verify supervisor 39000, one actual CUDA worker, 55 frozen hashes/53 protected authored files, local ground and exact actor/critic/Adam/exploration plus both teachers. Count 27,507 inherited updates plus completed and 50 verified R40 recoveries; exclude 32 unsaved R39 and all 512 matched, 12 smoke/reload and four passive diagnostic updates. Exclude the four saved diagnostic pauses only through each timing endpoint, including the reviewer interruption and remainder. Preserve the 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy TRAINING seconds; Fusion stays closed. This check’s native 5 ms probes cover 0.15/1.5 only and find no driving clips; later 10 Hz current-speed samples cannot inherit that attribution. Earlier 4/4.5/5 extrapolation fails gait/direction and proves no ceiling. Only qualified 5 m/s, or qualified 4 m/s with causal native limits blocking 5, ends training; every endpoint needs all unchanged gates and 32 independently seeded states per command.'
    ]
    evidence = dict(recorded_utc=stamp, summary=summary, owned_snapshot=str(OUT / 'final_cpu/live_handoff.json'), run=v['run'],
        exact_pair=pair, initial_pair=read(OUT / 'initial_gate_review.json')['latest'], decision=str(OUT / 'decision.json'),
        teacher_certificate=certificate, physical_diagnosis=str(OUT / 'neutral_teacher_response_analysis.json'),
        experiment_validation=str(OUT / 'experiment_validation.json'), matched_comparison=str(OUT / 'matched_coupled_teacher_review.json'),
        upper_comparison=str(OUT / 'upper_matched_comparison.json'), final_upper=str(OUT / 'final_upper/upper_joint_time_audit.json'),
        resumes=resumes, exact_main_receivers=receivers, retained_main_totals=total, measured_since_previous=delta,
        throughput={k:v[k] for k in ['active_PPO_samples_s', 'worker_effective_samples_s', 'whole_chain_samples_s', 'documented_diagnostic_pause_seconds_excluded']},
        diagnostic_updates_excluded_from_main=528, matched_updates=512, smoke_reload_updates=12, passive_updates=4,
        endpoint=False, production_change=False, computer_on=True)
    write(OUT / 'evidence.json', evidence)
    gate_table = ['|Command m/s|Signed nominal/pushed m/s|Genuine flight N/P %|Nominal / pushed failures|', '|---:|---:|---:|---|']
    for x, y in zip(pair['nominal_cases'], pair['pushed_cases']):
        gate_table.append(f"|{x['command']:g}|{x['signed_speed']:.4f}/{y['signed_speed']:.4f}|{100*x['flight']:.3f}/{100*y['flight']:.3f}|{', '.join(x['failures']) or 'pass'} / {', '.join(y['failures']) or 'pass'}|")
    experiment_table = ['|Seed|Branch|Updates|1.5 genuine flight N/P %|0.15 path efficiency N/P|3.5 lateral N/P m/s|', '|---:|---|---:|---:|---:|---:|']
    for r in matched['records']:
        x, y = matched_case(r, 'nominal', 1.5), matched_case(r, 'push', 1.5)
        px, py = matched_case(r, 'nominal', .15), matched_case(r, 'push', .15)
        fx, fy = matched_case(r, 'nominal', 3.5), matched_case(r, 'push', 3.5)
        experiment_table.append(f"|{r['seed']}|{r['branch']}|{r['updates']}|{100*x['true_flight']:.3f}/{100*y['true_flight']:.3f}|{px['statistics']['forward_path_efficiency']:.4f}/{py['statistics']['forward_path_efficiency']:.4f}|{fx['lateral_speed']:.4f}/{fy['lateral_speed']:.4f}|")
    report = f'''# Five-hour humanoid check — October 9, 2026, 15:48 UTC

The prolonged stall is real: current-speed direction and earlier skills are not jointly retained. This check freshly verifies an earlier checkpoint, diagnoses its arm/body compatibility, and tests a new coupled training target on the current full state. It does not count more elapsed training as qualification.

{decision['explanation']}

## Exact closing state and independent gates

{summary}

Checkpoint `{pair['checkpoint']}`, SHA256 `{pair['sha256']}`. Each matching nominal/pushed test has320 robots,20seconds,32 trials per command. Ordinary nominal clones do not establish final independent initial-state diversity. Classifiers, startup events and thresholds remain unchanged.

{chr(10).join(gate_table)}

Original actor1058 separately passes both current3.5 cases at signed3.511/3.517 and lateral0.113/0.114m/s, but still fails0.15 path,1.5 geometric flight and2.5 slip. Preserve that useful actor and separately qualified267/349. Gates never transfer to newer partial updates.

## Earlier checkpoint and physical compatibility

Neutral-head267, SHA `{certificate['checkpoint_sha256']}`, freshly passes every stopped/prior gate through3m/s in current original physics, ordinary nominal/pushed and separate32-independent-state-per-command tests. Its ordinary1.5 genuine flight is2.208%/2.135%. Its certificate supplies no labels above3m/s.

The640-robot20second physical comparison uses two seeds,32 paired individually seeded states per controller/command,truthful feedback,free base,gravity,self-collision,original gains/nativecaps. Every robot survives. Current1055 gives about0.387/0.399% genuine1.5 flight.267 body/arms with current head gives2.397% in both seeds and greater full mean return than current;267 body with current whole upper gives1.340/1.348% and lower return. At0.15,the body/arms+current-head composite tracks about0.187m/s. These are closed-loop component effects; they do not isolate mechanical coupling. Every composite remains diagnostic.

Fresh1058 repeats the bounded physical support before learning. Native5ms request/application attribution excludes terminal/reset samples; the recorded0.15/1.5 cohorts have zero driving-speed clips. This does not establish a3.5/4/5 ceiling. The preceding check’s explicit4/4.5/5 tests failed gait/direction with zero driving clips; that evidence remains bound to its earlier actor.

[Fresh certificate](neutral_teacher_certificate.json), [physical response and complete objective](neutral_teacher_response_analysis.json), [same-anchor support](coupled_training_trial/matched_coupled_teacher/current_physical_support.json).

![Truthful physical compatibility](neutral_teacher_response.png)

## Narrow coupled training target and exact controls

The new target operates only on settled exact0.15/1.5 commands in all five history frames, valid phase and current head within10degrees. Eligible truthful student states use the freshly certified267 labels for15body+6arm derivatives, excluding both head rows. Other commands retain original349 labels/conditioning; stopped teacher and upper refinement remain20batches/update. Original0.25MSEweight,23-output denominator and256selected bound remain. No teacher enters inference. Student observations,architecture,critic,Adam,exploration,rewards,physicaltargets,PPO,sampler,CADcaps and iteration4 remain. Shared hidden features can still change.

Twenty-three CPU eligibility/reset/finite/derivative checks pass. All51 unchanged inference replay arrays and all case reports are bitwise equal. Eight64-environment PPO updates plus four full-state reload updates pass; discarded. All matched branches start exact1058 SHA48cec6d2bcffac7ba8fcbfdb18619babb73857a02f92ca6c091359dec2088a3c. The4096-environment comparison has seeds229/241,64and128updates,unchanged320-robot20second nominal/pushed tests, and separately generated32initialstates/cmd at128. All512matched updates/50,331,648samples are diagnostic, not main progress. Original signatures explicitly do not certify the executed experimental target; no checkpoint enters production through a signature bypass.

{chr(10).join(experiment_table)}

Independent initial-state tests preserve the complete failures; targeted gains are assessed separately from whole-stage qualification. The candidate is rejected for immediate deployment, not declared permanently ineffective. No arbitrary weight increase or failed prior bucket/mean/gain/context/course/reward variant is repeated.

[Complete exact gate data](matched_coupled_teacher_review.json), [independent execution/load/diversity validation](experiment_validation.json), [target CPU checks](command_local_teacher_CPU_checks.json), [actual target/PPO gradient capture](command_local_gradient_analysis.json).

## Head, arms and shared features

All14 initial1049 plots,two fresh267 plots,four plots for each matched128actor and14closing plots are visually inspected with hash receipts. Actualheadyaw/pitch,shoulderpitch/roll,elbowangles,physicaltargets,velocities,bilateral amplitude,arm-leg phase,cadence,smoothness,saturation and noncancelling slowbias are saved byspeed. Population32/cmd sampling is10Hz;16selected representatives/gate are50Hz, with fewer representatives for some commands. Reset/terminal segments are preserved; population10Hz cannot certify fast jitter. Phase is interpreted alongside amplitude. Slow reversals are not normal periodic swing.

Closing3.5 nativeheadRMS {u['head_rms_deg']:.3f}degrees,p95 {u['head_p95_deg']:.3f}; shoulderpitch ranges {u['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{u['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}degrees. Held oppositebias,unequalroll/elbow/fastswing and neckoscillation remain. Existing upperfine-tuning continues20batches/update with35/1gains.

Four fixed-state target-only1e-4SGD probes reproduce small head-output changes from shared-feature gradients despite exactly zero directhead gradient. Last21row-only derivatives give exactly zero fixed-state headmean change. This is a local route, not proof of the observed physicalhead change or a learning-tested repair. Actual128 neckRMS and selected nativeworldheadaxis traces are saved separately. KnownIsaac/FKparity does not calibrate the originalCADface/cameraaxis.

![Actual upper angles and physical targets](final_upper/upper_joint_angles.png)

![Actual velocities](final_upper/upper_joint_velocities.png)

![Slow bias and faster residual](final_upper/upper_bias_decomposition.png)

Primary [human arm-motion study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/) Figure6 has shoulder/elbow time plots in10adults at1.25m/s walking; mapped usable numerical curves,axes,restposes and morphology remain unverified. Official [Unitree motion assets](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/main/src/assets/motions) reviewed at commit1425b15f contain danceCSVassets and do not provide verified speed-matched locomotionencoder references here. No fabricated referencecurve,naturalhuman/Unitree likeness or opticalgaze certification. [Sequential imitation research](https://proceedings.mlr.press/v15/ross11a.html) motivates student-state coverage; thisPPOauxiliary is notDAgger and inherits noformal guarantee.

[Population and representative upper comparison](upper_matched_comparison.json), [shared target/head probe](shared_target_head_effect.json), [nativeworldaxis comparison](native_head_axis_comparison.json), [reference availability](updated_motion_reference_review.json).

## Failures, corrections and safe recovery

The first CPU certificate reader importedpxr outsideSimulationApp after completedphysical tests; repaired literalgate-version reading retains strict validation. Failedlogs remain. The first float32bitwise autograd fixture differed4.66e-10 from arithmetic order; declaredatol1e-9/rtol1e-6passes without changing gates.

The CPU independentreviewer falsely reportedhead-indexmismatch usingURDFmetadataorder. ActualIsaacarticulationorder hasheadyaw/pitch11/16, and the experimentalhook correctly excludes them. Correspondinghistoryheadcolumns20/25,43/48,66/71are correct. The unnecessary stoppedcandidateevaluation, firstdriverfailure, originalsafe resume and correction remain. FiveCPU/runtime/contextchecks pass. Production sources and the experimental derivative helper are unchanged. The remainder reuses only exactcompletedcheckpoint/cleanupreceipts from the same1058anchor,while separately preserving newerproduction1061 at its secondpause. No partial result is overwritten. [Correction](joint_order_review_repair.json).

The final CPU validator initially expected a training-target annotation in diverse evaluations that execute the protected original worker with reset-only initial-state generation. The reviewer now validates that actual interface, its protected worker hash, reset-only argv and independent finite states. Ordinary wrapper evaluations still require the applied=false annotation. All32 comparison operations independently pass; the failed reviewer log remains. This is a diagnostic-accounting repair, with no production change or new pause. [Interface correction](diverse_validator_interface_repair.json).

The four saved pause durations are {', '.join(f"{r['pause_seconds']:.3f}" for r in resumes)}seconds, including worker drain. Every pause resumes the same39000supervisor and has a genuine completed exact actor/critic/Adam/exploration mainreceiver. Evaluation wrappers spanning pauses are not throughput. Documented intervals are excluded only through the measured endpoint.

Main retained totals: {total['updates']:,} updates, {total['samples']:,} samples; active collection + PPO {total['active_seconds']:.1f} s, worker wrapper {total['worker_seconds']:.1f} s. Since the preceding 13:24 check: {delta['updates']:,} retained updates, {delta['samples']:,} samples; active {delta['active_seconds']:.1f} s, worker {delta['worker_seconds']:.1f} s, whole-chain excluding documented pauses {delta['chain_seconds_excluding_documented_pauses']:.1f} s. These denominators yield {delta['active_samples_s']/1000:.2f}k/{delta['worker_effective_samples_s']/1000:.2f}k/{delta['chain_samples_s']/1000:.2f}k samples/s. The 32 unsaved R39 updates, capacity, smoke, reload, four passive and 512 matched diagnostic updates are excluded. Last completed main memory and separate partial progress are in [exact verification](final_learning_verification.json).

OneactualCUDAworker,55frozenhashes,53protected authoredfiles,localground,8192target/300MiBreserve,hostfallback and180continuoushealthyTRAINING-second upwardramp remain. Diagnostic4096/evaluationmemory does not prove8192capacity. Fusion staysclosed;queuedwork/videos preserved. Computeron. Noendpoint: onlyqualified5≥4.95 orqualified4≥3.96withcausalnative limitsblocking5,allunchangedgates and32independently seededstates/cmd ends training/automation.

## Next focused tests

{chr(10).join(f'{i+1}. {s}' for i,s in enumerate(focused))}
'''
    # Expand compact laboratory shorthand only in user-facing prose, never data or source paths.
    prose_pairs = {
        '320 robots,20seconds,32': '320 robots, 20 seconds, 32',
        '32-independent-state-per-command': '32 independent states per command',
        '0.15/1.5,the': '0.15/1.5, the', '0.15/1.5;the': '0.15/1.5; the',
        '640-robot20second': '640-robot, 20-second', '4096-environment': '4,096-environment',
        '32initialstates/cmd': '32 initial states per command', '32independently seededstates/cmd': '32 independently seeded states per command',
        '0.25MSEweight': '0.25 MSE weight', '256selected bound': '256 selected-sample bound',
        '15body+6arm': '15 body plus 6 arm', 'atol1e-9/rtol1e-6passes': 'atol 1e-9 / rtol 1e-6 passes',
        'original gains/nativecaps': 'original gains and native caps',
        'two seeds,32 paired': 'two seeds, 32 paired', 'controller/command,truthful': 'controller/command, truthful',
        'feedback,free base,gravity,self-collision': 'feedback, free base, gravity and self-collision',
        'Native5ms': 'Native 5 ms', 'native5ms': 'native 5 ms', 'Native5': 'Native 5',
        '64and128updates,unchanged320-robot20second': '64 and 128 updates, unchanged 320-robot, 20-second',
        'five history frames, valid phase': 'five history frames, valid phase', '10degrees': '10 degrees',
        '64-environment PPO': '64-environment PPO', 'Eight64-environment': 'Eight 64-environment',
        'All512matched updates/50,331,648samples': 'All 512 matched updates / 50,331,648 samples',
        'all14 initial1049 plots,two fresh267 plots,four plots': 'all 14 initial1049 plots, two fresh267 plots, four plots',
        'All14 initial1049 plots,two fresh267 plots,four plots': 'All 14 initial1049 plots, two fresh267 plots, four plots',
        'matched128actor and14closing plots': 'matched 128-update actor and 14 closing plots',
        'Actualheadyaw/pitch,shoulderpitch/roll,elbowangles,physicaltargets,velocities,bilateral amplitude,arm-leg phase,cadence,smoothness,saturation and noncancelling slowbias': 'Actual head yaw/pitch, shoulder pitch/roll, elbow angles, physical targets, velocities, bilateral amplitude, arm-leg phase, cadence, smoothness, saturation and noncancelling slow bias',
        'Population32/cmd sampling is10Hz;16selected representatives/gate are50Hz': 'Population sampling is 32 robots per command at 10 Hz; 16 selected representatives per gate are sampled at 50 Hz',
        'population10Hz': 'population sampling at 10 Hz', 'byspeed': 'by speed', 'actual128 neckRMS': 'actual 128-update neck RMS',
        'Closing3.5 nativeheadRMS': 'Closing 3.5 m/s native head RMS', 'shoulderpitch ranges': 'shoulder pitch ranges',
        'Held oppositebias,unequalroll/elbow/fastswing and neckoscillation': 'Held opposite bias, unequal roll/elbow/fast swing and neck oscillation',
        'upperfine-tuning continues20batches/update with35/1gains': 'upper fine-tuning continues at 20 batches/update with 35/1 gains',
        'target-only1e-4SGD': 'target-only 1e-4 SGD', 'directhead gradient': 'direct head gradient',
        'Last21row-only': 'Last-21-row-only', 'headmean change': 'head mean change',
        'physicalhead change': 'physical head change', 'Actual128 neckRMS': 'Actual 128-update neck RMS',
        'selected nativeworldheadaxis': 'selected native world head-axis', 'KnownIsaac/FKparity': 'Known Isaac/FK parity',
        'originalCADface/cameraaxis': 'original CAD face/camera axis', 'Figure6': 'Figure 6', 'in10adults at1.25m/s': 'in 10 adults at 1.25 m/s',
        'numerical curves,axes,restposes': 'numerical curves, axes, rest poses',
        'commit1425b15f contain danceCSVassets': 'commit1425b15f contain dance CSV assets',
        'speed-matched locomotionencoder': 'speed-matched locomotion encoder',
        'referencecurve,naturalhuman/Unitree': 'reference curve, natural human/Unitree',
        'opticalgaze': 'optical gaze', 'student-state coverage; thisPPOauxiliary is notDAgger and inherits noformal': 'student-state coverage; this PPO auxiliary is not DAgger and inherits no formal',
        'importedpxr outsideSimulationApp after completedphysical': 'imported pxr outside SimulationApp after completed physical',
        'literalgate-version': 'literal gate-version', 'Failedlogs': 'Failed logs',
        'first float32bitwise': 'first float32 bitwise', 'differed4.66e-10': 'differed by 4.66e-10', 'declaredatol': 'declared atol',
        'CPU independentreviewer': 'CPU independent reviewer', 'reportedhead-indexmismatch usingURDFmetadataorder': 'reported a head-index mismatch using URDF metadata order',
        'ActualIsaacarticulationorder hasheadyaw/pitch11/16': 'Actual Isaac articulation order has head yaw/pitch at 11/16',
        'experimentalhook': 'experimental hook', 'Correspondinghistoryheadcolumns20/25,43/48,66/71are': 'Corresponding history head columns20/25,43/48,66/71 are',
        'stoppedcandidateevaluation, firstdriverfailure, originalsafe resume': 'stopped candidate evaluation, first driver failure and original safe resume',
        'FiveCPU/runtime/contextchecks': 'Five CPU/runtime/context checks', 'exactcompletedcheckpoint/cleanupreceipts': 'exact completed checkpoint/cleanup receipts',
        'same1058anchor,while': 'same1058 anchor, while', 'newerproduction1061 at its secondpause': 'newer production1061 at its second pause',
        'same39000supervisor': 'same supervisor39000', 'genuine completed exact actor/critic/Adam/exploration mainreceiver': 'genuine completed exact actor/critic/Adam/exploration main receiver',
        'Evaluation wrappers spanning pauses are not throughput': 'Evaluation wrappers spanning pauses are excluded from training throughput',
        'activecollection+PPO': 'active collection + PPO', 'workerwrapper': 'worker wrapper',
        'documentedpauses': 'documented pauses', 'Differentdenominators': 'Different denominators',
        '32unsavedR39updates,capacity,smoke,reload,4passive and512matched diagnosticupdates': '32 unsaved R39 updates, capacity, smoke, reload, four passive and 512 matched diagnostic updates',
        'Lastrealmain': 'Last real main', 'partialprogress': 'partial progress',
        'OneactualCUDAworker,55frozenhashes,53protected authoredfiles,localground,8192target/300MiBreserve,hostfallback and180continuoushealthyTRAINING-second upwardramp': 'One actual CUDA worker, 55 frozen hashes, 53 protected authored files, local ground, 8,192 target / 300 MiB reserve, host fallback and180 continuous healthy TRAINING-second upward ramp',
        'Diagnostic4096/evaluationmemory': 'Diagnostic 4,096/evaluation memory', '8192capacity': '8,192 capacity',
        'Fusion staysclosed;queuedwork/videos': 'Fusion stays closed; queued work/videos', 'Computeron': 'Computer on',
        'Noendpoint: onlyqualified5': 'No endpoint: only qualified5', 'orqualified4': 'or qualified4',
        'withcausalnative limitsblocking5,allunchangedgates': 'with causal native limits blocking5, all unchanged gates',
        'same39000, one': 'same supervisor39000, one', '55 hashes/53': '55 hashes / 53',
        'Count27507': 'Count27,507', 'exclude32 unsaved': 'exclude32 unsaved',
        'all512 matched,12 smoke/reload and4 passive': 'all512 matched,12 smoke/reload and4 passive',
        'Keep8192 target,300MiBreserve,hostguards/fallback,180': 'Keep8,192 target,300 MiB reserve,host guards/fallback,180',
        'current-speed direction at64': 'current-speed direction at64 updates',
        'full-state64/128': 'full-state64/128-update', 'No267/349': 'No267/349',
    }
    def readable(text):
        for old, new in prose_pairs.items():
            text = text.replace(old, new)
        return text
    report = readable(report)
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    entry = f"\n<!-- {MARKER} -->\n\n## Humanoid five-hour check — October 9, 2026, 15:48 UTC\n\n{summary}\n\nMeasured since previous check: {delta['updates']:,} retained updates / {delta['samples']:,}samples; active/worker/chain rates{delta['active_samples_s']/1000:.2f}k/{delta['worker_effective_samples_s']/1000:.2f}k/{delta['chain_samples_s']/1000:.2f}k with explicit pause exclusions.\n\nReport: `{OUT / 'REPORT.md'}`. Evidence/decision/matched physical/learning/upper/load proofs and corrected reviewer are nearby.\n\n" + '\n'.join(f'{i+1}. {s}' for i,s in enumerate(focused)) + '\n'
    for index, p in enumerate(record_paths()):
        old = p.read_text(encoding='utf-8')
        assert MARKER not in old, 'Never duplicate a chronological check'
        p.write_text(old + readable(entry) if index < 3 else readable(entry) + '\n' + old, encoding='utf-8')
    old = read(OUT / 'previous_NEXT_CHECK.json')
    durable = ['schema', 'automation_id', 'highest_fully_qualified_stage', 'highest_fully_qualified_command_m_s',
        'inherited_main_updates_preserved', 'prior_source_and_history_scope', 'recording_request', 'fusion_user_preference',
        'resource_release_report', 'native_automation_same_thread_five_hours', 'endpoint_policy', 'historical_rejected_upper_comparison']
    handoff = {k: old[k] for k in durable if k in old}
    handoff.update(recorded_utc=stamp, report=str(OUT / 'REPORT.md'), decision=str(OUT / 'decision.json'), active_run=v['run'],
        summary=summary, closing_summary=summary, latest_observation=v['owned_snapshot_utc'], latest_liveness_utc=v['owned_snapshot_utc'],
        latest_liveness=str(OUT / 'final_cpu/live_handoff.json'), current_stage=v['stage'], current_stage_qualified=False,
        running_qualified=False, velocity_ceiling_proven=False, computer_on=True, production_change=False,
        completed_main_updates=v['completed_main_updates'], new_main_updates=v['completed_main_updates'], recovered_main_updates=50,
        actual_retained_current_run_main_updates=v['actual_retained_main_updates'], partial_observed_main_updates=v['observed_partial_main_updates'],
        current_owned_operation=v['operation_path'], current_checkpoint=live['state']['checkpoint'], current_checkpoint_sha256=sha(live['state']['checkpoint']),
        selected_actor=pair['checkpoint'], selected_actor_sha256=pair['sha256'], latest_matching_pair=pair,
        reviewed_speed_case_passes_both=a['passed'] and b['passed'], focused_tests=focused, next_tests=focused, focus=focused,
        resumed_learning_evidence=str(OUT / 'final_learning_verification.json'), final_verification=str(OUT / 'final_learning_verification.json'),
        retained_main_totals=total, measured_since_previous=delta,
        retained_main_throughput=evidence['throughput'], resources=dict(last_completed_main=block, target_environments=8192,
            reserve_MiB=300, healthy_training_ramp_seconds=180, evaluation_or_diagnostic_memory_is_not_capacity=True),
        mandatory_upper_audit=str(OUT / 'final_upper/upper_joint_time_audit.json'), mandatory_upper_bias_audit=str(OUT / 'final_upper/upper_bias_audit.json'),
        upper_target_bias_audit=str(OUT / 'final_upper/upper_target_bias_audit.json'), mandatory_all_speed_trace_checkpoint=pair['checkpoint'],
        mandatory_all_speed_trace_sha256=pair['sha256'], upper_matched_comparison=str(OUT / 'upper_matched_comparison.json'),
        synchronized_com_contact_audit=str(OUT / 'closing_com_contact_audit.json'), motor_speed_audit=str(OUT / 'closing_motor_nominal.json'),
        pushed_motor_speed_audit=str(OUT / 'closing_motor_pushed.json'), motor_audit_checkpoint=pair['checkpoint'],
        current_gradient_analysis=str(OUT / 'current_gradient_analysis.json'), shared_target_head_effect=str(OUT / 'shared_target_head_effect.json'),
        neutral_teacher_certificate=str(OUT / 'neutral_teacher_certificate.json'), neutral_teacher_physical_response=str(OUT / 'neutral_teacher_response_analysis.json'),
        matched_coupled_teacher_review=str(OUT / 'matched_coupled_teacher_review.json'), experiment_validation=str(OUT / 'experiment_validation.json'),
        diagnostic_resumes=[str(OUT / p) for p in ['diagnostic_resume.json', 'secondary_analysis/diagnostic_resume.json',
            'coupled_training_trial/diagnostic_resume.json', 'coupled_training_trial/remainder_pause/diagnostic_resume.json']],
        historical_diagnostic_paths=dict(previous_full_handoff=str(OUT / 'previous_NEXT_CHECK.json'), previous_check=str(OUT.parent / 'adaptive_check_20261009_1047/REPORT.md')),
        previous_full_handoff_snapshot=str(OUT / 'previous_NEXT_CHECK.json'), history_inventory=str(TRAIN / 'history/RUN_INVENTORY.json'),
        automation_verified_active=False)
    write(TRAIN / 'automation/NEXT_CHECK.json', handoff)
    before = tomllib.loads((OUT / 'previous_automation.toml').read_text(encoding='utf-8-sig'))
    current = tomllib.loads(AUTOMATION.read_text(encoding='utf-8-sig'))
    assert current['prompt'] == before['prompt']
    prefix = before['prompt'].split('Latest dated five-hour check', 1)[0]
    assert hashlib.sha256(prefix.encode()).hexdigest() == PREFIX_SHA
    prompt = (prefix + 'Latest dated five-hour check October 9, 15:48 UTC. ' + summary +
        '\n\nRead diagnostics/adaptive_check_20261009_1548/REPORT.md, decision.json, evidence.json, neutral_teacher_certificate.json, neutral_teacher_response_analysis.json, matched_coupled_teacher_review.json, experiment_validation.json, upper_matched_comparison.json, joint_order_review_repair.json, shared_target_head_effect.json and final_upper plots/data; then fresh ACTIVE_RUN/state.\n\nNext focused tests:\n' +
        '\n'.join(f'{i+1}. {s}' for i,s in enumerate(focused)) +
        '\n\nKeep the same ACTIVE five-hour heartbeat, thread, notification preference and verified 5 m/s or qualified 4 m/s plus causal native-limit endpoint. Preserve the full learned state, forward head, closed Fusion queued work and completed videos. Only at the verified endpoint save, pause training and automation, prevent restart and brief with the computer on.')
    args = dict(mode='update', id=before['id'], kind='heartbeat', name=before['name'], prompt=prompt, rrule=before['rrule'],
        status='ACTIVE', targetThreadId=before['target_thread_id'])
    if 'notification_policy' in before:
        args['notificationPolicy'] = before['notification_policy']
    write(OUT / 'automation_update_args.json', args)
    print(json.dumps(dict(records_written=True, checkpoint=pair['checkpoint'], retained_main=v['actual_retained_main_updates'],
        targeted_gains_replicated=replicated, deploy=False, endpoint=False), indent=2))

def verify():
    before = tomllib.loads((OUT / 'previous_automation.toml').read_text(encoding='utf-8-sig'))
    current = tomllib.loads(AUTOMATION.read_text(encoding='utf-8-sig'))
    args = read(OUT / 'automation_update_args.json')
    for key in ['id', 'kind', 'name', 'rrule', 'target_thread_id', 'created_at']:
        assert current[key] == before[key]
    assert current['status'] == 'ACTIVE' and current['prompt'] == args['prompt']
    assert hashlib.sha256(current['prompt'].split('Latest dated five-hour check', 1)[0].encode()).hexdigest() == PREFIX_SHA
    assert all(current.get(k) == before.get(k) for k in set(before) | set(current) if 'notification' in k or k in ['model', 'reasoning_effort', 'thinking'])
    assert all(p.read_text(encoding='utf-8').count(f'<!-- {MARKER} -->') == 1 for p in record_paths())
    inventory = read(TRAIN / 'history/RUN_INVENTORY.json')
    rows = inventory['physical_runs']
    differing = sum(x['actual_sha256'] is not None and not x['matches'] for r in rows for x in r['frozen_source_verification'])
    missing = sum(x['actual_sha256'] is None for r in rows for x in r['frozen_source_verification'])
    assert len(rows) == 40 and differing == 0 and not inventory['read_warnings']
    receipt = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), automation_id=current['id'], status='ACTIVE',
        same_thread=True, same_five_hour_cadence=True, durable_scope_and_endpoint_preserved=True, notification_and_model_preferences_preserved=True,
        prompt_matches_exact_update=True, record_marker_once_in_five_files=True, physical_runs=len(rows),
        physical_operations=sum(len(r['operations']) for r in rows), differing_hashes=differing,
        missing_historical_manifest_artifacts=missing, read_warnings=inventory['read_warnings'], inventory_generated_utc=inventory['generated_utc'], computer_on=True)
    write(OUT / 'automation_inventory_verification.json', receipt)
    handoff = read(TRAIN / 'automation/NEXT_CHECK.json')
    handoff.update(automation_verified_active=True, automation_verification=str(OUT / 'automation_inventory_verification.json'),
        inventory_verification=str(OUT / 'automation_inventory_verification.json'))
    write(TRAIN / 'automation/NEXT_CHECK.json', handoff)
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['prepare', 'verify'], required=True)
    a = p.parse_args()
    {'prepare': prepare, 'verify': verify}[a.mode]()
