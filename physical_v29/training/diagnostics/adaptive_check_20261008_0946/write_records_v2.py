"""Write the reviewed evidence and same-heartbeat handoff once, without runtime changes."""
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parents[1]
MARKER = 'adaptive_check_20261008_0946_record'
REFERENCES = [
    Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),
    Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),
]
JOURNALS = [TRAIN / 'ITERATION_HISTORY.md', ROOT / 'docs/robot/TRAINING_JOURNAL_20261001.md',
    Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

live = read(OUT / 'liveness_cpu/live_handoff.json')
v = read(OUT / 'liveness_learning_verification.json')
pair = read(OUT / 'closing_gate_review.json')['latest']
cycle = read(OUT / 'individual_cycle_analysis.json')
ctx = read(OUT / 'state_context_sensitivity.json')
validation = read(OUT / 'diagnostic_validation.json')
resume = read(OUT / 'diagnostic_resume.json')
failed = read(OUT / 'attempt01_initial_import_failure/diagnostic_resume.json')
upper = read(OUT / 'final_upper/upper_joint_time_audit.json')
u = next(r for r in upper['records'] if r['actor_tag'] == 'latest' and r['test'] == 'nominal' and r['command_m_s'] == 3.5)
old = read(OUT / 'previous_NEXT_CHECK.json')
previous = read(old['resumed_learning_evidence'])
before = tomllib.loads((OUT / 'previous_automation.toml').read_text(encoding='utf-8'))

# Validate every lineage binding before writing chronological records.
assert live['state']['status'] == 'running' and live['state']['supervisor_pid'] == 39000
assert len(live['actual_worker_pids']) == 1
assert len(live['frozen_source_integrity']) == 55 and all(live['frozen_source_integrity'].values())
assert validation['passed'] and ctx['first_actual_minibatch_actor_bitwise_identical']
for d in (cycle, ctx, validation, u):
    assert Path(d['checkpoint']) == Path(pair['checkpoint']) and d['sha256'] == pair['sha256']
assert Path(upper['gate_review']) == OUT / 'closing_gate_review.json'
assert hashlib.sha256(Path(pair['checkpoint']).read_bytes()).hexdigest() == pair['sha256']
first_resumed = next(b for b in v['blocks'] if Path(b['folder']).name == '608_run_350_train_200')
assert first_resumed['updates'] == 64 and first_resumed['environments'] == 6144
assert first_resumed['full_load_proof']['source_sha256'] == pair['sha256']
assert all(first_resumed['full_load_proof'][k] for k in ('actor_equal', 'critic_equal', 'optimizer_equal', 'exploration_equal'))
assert first_resumed['all_preupdate_KL_and_command_mismatch_zero']
assert first_resumed['three_auxiliaries_batches_per_update'] == 20
assert not (OUT / 'REPORT.md').exists(), 'Records already written; do not duplicate chronological entries.'
for p in REFERENCES:
    assert MARKER not in p.read_text(encoding='utf-8')
for p in JOURNALS:
    assert p.exists()

def totals(d):
    return {k: sum(b[k] for b in d['blocks']) for k in ('updates', 'samples', 'active_seconds', 'worker_seconds')}

tot = totals(v)
prev_tot = totals(previous)
assert tot['updates'] == v['actual_retained_main_updates']
delta = {k: tot[k] - prev_tot[k] for k in tot}
delta['chain_seconds_excluding_documented_pauses'] = (
    v['whole_chain_measurement_endpoint_unix'] - previous['whole_chain_measurement_endpoint_unix']
    - v['documented_diagnostic_pause_seconds_excluded'] + previous['documented_diagnostic_pause_seconds_excluded'])
delta['active_samples_s'] = delta['samples'] / delta['active_seconds']
delta['worker_effective_samples_s'] = delta['samples'] / delta['worker_seconds']
delta['chain_samples_s'] = delta['samples'] / delta['chain_seconds_excluding_documented_pauses']

a, b = pair['nominal_cases'][-1], pair['pushed_cases'][-1]
assert a['command'] == b['command'] == 3.5 and a['passed'] and b['passed']
failures = {label: [dict(command=c['command'], failures=c['failures'])
    for c in pair[label + '_cases'][:-1] if not c['passed']] for label in ('nominal', 'pushed')}
now = datetime.now(timezone.utc).isoformat()
next_tests = [
    'Preserve actor 605, whose 3.5 m/s running case passes both tests, and compare newer exact actors with all stopped and earlier-speed gates. Actor 605 still fails 0.15 net travel, 1.5 flight and 2.5 slip. Audit their actual per-command advantages, teacher targets and landing/force events before a narrow retention continuation. Current-speed success alone does not promote the stage; preserve actor 267 and separately certified 349.',
    'Continue mandatory upper angle, target, velocity, bilateral and arm-leg timelines. Same actor 605 gives phase-balanced 3.5 m/s shoulder outputs (-16.04, -28.52 degrees) on stable-command training states versus (-28.15, +9.90) on deterministic evaluation states. Measure truthful sequential per-robot cycle coverage, command transitions, exploration and arm-history/hidden-feature response. Stage an individual-cycle or state-conditioned two-final-row derivative only after reset-safe CPU fixtures, physical response, exact full-state 64/128 controls, all gates and independent seeds. The ensemble-mean candidate remains rejected. Preserve forward head, legs, teachers, 35/1 gains and all learned state; no observation substitution, hybrid or imposed reference curve.',
    'Verify supervisor 39000, one worker, 55 hashes, local ground, exact full-state teachers and actual retained samples. Preserve 27,507 inherited updates plus completed and 50 verified R40 recoveries; exclude 32 unsaved R39 updates. Exclude this check\'s 110.1-second failed-import and 197.7-second successful-capture pauses through each timing endpoint. Keep target 8192, reserve 300 MiB, host guards and 180 continuous healthy training seconds; do not force a ramp from evaluation memory. Correctly mapped actor-605 motor samples show right ankle motor 2 at/above cap in 1.96% of 10 Hz samples. If a high-speed plateau appears, measure current-actor 5 ms requested/applied torque and sustained driving clips before any limit conclusion. Only qualified 5 m/s or qualified 4 m/s with causal limits blocking 5 ends the same five-hour heartbeat.'
]
summary = (
    f"The same R40 learner and supervisor 39000 are running at {v['owned_snapshot_utc']}, with one actual worker and 55 verified hashes. "
    f"It retains {v['completed_main_updates']:,} completed plus 50 recovered = {v['actual_retained_main_updates']:,} current-run main updates, plus 27,507 inherited; "
    f"{v['observed_partial_main_updates']} partial updates are separate. Reviewed actor 605, nominal 606/pushed 607, passes both 3.5 m/s running cases: "
    f"signed speed {a['signed_speed']:.3f}/{b['signed_speed']:.3f}, lateral {a['lateral_speed']:.3f}/{b['lateral_speed']:.3f} m/s, "
    f"genuine flight {a['flight']*100:.2f}/{b['flight']*100:.2f}%. The whole stage remains unqualified because 0.15 net travel, 1.5 flight and 2.5 slip fail both tests. "
    f"Saved gates never qualify newer partial actors. Required actor-605 upper timelines were inspected; head RMS {u['head_rms_deg']:.2f} degrees, held shoulder biases and asymmetry remain. "
    "No human/Unitree motion or optical-gaze certification. Passive truthful-state and actual PPO captures establish a same-actor state-distribution/arm-feedback lead; no new learning candidate was deployed. "
    "All 51 unchanged physical replay arrays and cases are bitwise equal. Four diagnostic PPO updates pass full-load, KL and command checks. "
    "The same supervisor resumed after 110.1 seconds for the failed import attempt and 197.7 seconds for successful capture; failed evidence is preserved. "
    "Resumed block 608 completes 64 genuine 6144-environment updates with exact actor/critic/Adam/exploration. "
    f"Retained rates are {v['active_PPO_samples_s']/1000:.2f}k active, {v['worker_effective_samples_s']/1000:.2f}k worker-effective and {v['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. "
    "Existing upper and both prior auxiliaries remain at 20 batches/update; 35/1 gains and all production sources, rewards, native caps and learned state remain unchanged. "
    "A CPU motor-case attribution error was corrected; no production fault or actuator ceiling is established. The same five-hour automation stays active; computer on."
)
decision = dict(recorded_utc=now, decision='PRESERVE_IMPROVING_ORIGINAL_FULL_LEARNER',
    explanation='Preserve the useful current-speed direction improvement. Full retention and upper motion remain incomplete. Four passive gradient batches show both signs of alignment; they do not justify a new deployed loss. Continue existing refinements and test truthful sequential individual-cycle/state-conditioned coverage before a narrow candidate.',
    production_change=False, diagnostic_actors_deployed=False, new_candidate_learning_tested=False, next_tests=next_tests)
evidence = dict(recorded_utc=now, summary=summary, reviewed_checkpoint=pair['checkpoint'], reviewed_sha256=pair['sha256'],
    owned=str(OUT/'liveness_cpu/live_handoff.json'), learning=str(OUT/'liveness_learning_verification.json'),
    exact_pair=str(OUT/'closing_gate_review.json'), upper=str(OUT/'final_upper'), cycles=str(OUT/'individual_cycle_analysis.json'),
    state_response=str(OUT/'state_context_sensitivity.json'), actual_gradients=str(OUT/'current_gradient_analysis.json'),
    validation=str(OUT/'diagnostic_validation.json'), motor_audit=str(OUT/'motor_speed_audit.json'),
    retained_totals=tot, previous_retained_totals=prev_tot, measured_delta=delta,
    comparison_previous_snapshot=previous['owned_snapshot_utc'], decision=decision, production_change=False)

table = ['| Command m/s | Nominal signed / lateral m/s | Pushed signed / lateral m/s | Failures |', '|---|---|---|---|']
for n, p in zip(pair['nominal_cases'], pair['pushed_cases']):
    table.append(f"| {n['command']:g} | {n['signed_speed']:.3f} / {n['lateral_speed']:.3f} | {p['signed_speed']:.3f} / {p['lateral_speed']:.3f} | {'; '.join(sorted(set(n['failures']+p['failures']))) or 'pass both'} |")
contexts = ['| Command m/s | Training effective shoulder pitch means, deg | Evaluation effective shoulder pitch means, deg |', '|---|---|---|']
for r in ctx['rows']:
    t, e = r['training_phase_balanced_effective_mean_deg'], r['evaluation_phase_balanced_effective_mean_deg']
    contexts.append(f"| {r['command']:g} | {t[0]:.2f} / {t[1]:.2f} | {e[0]:.2f} / {e[1]:.2f} |")
report = f'''# Five-hour check — October 8, 09:46 UTC

{summary}

## Decision and exact qualification

{decision['explanation']} Preserve actor 605 SHA `{pair['sha256']}` with matching 606/607 gates, actor 267's all-prior pair and original 349's separate 3 m/s certification. No 5 m/s, conditional 4 m/s or actuator-limit endpoint is established. The owned operation `{Path(v['operation_path']).name}` is recorded separately. Evaluation robot count is not PPO capacity.

The reviewed pair uses 320 robots, 20 seconds and 32 trials per command, with 100% survival. Nominal clones do not establish the independently seeded initial-state diversity required at the final endpoint.

{chr(10).join(table)}

Compared with the previous check's actor 524, 3.5 lateral speed improves from 0.184/0.163 to 0.117/0.120 m/s. The 3 m/s case recovers direction. The 0.15 net-travel and 1.5 flight failures persist, while 2.5 slip regresses after actor 602 briefly passed it. These nonmonotonic prior results prevent promotion despite both current-speed cases passing.

## Upper motion over time

Actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets and velocities were plotted and visually inspected for actors 602 and 605. The audits include bilateral amplitudes, ipsilateral hip phase, elbow coordination, cadence, smoothness, saturation and a descriptive 0.5 Hz slow/faster decomposition. Thirty-two robots per command at 10 Hz are distinguished from 16 total representatives at 50 Hz, one or two per command. Resets and terminal rows are segmented. Population sampling at 10 Hz cannot certify fast jitter.

At actor 605's 3.5 m/s command, head RMS is {u['head_rms_deg']:.3f} degrees; shoulder pitch ranges are 9.59/7.20 degrees and arm-versus-ipsilateral-hip phases are 169.7/-174.0 degrees at 2.67 Hz. Phase alone does not certify natural motion. Earlier slow 2.5/3 m/s reversals become more nearly held offsets in 605; physical targets carry those offsets. Roll asymmetry, elbow coordination and gait-linked head oscillation persist. All angle/target, velocity, bias and target-bias plots in `final_upper` were inspected and QA receipts saved. The [individual cycle plot](individual_cycle_means.png) was also inspected.

The unchanged passive replay captures truthful 400-input, five-frame histories at 10 Hz for all 320 robots and 50 Hz for 16 representatives. It returns identical actions. All cases and 51 physical arrays match original nominal 606 bitwise. CPU actor replay differs by at most {cycle['CPU_action_replay_max_error']:.3g} action units from saved GPU outputs.

There are 256 complete eight-bin first-episode cycles between 5 and 17 seconds. Individual cycle means avoid cancellation between robots and slow bias reversals. At 3.5 m/s the representative's effective policy cycle-mean RMS is 28.14/9.99 degrees and actual joint cycle-mean RMS is 21.81/11.11 degrees. This is one robot over 31 cycles, not 32 independent robots. The five-degree reporting band is diagnostic only. Three CPU counterexamples cover quiet motion, zero-mean opposed swing and ensemble cancellation. An undeployed detached-feature temporal derivative affecting only the two final shoulder-pitch output rows is quantified; no production temporal loss or 64/128-update candidate learning was tested this check.

## Same-actor training-state diagnosis

The first actual 4096-environment PPO minibatch mean-network tensors are bitwise equal to actor 605. Restricting both datasets to the exact command in all five frames and balancing eight phase bins still gives:

{chr(10).join(contexts)}

CPU-only phase-matched history substitutions measure fixed-state policy sensitivity. At 3.5 m/s, borrowing actual training arm q/qd/previous-action histories changes shoulder outputs by 1.708 RMS action units and lower outputs by 0.439 RMS; head-history substitution changes shoulders by only 0.038 RMS. These artificial combinations are not physically consistent trajectories, proof that mechanical effects are absent, or deployable observations. The lower-output response rejects an immediate history-spoof or hybrid approach. The reproduced training/evaluation state difference is a research lead, not an observation bug or established causal repair. Production inference remains truthful.

Four actual PPO minibatches have shoulder final-row PPO norms 4.29–15.74 versus existing upper auxiliary norms 2.50–2.92. Alignment with the proposed evaluation-cycle derivative ranges from -0.915 to +0.924; no universal opposition is established. Per-command advantages, ratios, teacher coverage, effective targets and pre-auxiliary derivatives are saved. The current 256-row auxiliary has only 1–8 exact 2.5 m/s samples in these batches. This describes coverage; it does not prove that increasing weight or sample count fixes motion. CPU stopped-hook subtraction matches directly captured pre-standing PPO derivatives; no production hook fault is inferred.

The [primary human arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) uses wrist/lower-back IMUs and supports feature comparisons of amplitude, asymmetry and smoothness by speed. It supplies no morphology-matched shoulder encoder targets. The [official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab) was revisited; mapped, speed-matched G1/H1 encoder curves were not obtained. No reference curves are fabricated and human/Unitree likeness remains uncertified. Native head zero is torso-relative. Known Isaac/FK parity does not calibrate the original CAD face/camera optical axis or certify world gaze.

## Synchronized physics and corrected motor attribution

`closing_com_contact_audit.json` verifies actor 605's representative 50 Hz COM, joints, physical targets, actions, forces, contacts and commands: finite values, no resets and zero observed/action command difference. Full-population contact/support gates remain unchanged. Moving COM offsets and simplified collision envelopes are not full-body pressure certification.

The older 0741 CPU motor helper assumed contiguous cohorts, but evaluation saves interleaved `case_labels`. The corrected audit stably sorts q/qd by saved labels before the unchanged exact CAD Jacobian/reduction mapping. A synthetic label fixture and directional finite-difference checks pass. This is a diagnostic accounting correction; production physics, native caps and aggregate samples are unchanged. Preserve older artifacts and treat their per-command attribution as unreliable. No historical ceiling had been established from those samples.

Correctly attributed actor-605/3.5 right ankle motor 2 has p99 speed 1.301 times its 400 RPM cap, with 1.962% of 10 Hz samples at/above cap; left motor 2 p99 is 0.853. These pose-derived samples are not native 5 ms driving-clipping or available-torque evidence. A fresh actor-specific substep audit is required if a plateau appears; older different-actor zero-clip probes cannot certify this actor. No sustained relevant ceiling or conditional 4 m/s success is inferred.

## Failure preservation and exact continuation

The first passive wrapper imported Isaac math before AppLauncher and failed with missing `pxr`. Its cleanup resumed the original supervisor after {failed['pause_seconds']:.1f} seconds, including {failed['worker_drain_seconds']:.1f} seconds for drain. Logs and receipts are preserved under `attempt01_initial_import_failure`. The saved script copy there was made after the diagnostic import correction; failed logs remain exact. Only diagnostic import ordering changed. The successful retry completed passive evaluation plus four actual PPO/full-load updates, with zero KL and command mismatch; the smoke actor is discarded. The same supervisor resumed after {resume['pause_seconds']:.1f} seconds, including {resume['worker_drain_seconds']:.1f} seconds for drain. A separate CPU analysis syntax error was corrected without simulator impact. The successful retry harness exited zero, with wait/drain budgets separated from diagnostic execution.

First resumed block 608 finishes 64 genuine 6144-environment updates and loads actor 605's actor, critic, Adam and exploration exactly. Both teachers and the upper auxiliary remain at 20 batches/update. Minimum free GPU is {first_resumed['minimum_free_gpu_MiB']} MiB, available host memory {first_resumed['minimum_host_available_MiB']} MiB, private-memory growth {first_resumed['private_memory_growth_MiB']} MiB and Torch allocation growth {first_resumed['torch_allocation_growth_MiB']} MiB. All pre-update KL and command differences are zero. One worker and 55 frozen hashes verify.

## Actual samples, time and resource guards

At {v['owned_snapshot_utc']}, the learner retains {v['completed_main_updates']:,} completed plus 50 recovered = {v['actual_retained_main_updates']:,} current-run updates, plus 27,507 inherited. Thirty-two interrupted unsaved R39 updates are excluded; {v['observed_partial_main_updates']} observed partial updates are separate. Capacity, smoke and diagnostic actors never count as main learning.

Current retained main totals are {tot['samples']:,} samples, {tot['active_seconds']:,.1f} collection-plus-PPO seconds and {tot['worker_seconds']:,.1f} worker seconds. Since the previous owned observation at {previous['owned_snapshot_utc']}, this adds {delta['updates']:,} retained updates and {delta['samples']:,} samples in {delta['active_seconds']:,.1f} active seconds, {delta['worker_seconds']:,.1f} worker seconds and {delta['chain_seconds_excluding_documented_pauses']:,.1f} chain seconds excluding documented pauses. The interval rates are {delta['active_samples_s']:,.0f}/{delta['worker_effective_samples_s']:,.0f}/{delta['chain_samples_s']:,.0f} samples/s respectively.

Cumulative retained rates are {v['active_PPO_samples_s']:,.0f} active, {v['worker_effective_samples_s']:,.0f} worker-effective and {v['whole_chain_samples_s']:,.0f} whole-chain samples/s. Whole-chain excludes {v['documented_diagnostic_pause_seconds_excluded']:,.1f} documented suspension seconds through its endpoint. Denominators differ; startup, capacity and gates remain in chain time. An earlier untimestamped short bookkeeping pause remains conservatively included. Steady training uses 6144 environments with about 1.9 GiB free GPU; evaluation memory is not capacity. Keep target 8192, reserve 300 MiB, host guards/fallback and 180 continuous healthy training seconds before upward ramps. Fusion stays closed, queued work and completed videos are preserved, and the computer remains on.

## Next focused tests

{chr(10).join(f'{i+1}. {s}' for i,s in enumerate(next_tests))}

Keep the same active five-hour heartbeat, thread, notification preference and durable authority. Finish only after actual signed 5 m/s speed at least 4.95 with every matching nominal/pushed/stopped/prior gate and 32 independently seeded states per command; or signed 4 m/s at least 3.96 with every gate plus controlled native limits proving 5 is blocked. At that verified endpoint save exact evidence, safely pause training and automation, prevent restart, brief the user and leave the computer on.
'''
write(OUT / 'decision.json', decision)
write(OUT / 'evidence.json', evidence)
(OUT / 'REPORT.md').write_text(report, encoding='utf-8')
entry = f'\n## Five-hour check October 8, 09:46 UTC — {now}\n\n{summary}\n\n{decision["explanation"]}\n\nReport: {OUT / "REPORT.md"}. No production change. Passive physics/load/gradient evidence, failed-import cleanup, individual-cycle/state-response data and corrected motor-case attribution are preserved. Next focused tests:\n\n' + '\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests)) + '\n'
for p in JOURNALS:
    with p.open('a', encoding='utf-8') as f:
        f.write(entry)
for p in REFERENCES:
    text = p.read_text(encoding='utf-8')
    p.write_text(f'<!-- {MARKER} -->\n{summary}\nRead {OUT / "REPORT.md"}, then fresh ACTIVE_RUN/state. Earlier entries are preserved.\n<!-- /{MARKER} -->\n\n' + text, encoding='utf-8')

# Preserve recording/Fusion/endpoint scope; clearly archive prior comparison paths.
historical = {k: old.pop(k) for k in ('upper_comparison', 'matched_shoulder_comparison', 'upper_training_evaluation_gap') if k in old}
old.pop('automation_verification', None)
old.pop('closeout_verification', None)
old['automation_verified_active'] = False  # Replaced by the actual native verification after update.
old.update(recorded_utc=now, report=str(OUT/'REPORT.md'), latest_observation=v['owned_snapshot_utc'], latest_liveness_utc=v['owned_snapshot_utc'],
    active_run=v['run'], current_stage=live['state']['stage'], completed_main_updates=v['completed_main_updates'], new_main_updates=v['completed_main_updates'],
    recovered_main_updates=50, actual_retained_current_run_main_updates=v['actual_retained_main_updates'], partial_observed_main_updates=v['observed_partial_main_updates'],
    current_owned_operation=v['operation_path'], selected_actor=pair['checkpoint'], selected_actor_sha256=pair['sha256'], latest_matching_pair=pair,
    reviewed_speed_case_passes_both=True, current_stage_qualified=False, running_qualified=False, velocity_ceiling_proven=False,
    summary=summary, focused_tests=next_tests, next_tests=next_tests, focus=next_tests, decision=str(OUT/'decision.json'),
    latest_liveness=str(OUT/'liveness_cpu/live_handoff.json'), resumed_learning_evidence=str(OUT/'liveness_learning_verification.json'),
    upper_joint_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'), mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),
    mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'), upper_target_bias_audit=str(OUT/'final_upper/upper_target_bias_audit.json'),
    individual_upper_cycles=str(OUT/'individual_cycle_analysis.json'), upper_state_context_sensitivity=str(OUT/'state_context_sensitivity.json'),
    historical_rejected_upper_comparison=historical, synchronized_com_contact_audit=str(OUT/'closing_com_contact_audit.json'), motor_speed_audit=str(OUT/'motor_speed_audit.json'),
    diagnostic_resume=str(OUT/'diagnostic_resume.json'), previous_full_handoff_snapshot=str(OUT/'previous_NEXT_CHECK.json'),
    retained_main_totals=tot, measured_since_previous=delta,
    retained_main_throughput={k:v[k] for k in ('active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded')},
    resources=dict(last_completed_main=first_resumed, target_environments=8192, reserve_MiB=300, healthy_training_ramp_seconds=180, evaluation_memory_is_not_capacity=True),
    production_change=False, computer_on=True, history_inventory=str(OUT/'inventory.log'))
old.pop('current_speed_case_passes_both', None)
write(TRAIN/'automation/NEXT_CHECK.json', old)
prefix = before['prompt'].split('Latest dated five-hour check', 1)[0]
assert hashlib.sha256(prefix.encode()).hexdigest() == '46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
prompt = prefix + 'Latest dated five-hour check October 8, 09:46 UTC: ' + summary + '\nRead diagnostics/adaptive_check_20261008_0946/REPORT.md, decision.json, individual_cycle_analysis.json, state_context_sensitivity.json, diagnostic_validation.json, motor_speed_audit.json, final_upper and liveness_learning_verification.json, then fresh ACTIVE_RUN/state.\n\nNext focused tests:\n' + '\n'.join(f'{i+1}. {s}' for i,s in enumerate(next_tests)) + '\n\nPreserve the improving exact full learner and all rejected/failed evidence. Keep the same ACTIVE five-hour heartbeat/thread/notifications and verified 5 or qualified 4 plus causal native-limit endpoint. At the verified endpoint save, pause both, prevent restart, brief and leave the computer on. Fusion stays closed, queued work is preserved and videos are fulfilled.'
args = dict(id='humanoid-training-research-and-repair-every-four-hours', mode='update', kind='heartbeat', name=before['name'],
    prompt=prompt, rrule=before['rrule'], status='ACTIVE', targetThreadId=before['target_thread_id'])
if 'notification_policy' in before:
    args['notificationPolicy'] = before['notification_policy']
write(OUT/'automation_update_args.json', args)
print(json.dumps(dict(report=str(OUT/'REPORT.md'), retained_updates=v['actual_retained_main_updates'], reviewed_actor=605,
    current_cases_pass=True, whole_stage_pass=False, delta_samples=delta['samples'], next_tests=len(next_tests)), indent=2))
