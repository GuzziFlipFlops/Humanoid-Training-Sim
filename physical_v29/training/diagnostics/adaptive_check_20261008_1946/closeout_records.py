"""Add one fresh owned closeout to existing records without repeating the check."""
import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

live = read(OUT / 'final_owned_cpu/live_handoff.json')
v = read(OUT / 'final_owned_learning_verification.json')
pair = read(OUT / 'final_owned_gate_review.json')['latest']
upper = read(OUT / 'closing_newer_upper/upper_joint_time_audit.json')
bias = read(OUT / 'closing_newer_upper/upper_bias_audit.json')
u = next(r for r in upper['records'] if r['actor_tag'] == 'latest' and r['test'] == 'nominal' and r['command_m_s'] == 3.5)
ub = next(r for r in bias['records'] if r['test'] == 'nominal' and r['speed_m_s'] == 3.5)
assert live['state']['status'] == 'running' and live['state']['supervisor_pid'] == 39000
assert len(live['actual_worker_pids']) == 1
assert len(live['frozen_source_integrity']) == 55 and all(live['frozen_source_integrity'].values())
assert v['protected_authored_worker_files_equal'] == 53
assert v['inherited_actual_main_updates'] == 27507 and v['hash_proven_recovered_main_updates'] == 50
assert u['sha256'] == pair['sha256'] == bias['sha256']
for block in v['blocks']:
    assert all(block['full_load_proof'][k] for k in ['actor_equal', 'critic_equal', 'optimizer_equal', 'exploration_equal'])
    assert block['all_preupdate_KL_and_command_mismatch_zero']
    assert block['three_auxiliaries_batches_per_update'] == 20
assert all(v['partial_full_load_proof'][k] for k in ['actor_equal', 'critic_equal', 'optimizer_equal', 'exploration_equal'])
assert not v['diagnostic_gains_and_actors_deployed']

observations = (
    'Exact 797 nominal representatives: head yaw/pitch are gait-linked; shoulder pitch has held unequal biases and unequal swing, '
    'with roll and elbow asymmetry. The 2.5 m/s representative holds opposite shoulder biases; the 3 m/s representative slowly '
    'reverses them later in the test. At 3.5, actual shoulder targets and joints both carry held biases. Velocity traces show '
    'gait-linked peaks in these selected 50 Hz robots; 10 Hz population data cannot certify fast jitter. The four plots were '
    'visually inspected. No natural human/Unitree or optical-gaze certification is inferred.'
)
subprocess.run([sys.executable, '-X', 'utf8', str(OUT / 'record_plot_qa.py'), '--folder', 'closing_newer_upper',
                '--observations', observations], check=True)
audit = read(OUT / 'closing_newer_upper/audit_set.json')
audit.update(all_plots_visually_inspected=True, inspected_plot_count=4,
             inspection_receipt=str(OUT / 'closing_newer_upper/visual_inspection.json'),
             sampling_scope='All-command population metrics; four representative plot panels, not all-command trace QA. Full all-command nominal/pushed trace QA remains exact 794.')
write(OUT / 'closing_newer_upper/audit_set.json', audit)

inv = json.loads((OUT / 'inventory.log').read_text(encoding='utf-8-sig'))
assert inv['physical_runs'] == 40 and inv['differing_manifest_hashes'] == 0 and not inv['read_warnings']
assert inv['missing_manifest_artifacts'] == 8
inventory = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), passed=True, inventory=inv,
                 historical_missing_manifests_preserved=True, source=str(OUT / 'inventory.log'))
write(OUT / 'inventory_verification.json', inventory)

totals = lambda d: {k: sum(b[k] for b in d['blocks']) for k in ['updates', 'samples', 'active_seconds', 'worker_seconds']}
total = totals(v)
old = read(OUT.parent / 'adaptive_check_20261008_1446/closing_owned_learning_verification.json')
delta = {k: total[k] - totals(old)[k] for k in total}
delta['chain_seconds_excluding_documented_pauses'] = (v['whole_chain_measurement_endpoint_unix'] - old['whole_chain_measurement_endpoint_unix']
    - v['documented_diagnostic_pause_seconds_excluded'] + old['documented_diagnostic_pause_seconds_excluded'])
for key, denominator in [('active_samples_s', 'active_seconds'), ('worker_effective_samples_s', 'worker_seconds'),
                         ('chain_samples_s', 'chain_seconds_excluding_documented_pauses')]:
    assert delta[denominator] > 0
    delta[key] = delta['samples'] / delta[denominator]
delta['previous_learning_evidence'] = str(OUT.parent / 'adaptive_check_20261008_1446/closing_owned_learning_verification.json')

nom, push = pair['nominal_cases'][-1], pair['pushed_cases'][-1]
assert nom['command'] == push['command'] == 3.5
assert not nom['passed'] and not push['passed']
failures = {label: [dict(command=c['command'], failures=c['failures']) for c in pair[label + '_cases'] if not c['passed']]
            for label in ['nominal', 'pushed']}
f15 = [next(c for c in pair[label + '_cases'] if c['command'] == 1.5)['flight'] * 100 for label in ['nominal', 'pushed']]
last = v['blocks'][-1]
summary = (
    f"Closing owned evidence {v['owned_snapshot_utc']}: the same R40 supervisor 39000 has one actual worker and 55 verified frozen hashes. "
    f"It retains {v['completed_main_updates']:,} completed plus 50 recovered = {v['actual_retained_main_updates']:,} current-run main updates, "
    f"plus 27,507 inherited; {v['observed_partial_main_updates']} observed partial updates remain separate. "
    f"Exact actor 797, matching nominal 798/pushed 799, reaches {nom['signed_speed']:.3f}/{push['signed_speed']:.3f} m/s signed at command 3.5, "
    f"with {nom['lateral_speed']:.3f}/{push['lateral_speed']:.3f} m/s lateral drift. Both direction tests still fail. "
    f"Retained 0.15 net travel, 1.5 geometric flight and 2/3 m/s direction now fail both tests; all other saved cases pass and all robots survive. "
    f"Retained 1.5 true flight is {f15[0]:.4f}/{f15[1]:.4f}%. Improvement is nonmonotonic: preserve 794's separately recovered 2/3 m/s cases. "
    f"Four newer 797 angle/target, velocity and slow-bias plots were visually inspected; all-command population statistics are saved. "
    f"Running head RMS is {u['head_rms_deg']:.2f} degrees, but held shoulder bias and unequal swing/elbow/roll remain. "
    f"Full all-command nominal/pushed trace QA stays explicitly bound to 794; the 773 native 5 ms motor attribution stays separate. "
    f"Block 797 completed 64 genuine updates at 8,192 environments, loaded exact 794 actor/critic/Adam/exploration, had zero KL/command mismatch "
    f"and all three auxiliaries at 20 batches/update. Minimum free GPU/host memory was {last['minimum_free_gpu_MiB']:.0f}/{last['minimum_host_available_MiB']:.0f} MiB, "
    f"private growth {last['private_memory_growth_MiB']:.0f} MiB and Torch allocation growth {last['torch_allocation_growth_MiB']:.0f} MiB. "
    f"Retained rates are {v['active_PPO_samples_s']/1000:.2f}k active / {v['worker_effective_samples_s']/1000:.2f}k worker-effective / "
    f"{v['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s, with {v['documented_diagnostic_pause_seconds_excluded']:.1f}s documented pauses excluded. "
    f"No diagnostic actor, sampler, reward, gain, cap or optimizer reset entered production. The current operation is 800 at 8,192. "
    f"Inventory: 40 physical runs, zero differing hashes/read warnings; eight historical missing manifests remain documented. "
    f"Read final_owned_gate_review.json, final_owned_learning_verification.json, final_owned_cpu/live_handoff.json and closing_newer_upper, then fresh ACTIVE_RUN/state. "
    f"The same five-hour check remains active toward qualified 5 m/s or qualified 4 m/s with causal native limits blocking 5; computer on."
)
record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), summary=summary, owned_snapshot=str(OUT / 'final_owned_cpu/live_handoff.json'),
              matching_pair=pair, prior_and_current_failures=failures, retained_main_totals=total, measured_since_previous=delta,
              last_main_block=last, upper_head_RMS_deg=u['head_rms_deg'], upper_record=u, upper_bias_record=ub,
              inventory=inventory, no_production_change=True, computer_on=True)
write(OUT / 'closing_owned_addendum.json', record)

marker = 'adaptive_check_20261008_1946_closing_owned'
entry = f'\n<!-- {marker} -->\n\n{summary}\n\nEvidence: {OUT / "closing_owned_addendum.json"}.\n<!-- /{marker} -->\n'
paths = [OUT / 'REPORT.md', TRAIN / 'ITERATION_HISTORY.md', ROOT / 'docs/robot/TRAINING_JOURNAL_20261001.md',
         Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]
refs = [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),
        Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
assert all(p.is_file() and marker not in p.read_text(encoding='utf-8') for p in paths + refs)
for p in paths:
    with p.open('a', encoding='utf-8') as f:
        f.write(entry)
for p in refs:
    p.write_text(entry + '\n' + p.read_text(encoding='utf-8'), encoding='utf-8')

next_path = TRAIN / 'automation/NEXT_CHECK.json'
n = read(next_path)
n.update(recorded_utc=record['recorded_utc'], latest_observation=v['owned_snapshot_utc'], latest_liveness_utc=v['owned_snapshot_utc'],
         completed_main_updates=v['completed_main_updates'], new_main_updates=v['completed_main_updates'],
         actual_retained_current_run_main_updates=v['actual_retained_main_updates'], partial_observed_main_updates=v['observed_partial_main_updates'],
         current_owned_operation=v['operation_path'], selected_actor=pair['checkpoint'], selected_actor_sha256=pair['sha256'], latest_matching_pair=pair,
         latest_liveness=str(OUT / 'final_owned_cpu/live_handoff.json'), resumed_learning_evidence=str(OUT / 'final_owned_learning_verification.json'),
         final_owned_observation=str(OUT / 'final_owned_cpu/live_handoff.json'), final_verification=str(OUT / 'final_owned_learning_verification.json'),
         closing_summary=summary, closing_owned_addendum=str(OUT / 'closing_owned_addendum.json'),
         closing_upper=str(OUT / 'closing_newer_upper'), closing_upper_checkpoint=pair['checkpoint'], closing_upper_sha256=pair['sha256'],
         retained_main_totals=total, measured_since_previous=delta,
         retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s', 'worker_effective_samples_s', 'whole_chain_samples_s', 'documented_diagnostic_pause_seconds_excluded']},
         resources=dict(last_completed_main=last, target_environments=8192, reserve_MiB=300, healthy_training_ramp_seconds=180, evaluation_memory_is_not_capacity=True),
         inventory_verification=str(OUT / 'inventory_verification.json'))
n['summary'] += '\n\n' + summary
for key in ['focused_tests', 'next_tests', 'focus']:
    assert len(n[key]) == 3
    n[key][0] = 'Newer exact 797 also fails both retained 2/3 m/s direction tests; 794 passed them separately. Compare those fresh cases too. ' + n[key][0]
write(next_path, n)
e = read(OUT / 'evidence.json')
e['closing_owned'] = record
write(OUT / 'evidence.json', e)
g = read(OUT / 'gate_trends.json')
g['final_owned_exact_pair'] = pair
g['final_owned_nonmonotonic_note'] = '797 direction improves at 3.5 but fails both 2/3 m/s direction and retained 1.5 flight; 794 remains separately scoped.'
write(OUT / 'gate_trends.json', g)
d = read(OUT / 'decision.json')
d['closing_owned'] = dict(recorded_utc=v['owned_snapshot_utc'], checkpoint=pair['checkpoint'], production_change=False, failures=failures)
d['next_tests'] = n['next_tests']
write(OUT / 'decision.json', d)

args = read(OUT / 'automation_update_args.json')
before = tomllib.loads((OUT / 'previous_automation.toml').read_text(encoding='utf-8'))
prefix = args['prompt'].split('Latest dated five-hour check', 1)[0]
assert hashlib.sha256(prefix.encode()).hexdigest() == '46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
args['prompt'] = args['prompt'].replace('\n\nNext focused tests:\n', '\n\n' + summary + '\n\nNext focused tests:\n', 1)
args['prompt'] = args['prompt'].replace('1. Preserve the current R40 full state', '1. Newer exact 797 also fails both retained 2/3 m/s direction tests; 794 passed them separately. Compare those fresh cases too. Preserve the current R40 full state', 1)
args['prompt'] = args['prompt'].replace('Keep same ACTIVEfive-hour heartbeat/thread/notifications,verified5 or qualified4+causal native limits endpoint. Save,pause both,prevent restart,brief,computeron only at that verified endpoint. Preserve full learned state,forward head,fulfilled videos and closedFusion queued work.',
    'Keep the same ACTIVE five-hour heartbeat, thread and notification preference. Only qualified 5 m/s, or qualified 4 m/s with causal native limits blocking 5, ends training. At that verified endpoint save, pause training and automation, prevent restart and brief the user with the computer on. Preserve the full learner, forward head, fulfilled videos and closed Fusion queued work.')
assert args['rrule'] == before['rrule'] == 'FREQ=HOURLY;INTERVAL=5'
assert args['targetThreadId'] == before['target_thread_id']
write(OUT / 'automation_update_args.json', args)
print(json.dumps(dict(summary=summary, retained_main_samples=total['samples'], interval=delta,
                      inventory=inv, upper3_5=dict(head_RMS=u['head_rms_deg'],
                        shoulder_ranges=[u['joints'][j]['median_per_robot_p95_p05_range_deg'] for j in ['left_shoulder_pitch','right_shoulder_pitch']],
                        shoulder_slow_means=[ub['joints'][j]['slow_bias_mean_deg'] for j in ['left_shoulder_pitch','right_shoulder_pitch']])), indent=2))
