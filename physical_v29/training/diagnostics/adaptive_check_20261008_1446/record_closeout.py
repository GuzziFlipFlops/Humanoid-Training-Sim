"""Append final owned liveness without attributing old gates to a newer actor."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
write = lambda p, d: Path(p).write_text(json.dumps(d, indent=2) + '\n', encoding='utf-8')
v = read(OUT / 'closing_owned_learning_verification.json')
live = read(OUT / 'final_owned_cpu/live_handoff.json')
last = v['blocks'][-1]
assert live['state']['status'] == 'running' and live['state']['supervisor_pid'] == 39000
assert len(live['actual_worker_pids']) == 1 and len(live['frozen_source_integrity']) == 55
assert all(live['frozen_source_integrity'].values())
assert last['environments'] == 8192 and last['updates'] == 64
assert last['all_preupdate_KL_and_command_mismatch_zero']
assert last['three_auxiliaries_batches_per_update'] == 20
assert all(last['full_load_proof'][k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
note = (f"Closing owned evidence {v['owned_snapshot_utc']}: same R40 supervisor 39000, one actual worker and 55 verified frozen hashes. "
    f"It retains {v['completed_main_updates']:,} completed + 50 recovered = {v['actual_retained_main_updates']:,} current-run main updates, plus 27,507 inherited. "
    "Both resumed blocks 719 and 722 complete 64 genuine updates at 8,192 environments with exact actor/critic/Adam/exploration, zero pre-update KL/command mismatch and all three auxiliaries at 20 batches/update. "
    f"Last block minimum GPU {last['minimum_free_gpu_MiB']} MiB, host {last['minimum_host_available_MiB']} MiB, private growth {last['private_memory_growth_MiB']} MiB and Torch allocation growth {last['torch_allocation_growth_MiB']} MiB. "
    "The snapshot is evaluating newer actor 722; its matching pair is incomplete and no qualification is inferred. The reviewed gate/upper pair remains exact 719, all-command traces and motor audit remain exact 716. "
    "Current selected next training count is 8,192; evaluation's 320 robots/free memory do not establish training capacity. Read closing_owned_learning_verification.json and final_owned_cpu/live_handoff.json, then fresh ACTIVE_RUN/state. "
    "No production source/reward/gain/cap/actor/optimizer change or diagnostic actor is deployed. Same active five-hour endpoint; computer on.")
with (OUT / 'REPORT.md').open('a', encoding='utf-8') as f:
    f.write('\n## Closing owned verification\n\n' + note + '\n')
for path in [TRAIN / 'ITERATION_HISTORY.md', ROOT / 'docs/robot/TRAINING_JOURNAL_20261001.md',
        Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with path.open('a', encoding='utf-8') as f:
        f.write('\n' + note + '\n')
evidence_path = OUT / 'evidence.json'
evidence = read(evidence_path)
evidence.update(closing_owned_summary=note, closing_owned_evidence=str(OUT / 'closing_owned_learning_verification.json'))
write(evidence_path, evidence)
next_path = TRAIN / 'automation/NEXT_CHECK.json'
next_data = read(next_path)
previous = read(OUT / 'previous_NEXT_CHECK.json')
old = read(previous['resumed_learning_evidence'])
keys = ['updates','samples','active_seconds','worker_seconds']
total = {k: sum(b[k] for b in v['blocks']) for k in keys}
delta = {k: total[k] - sum(b[k] for b in old['blocks']) for k in keys}
delta['chain_seconds_excluding_documented_pauses'] = (v['whole_chain_measurement_endpoint_unix'] - old['whole_chain_measurement_endpoint_unix']
    - v['documented_diagnostic_pause_seconds_excluded'] + old['documented_diagnostic_pause_seconds_excluded'])
for key, denominator in [('active_samples_s','active_seconds'),('worker_effective_samples_s','worker_seconds'),('chain_samples_s','chain_seconds_excluding_documented_pauses')]:
    delta[key] = delta['samples'] / delta[denominator]
next_data.update(latest_observation=v['owned_snapshot_utc'], latest_liveness_utc=v['owned_snapshot_utc'],
    completed_main_updates=v['completed_main_updates'], new_main_updates=v['completed_main_updates'],
    actual_retained_current_run_main_updates=v['actual_retained_main_updates'], partial_observed_main_updates=0,
    current_owned_operation=v['operation_path'], latest_liveness=str(OUT / 'final_owned_cpu/live_handoff.json'),
    resumed_learning_evidence=str(OUT / 'closing_owned_learning_verification.json'), closing_summary=note,
    summary=next_data['summary'] + '\n\n' + note, retained_main_totals=total, measured_since_previous=delta,
    retained_main_throughput={k: v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
    resources=dict(last_completed_main=last,target_environments=8192,reserve_MiB=300,healthy_training_ramp_seconds=180,evaluation_memory_is_not_capacity=True),
    final_owned_observation=str(OUT / 'final_owned_cpu/live_handoff.json'), final_verification=str(OUT / 'closing_owned_learning_verification.json'))
write(next_path, next_data)
args_path = OUT / 'automation_update_args.json'
args = read(args_path)
args['prompt'] += '\n\n' + note
write(args_path, args)
print(note)
