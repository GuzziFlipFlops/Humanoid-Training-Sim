"""Final owned observation; do not chase or certify a newer gate pair."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
write = lambda p, value: Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
spec = importlib.util.spec_from_file_location('closing_owned_recorder', TRAIN / 'record_owned_observation_handoff.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = OUT / 'final_liveness_cpu'
module.main()
live = read(module.OUT / 'live_handoff.json')
state = live['state']
assert state['status'] == 'running' and state['supervisor_pid'] == 39000
assert len(live['actual_worker_pids']) == 1
assert len(live['frozen_source_integrity']) == 55 and all(live['frozen_source_integrity'].values())
attempt = next(a for a in reversed(state['attempts']) if a['completed'] and a['operation'] == 'train'
               and re.search(r'_run_\d+_train_\d+', Path(a['folder']).name))
folder = Path(attempt['folder'])
rows = [json.loads(line) for line in (folder / 'progress.jsonl').read_text().splitlines()]
proof = read(folder / 'full_load_proof.json')
assert all(proof[k] for k in ['actor_equal', 'critic_equal', 'optimizer_equal', 'exploration_equal'])
assert all(r['ppo']['pre_update_kl'] == 0 and r['behavior']['action_command_difference'] == 0 for r in rows)
assert all(r['retention']['batches'] == 20 and r['locomotion_retention']['batches'] == 20
           and r['upper_refinement']['batches'] == 20 for r in rows)
earlier = read(OUT / 'closeout_learning_verification.json')
extra_attempts = [a for a in state['attempts'] if a['operation'] == 'train'
    and re.search(r'_run_\d+_train_\d+', Path(a['folder']).name)
    and Path(a['folder']) not in [Path(b['folder']) for b in earlier['blocks']]]
assert all(a['completed'] for a in extra_attempts), 'Any new failed main block needs a separate recovery audit'
receipt = dict(recorded_utc=live['recorded_utc'], supervisor_pid=39000, actual_worker_pids=live['actual_worker_pids'],
    frozen_hashes=55, all_frozen_hashes_equal=True, operation_path=state['operation_path'],
    completed_main_updates=state['total_training_iterations'], recovered_main_updates=50,
    actual_retained_main_updates=state['total_training_iterations'] + 50, inherited_main_updates=27507,
    current_checkpoint=state['checkpoint'], current_checkpoint_sha256=hashlib.sha256(Path(state['checkpoint']).read_bytes()).hexdigest(),
    latest_completed_main=dict(folder=str(folder), updates=len(rows), samples=sum(r['num_envs'] * 24 for r in rows),
        environments=rows[0]['num_envs'], full_load_proof=proof, all_KL_and_command_checks_zero=True,
        three_auxiliaries_batches_per_update=20,
        minimum_free_gpu_MiB=min(r['gpu']['free_mb'] for r in rows),
        minimum_host_available_MiB=min(r['gpu']['ram_available_mb'] for r in rows),
        private_growth_MiB=rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb'],
        torch_allocation_growth_MiB=rows[-1]['gpu']['torch_allocated_mb']-rows[0]['gpu']['torch_allocated_mb']),
    current_gpu=live['gpu'], current_host=live['system_memory'],
    latest_reviewed_gate_actor='1064', mandatory_upper_and_population_motor_actor='1061',
    newer_checkpoint_qualification_inferred=False, diagnostic_actors_deployed=False, endpoint=False,
    samples_and_timing_report=str(OUT / 'closeout_learning_verification.json'),
    scope='Final process, source, saved-checkpoint and completed-main receipt. Full timing totals remain dated in the closeout verification; evaluation memory is not steady training capacity.')
write(OUT / 'final_liveness.json', receipt)
handoff = read(TRAIN / 'automation/NEXT_CHECK.json')
handoff.update(final_owned_receipt=str(OUT / 'final_liveness.json'), final_owned_observation_utc=receipt['recorded_utc'],
    current_checkpoint=receipt['current_checkpoint'], current_checkpoint_sha256=receipt['current_checkpoint_sha256'],
    current_owned_operation=receipt['operation_path'], latest_owned_saved_main_updates=receipt['actual_retained_main_updates'],
    latest_owned_completed_main_updates=receipt['completed_main_updates'], latest_owned_main_receipt=receipt['latest_completed_main'])
write(TRAIN / 'automation/NEXT_CHECK.json', handoff)
evidence = read(OUT / 'evidence.json')
evidence['final_owned_receipt'] = receipt
write(OUT / 'evidence.json', evidence)
report = (OUT / 'REPORT.md').read_text(encoding='utf-8')
report += (f"\nFinal owned observation {receipt['recorded_utc']}: same supervisor, one actual worker and 55 unchanged hashes. "
    f"Saved main count is {receipt['completed_main_updates']:,} completed plus 50 recovered = {receipt['actual_retained_main_updates']:,}, plus 27,507 inherited. "
    f"Latest completed block {folder.name} has {len(rows)} genuine updates at {rows[0]['num_envs']:,}, exact full loading and zero KL/command mismatch; all three auxiliaries remain at 20 batches/update. "
    "The reviewed gate pair remains 1064 and full upper/motor timeline QA remains 1061; no newer qualification is inferred. [Final owned receipt](final_liveness.json).\n")
(OUT / 'REPORT.md').write_text(report, encoding='utf-8')
print(json.dumps({k: receipt[k] for k in ['recorded_utc', 'completed_main_updates', 'actual_retained_main_updates',
    'operation_path', 'current_checkpoint', 'latest_completed_main']}, indent=2))
