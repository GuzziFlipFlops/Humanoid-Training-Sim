"""CPU-only final verification of owned continuation and saved records."""
import json, time
from pathlib import Path
from datetime import datetime, timezone
import psutil

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

pause = read(OUT / 'diagnostic_pause.json')
resume = read(OUT / 'diagnostic_resume.json')
run = Path(read(TRAIN / 'ACTIVE_RUN.json')['run'])
state = read(run / 'state.json')
sup = psutil.Process(state['supervisor_pid'])
assert sup.pid == 39000 and sup.create_time() == pause['supervisor_create_time']
assert sup.status() != psutil.STATUS_STOPPED and state['status'] == 'running'
deadline = time.monotonic() + 30
while True:
    state = read(run / 'state.json')
    workers = [p for p in psutil.process_iter(['cmdline'])
               if any(Path(s).name == 'worker_unitree.py' for s in p.info['cmdline'] or [])]
    if workers or time.monotonic() >= deadline:
        break
    assert state['status'] == 'running' and sup.status() != psutil.STATUS_STOPPED
    time.sleep(1)
leaves = {p.pid for p in workers} - {p.ppid() for p in workers}
assert len(leaves) == 1
actual = next(p for p in workers if p.pid in leaves)
assert all(Path(p.cmdline()[p.cmdline().index('--output') + 1]).resolve()
           == Path(state['operation_path']).resolve() for p in workers)
diagnostic_names = {'shoulder_candidate.py', 'diverse_eval.py', 'run_matched.py'}
assert not any(any(Path(s).name in diagnostic_names for s in p.info['cmdline'] or [])
               for p in psutil.process_iter(['cmdline']))
live = read(OUT / 'final_cpu/live_handoff.json')
assert len(live['frozen_source_integrity']) == 55 and all(live['frozen_source_integrity'].values())
verification = read(OUT / 'final_learning_verification.json')
blocks = [b for b in verification['blocks'] if b['full_load_proof']['source_sha256'] == pause['checkpoint_sha256']]
assert len(blocks) == 1
block = blocks[0]
assert block['updates'] == 64 and block['all_preupdate_KL_and_command_mismatch_zero']
assert block['three_auxiliaries_batches_per_update'] == 20
inventory = read(OUT / 'inventory.log')
assert inventory['differing_manifest_hashes'] == 0 and not inventory['read_warnings']
write(OUT / 'inventory_verification.json', inventory)
automation = read(OUT / 'automation_verification.json')
assert automation['status'] == 'ACTIVE' and automation['five_hour_cadence'] and automation['same_thread']
assert automation['durable_scope_endpoint_prefix_unchanged']
for file in [TRAIN / 'ITERATION_HISTORY.md', ROOT / 'docs/robot/TRAINING_JOURNAL_20261001.md',
             Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    assert 'Five-hour check October8 04:46 UTC' in file.read_text(encoding='utf-8')
record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), supervisor=39000,
    actual_worker=actual.pid, worker_launcher_pids=[p.pid for p in workers if p.pid not in leaves],
    operation=state['operation_path'], status=state['status'], sources_verified=55,
    resumed_completed_block=block['folder'], resumed_full_state_exact=True,
    resumed_updates=64, resumed_zero_KL_and_command_mismatch=True,
    resumed_memory={k: block[k] for k in ['minimum_free_gpu_MiB', 'minimum_host_available_MiB',
        'private_memory_growth_MiB', 'torch_allocation_growth_MiB']},
    retained_main_snapshot=verification['actual_retained_main_updates'], inherited=27507,
    diagnostic_pause_seconds=resume['pause_seconds'], inventory=inventory,
    native_automation_active_same_thread_5h=True, production_change=False, computer_on=True,
    reviewed_matching_pair=read(OUT / 'final_gate_review.json')['latest']['checkpoint'],
    qualification_scope='Saved matching actor only; newer partial learning remains unqualified.')
write(OUT / 'closeout_verification.json', record)
with (OUT / 'REPORT.md').open('a', encoding='utf-8') as f:
    f.write('\n## Final continuation verification\n\n' + json.dumps(record, indent=2) + '\n')
n = read(TRAIN / 'automation/NEXT_CHECK.json')
n.update(automation_verified_active=True, automation_verification=str(OUT / 'automation_verification.json'),
    closeout_verification=str(OUT / 'closeout_verification.json'), history_inventory=str(OUT / 'inventory_verification.json'),
    current_owned_operation=state['operation_path'])
write(TRAIN / 'automation/NEXT_CHECK.json', n)
print(json.dumps(record, indent=2))
