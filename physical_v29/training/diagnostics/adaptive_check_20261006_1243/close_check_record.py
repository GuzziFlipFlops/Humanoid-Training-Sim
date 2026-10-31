"""Save the final CPU-only observation without touching frozen training sources."""
import json
from pathlib import Path

out = Path(__file__).resolve().parent
snapshot = json.loads((out / 'final_cpu/live_handoff.json').read_text(encoding='utf-8'))
inventory = json.loads((out / 'history_inventory_stdout.json').read_text(encoding='utf-8-sig'))
state = snapshot['state']
assert state['status'] == 'running' and state['supervisor_pid'] == 35688
assert len(snapshot['actual_worker_pids']) == 1
assert len(snapshot['frozen_source_integrity']) == 55
assert inventory['differing_manifest_hashes'] == 0 and not inventory['read_warnings']

next_path = out.parents[1] / 'automation/NEXT_CHECK.json'
next_check = json.loads(next_path.read_text(encoding='utf-8'))
next_check['recorded_utc'] = snapshot['recorded_utc']
next_check['latest_observation'] = str(out / 'final_cpu/live_handoff.json')
next_check['new_main_updates'] = state['total_training_iterations']
next_check['actual_retained_current_run_main_updates'] = state['total_training_iterations'] + next_check['recovered_main_updates']
next_check['latest_verified_state'] = state
next_check['history_inventory'] = inventory
next_check['selected_actor']['scope'] = 'Exact132 nominal133/pushed134. These saved gates never qualify newer learned actors or current partial operation141.'
next_path.write_text(json.dumps(next_check, indent=2) + '\n', encoding='utf-8')

verification = {
    'recorded_utc': snapshot['recorded_utc'],
    'same_supervisor_pid': state['supervisor_pid'],
    'actual_worker_pids': snapshot['actual_worker_pids'],
    'status': state['status'],
    'stage': state['stage'],
    'operation': state['operation_path'],
    'completed_operation_main_updates': state['total_training_iterations'],
    'verified_recovered_updates': next_check['recovered_main_updates'],
    'current_run_retained_updates': next_check['actual_retained_current_run_main_updates'],
    'authored_frozen_hashes_verified': 55,
    'inventory': inventory,
    'automation_verification': str(out / 'automation_verification.json'),
    'scope': 'CPU liveness and provenance. Gates reviewed in this check bind exact132, not newer partial actors. Startup memory is not steady training memory.',
}
(out / 'final_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
report = out / 'REPORT.md'
text = report.read_text(encoding='utf-8')
marker = '\n## Final CPU observation\n'
assert marker not in text
text += marker + f"\nAt {snapshot['recorded_utc']}, the same supervisor35688 owns one worker, all55 frozen hashes verify, and operation141 is starting another6144-environment main block at3.5m/s. Completed-operation main updates are{state['total_training_iterations']}; adding33 verified recovered gives{next_check['actual_retained_current_run_main_updates']} retained R39 updates, plus16936 inherited. The independent132 gates reported above do not qualify these newer actors. [Final observation](final_cpu/live_handoff.json), [verification](final_verification.json).\n\nCPU history inventory records39 physical runs and2085 operations, zero differing manifest hashes and no read warnings. Eight historical missing manifests remain documented. The verified same heartbeat is ACTIVE at a five-hour interval; no duplicate, shutdown or endpoint pause was performed.\n"
report.write_text(text, encoding='utf-8')
print(json.dumps(verification, indent=2))
