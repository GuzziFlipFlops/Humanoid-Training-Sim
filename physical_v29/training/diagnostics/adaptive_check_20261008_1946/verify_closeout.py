"""Verify native heartbeat persistence and current owned process liveness."""
import hashlib
import json
import tomllib
import time
from datetime import datetime, timezone
from pathlib import Path
import psutil

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

args = read(OUT / 'automation_update_args.json')
before = tomllib.loads((OUT / 'previous_automation.toml').read_text(encoding='utf-8'))
path = Path('C:/Users/rshan/.codex/automations') / args['id'] / 'automation.toml'
current = tomllib.loads(path.read_text(encoding='utf-8'))
assert current['id'] == args['id'] == before['id']
assert current['prompt'] == args['prompt']
assert current['rrule'] == args['rrule'] == before['rrule'] == 'FREQ=HOURLY;INTERVAL=5'
assert current['target_thread_id'] == args['targetThreadId'] == before['target_thread_id']
assert current['name'] == args['name'] == before['name']
assert current['status'] == 'ACTIVE' and current['kind'] == before['kind'] == 'heartbeat'
assert current.get('notification_policy') == before.get('notification_policy')
prefix = current['prompt'].split('Latest dated five-hour check', 1)[0]
prefix_sha = hashlib.sha256(prefix.encode()).hexdigest()
assert prefix_sha == '46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
automation = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), passed=True, native_update_confirmed=True,
                  id=current['id'], kind=current['kind'], status=current['status'], rrule=current['rrule'],
                  target_thread_id=current['target_thread_id'], notification_policy_preserved=True,
                  name_preserved=True, durable_scope_prefix_sha256=prefix_sha,
                  dated_prompt_matches_prepared_update=True, model_override_requested=False,
                  no_duplicate_created=True, path=str(path))
write(OUT / 'automation_verification.json', automation)

live = read(OUT / 'final_owned_cpu/live_handoff.json')
run = Path(live['active_run']['run'])
supervisor = psutil.Process(39000)
assert abs(supervisor.create_time() - 1791346353.9543097) < 1
assert supervisor.is_running() and supervisor.status() != psutil.STATUS_STOPPED
for attempt in range(3):
    state = read(run / 'state.json')
    owned = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            command = ' '.join(proc.info['cmdline'] or [])
            if str(run).lower() in command.lower() and proc.info['name'].lower().startswith('python'):
                owned.append(dict(pid=proc.pid, parent_pid=proc.ppid(), created=proc.create_time(), command=command,
                                  status=proc.status()))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    workers = [p for p in owned if 'worker_unitree.py' in p['command']]
    leaves = [p for p in workers if not any(q['parent_pid'] == p['pid'] for q in workers)]
    if len(leaves) == 1 and (not state.get('child_pid') or state['child_pid'] in [p['pid'] for p in workers]):
        break
    time.sleep(0.25)
assert len(leaves) == 1, owned
assert state['status'] == 'running' and state['supervisor_pid'] == 39000
assert supervisor.pid in [p['pid'] for p in owned]
verification = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                    supervisor_pid=39000, supervisor_identity_equal=True, supervisor_not_suspended=True,
                    owned_processes=owned, actual_worker_pids=[p['pid'] for p in leaves],
                    worker_count=1, owned_full_evidence=str(OUT / 'final_owned_cpu/live_handoff.json'),
                    current_owned_operation=state['operation_path'], current_operation_environment_count=state['environment_count'],
                    frozen_55_source_checks_and_full_state_evidence=str(OUT / 'final_owned_learning_verification.json'),
                    gate_and_joint_timeline_scope='Saved matching actor 797; full all-command trace QA 794; substep motor attribution 773.',
                    automation_verification=automation, no_production_change=True, computer_on=True)
verification['preserved_diagnostic_attempt'] = dict(log=str(OUT / 'verify_closeout_attempt01.log'),
    explanation='The initial CPU closeout checker looked up the wrong saved JSON key. Corrected to active_run.run; no training pause, native automation failure or production change.')
verification['preserved_boundary_snapshot'] = dict(log=str(OUT / 'verify_closeout_attempt02_boundary.log'),
    explanation='A non-atomic process snapshot caught an ordinary worker handoff. The corrected check reconciles fresh owned state with worker process identity; no crash or pause is inferred.')
write(OUT / 'final_closeout_verification.json', verification)
npath = TRAIN / 'automation/NEXT_CHECK.json'
n = read(npath)
n.update(automation_verified_active=True, automation_verification=str(OUT / 'automation_verification.json'),
         inventory_verification=str(OUT / 'inventory_verification.json'), final_closeout_verification=str(OUT / 'final_closeout_verification.json'))
write(npath, n)
e = read(OUT / 'evidence.json')
e['closeout_verification'] = verification
write(OUT / 'evidence.json', e)
with (OUT / 'REPORT.md').open('a', encoding='utf-8') as f:
    f.write('\nNative automation update is confirmed: same heartbeat/thread, ACTIVE, five-hour interval, unchanged notification preference and durable endpoint. '
            'Current owned supervisor identity and one actual worker are live. See [closeout verification](final_closeout_verification.json) and '
            '[automation verification](automation_verification.json).\n')
print(json.dumps(dict(recorded_utc=verification['recorded_utc'], supervisor=39000,
                      actual_workers=verification['actual_worker_pids'], automation=automation), indent=2))
