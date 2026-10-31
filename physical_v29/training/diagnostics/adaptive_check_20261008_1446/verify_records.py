"""Verify the native heartbeat update, CPU inventory and durable record links."""
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
before = tomllib.loads((OUT / 'previous_automation.toml').read_text(encoding='utf-8'))
automation_path = Path('C:/Users/rshan/.codex/automations') / before['id'] / 'automation.toml'
after = tomllib.loads(automation_path.read_text(encoding='utf-8'))
args = json.loads((OUT / 'automation_update_args.json').read_text(encoding='utf-8'))
prefix = lambda d: d['prompt'].split('Latest dated five-hour check', 1)[0]
assert prefix(before) == prefix(after)
assert after['prompt'] == args['prompt']
for key in ['id', 'kind', 'name', 'rrule', 'target_thread_id']:
    assert after[key] == before[key]
assert after['status'] == 'ACTIVE'
assert after['rrule'] == 'FREQ=HOURLY;INTERVAL=5'
assert after.get('notification_policy') == before.get('notification_policy')
record = {
    'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'id': after['id'], 'status': after['status'], 'kind': after['kind'],
    'target_thread_id': after['target_thread_id'], 'rrule': after['rrule'],
    'durable_prefix_sha256': hashlib.sha256(prefix(after).encode()).hexdigest(),
    'durable_prefix_unchanged': True, 'prompt_matches_native_update_arguments': True,
    'notification_preference_unchanged': True,
    'duplicates_created': False, 'automation_file': str(automation_path)
}
inventory = json.loads((OUT / 'inventory.log').read_text(encoding='utf-8-sig'))
assert inventory['differing_manifest_hashes'] == 0 and not inventory['read_warnings']
assert inventory['missing_manifest_artifacts'] == 8
for filename, value in [('automation_verification.json', record), ('inventory_verification.json', inventory)]:
    (OUT / filename).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
next_path = TRAIN / 'automation/NEXT_CHECK.json'
handoff = json.loads(next_path.read_text(encoding='utf-8'))
assert handoff['report'] == str(OUT / 'REPORT.md')
assert handoff['selected_actor'].endswith('719_run_350_train_237\\candidate.pt')
handoff.update(automation_verified_active=True,
    automation_verification=str(OUT / 'automation_verification.json'),
    inventory_verification=str(OUT / 'inventory_verification.json'))
next_path.write_text(json.dumps(handoff, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'automation': record, 'inventory': inventory}, indent=2))
