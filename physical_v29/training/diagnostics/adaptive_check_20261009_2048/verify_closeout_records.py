"""Verify the native heartbeat update and CPU inventory after evidence publication."""
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
AUTO = Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
write = lambda p, d: Path(p).write_text(json.dumps(d, indent=2, allow_nan=False) + '\n', encoding='utf-8')
before = tomllib.loads((OUT / 'previous_automation.toml').read_text(encoding='utf-8-sig'))
current = tomllib.loads(AUTO.read_text(encoding='utf-8-sig'))
args = read(OUT / 'automation_update_args.json')
for key in ['id', 'kind', 'name', 'rrule', 'target_thread_id', 'created_at']:
    assert current[key] == before[key]
assert current['status'] == 'ACTIVE' and current['prompt'] == args['prompt']
assert current['rrule'] == 'FREQ=HOURLY;INTERVAL=5'
prefix_sha = hashlib.sha256(current['prompt'].split('Latest dated five-hour check', 1)[0].encode()).hexdigest()
assert prefix_sha == '46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
pref_keys = [k for k in set(before) | set(current)
             if 'notification' in k or k in ['model', 'reasoning_effort', 'thinking']]
assert all(current.get(k) == before.get(k) and (k in current) == (k in before) for k in pref_keys)
records = read(OUT / 'records_receipt.json')
marker = '<!-- adaptive_check_20261009_2048_record -->'
assert len(records['records']) == 5
assert all(Path(r['path']).read_text(encoding='utf-8-sig').count(marker) == 1 for r in records['records'])
inventory = read(TRAIN / 'history/RUN_INVENTORY.json')
runs = inventory['physical_runs']
different = sum(x['actual_sha256'] is not None and not x['matches']
                for r in runs for x in r['frozen_source_verification'])
missing = sum(x['actual_sha256'] is None for r in runs for x in r['frozen_source_verification'])
assert len(runs) == 40 and different == 0 and not inventory['read_warnings']
assert inventory['generated_utc'] >= records['recorded_utc']
handoff = read(TRAIN / 'automation/NEXT_CHECK.json')
assert handoff['current_check'] == str(OUT)
assert handoff['selected_actor_sha256'] == read(OUT / 'closing_gate_review.json')['latest']['sha256']
assert not handoff['running_qualified'] and not handoff['velocity_ceiling_proven']
receipt = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), automation_id=current['id'],
    status=current['status'], same_thread=True, same_five_hour_cadence=True,
    durable_scope_and_endpoint_preserved=True, durable_prefix_sha256=prefix_sha,
    notification_and_model_preferences_preserved=True,
    stored_preference_fields={k: current.get(k) for k in pref_keys},
    preference_scope='Stored fields and their absence preserved; global/thread model settings are not inferred from absent automation fields.',
    prompt_matches_exact_update=True, record_marker_once_in_five_files=True,
    physical_runs=len(runs), physical_operations=sum(len(r['operations']) for r in runs),
    differing_hashes=different, missing_historical_manifest_artifacts=missing,
    read_warnings=inventory['read_warnings'], inventory_generated_utc=inventory['generated_utc'],
    computer_on=True)
write(OUT / 'automation_inventory_verification.json', receipt)
handoff.update(automation_verified_active=True,
    automation_verification=str(OUT / 'automation_inventory_verification.json'),
    inventory_verification=str(OUT / 'automation_inventory_verification.json'))
write(TRAIN / 'automation/NEXT_CHECK.json', handoff)
records.update(native_update_pending=False,
    native_update_verification=str(OUT / 'automation_inventory_verification.json'))
write(OUT / 'records_receipt.json', records)
print(json.dumps(receipt, indent=2))
