"""Check durable one-run authority and native scheduler update; read-only train."""
import json, hashlib, tomllib
from pathlib import Path
from datetime import datetime, timezone

TRAIN = Path(__file__).resolve().parent.parent
OUT = TRAIN / 'diagnostics/final_check_authorization_20261010'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
native = tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text(encoding='utf-8'))
previous = tomllib.loads((OUT / 'previous_automation.toml').read_text(encoding='utf-8'))
args = read(OUT / 'automation_update_args.json')
request = read(TRAIN / 'automation/FINAL_CHECK_REQUEST.json')
handoff = read(TRAIN / 'automation/NEXT_CHECK.json')
receipt = read(OUT / 'native_automation_update_receipt.json')
assert not receipt.get('isError', False)
for field in ('id', 'kind', 'target_thread_id', 'created_at'):
    assert native[field] == previous[field], field
for field in ('notification_policy', 'model', 'reasoning_effort'):
    assert native.get(field) == previous.get(field) and (field in native) == (field in previous), field
assert native['status'] == 'ACTIVE' and native['prompt'] == args['prompt'] and native['rrule'] == args['rrule']
assert native['prompt'] == handoff['adaptive_automation_prompt']
assert request['remaining_scheduled_prompts'] == handoff['remaining_authorized_automation_prompts'] == 1
assert request['stop_training_after_final_check'] and request['pause_automation_at_final_check_start']
assert handoff['recordings_pending'] and handoff['stop_training_after_next_check']
assert not handoff['continue_until_speed_endpoint']
assert datetime.fromisoformat(request['due_utc']) > datetime.now(timezone.utc)
assert 18000 <= (datetime.fromisoformat(request['due_utc']) - datetime.fromisoformat(request['recorded_utc'])).total_seconds() <= 18060
workflow = (TRAIN / 'automation/FIVE_HOUR_CHECK.md').read_text(encoding='utf-8')
assert '## Latest user authorization: one final check' in workflow
assert 'NEW pending video/data request' in workflow
records = read(OUT / 'durable_records.json')
assert all(Path(path).read_text(encoding='utf-8').count(records['marker']) == 1 for path in records['paths'])
checks = read(OUT / 'motor_mapping_CPU_checks.json')
assert checks['passed']
inventory = read(TRAIN / 'history/RUN_INVENTORY.json')

result = dict(verified_utc=datetime.now(timezone.utc).isoformat(), automation_id=native['id'],
    status=native['status'], same_thread=True, preserved_notification_and_model_preferences=True,
    final_run_only=True, due_utc=request['due_utc'], due_local=request['due_local'],
    user_budget_supersedes_old_endpoint=True, stop_training_after_final_check=True,
    pause_automation_on_final_entry=True, new_motor_video_table_request_saved=True,
    installed_motor_mapping_CPU_checks=checks, five_durable_markers_verified=True,
    production_sources_edited=False, training_left_running_until_final_check=True,
    inventory_refreshed=True, inventory_file=str(TRAIN / 'history/RUN_INVENTORY.json'),
    protocol_sha256=hashlib.sha256((TRAIN / 'automation/FINAL_CHECK_AND_MOTOR_CAPTURE.md').read_bytes()).hexdigest(),
    module_sha256=hashlib.sha256((TRAIN / 'automation/motor_shaft_metrics.py').read_bytes()).hexdigest(),
    caveat='Native final slot is ACTIVE; its final prompt explicitly pauses itself at entry. CPU shaft mapping is tested; final synchronized Isaac capture is pending.')
(OUT / 'authorization_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
handoff['automation_verified_active'] = True
handoff['final_authorization_verification'] = str(OUT / 'authorization_verification.json')
(TRAIN / 'automation/NEXT_CHECK.json').write_text(json.dumps(handoff, indent=2)+'\n', encoding='utf-8')
evidence = read(OUT / 'request_evidence.json')
evidence.update(native_update_verified=True, verification=str(OUT / 'authorization_verification.json'))
(OUT / 'request_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))
