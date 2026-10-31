"""CPU-only verification and local record of the user-authorized heartbeat update."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[4]
RECORD = Path(__file__).resolve().parent
AUTOMATION = Path(r'C:\Users\rshan\.codex\automations\humanoid-training-research-and-repair-every-four-hours\automation.toml')
AUTOMATION_ID = 'humanoid-training-research-and-repair-every-four-hours'
THREAD = '01a08b12-e85a-7f73-8677-315c708cd9f7'


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify():
    before = tomllib.loads((RECORD / 'automation_before.toml').read_text(encoding='utf-8-sig'))
    current = tomllib.loads(AUTOMATION.read_text(encoding='utf-8-sig'))
    assert current['id'] == AUTOMATION_ID
    assert current['kind'] == before['kind'] == 'heartbeat'
    assert current['status'] == 'ACTIVE'
    assert current['rrule'] == 'FREQ=HOURLY;INTERVAL=3'
    assert current['target_thread_id'] == before['target_thread_id'] == THREAD
    assert current.get('notification_policy') == before.get('notification_policy')
    for marker in ('NEXT_CHECK.json', 'automation_update', '5 m/s', 'velocity ceiling',
                   'leave the computer on', '300 MiB', '180-second', 'GPT-6.1 Sol', 'max reasoning'):
        assert marker in current['prompt'], marker
    assert current['name'] == 'Humanoid training — adaptive three-hour check'

    matching = []
    for path in AUTOMATION.parent.parent.glob('*/automation.toml'):
        item = tomllib.loads(path.read_text(encoding='utf-8-sig'))
        if 'humanoid' in item.get('id', '').lower() or 'humanoid' in item.get('name', '').lower():
            matching.append({'id': item['id'], 'status': item['status'], 'name': item['name']})
    active = [item for item in matching if item['status'] == 'ACTIVE']
    assert [item['id'] for item in active] == [AUTOMATION_ID], active

    pointer = read_json(ROOT / 'physical_v29/training/ACTIVE_RUN.json')
    state = read_json(Path(pointer['run']) / 'state.json')
    next_check = read_json(ROOT / 'physical_v29/training/automation/NEXT_CHECK.json')
    assert next_check['goal_actual_speed_m_s'] == 5.0
    assert state['status'] == 'running', state['status']

    (RECORD / 'automation_after.toml').write_bytes(AUTOMATION.read_bytes())
    result = {
        'passed': True, 'observed_utc': datetime.now(timezone.utc).isoformat(),
        'automation_id': AUTOMATION_ID, 'status': current['status'], 'interval_hours': 3,
        'thread_preserved': True, 'notification_preference_preserved': True,
        'prompt_sha256': hashlib.sha256(current['prompt'].encode()).hexdigest(),
        'same_heartbeat_updated_no_duplicate': True,
        'other_humanoid_schedules': matching,
        'active_run': pointer['run'], 'training_status': state['status'],
        'stage': state['stage'], 'stage_speed': state['stage_speed'],
        'environment_count': state['environment_count'],
        'completed_main_updates': state['total_training_iterations'],
        'checkpoint': state['checkpoint'],
        'scope': 'Schedule and durable instructions verified; no source, checkpoint, reward or gait qualification changed.',
        'endpoint': 'Verified actual 5 m/s or reproduced locomotion-relevant actuator velocity ceiling; pause training and heartbeat, save report and brief user; no shutdown.',
    }
    (RECORD / 'VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
