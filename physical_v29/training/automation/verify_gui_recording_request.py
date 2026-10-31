"""Verify the app applied the requested same-heartbeat prompt update."""
import json
import tomllib
from pathlib import Path
from datetime import datetime, timezone

here = Path(__file__).resolve().parent
audit = here/'recording_request_20261004'
before = json.loads((audit/'automation_before.json').read_text(encoding='utf-8'))
args = json.loads((audit/'automation_update_args.json').read_text(encoding='utf-8'))
after = tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text(encoding='utf-8-sig'))
handoff = json.loads((here/'NEXT_CHECK.json').read_text(encoding='utf-8'))
checks = {
    'same_id':after['id']==before['id']==args['id'],
    'same_heartbeat':after['kind']==before['kind']=='heartbeat',
    'same_thread':after['target_thread_id']==before['target_thread_id']==args['targetThreadId'],
    'same_three_hour_cadence':after['rrule']==before['rrule']=='FREQ=HOURLY;INTERVAL=3',
    'active':after['status']=='ACTIVE',
    'exact_updated_prompt':after['prompt']==args['prompt'],
    'previous_scope_preserved':after['prompt'].endswith(before['prompt']),
    'notification_preference_preserved':after.get('notification_policy')==before.get('notification_policy') and after.get('notificationPolicy')==before.get('notificationPolicy'),
    'one_time_request_pending':handoff['recording_request']['status']=='pending',
    'workflow_exists':Path(handoff['recording_request']['workflow']).is_file(),
    'ten_second_motion_and_playback':handoff['recording_request']['normal_playback_seconds_per_speed']==handoff['recording_request']['simulated_motion_seconds_per_speed']==10,
    'new_running_speed_included':1.5 in handoff['recording_request']['commanded_speeds_m_s'],
    'safe_pause_resume_requested':handoff['recording_request']['pause_training'] and handoff['recording_request']['resume_training_after_capture']}
assert all(checks.values()),checks
proof={'verified_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,
       'automation_id':after['id'],'pending_workflow':handoff['recording_request']['workflow'],
       'new_videos_created':False,'training_paused_now':False}
(audit/'VERIFIED.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof,indent=2))
