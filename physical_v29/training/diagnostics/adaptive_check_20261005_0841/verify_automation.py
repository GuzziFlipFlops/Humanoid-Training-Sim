import json,tomllib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
args=json.loads((OUT/'automation_update_args.json').read_text())
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text(encoding='utf-8'))
before=json.loads((OUT/'automation_current_before_update.json').read_text())
assert cfg['id']==args['id'] and cfg['kind']=='heartbeat' and cfg['status']=='ACTIVE'
assert cfg['rrule']==before['rrule']=='FREQ=HOURLY;INTERVAL=3' and cfg['target_thread_id']==before['target_thread_id']==args['targetThreadId']
assert cfg['prompt'].strip()==args['prompt'].strip() and cfg['prompt'].split('Latest dated focus')[0]==before['prompt'].split('Latest dated focus')[0]
assert cfg.get('notification_policy')==before.get('notification_policy')
r=dict(verified=True,recorded_utc=datetime.now(timezone.utc).isoformat(),same_id=cfg['id'],status=cfg['status'],same_thread=cfg['target_thread_id'],
 three_hour_cadence=True,durable_scope_unchanged=True,notification_policy_preserved=True,dated_focus_updated=True,updated_at=cfg['updated_at'])
(OUT/'automation_verified.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
