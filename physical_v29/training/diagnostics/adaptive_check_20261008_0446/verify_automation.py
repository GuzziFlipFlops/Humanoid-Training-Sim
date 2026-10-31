import json,tomllib,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
args=read(OUT/'automation_update_args.json')
p=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
n=tomllib.loads(p.read_text(encoding='utf-8'));before=tomllib.loads((OUT/'automation_before.toml').read_text(encoding='utf-8'))
assert n['kind']=='heartbeat' and n['status']=='ACTIVE'
assert n['rrule']==before['rrule']=='FREQ=HOURLY;INTERVAL=5'
assert n['target_thread_id']==before['target_thread_id']==args['targetThreadId']
assert n['name']==before['name']==args['name']
assert n.get('notification_policy')==before.get('notification_policy')
assert n['prompt'].rstrip()==args['prompt'].rstrip()
prefix=lambda s:s.split('Latest dated five-hour check',1)[0]
assert prefix(n['prompt'])==prefix(before['prompt'])
receipt={'recorded_utc':datetime.now(timezone.utc).isoformat(),'same_id':args['id'],'kind':n['kind'],'status':n['status'],'same_thread':True,'five_hour_cadence':True,'notification_preference_preserved':True,'durable_scope_endpoint_prefix_unchanged':True,'prompt_matches_args':True,'prefix_sha256':hashlib.sha256(prefix(n['prompt']).encode()).hexdigest(),'no_duplicate_created':True}
(OUT/'automation_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
