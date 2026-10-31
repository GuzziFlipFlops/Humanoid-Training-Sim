"""Read-only final automation check after native update; save a local receipt."""
import json,tomllib,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
args=read(OUT/'automation_update_args.json');before=read(OUT/'automation_current_before_update.json')
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text())
assert cfg['prompt']==args['prompt'] and cfg['status']=='ACTIVE'
assert cfg['id']==before['id'] and cfg['kind']==before['kind']=='heartbeat' and cfg['rrule']==before['rrule']=='FREQ=HOURLY;INTERVAL=3'
assert cfg['target_thread_id']==before['target_thread_id'] and cfg.get('notification_policy')==before.get('notification_policy')
assert cfg['prompt'].split('Latest dated focus',1)[0].rstrip()==before['prompt'].split('Latest dated focus',1)[0].rstrip()
report=(OUT/'REPORT.md').read_text();assert 'closing_summary_2046' in report and 'Clipped Action Policy Gradient' in report
n=read(OUT.parents[1]/'automation/NEXT_CHECK.json')
assert n['recording_request']['status']=='complete' and not n['running_qualified'] and not n['velocity_ceiling_proven']
assert n['latest_verified_state']['actual_workers']==read(OUT/'final_liveness.json')['one_actual_worker']
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),automation_id=cfg['id'],status='ACTIVE',
    same_thread=True,same_cadence=True,same_notification_preference=True,same_durable_scope=True,prompt_exactly_saved=True,
    report_sha256=hashlib.sha256((OUT/'REPORT.md').read_bytes()).hexdigest(),video_request_still_complete=True)
(OUT/'automation_final_verification.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'diagnostic_source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')},indent=2)+'\n')
print(json.dumps(result,indent=2))
