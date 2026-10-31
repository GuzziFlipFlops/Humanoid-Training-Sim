import json,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_focused_prior'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=[]
for p in ROOT.glob('*'):
 if not p.is_dir() or not (p/'process.json').exists():continue
 r=dict(folder=p.name,completed=(p/'operation_receipt.json').exists())
 if (p/'progress.jsonl').exists():
  s=[json.loads(v) for v in (p/'progress.jsonl').read_text().splitlines()];last=s[-1]
  r.update(observed_updates=len(s),GPU=last['gpu'],pre_update_KL=last['ppo']['pre_update_kl'],command_mismatch=last['behavior']['action_command_difference'])
 rows.append(r)
active=[]
names={'focused_prior_candidate.py','worker_unitree.py','motor_retention_probe.py','capture_current_gradients.py','diverse_eval.py'}
for p in psutil.process_iter(['cmdline']):
 if any(Path(x).name in names for x in p.info['cmdline'] or []):active.append(dict(pid=p.pid,created=p.create_time(),file=next(Path(x).name for x in p.info['cmdline'] if Path(x).name in names)))
r=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),supervisor_status=psutil.Process(39000).status(),diagnostic_processes=active,operations=rows,
 diagnostic_updates_are_not_main=True,host_available_MiB=psutil.virtual_memory().available/2**20)
(OUT/'diagnostic_monitor.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(time=r['recorded_utc'],supervisor=r['supervisor_status'],active=active,last_operations=rows[-2:],host_available_MiB=r['host_available_MiB'])))
