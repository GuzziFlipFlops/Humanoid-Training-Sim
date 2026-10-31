"""Scope diagnostic memory separately from live PPO capacity."""
import json, subprocess
from pathlib import Path
from datetime import datetime, timezone
import psutil
out=Path(__file__).resolve().parent
items=[]
for p in psutil.process_iter(['pid','cmdline','create_time']):
    args=p.info['cmdline'] or []
    if any(Path(a).name in ['teacher_body_probe.py','capture_current_gradients.py','worker_unitree.py'] for a in args):
        items.append(dict(pid=p.pid,created=p.create_time(),command=args,
            leaf=not any(any(Path(a).name in ['teacher_body_probe.py','capture_current_gradients.py','worker_unitree.py']
                for a in c.cmdline()) for c in p.children()),private_MiB=getattr(p.memory_info(),'private',p.memory_info().rss)/1024**2))
leaves=[p for p in items if p['leaf']]
assert len(leaves)==1
assert any(Path(a).name=='teacher_body_probe.py' for a in leaves[0]['command'])
memory=psutil.virtual_memory()
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used,memory.total','--format=csv,noheader,nounits'],
    capture_output=True,text=True,check=True).stdout.strip()
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=39000,
    supervisor_status=psutil.Process(39000).status(),actual_diagnostic_leaf=leaves,
    host_available_MiB=memory.available/1024**2,gpu_free_used_total_MiB=gpu,
    scope='1152-robot diagnostic; this is not steady 8192 PPO memory or a ramp qualification.')
assert record['supervisor_status']==psutil.STATUS_STOPPED
(out/'diagnostic_memory_observation.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
