"""Read-only diagnostic memory observation; never training-capacity certification."""
import json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(OUT/'monitor_check.py')],check=True,stdout=subprocess.DEVNULL)
s=json.loads((OUT/'monitor_snapshot.json').read_text());assert s['actual_worker_count']==1
w=s['actual_workers'][0];assert Path(w['output']).is_relative_to(OUT/'matched_full_state')
sup=psutil.Process(39000);assert sup.status()==psutil.STATUS_STOPPED
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
data=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=39000,supervisor_status=sup.status(),
 actual_diagnostic_worker=w,host_available_MiB=psutil.virtual_memory().available/1024**2,gpu_free_used_total_MiB=gpu,
 scope='Owned diagnostic4096-env PPO or320-robot evaluation; not8192 main capacity or180-second ramp qualification.')
(OUT/'diagnostic_memory_observation.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
