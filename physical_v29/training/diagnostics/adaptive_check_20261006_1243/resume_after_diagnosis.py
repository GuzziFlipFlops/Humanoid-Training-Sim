"""Resume only the exact suspended supervisor after diagnostic workers exit."""
import json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run']);p=psutil.Process(pause['supervisor_pid'])
assert p.create_time()==pause['supervisor_create_time'] and p.status()==psutil.STATUS_STOPPED
for proc in psutil.process_iter(['pid','cmdline']):
 a=proc.info['cmdline'] or []
 assert not any(Path(x).name in ['steering_probe.py','capture_current_gradients.py','worker_unitree.py'] for x in a),'Diagnostic worker still active'
s=read(OUT/'state_before_diagnosis.json');assert hashlib.sha256(Path(s['checkpoint']).read_bytes()).hexdigest()==pause['checkpoint_sha256']
resume_unix=time.time();suspend_proxy=(OUT/'state_before_diagnosis.json').stat().st_mtime
s['updated_unix']=resume_unix;s['controlled_diagnosis_pause_seconds']=resume_unix-suspend_proxy;write(run/'state.json',s)
a=read(HERE/'ACTIVE_RUN.json');assert Path(a['run'])==run;a['status']='running';write(HERE/'ACTIVE_RUN.json',a);p.resume()
r=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),same_supervisor=True,supervisor_pid=p.pid,checkpoint=s['checkpoint'],checkpoint_sha256=pause['checkpoint_sha256'],pause_seconds=s['controlled_diagnosis_pause_seconds'],supervisor_suspension_time_proxy_unix=suspend_proxy,post_worker_drain_seconds=resume_unix-pause['paused_unix'],worker_drain_seconds=pause['paused_unix']-suspend_proxy,pause_timing_scope='Saved-state mtime immediately after supervisor.suspend bounds the suspension start; total excludes the complete suspended interval, including worker drain. Paused evaluation wrapper includes this interval.',training_source_changed=False,diagnostic_actors_deployed=False)
write(OUT/'diagnostic_resume.json',r);print(json.dumps(r,indent=2))
