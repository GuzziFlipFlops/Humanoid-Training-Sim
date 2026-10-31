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
 assert not any(Path(x).name in ['physical_takeoff_probe.py','worker_unitree.py'] for x in a),'Diagnostic worker still active'
s=read(OUT/'state_before_diagnosis.json');assert hashlib.sha256(Path(s['checkpoint']).read_bytes()).hexdigest()==pause['checkpoint_sha256']
s['updated_unix']=time.time();s['controlled_diagnosis_pause_seconds']=time.time()-pause['paused_unix'];write(run/'state.json',s)
a=read(HERE/'ACTIVE_RUN.json');assert Path(a['run'])==run;a['status']='running';write(HERE/'ACTIVE_RUN.json',a);p.resume()
r=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),same_supervisor=True,supervisor_pid=p.pid,checkpoint=s['checkpoint'],checkpoint_sha256=pause['checkpoint_sha256'],pause_seconds=s['controlled_diagnosis_pause_seconds'],training_source_changed=False,diagnostic_actors_deployed=False)
write(OUT/'diagnostic_resume.json',r);print(json.dumps(r,indent=2))
