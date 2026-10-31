"""Resume the exact suspended supervisor after GUI recording has exited."""
import json,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text())
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
pause=read(OUT/'video_pause.json');run=Path(pause['run']);p=psutil.Process(pause['supervisor_pid'])
assert p.create_time()==pause['supervisor_create_time'] and p.status()==psutil.STATUS_STOPPED
assert read(OUT/'video_verification.json')['passed']
started=read(HERE/'videos/walking_gui_20261004/capture_started.json')
try:
    recording=psutil.Process(started['pid'])
    assert recording.create_time()!=started['created_time'],'GUI recorder still active'
except psutil.NoSuchProcess:pass
for proc in psutil.process_iter(['cmdline']):
    assert not any(Path(a).name=='record_walking_gui.py' for a in (proc.info['cmdline'] or [])),'Recorder launcher still active'
state=read(OUT/'state_before_video_pause.json');state['updated_unix']=time.time()
state['video_capture_pause_seconds']=time.time()-pause['paused_unix']
state['video_capture_manifest']=str(HERE/'videos/walking_gui_20261004/manifest.json')
write(run/'state.json',state)
active=read(HERE/'ACTIVE_RUN.json');active['status']='running';write(HERE/'ACTIVE_RUN.json',active)
p.resume()
write(OUT/'video_resume.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'supervisor_pid':p.pid,
    'same_supervisor':True,'same_checkpoint':state['checkpoint'],'pause_seconds':state['video_capture_pause_seconds'],
    'scope':'Exact in-memory chain continues; no reset, restart, reward/physics change or video-trained actor'})
print('Same owned supervisor resumed; waiting for fresh worker/status evidence')
