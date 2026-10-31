"""Resume the exact owned supervisor after GUI exit, including capture failures."""
import json,time,hashlib
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
pause=read(OUT/'video_pause.json');run=Path(pause['run']);dest=Path(read(OUT/'capture_config.json')['destination'])
p=psutil.Process(pause['supervisor_pid'])
assert p.create_time()==pause['supervisor_create_time'] and p.status()==psutil.STATUS_STOPPED
started_path=dest/'capture_started.json'
if started_path.exists():
    started=read(started_path)
    try:
        recording=psutil.Process(started['pid'])
        assert recording.create_time()!=started['created_time'],'GUI recorder still active'
    except psutil.NoSuchProcess:pass
for proc in psutil.process_iter(['cmdline']):
    assert not any(Path(a).name=='record_full_body_gui.py' for a in (proc.info['cmdline'] or [])),'GUI launcher still active'
completed=(OUT/'video_verification.json').exists() and read(OUT/'video_verification.json')['passed']
state=read(OUT/'state_before_video_pause.json')
assert hashlib.sha256(Path(state['checkpoint']).read_bytes()).hexdigest()==pause['checkpoint_sha256']
state['updated_unix']=time.time();state['video_capture_pause_seconds']=time.time()-pause['paused_unix']
state['video_capture_manifest']=str(dest/'manifest.json') if completed else None
write(run/'state.json',state)
active=read(HERE/'ACTIVE_RUN.json');assert Path(active['run'])==run
active['status']='running';write(HERE/'ACTIVE_RUN.json',active);p.resume()
audit={'recorded_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'supervisor_pid':p.pid,
    'same_supervisor':True,'same_checkpoint':state['checkpoint'],'checkpoint_sha256':pause['checkpoint_sha256'],
    'pause_seconds':state['video_capture_pause_seconds'],'verified_videos_complete':completed,
    'scope':'Exact same in-memory supervisor continues; no training restart, source/physics/reward change or diagnostic policy update'}
write(OUT/'video_resume.json',audit);print(json.dumps(audit,indent=2))
