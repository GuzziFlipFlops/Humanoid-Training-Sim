import json, subprocess, sys, time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
assert (OUT/'evaluation_complete.json').exists()
for p in psutil.process_iter(['pid','cmdline']):
    cmd=p.info['cmdline'] or []
    assert not any(Path(a).name in ('worker_unitree.py','eval_entry.py','record_motor_gui.py') for a in cmd), p.info
with (OUT/'gui_capture_stdout.log').open('w',encoding='utf-8') as stdout, (OUT/'gui_capture_stderr.log').open('w',encoding='utf-8') as stderr:
    p=subprocess.Popen([sys.executable,'-X','utf8','-u',str(OUT/'record_motor_gui.py'),'--device','cuda:0','--enable_cameras'],cwd=str(OUT),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    rec={'pid':p.pid,'created_time':psutil.Process(p.pid).create_time(),'command':[sys.executable,'-X','utf8','-u',str(OUT/'record_motor_gui.py'),'--device','cuda:0','--enable_cameras'],'no_learning':True,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (OUT/'gui_capture_identity.json').write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rec),flush=True)
