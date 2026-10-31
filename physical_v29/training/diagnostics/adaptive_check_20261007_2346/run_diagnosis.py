"""Pause at saved boundary, bounded physical experiment, always resume."""
import json,subprocess,sys,psutil,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
def launch(name,timeout):
    with (OUT/(name+'.stdout.log')).open('w',encoding='utf-8') as so,(OUT/(name+'.stderr.log')).open('w',encoding='utf-8') as se:
        p=subprocess.Popen([sys.executable,'-u','-X','utf8',str(OUT/name)],stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        owned=psutil.Process(p.pid);identity=owned.create_time()
        try:code=p.wait(timeout=timeout)
        except BaseException:
            if owned.is_running() and owned.create_time()==identity:
                tree=owned.children(recursive=True)+[owned]
                for proc in tree:
                    if proc.is_running():proc.terminate()
                psutil.wait_procs(tree,timeout=20)
            raise
    assert code==0,(name,code)
launch('prepare_and_pause.py',870)
try:
    launch('run_timing_probe.py',480)
finally:
    launch('resume_after_diagnosis.py',45)
print('Same original training supervisor resumed; diagnostic command laws not deployed.')
