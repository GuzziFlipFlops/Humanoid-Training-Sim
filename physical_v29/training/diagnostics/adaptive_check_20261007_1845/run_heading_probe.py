import json,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
names={'worker_unitree.py','heading_probe.py'}
assert not any(any(Path(x).name in names for x in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
folder=OUT/'heading_response';folder.mkdir(exist_ok=True)
with (folder/'stdout.log').open('w',encoding='utf-8') as so,(folder/'stderr.log').open('w',encoding='utf-8') as se:
 p=subprocess.Popen([sys.executable,'-u','-X','utf8',str(OUT/'heading_probe.py'),'--headless','--device','cuda:0'],stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
 owned=psutil.Process(p.pid);(folder/'process.json').write_text(json.dumps({'pid':p.pid,'create_time':owned.create_time()}))
 deadline=time.monotonic()+300
 try:
  while p.poll() is None and not (folder/'result.json').exists():
   assert not (OUT/'physical_probe_failure.json').exists(),'Physical diagnostic failed; exit promptly and resume training'
   assert time.monotonic()<deadline,'Bounded physical probe timeout';time.sleep(2)
  if p.poll() is None:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:
    assert json.loads((folder/'result.json').read_text())['completed']
    tree=owned.children(recursive=True)+[owned]
    for proc in tree:
     if proc.is_running():proc.terminate()
    psutil.wait_procs(tree,timeout=20)
    (folder/'exit_cleanup.json').write_text(json.dumps({'reason':'Kit shutdown after completed result','only_after_completed_result':True}))
 except BaseException:
  tree=owned.children(recursive=True)+[owned]
  for proc in tree:
   if proc.is_running():proc.terminate()
  psutil.wait_procs(tree,timeout=20);raise
assert p.returncode==0 or (folder/'exit_cleanup.json').exists()
assert json.loads((folder/'result.json').read_text())['completed']
print('Physical heading-command test complete; diagnostic worker exited.')
