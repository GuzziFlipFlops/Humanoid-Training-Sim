"""Additional independently seeded reset-state tests after matched learning."""
import json,sys,subprocess,time,hashlib
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_context'
assert json.loads((ROOT/'completion.json').read_text())['completed']
names={'worker_unitree.py','teacher_context_candidate.py','matched_context_test.py','capture_current_gradients.py','diverse_initial_eval.py'}
assert not any(any(Path(x).name in names for x in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
if not (ROOT/'completion_before_diversity.json').exists():
 (ROOT/'completion_before_diversity.json').write_bytes((ROOT/'completion.json').read_bytes())
for branch in ['control','candidate']:
 folder=ROOT/f'{branch}_128_diverse_nominal'
 if (folder/'result.json').exists():assert json.loads((folder/'result.json').read_text())['completed'];continue
 cp=ROOT/f'{branch}_128_train/candidate.pt';folder.mkdir(exist_ok=True)
 argv=[str(OUT/'diverse_initial_eval.py'),'--mode','eval','--output',str(folder),'--checkpoint',str(cp),
  '--num_envs','320','--speed','3.5','--duration','20','--seed','2301','--headless','--device','cuda:0','--reserve_mb','300']
 with (folder/'stdout.log').open('w',encoding='utf-8') as so,(folder/'stderr.log').open('w',encoding='utf-8') as se:
  p=subprocess.Popen([sys.executable,'-u','-X','utf8']+argv,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
  owned=psutil.Process(p.pid);(folder/'process.json').write_text(json.dumps(dict(pid=p.pid,create_time=owned.create_time(),arguments=argv)))
  deadline=time.monotonic()+240
  try:
   while p.poll() is None and not (folder/'result.json').exists():
    assert time.monotonic()<deadline,'Diverse eval timeout';time.sleep(2)
   if p.poll() is None:
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:
     assert json.loads((folder/'result.json').read_text())['completed']
     ps=owned.children(recursive=True)+[owned]
     for proc in ps:
      if proc.is_running():proc.terminate()
     psutil.wait_procs(ps,timeout=20)
     (folder/'exit_cleanup.json').write_text(json.dumps(dict(reason='Kit shutdown after completed result',only_after_completed_result=True)))
  except BaseException:
   ps=owned.children(recursive=True)+[owned]
   for proc in ps:
    if proc.is_running():proc.terminate()
   psutil.wait_procs(ps,timeout=20);raise
 assert (p.returncode==0 or (folder/'exit_cleanup.json').exists()) and json.loads((folder/'result.json').read_text())['completed']
 receipt=json.loads((folder/'initial_diversity.json').read_text());assert all(n==32 for n in receipt['unique_initial_states_by_command'].values())
 print(json.dumps(dict(branch=branch,initial_states=receipt['unique_initial_states_by_command'])),flush=True)
(ROOT/'diversity_completion.json').write_text(json.dumps(dict(completed=True,seed=2301,unchanged_classifiers=True,independent_initial_joint_states_per_command=32,
 scope='Additional matched nominal-physics reset robustness; does not change production promotion gates or certify an unqualified speed'),indent=2))
