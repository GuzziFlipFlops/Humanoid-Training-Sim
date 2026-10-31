"""Deploy only proved termination correction; keep old supervisor safely suspended until new startup verifies."""
import hashlib,json,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pause=read(OUT/'diagnostic_pause.json');proof=read(OUT/'termination_continuation_proof.json')
assert read(OUT/'migration_cpu_tests.json')['passed']
oldrun=Path(pause['run']);old=psutil.Process(pause['supervisor_pid'])
assert old.create_time()==pause['supervisor_create_time'] and old.status()==psutil.STATUS_STOPPED
assert read(oldrun/'state.json')['child_pid'] is None
for p in psutil.process_iter(['cmdline']):
 assert not any(Path(x).name=='worker_unitree.py' for x in p.info['cmdline'] or []),'Another actual worker is active'
assert sha(pause['checkpoint'])==pause['checkpoint_sha256']
newrun=HERE/'runs/locomotion_20261006_r38_commanded_travel_budget';assert not newrun.exists()
backup=OUT/'canonical_before_guard';backup.mkdir()
changes=[('environment.py',OUT/'candidate_source/physical_v29/training/environment.py'),
 ('worker_unitree.py',OUT/'candidate_source/physical_v29/training/worker_unitree.py'),
 ('supervisor_unitree.py',OUT/'tested_supervisor_unitree.py')]
for name,path in changes:
 canonical=HERE/name
 if name!='supervisor_unitree.py':assert canonical.read_bytes()==(oldrun/'source/physical_v29/training'/name).read_bytes()
 else:assert sha(canonical)==read(OUT/'supervisor_patch.json')['before_sha256']
 shutil.copy2(canonical,backup/name);shutil.copyfile(path,canonical)
assert sha(HERE/'supervisor_unitree.py')==proof['supervisor_sha256']
newrun.mkdir()
argv=[sys.executable,'-u','-X','utf8',str(HERE/'supervisor_unitree.py'),'--output',str(newrun),
 '--reserve_mb','300','--max_hours','18','--iterations_per_block','64','--train_envs','8192',
 '--learning_rate','0.0001','--migrate_run',str(oldrun),'--initial_checkpoint',pause['checkpoint'],
 '--optimization_proof',str(OUT/'termination_continuation_proof.json')]
with (newrun/'supervisor_stdout.log').open('w') as out,(newrun/'supervisor_stderr.log').open('w') as err:
 child=subprocess.Popen(argv,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
identity={'pid':child.pid,'create_time':psutil.Process(child.pid).create_time()}
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),new_run=str(newrun),argv=argv,
 new_supervisor=identity,old_supervisor_suspended=True,original_full_checkpoint=pause['checkpoint'],
 original_sha256=pause['checkpoint_sha256'],diagnostic_actors_deployed=False,
 retains_actor_critic_Adam_exploration=True,correction_only_termination_and_explicit_load_provenance=True,
 old_frozen_sources_unchanged=True,proof=str(OUT/'termination_continuation_proof.json'))
(OUT/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
active=dict(run=str(newrun),status='starting',recorded_utc=receipt['recorded_utc'],prior_run=str(oldrun),
 initial_full_checkpoint=pause['checkpoint'],selected_full_checkpoint=pause['checkpoint'],
 selected_sha256=pause['checkpoint_sha256'],reason='Correct reproduced unconditional60m commanded-travel termination; preserve full learned state',
 repair_report=str(OUT/'REPORT.md'),same_learned_state_and_stage=True,supervisor_pid=child.pid,
 migration_proof=str(OUT/'termination_continuation_proof.json'))
sys.path.insert(0,str(HERE));from status_io import atomic_json
atomic_json(HERE/'ACTIVE_RUN.json',active)
deadline=time.monotonic()+30
while time.monotonic()<deadline:
 if (newrun/'state.json').exists():
  s=read(newrun/'state.json');assert s['status']!='failed',s
  if (newrun/'chain_config.json').exists():break
 assert child.poll() is None,'New supervisor exited during freeze'
 time.sleep(1)
else:raise RuntimeError('No fresh source-frozen startup within30s; old supervisor remains suspended for safe fallback')
receipt['freeze_created']=True;receipt['startup_state']=s
(OUT/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'run':str(newrun),'supervisor':identity,'status':s['status'],'old_supervisor_still_suspended':True},indent=2))
