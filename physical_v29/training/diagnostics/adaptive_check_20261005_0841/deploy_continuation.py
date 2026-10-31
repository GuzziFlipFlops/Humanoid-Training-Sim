"""Freeze the tested walking teacher-target correction; exact R35 full state."""
import hashlib,json,shutil,subprocess,sys,time,psutil
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];ROOT=HERE.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
pause=read(OUT/'diagnostic_pause.json');old=Path(pause['run']);source=OUT/'all_prior_source/physical_v29'
sup=psutil.Process(pause['supervisor_pid'])
assert sup.create_time()==pause['supervisor_create_time'] and sup.status()==psutil.STATUS_STOPPED
for p in psutil.process_iter(['cmdline']):
 assert not any(Path(a).name in ['worker_unitree.py','run_matched_all_prior.py','run_all_prior_parity.py'] for a in p.info['cmdline'] or []),'Diagnostic worker active'
state=read(old/'state.json');assert state['checkpoint']==pause['checkpoint'] and sha(state['checkpoint'])==pause['checkpoint_sha256']
assert state['status'].startswith('paused_')
assert read(OUT/'migration_cpu_tests.json')['passed']
proof=read(OUT/'teacher_context_continuation_proof.json')
assert sha(HERE/'supervisor_unitree.py')==proof['supervisor_sha256']
assert state['total_training_iterations']==4352 and state['prior_training_iterations']==1473
sys.path.insert(0,str(HERE));from supervisor_unitree import verify_teacher_context_migration
for name in ['upper_body_refinement.py','worker_unitree.py']:
 assert (HERE/name).read_bytes()==(old/'source/physical_v29/training'/name).read_bytes(),name
for name in ['teacher_head_context.py','teacher_head_context.json']:assert not (HERE/name).exists()
run=HERE/'runs/locomotion_20261005_r36_posture_conditioned_priors';assert not run.exists()
backup=OUT/'canonical_before_continuation';backup.mkdir()
shutil.copy2(HERE/'upper_body_refinement.py',backup/'upper_body_refinement.py')
shutil.copy2(HERE/'supervisor_unitree.py',OUT/'tested_supervisor_unitree.py')
write(OUT/'state_preserved_before_supersede.json',state);active=read(HERE/'ACTIVE_RUN.json');write(OUT/'active_before_continuation.json',active)
state.update(status='superseded_for_diagnostic_repair',child_pid=None,updated_unix=time.time(),
 pause_reason='Exact full-state continuation; fix reproduced tilted-head feedback dependence of walking teacher targets')
write(old/'state.json',state)
try:verify_teacher_context_migration(old,source,OUT/'teacher_context_continuation_proof.json')
except BaseException:
 write(old/'state.json',read(OUT/'state_preserved_before_supersede.json'));raise
changes={}
for name in ['upper_body_refinement.py','teacher_head_context.py','teacher_head_context.json']:
 target=HERE/name;changes[name]=dict(before=sha(target) if target.exists() else None,after=sha(source/'training'/name))
 shutil.copy2(source/'training'/name,target)
run.mkdir()
argv=[sys.executable,'-u','-X','utf8',str(HERE/'supervisor_unitree.py'),'--output',str(run),
 '--reserve_mb','300','--max_hours','18','--iterations_per_block','64','--train_envs','8192',
 '--learning_rate','0.0001','--migrate_run',str(old),'--initial_checkpoint',pause['checkpoint'],
 '--retention_proof',str(OUT/'teacher_context_continuation_proof.json')]
write(OUT/'launch_args.json',dict(argv=argv))
with (run/'supervisor_stdout.log').open('w') as log,(run/'supervisor_stderr.log').open('w') as err:
 new=subprocess.Popen(argv,stdout=log,stderr=err,cwd=str(ROOT),creationflags=subprocess.CREATE_NO_WINDOW)
write(HERE/'ACTIVE_RUN.json',dict(run=str(run),status='starting',recorded_utc=datetime.now(timezone.utc).isoformat(),
 prior_run=str(old),initial_full_checkpoint=pause['checkpoint'],selected_full_checkpoint=pause['checkpoint'],selected_sha256=pause['checkpoint_sha256'],
 reason=proof['reason'],repair_report=str(OUT/'REPORT.md'),same_learned_state_and_stage=True))
deadline=time.monotonic()+45
while not (run/'chain_config.json').exists() and new.poll() is None and time.monotonic()<deadline:time.sleep(.5)
if new.poll() is not None or not (run/'chain_config.json').exists():
 # Preserve failed run and restore the original owned learner, never a new actor.
 new.terminate() if new.poll() is None else None
 write(old/'state.json',read(OUT/'state_before_diagnosis.json'));write(HERE/'ACTIVE_RUN.json',active)
 sup.resume();raise RuntimeError('Fresh source startup failed; original exact learner resumed. Failed evidence retained.')
assert sup.create_time()==pause['supervisor_create_time'];sup.kill();sup.wait(timeout=10)
deployment=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),supervisor_launcher_pid=new.pid,
 supervisor_launcher_create_time=psutil.Process(new.pid).create_time(),old_run=str(old),exact_continued_checkpoint=pause['checkpoint'],
 checkpoint_sha256=pause['checkpoint_sha256'],retained_main_updates=4352,prior_main_updates_preserved=1473,
 same_stage='run_150',same_speed=1.5,changes=changes,no_actor_warm_start=True,
 critic_Adam_exploration_retained=True,smoke_and_matched_actors_discarded=True,
 fresh_matching_current_and_prior_gates_required=True,training_pause_seconds=time.time()-pause['paused_unix'],
 learning_rate_ceiling=.0001,running_qualified=False)
write(OUT/'deployment.json',deployment);print(json.dumps(deployment,indent=2))
