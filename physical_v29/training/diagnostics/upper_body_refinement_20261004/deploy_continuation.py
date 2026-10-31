"""Freeze the tested upper refinement and continue the exact drained full state."""
import hashlib,json,shutil,subprocess,sys,time,psutil
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
pause=read(OUT/'diagnostic_pause.json');old=Path(pause['run']);source=OUT/'validation_source/physical_v29'
assert read(OUT/'matched_control.json')['passed']
assert read(OUT/'physical_validation.json')['passed'] and read(OUT/'full_state_smoke.json')['passed']
sup=psutil.Process(pause['supervisor_pid']);assert sup.create_time()==pause['supervisor_create_time'] and sup.status()==psutil.STATUS_STOPPED
assert not any(any(Path(a).name=='worker_unitree.py' for a in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
state=read(old/'state.json');assert state['checkpoint']==pause['checkpoint'] and sha(state['checkpoint'])==pause['checkpoint_sha256']
assert state['status']=='paused_for_upper_body_validation'
sys.path.insert(0,str(source/'training'))
from curriculum_provenance import training_reward_signature
from upper_body_refinement import verify_upper_body_migration
proof=dict(schema='upper_body_output_refinement_v1',prior_run=str(old),checkpoint=pause['checkpoint'],
 checkpoint_sha256=pause['checkpoint_sha256'],supervisor_sha256=sha(OUT/'candidate_supervisor_unitree.py'),
 implementation_hashes={k:sha(source/k) for k in ['training/worker_unitree.py','training/upper_body_refinement.py']},
 reward_signature=training_reward_signature(source/'training'),actor_critic_Adam_exploration_retained=True,
 qualifies_new_posture=False,reason='User requested head-forward and calmer arms while continuing learned running, without resetting training',
 retained_main_updates=state['total_training_iterations'],same_stage=state['stage'],same_speed=state['stage_speed'])
for key in ['cpu_tests','physical_validation','full_state_smoke']:
 p=OUT/(key+'.json');proof[key]=dict(path=p.name,sha256=sha(p))
write(OUT/'upper_continuation_proof.json',proof)
files={'worker_unitree.py':source/'training/worker_unitree.py','upper_body_refinement.py':source/'training/upper_body_refinement.py',
 'supervisor_unitree.py':OUT/'candidate_supervisor_unitree.py'}
for name in ['worker_unitree.py','supervisor_unitree.py']:
 expected=old/name if name=='supervisor_unitree.py' else old/'source/physical_v29/training'/name
 assert (HERE/name).read_bytes()==expected.read_bytes(),name
assert not (HERE/'upper_body_refinement.py').exists()
run=HERE/'runs/locomotion_20261004_r35_upper_body_refinement';assert not run.exists()
write(OUT/'state_preserved_before_supersede.json',state)
active=read(HERE/'ACTIVE_RUN.json');write(OUT/'active_before_continuation.json',active)
state.update(status='superseded_for_diagnostic_repair',child_pid=None,updated_unix=time.time(),
 pause_reason='Exact saved full state continues at run_150 with training-only head/arm refinement; no actor/critic/Adam reset')
write(old/'state.json',state)
try:verified=verify_upper_body_migration(old,source,OUT/'upper_continuation_proof.json')
except BaseException:
 write(old/'state.json',read(OUT/'state_preserved_before_supersede.json'));sup.resume();raise
assert verified['checkpoint']==pause['checkpoint']
backup=OUT/'canonical_before';backup.mkdir()
changes={}
for name,p in files.items():
 target=HERE/name
 if target.exists():shutil.copy2(target,backup/name)
 changes[name]=dict(before=sha(target) if target.exists() else None,after=sha(p));shutil.copy2(p,target)
run.mkdir()
argv=[sys.executable,'-u','-X','utf8',str(HERE/'supervisor_unitree.py'),'--output',str(run),
 '--reserve_mb','300','--max_hours','18','--iterations_per_block','64','--train_envs','8192',
 '--learning_rate','0.0001','--migrate_run',str(old),'--initial_checkpoint',pause['checkpoint'],
 '--retention_proof',str(OUT/'upper_continuation_proof.json')]
write(OUT/'launch_args.json',dict(argv=argv))
with (run/'supervisor_stdout.log').open('w') as log,(run/'supervisor_stderr.log').open('w') as err:
 new=subprocess.Popen(argv,stdout=log,stderr=err,cwd=str(ROOT),creationflags=subprocess.CREATE_NO_WINDOW)
write(HERE/'ACTIVE_RUN.json',dict(run=str(run),status='starting',recorded_utc=datetime.now(timezone.utc).isoformat(),
 prior_run=str(old),initial_full_checkpoint=pause['checkpoint'],selected_full_checkpoint=pause['checkpoint'],selected_sha256=pause['checkpoint_sha256'],
 reason=proof['reason'],repair_report=str(OUT/'REPORT.md'),same_learned_state_and_stage=True))
deadline=time.monotonic()+45
while not (run/'chain_config.json').exists() and new.poll() is None and time.monotonic()<deadline:time.sleep(.5)
assert new.poll() is None and (run/'chain_config.json').exists(),'Fresh supervisor startup failed; old supervisor remains safely suspended for recovery'
assert sup.create_time()==pause['supervisor_create_time'];sup.kill();sup.wait(timeout=10)
deployment=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),supervisor_launcher_pid=new.pid,
 supervisor_create_time=psutil.Process(new.pid).create_time(),old_run=str(old),exact_continued_checkpoint=pause['checkpoint'],
 checkpoint_sha256=pause['checkpoint_sha256'],retained_main_updates=proof['retained_main_updates'],
 same_stage=proof['same_stage'],same_speed=proof['same_speed'],changes=changes,
 no_actor_warm_start=True,critic_Adam_exploration_retained=True,smoke_actors_discarded=True,
 fresh_matching_current_and_prior_gates_required=True,training_pause_seconds=time.time()-pause['paused_unix'],
 learning_rate_ceiling=.0001,new_posture_qualified=False)
write(OUT/'deployment.json',deployment);print(json.dumps(deployment,indent=2))
