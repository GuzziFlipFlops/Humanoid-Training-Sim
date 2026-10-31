"""Deploy only validated startup maintenance, retaining the exact failed learner."""
import hashlib,importlib.util,json,shutil,subprocess,sys,time,psutil
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];ROOT=HERE.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
original=read(OUT/'initial_evidence.json');old=Path(original['active']['run'])
proof=read(OUT/'offline_ground_continuation_proof.json');state=read(old/'state.json')
assert state['status']=='failed' and state.get('child_pid') is None
assert Path(state['checkpoint']).resolve()==Path(original['checkpoint']).resolve()
assert sha(state['checkpoint'])==proof['checkpoint_sha256']==original['checkpoint_sha256']
assert state['total_training_iterations']==1536 and state['prior_training_iterations']==5825
assert Path(read(HERE/'ACTIVE_RUN.json')['run']).resolve()==old.resolve()
comparison=read(OUT/'matched_comparison.json')
assert set(comparison['completed_boundaries'])=={'64','128'}
assert not any(d['deploy_supported'] for d in comparison['decisions'].values()),'Candidate requires a separate gradient migration; do not mix with startup repair'
assert read(OUT/'physical_speed_response/result.json')['completed']
for p in psutil.process_iter(['pid','cmdline']):
 args=p.info['cmdline'] or []
 assert not any(Path(a).name in ['worker_unitree.py','supervisor_unitree.py','physical_speed_probe.py','run_matched_extension.py','run_matched_alias.py'] for a in args),'Another owned simulator or chain is active'
assert read(OUT/'offline_migration_cpu_tests.json')['passed']
assert sha(OUT/'tested_supervisor_unitree.py')==proof['supervisor_sha256']
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('tested_offline_supervisor',OUT/'tested_supervisor_unitree.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.verify_offline_ground_migration(old,OUT/'offline_source/physical_v29',OUT/'offline_ground_continuation_proof.json')
for p in (old/'source/physical_v29').rglob('*'):
 if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc' or 'usd' in p.parts:continue
 relative=p.relative_to(old/'source/physical_v29');assert (HERE.parent/relative).read_bytes()==p.read_bytes(),relative
backup=OUT/'canonical_before_offline_continuation';backup.mkdir()
shutil.copy2(HERE/'supervisor_unitree.py',backup/'supervisor_unitree.py')
shutil.copy2(HERE/'worker_unitree.py',backup/'worker_unitree.py')
write(OUT/'state_preserved_before_supersede.json',state);write(OUT/'active_preserved_before_continuation.json',read(HERE/'ACTIVE_RUN.json'))
changes={}
for relative,h in proof['implementation_hashes'].items():
 source=OUT/'offline_source/physical_v29'/relative;target=HERE.parent/relative
 changes[relative]=dict(before=sha(target) if target.exists() else None,after=h)
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(target)==h
shutil.copy2(OUT/'tested_supervisor_unitree.py',HERE/'supervisor_unitree.py');assert sha(HERE/'supervisor_unitree.py')==proof['supervisor_sha256']
run=HERE/'runs/locomotion_20261005_r37_offline_ground_full_state';assert not run.exists();run.mkdir()
argv=[sys.executable,'-u','-X','utf8',str(HERE/'supervisor_unitree.py'),'--output',str(run),
 '--reserve_mb','300','--max_hours','18','--iterations_per_block','64','--train_envs','8192',
 '--learning_rate','0.0001','--migrate_run',str(old),'--initial_checkpoint',state['checkpoint'],
 '--optimization_proof',str(OUT/'offline_ground_continuation_proof.json')]
write(OUT/'launch_args.json',dict(argv=argv))
with (run/'supervisor_stdout.log').open('w') as log,(run/'supervisor_stderr.log').open('w') as err:
 new=subprocess.Popen(argv,stdout=log,stderr=err,cwd=str(ROOT),creationflags=subprocess.CREATE_NO_WINDOW)
write(HERE/'ACTIVE_RUN.json',dict(run=str(run),status='starting',recorded_utc=datetime.now(timezone.utc).isoformat(),
 prior_run=str(old),initial_full_checkpoint=state['checkpoint'],selected_full_checkpoint=state['checkpoint'],selected_sha256=proof['checkpoint_sha256'],
 reason=proof['reason'],repair_report=str(OUT/'REPORT.md'),same_learned_state_and_stage=True))
deadline=time.monotonic()+45
while not (run/'chain_config.json').exists() and new.poll() is None and time.monotonic()<deadline:time.sleep(.5)
if new.poll() is not None or not (run/'chain_config.json').exists():
 write(OUT/'deployment_failure.json',dict(run=str(run),exit=new.poll(),time=time.time()))
 raise RuntimeError('Source freeze/startup failed; exact old checkpoint retained. Inspect before any retry.')
state.update(status='superseded_for_offline_ground',child_pid=None,updated_unix=time.time(),
 continuation_run=str(run),supersede_reason='Exact full-state continuation after online ground resolver failure; original failure and checkpoints preserved')
write(old/'state.json',state)
deployment=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),supervisor_launcher_pid=new.pid,
 supervisor_launcher_create_time=psutil.Process(new.pid).create_time(),old_run=str(old),exact_continued_checkpoint=state['checkpoint'],
 checkpoint_sha256=proof['checkpoint_sha256'],retained_main_updates=1536,inherited_main_updates_preserved=5825,
 inherited_total_main_updates=7361,same_stage='run_150',same_speed=1.5,changes=changes,
 no_actor_warm_start=True,actor_critic_Adam_exploration_retained=True,diagnostic_actors_discarded=True,
 training_gradient_repair=False,reward_physics_and_caps_unchanged=True,fresh_matching_gates_required=True,
 learning_rate_ceiling=.0001,running_qualified=False)
write(OUT/'deployment.json',deployment);print(json.dumps(deployment,indent=2))
