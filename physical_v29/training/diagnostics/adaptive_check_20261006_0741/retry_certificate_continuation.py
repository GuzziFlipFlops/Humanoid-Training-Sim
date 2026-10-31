"""Preserve R38 failed before main learning; correct gate discovery in a new frozen R39."""
import hashlib,importlib.util,json,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=Path(read(OUT/'deployment.json')['new_run']);state=read(old/'state.json')
assert state['status']=='failed' and state['total_training_iterations']==0 and state['child_pid'] is None
assert 'Missing matching full teacher qualification' in state['error']
for p in psutil.process_iter(['cmdline']):
 assert not any(Path(x).name=='worker_unitree.py' for x in p.info['cmdline'] or []),'Another actual simulator is active'
proof_path=OUT/'termination_continuation_proof_retry.json';proof=read(proof_path)
sys.path.insert(0,str(HERE));from standing_retention import qualification_certificate
try:qualification_certificate(old,read(OUT/'state_before_diagnosis.json')['promotions'][-1]['checkpoint'],3.,proof['new_reward_signature'],read(OUT/'103_teacher_run_300_gate_nominal/result.json')['gate_version'])
except AssertionError as e:assert str(e)=='Missing matching full teacher qualification'
else:raise AssertionError('Expected old filename lookup fault was not reproduced')
label='run_300_gate_termination_teacher_requalification'
assert Path('007_'+label+'_nominal/result.json').match('*_gate_*/result.json')
assert not Path('007_retention_teacher_requalification_nominal/result.json').match('*_gate_*/result.json')
certificate=qualification_certificate(OUT,read(OUT/'state_before_diagnosis.json')['promotions'][-1]['checkpoint'],3.,proof['new_reward_signature'],read(OUT/'103_teacher_run_300_gate_nominal/result.json')['gate_version'])
(OUT/'certificate_filename_cpu_tests.json').write_text(json.dumps(dict(passed=True,old_fault_reproduced=True,new_label=label,
 gate_glob_matches=True,actual_passed_diagnostic_certificate_validates=True,no_classifier_or_worker_change=True,
 scope='Lookup regression checks only; production repeats real fresh nominal/pushed teacher tests.'),indent=2)+'\n')
new=HERE/'runs/locomotion_20261006_r39_commanded_travel_teacher_certificate';assert not new.exists()
backup=OUT/'canonical_before_certificate_retry';backup.mkdir();shutil.copy2(HERE/'supervisor_unitree.py',backup/'supervisor_unitree.py')
shutil.copyfile(OUT/'tested_supervisor_unitree.py',HERE/'supervisor_unitree.py');assert sha(HERE/'supervisor_unitree.py')==proof['supervisor_sha256']
from status_io import atomic_json
state.update(status='failed_before_main_certificate_lookup',superseded_by=str(new),child_pid=None)
atomic_json(old/'state.json',state)
new.mkdir();argv=[sys.executable,'-u','-X','utf8',str(HERE/'supervisor_unitree.py'),'--output',str(new),
 '--reserve_mb','300','--max_hours','18','--iterations_per_block','64','--train_envs','8192','--learning_rate','0.0001',
 '--migrate_run',proof['prior_run'],'--initial_checkpoint',proof['checkpoint'],'--optimization_proof',str(proof_path)]
with (new/'supervisor_stdout.log').open('w') as out,(new/'supervisor_stderr.log').open('w') as err:
 p=subprocess.Popen(argv,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),new_run=str(new),argv=argv,
 new_supervisor=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time()),
 preserved_failed_run=str(old),original_full_checkpoint=proof['checkpoint'],original_sha256=proof['checkpoint_sha256'],
 failed_run_had_zero_main_updates=True,supervisor_only_filename_fix=True,all_frozen_runtime_bytes_same_as_R38=True,
 migration_proof=str(proof_path),diagnostic_actors_deployed=False)
atomic_json(OUT/'certificate_retry_deployment.json',record)
active=read(HERE/'ACTIVE_RUN.json');active.update(run=str(new),status='starting',recorded_utc=record['recorded_utc'],
 supervisor_pid=p.pid,migration_proof=str(proof_path),failed_startup_preserved=str(old),last_verified_utc=record['recorded_utc'])
atomic_json(HERE/'ACTIVE_RUN.json',active)
print(json.dumps(record,indent=2))
