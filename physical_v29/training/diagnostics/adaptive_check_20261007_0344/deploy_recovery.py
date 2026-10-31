"""Start fresh source-frozen R40 only after sequential diagnostics have exited."""
import json,subprocess,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import psutil
out=Path(__file__).resolve().parent;train=out.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert read(out/'resume_cpu_tests.json')['passed']
assert read(out/'matched_gain/completion.json')['completed']
for p in psutil.process_iter(['cmdline']):
 assert not any(Path(x).name in ['worker_unitree.py','shoulder_gain_candidate.py'] for x in (p.info['cmdline'] or [])),'Existing CUDA worker'
old=read(out/'initial_owned_observation.json');prior=Path(old['active_run']['run']);state=read(prior/'state.json')
assert state['status']=='failed' and state.get('child_pid') is None
checkpoint=state['checkpoint'];run=train/'runs/locomotion_20261007_r40_goal_budget_full_state'
assert not run.exists(),'Unknown preexisting run; do not duplicate'
argv=[sys.executable,'-u','-X','utf8',str(train/'supervisor_unitree.py'),'--output',str(run),'--reserve_mb','300','--max_hours','0','--iterations_per_block','64','--train_envs','8192','--learning_rate','.0001','--resume_run',str(prior)]
run.mkdir()
with (run/'supervisor_stdout.log').open('w',encoding='utf-8') as so,(run/'supervisor_stderr.log').open('w',encoding='utf-8') as se:
 proc=subprocess.Popen(argv,stdout=so,stderr=se,cwd=str(train.parents[1]),creationflags=subprocess.CREATE_NO_WINDOW)
 identity=psutil.Process(proc.pid)
 (out/'deployment.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),prior_run=str(prior),checkpoint=checkpoint,checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),launcher_pid=proc.pid,launcher_create_time=identity.create_time(),argv=argv,supervisor_only_maintenance=True,shoulder_gain_candidate_deployed=False,full_actor_critic_Adam_exploration_and_teachers_retained=True,max_hours=0,elapsed_budget_disabled=True,stall_and_memory_guards_retained=True,unqualified_stage='run_350'),indent=2))
deadline=time.monotonic()+35
while not (run/'state.json').exists():
 assert time.monotonic()<deadline and proc.poll() is None;time.sleep(1)
s=read(run/'state.json');assert s['status'] in ('running','starting')
active=dict(run=str(run),status=s['status'],recorded_utc=datetime.now(timezone.utc).isoformat(),prior_run=str(prior),initial_full_checkpoint=checkpoint,selected_full_checkpoint=checkpoint,selected_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),reason='Goal-driven budget and exact full-state teacher-context recovery after18h elapsed guard interrupted R39; no physical/reward/gain change',repair_report=str(out/'REPORT.md'),same_learned_state_and_stage=True,supervisor_pid=s['supervisor_pid'],launcher_pid=proc.pid,full_context_proof=str(out/'resume_context_cpu.json'))
t=(train/'ACTIVE_RUN.json').with_suffix('.tmp');t.write_text(json.dumps(active,indent=2)+'\n');t.replace(train/'ACTIVE_RUN.json')
print(json.dumps(dict(new_run=str(run),supervisor_pid=s['supervisor_pid'],status=s['status'],checkpoint=checkpoint)))
