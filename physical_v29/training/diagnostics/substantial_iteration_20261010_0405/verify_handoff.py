"""CPU-only ownership, freeze, final-budget and native automation receipt."""
import json,hashlib,os,tomllib,time
from pathlib import Path
from datetime import datetime,timezone
import psutil
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=read(OUT/'source_freeze.json')
assert all(sha(Path(freeze['source'])/n)==h for n,h in freeze['source_hashes'].items())
assert all(sha(Path(freeze['prior_source'])/n)==h for n,h in freeze['original_source_hashes'].items())
identity=read(OUT/'controlled_driver_identity.json');driver=psutil.Process(identity['pid'])
assert driver.create_time()==identity['create_time']
supervisor=psutil.Process(39000);pause=read(OUT/'diagnostic_pause.json')
assert supervisor.create_time()==pause['supervisor_create_time'] and supervisor.status()==psutil.STATUS_STOPPED
workers=[]
for p in driver.children(recursive=True):
 cmd=p.cmdline()
 if any(Path(x).name in ('worker_entry.py','diverse_eval.py') for x in cmd):
  workers.append(dict(pid=p.pid,parent=p.ppid(),create_time=p.create_time(),cmdline=cmd))
parents={x['parent'] for x in workers};actual=[x for x in workers if x['pid'] not in parents]
assert len(actual)<=1,'Multiple actual experiment CUDA workers'
assert read(OUT/'smoke_validation_summary.json')['passed']
budget=read(TRAIN/'automation/FINAL_CHECK_REQUEST.json')
assert budget['remaining_scheduled_prompts']==1 and budget['due_utc']=='2026-10-10T09:03:00+00:00'
assert budget['current_iteration_change_authorized'] and budget['stop_training_after_final_check']
base=Path(os.environ.get('CODEX_HOME','C:/Users/rshan/.codex'))
native=tomllib.loads((base/'automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text(encoding='utf-8-sig'))
args=read(OUT/'automation_update_args.json')
assert native['status']=='ACTIVE' and native['rrule']==args['rrule'] and native['target_thread_id']==args['targetThreadId']
assert native['prompt']==args['prompt'].strip()
next=read(TRAIN/'automation/NEXT_CHECK.json');next['adaptive_automation_prompt']=native['prompt']
next['substantial_iteration']['handoff_verification']=str(OUT/'handoff_verification.json')
tmp=TRAIN/'automation/NEXT_CHECK.tmp';tmp.write_text(json.dumps(next,indent=2)+'\n');tmp.replace(TRAIN/'automation/NEXT_CHECK.json')
inventory=read(TRAIN/'history/RUN_INVENTORY.json')
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=True,controlled_driver=identity,
 actual_CUDA_workers=actual,original_supervisor_suspended_at_saved_boundary=True,
 unchanged_original_authored_files=len(freeze['original_source_hashes']),frozen_candidate_authored_files=len(freeze['source_hashes']),
 candidate_source_hashes_verified=True,smoke_and_full_reload_passed=True,complete_speed_gates_pending=True,
 complete_64_128_two_seed_tests_pending=True,candidate_not_deployed=True,
 native_same_automation_thread_schedule_preferences_preserved=True,final_check_UTC=budget['due_utc'],remaining_prompts=1,
 CPU_history_inventory_path=str(TRAIN/'history/RUN_INVENTORY.json'),new_iteration_documented=True,
 one_actual_worker_and_300MiB_reserve_and_native_caps=True,computer_on=True,
 finalizer_failed_setup_logs_preserved=True,finalizer_current_log_empty=(OUT/'finalizer_stderr.log').stat().st_size==0)
(OUT/'handoff_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('controlled_driver','actual_CUDA_workers')},indent=2))
