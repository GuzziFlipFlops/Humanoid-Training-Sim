"""Final CPU-only receipts. This file has no process-launch or resume pathway."""
import csv,datetime,gzip,hashlib,json,re,subprocess,sys,tomllib
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;T=OUT.parents[1];R=T/'runs/locomotion_20261007_r40_goal_budget_full_state';V=T/'videos/final_motor_capture_20261010_0905'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
sys.path.insert(0,str(T/'history'));from record_inventory import frozen_path
state=read(R/'state.json');active=read(T/'ACTIVE_RUN.json');request=read(T/'automation/FINAL_CHECK_REQUEST.json');handoff=read(T/'automation/NEXT_CHECK.json')
assert state['status']=='stopped_by_user_budget' and state['total_training_iterations']==26023
assert active['status']=='stopped_by_user_budget' and state['resume_requires_new_user_authorization']
assert request['remaining_scheduled_prompts']==0 and request['final_check_status']=='completed_user_budget_stop'
assert handoff['automation_status']=='PAUSED' and not handoff['learning_authorized'] and not handoff['next_focused_tests']
assert (R/'NO_AUTOMATIC_RESTART.json').exists()
processes=[];fusion=[]
for p in psutil.process_iter(['pid','name','cmdline'],ad_value=None):
 cmd=p.info.get('cmdline') or []
 if any(Path(a).name in ('worker_unitree.py','eval_entry.py','record_motor_gui.py','supervisor_unitree.py','run_command_advantage_trial.py') for a in cmd):processes.append(p.info)
 if (p.info.get('name') or '').lower() in ('fusion360.exe','fusion.exe'):fusion.append(p.info['pid'])
assert not processes,processes
assert not fusion,fusion
config=read(R/'chain_config.json');sources=[]
for name,expected in config['sources_sha256'].items():
 path=frozen_path(R,name);actual=sha(path);assert actual==expected
 sources.append(dict(name=name,path=str(path),sha256=actual))
assert len(sources)==55
context=read(R/'resume_context.json')
for relative,expected in context['protected_hashes'].items():assert sha(R/'source/physical_v29'/relative)==expected
assert len(context['protected_hashes'])==53
cp=read(OUT/'full_state_save_receipt.json');assert sha(cp['saved_copy'])==sha(cp['source'])==cp['sha256']
automation_path=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
a=tomllib.loads(automation_path.read_text());assert a['status']=='PAUSED' and a['target_thread_id']=='01a08b12-e85a-7f73-8677-315c708cd9f7'
assert a['id']=='humanoid-training-research-and-repair-every-four-hours'
automation=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),app_update_confirmed=True,id=a['id'],status=a['status'],name=a['name'],rrule_preserved=a['rrule'],same_thread=a['target_thread_id'],notification_preference=a.get('notification_policy'),toml_sha256=sha(automation_path),prompt_sha256=hashlib.sha256(a['prompt'].encode()).hexdigest(),zero_remaining_scheduled_prompts=True)
write(OUT/'automation_pause_receipt.json',automation)
manifest=read(V/'manifest.json');assert len(manifest['clips'])==10 and len(manifest['encoded_verified'])==10
assert read(OUT/'decode_verification.json')['passed'] and read(OUT/'video_visual_QA.json')['all_ten_clips_inspected']
csv_counts=[]
for p in [OUT/'selected_diverse_nominal/motor_samples_100.csv.gz',OUT/'selected_diverse_nominal/motor_samples_350.csv.gz']+[V/f"locomotion_{round(c['speed']*100):03d}"/f"motor_samples_{round(c['speed']*100):03d}.csv.gz" for c in manifest['clips']]:
 with gzip.open(p,'rt',encoding='utf-8') as f:
  header=next(f);count=sum(1 for _ in f)
 expected=128000 if p.parent.name=='selected_diverse_nominal' else 2400
 assert count==expected,(p,count,expected);csv_counts.append(dict(path=str(p),rows=count,columns=len(next(csv.reader([header])))))
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
memory=psutil.virtual_memory()
inventory=read(T/'history/RUN_INVENTORY.json')
inventory_summary=dict(physical_runs=len(inventory['physical_runs']),physical_operations=sum(len(r['operations']) for r in inventory['physical_runs']),missing_manifest_artifacts=sum(p['actual_sha256'] is None for r in inventory['physical_runs'] for p in r['frozen_source_verification']),differing_manifest_hashes=sum(p['actual_sha256'] is not None and not p['matches'] for r in inventory['physical_runs'] for p in r['frozen_source_verification']),read_warnings=inventory['read_warnings'])
assert inventory_summary['differing_manifest_hashes']==0 and not inventory_summary['read_warnings']
receipt=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),classification='completed user-budget stop; not speed success or actuator ceiling',owned_supervisor_and_workers_absent=True,matching_processes=processes,fusion_gui_pids=fusion,automation=automation,request_status=request['final_check_status'],remaining_scheduled_prompts=0,learning_authorized=False,restart_requires_new_explicit_user_authorization=True,computer_on=True,active_status=active['status'],run_status=state['status'],checkpoint_sha256=cp['sha256'],frozen_sources=sources,protected_authored_files_verified=53,video_count=10,all_decode_visual_sample_alignment_checks_pass=True,population_trials_per_command=32,native_motor_samples_Hz=200,csv_counts=csv_counts,system_memory=dict(available_MiB=memory.available//1048576,total_MiB=memory.total//1048576),gpu=dict(query_returncode=gpu.returncode,total_used_free_utilization=gpu.stdout.strip(),error=gpu.stderr.strip()),inventory=inventory_summary)
write(OUT/'final_status_receipt.json',receipt)
# Verify every deliverable's local link after both final receipts exist.
missing=[]
for p in [OUT/'REPORT.md',OUT/'MOTOR_TABLES.md',V/'README.md']:
 for target in re.findall(r'\]\(([^)]*)\)',p.read_text(encoding='utf8')):
  if target.startswith(('http:','https:','#')):continue
  destination=(p.parent/target.split('#')[0]).resolve()
  if not destination.exists():missing.append(dict(document=str(p),target=target))
 # artifact_hashes is generated below.
missing=[m for m in missing if m['target']!='artifact_hashes.json'];assert not missing,missing
artifacts=[]
for base in [OUT,V]:
 for p in sorted(base.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or p.name=='artifact_hashes.json' or 'playback_asset' in p.parts:continue
  if p.suffix.lower() not in ('.pt','.json','.npz','.csv','.gz','.md','.mp4','.png','.jpg','.svg','.log','.txt','.py'):continue
  artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
write(OUT/'artifact_hashes.json',dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifacts=artifacts,frozen_source_hashes=sources,hash_manifest_excludes_itself=True))
print(json.dumps({k:v for k,v in receipt.items() if k not in ['frozen_sources','csv_counts']},indent=2))
print(f'Final local links and {len(artifacts)} artifact hashes verified.')
