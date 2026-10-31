"""Check native automation, chronological records and fresh owned training evidence."""
import hashlib,json,re,tomllib
from pathlib import Path
from datetime import datetime,timezone
out=Path(__file__).resolve().parent;train=out.parents[1];root=train.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,value):
 assert value,name
 checks.append(name)
before=tomllib.loads((out/'previous_automation.toml').read_text(encoding='utf-8'))
native=Path('C:/Users/rshan/.codex/automations')/before['id']/'automation.toml'
after=tomllib.loads(native.read_text(encoding='utf-8'));args=read(out/'automation_update_args.json')
for key in ['id','kind','name','target_thread_id','rrule','created_at','notification_policy','model','reasoning_effort']:
 check('native automation preserves '+key,after.get(key)==before.get(key))
check('same ACTIVE five-hour cadence',after['status']=='ACTIVE' and after['rrule']=='FREQ=HOURLY;INTERVAL=5')
check('native automation prompt equals prepared update',after['prompt']==args['prompt'])
prefix=after['prompt'].split('Latest dated five-hour check',1)[0]
check('durable authority/upper audits/5 or conditional4 endpoint preserved',hashlib.sha256(prefix.encode()).hexdigest()=='46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3')
automation=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=True,path=str(native),automation_id=after['id'],kind=after['kind'],status=after['status'],target_thread_id=after['target_thread_id'],rrule=after['rrule'],prompt_sha256=hashlib.sha256(after['prompt'].encode()).hexdigest(),notification_preference_unchanged=True,duplicate_created=False)
write(out/'automation_verification.json',automation)
inventory=read(out/'inventory.log')
check('40 physical runs inventoried',inventory['physical_runs']==40)
check('zero differing inventory hashes',inventory['differing_manifest_hashes']==0)
check('zero inventory read warnings',not inventory['read_warnings'])
check('eight historical missing manifests still explicit',inventory['missing_manifest_artifacts']==8)
write(out/'inventory_verification.json',dict(recorded_utc=automation['recorded_utc'],passed=True,**inventory))
v=read(out/'liveness_learning_verification.json');live=read(out/'liveness_cpu/live_handoff.json')
reviewed=read(out/'closing_gate_review.json')['latest'];newer=read(out/'liveness_gate_review.json')['latest']
check('fresh same supervisor running',v['supervisor_pid']==39000 and live['state']['status']=='running')
check('fresh one actual Isaac worker',len(live['actual_worker_pids'])==1)
check('fresh all55 frozen hashes',len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values()))
check('fresh53 authored worker files unchanged',v['protected_authored_worker_files_equal']==53)
check('recovered50 inherited27507 and unsaved32 excluded',v['hash_proven_recovered_main_updates']==50 and v['inherited_actual_main_updates']==27507 and v['previous_interrupted_unsaved_updates_excluded']==32)
check('no production diagnostic actor or gains',not v['diagnostic_gains_and_actors_deployed'])
check('every retained main full-load/KL/command/auxiliary check',all(all(b['full_load_proof'][k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal']) and b['all_preupdate_KL_and_command_mismatch_zero'] and b['three_auxiliaries_batches_per_update']==20 for b in v['blocks']))
check('completed923 resumed exact920',any(Path(b['folder']).name=='923_run_350_train_305' and b['updates']==64 and b['environments']==8192 and b['full_load_proof']['source_sha256']==reviewed['sha256'] for b in v['blocks']))
check('last real main memory reserve',v['blocks'][-1]['minimum_free_gpu_MiB']>=300)
decision=read(out/'decision.json')
check('safe no-deploy/no-endpoint decision',not decision['deploy'] and not decision['production_change'] and not decision['endpoint_achieved'])
check('60 preserved cross-evidence validations',read(out/'physical_validation.json')['passed'] and read(out/'physical_validation.json')['count']==60)
plot_count=0
for folder in ['initial_upper','final_upper']:
 audit=read(out/folder/'audit_set.json')
 check(folder+' explicit visual QA complete',audit['all_plots_visually_inspected'] and not audit['plots_need_visual_QA'])
 n=0
 for path in audit['visual_QA_receipts']:
  for p in read(path)['plots']:
   check('preserved inspected upper image '+p['path'],p['visually_inspected'] and sha(p['path'])==p['sha256']);n+=1
 check(folder+'14 inspected plots',n==14);plot_count+=n
for p in read(out/'diagnostic_plot_visual_QA.json')['plots']:
 check('preserved inspected diagnostic plot '+p['path'],p['visually_inspected'] and sha(p['path'])==p['sha256'])
histories=[train/'ITERATION_HISTORY.md',root/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
for p in histories:check('new chronological entry exactly once '+str(p),p.read_text(encoding='utf-8').count('<!-- adaptive_check_20261009_0547_record -->')==1)
links=0
for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(out/'REPORT.md').read_text(encoding='utf-8')):
 if target.startswith(('http:','https:','#')):continue
 check('report local artifact exists '+target,(out/target).exists());links+=1
h=read(train/'automation/NEXT_CHECK.json')
check('reviewed920 gate and all-command upper actor bound',h['selected_actor_sha256']==h['mandatory_all_speed_trace_sha256']==reviewed['sha256'])
check('917 native physical attribution remains separate',h['physical_coupling_sha256']=='58ec1bf8f971433a3bd21e20cc82b5ad0f8d0b74aa27355a0ce76e4b3de0e746')
check('fulfilled video request preserved',h['recording_request']['status']=='complete')
check('no stage/limit qualification inferred',not h['current_stage_qualified'] and not h['velocity_ceiling_proven'])
newcases={t:[{k:c[k] for k in ['command','passed','failures','signed_speed','lateral_speed','flight']} for c in newer[t+'_cases']] for t in ['nominal','pushed']}
fresh=dict(recorded_utc=v['owned_snapshot_utc'],checkpoint=newer['checkpoint'],sha256=newer['sha256'],cases=newcases,
 fully_upper_reviewed_checkpoint=reviewed['checkpoint'],full_upper_review_not_attributed_to_newer=newer['sha256']!=reviewed['sha256'],
 qualification_never_attributed_to_partial_actors=True)
write(out/'closing_newer_pair.json',fresh)
h.update(latest_observation=v['owned_snapshot_utc'],latest_liveness_utc=v['owned_snapshot_utc'],latest_liveness=str(out/'liveness_cpu/live_handoff.json'),final_owned_observation=str(out/'liveness_cpu/live_handoff.json'),
 completed_main_updates=v['completed_main_updates'],new_main_updates=v['completed_main_updates'],actual_retained_current_run_main_updates=v['actual_retained_main_updates'],partial_observed_main_updates=v['observed_partial_main_updates'],
 current_owned_operation=v['operation_path'],current_checkpoint=live['state']['checkpoint'],current_checkpoint_sha256=sha(live['state']['checkpoint']),
 newest_complete_pair=str(out/'closing_newer_pair.json'),newest_complete_pair_checkpoint=newer['checkpoint'],newest_complete_pair_sha256=newer['sha256'],
 resumed_learning_evidence=str(out/'liveness_learning_verification.json'),automation_verified_active=True,automation_verification=str(out/'automation_verification.json'),inventory_verification=str(out/'inventory_verification.json'),final_verification=str(out/'closeout_verification.json'))
write(train/'automation/NEXT_CHECK.json',h)
result=dict(recorded_utc=automation['recorded_utc'],passed=True,checks=checks,owned_snapshot_utc=v['owned_snapshot_utc'],supervisor_pid=39000,actual_worker_pids=live['actual_worker_pids'],frozen_files_verified=55,
 current_operation=v['operation_path'],completed_main_updates=v['completed_main_updates'],recovered_main_updates=50,retained_main_updates=v['actual_retained_main_updates'],inherited_actual_main_updates=27507,
 partial_observed_main_updates=v['observed_partial_main_updates'],fully_reviewed_actor=reviewed['checkpoint'],newest_matching_actor=newer['checkpoint'],last_completed_main=v['blocks'][-1],upper_QA_images=plot_count,verified_report_links=links,automation=automation,inventory=inventory)
write(out/'closeout_verification.json',result)
e=read(out/'evidence.json');e.update(closeout_verification=str(out/'closeout_verification.json'),automation_verification=str(out/'automation_verification.json'),inventory_verification=str(out/'inventory_verification.json'),closing_newer_pair=str(out/'closing_newer_pair.json'));write(out/'evidence.json',e)
print(json.dumps(dict(passed=True,checks=len(checks),retained_main_updates=v['actual_retained_main_updates'],current_operation=v['operation_path'],newest_complete_actor=Path(newer['checkpoint']).parent.name,newer_cases=newcases),indent=2))
