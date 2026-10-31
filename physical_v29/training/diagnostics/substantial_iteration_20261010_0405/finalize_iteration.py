"""CPU-only completion records. No learner launch, source edit or gate relaxation."""
import json,hashlib,subprocess,sys,time,shutil
from pathlib import Path
from datetime import datetime,timezone
import psutil
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=OUT/'matched_command_advantage'
RUN=TRAIN/'runs/locomotion_20261010_r41_command_conditioned_advantage'
OLD=TRAIN/'diagnostics/adaptive_check_20261010_0149'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
identities=[dict(pid=p.pid,create_time=p.info['create_time'],cmdline=p.info['cmdline'])
 for p in psutil.process_iter(['cmdline','create_time'])
 if any(Path(x).name=='run_comparison.py' and Path(x).resolve()==(OUT/'run_comparison.py').resolve() for x in p.info['cmdline'] or [])]
shim_identities=identities
parents={psutil.Process(x['pid']).ppid() for x in shim_identities}
identities=[x for x in shim_identities if x['pid'] not in parents]
assert len(identities)==1
identities[0]['launcher_and_actual_driver']=[dict(x) for x in shim_identities]
write(OUT/'controlled_driver_identity.json',identities[0])
shutil.copy2(OUT/'run_comparison.py',RUN/'supervisor_unitree.py')
shutil.copy2(TRAIN/'runs/locomotion_20261007_r40_goal_budget_full_state/vram_ramp.py',RUN/'vram_ramp.py')
oldconfig=read(TRAIN/'runs/locomotion_20261007_r40_goal_budget_full_state/chain_config.json')
config=dict(oldconfig,experiment_only=True,max_hours=0,
 exact_user_stop_utc='2026-10-10T09:03:00+00:00',comparison_path=str(ROOT),
 executed_driver=str(OUT/'run_comparison.py'),driver_identity=identities[0],
 source_parent='locomotion_20261007_r40_goal_budget_full_state',
 actual_comparison_environment_count=4096,target_environments_after_qualification=8192,
 executed_training_update='command_conditioned_advantage_v1',source_freeze_manifest=str(RUN/'training_update_manifest.json'))
config['sources_sha256']=dict(oldconfig['sources_sha256'])
config['sources_sha256'].update({'worker_unitree.py':sha(RUN/'source/physical_v29/training/worker_unitree.py'),
 'command_advantage.py':sha(RUN/'source/physical_v29/training/command_advantage.py'),
 'supervisor_unitree.py':sha(RUN/'supervisor_unitree.py')})
write(RUN/'chain_config.json',config)
write(RUN/'state.json',dict(status='controlled_iteration_in_progress',supervisor_pid=identities[0]['pid'],
 supervisor_create_time=identities[0]['create_time'],actual_driver=str(OUT/'run_comparison.py'),
 checkpoint=None,stage='run_350',experiment_only=True,not_deployed=True,
 diagnostics_not_main_samples=True,total_training_iterations=0,report=str(OUT/'REPORT.md')))
deadline=time.monotonic()+10200
while not ((ROOT/'completion.json').exists() or (ROOT/'failure.json').exists()):
 assert time.monotonic()<deadline,'Comparison completion missing; final check must inspect actual ownership'
 time.sleep(10)
# Wait for the finally receipt: preserve whether the original learner actually resumed.
deadline=time.monotonic()+180
while not (OUT/'diagnostic_resume.json').exists():
 assert time.monotonic()<deadline,'Original resume receipt missing; do not claim liveness'
 time.sleep(2)
subprocess.run([sys.executable,str(OUT/'watch_comparison.py')],check=True,stdout=subprocess.DEVNULL)
data=read(OUT/'live_comparison.json');freeze=read(OUT/'source_freeze.json')
assert all(sha(Path(freeze['source'])/n)==h for n,h in freeze['source_hashes'].items())
assert all(sha(Path(freeze['prior_source'])/n)==h for n,h in freeze['original_source_hashes'].items())
checks=[]
if data['completed']:
 import torch
 for seed in (347,359):
  a,b=[torch.load(ROOT/f'seed{seed}_{branch}_064_train/first_rollout_normalization.pt',weights_only=False,map_location='cpu') for branch in ('control','candidate')]
  parity={k:torch.equal(a[k],b[k]) for k in ('raw_advantage','old_global_normalized','labels','policy_observations','returns','values','rewards','dones','actions')}
  assert all(parity.values()),parity
  checks.append(dict(seed=seed,first_preupdate_rollout_bitwise_equal=parity,
   advantages_intentionally_differ=not torch.equal(a['actual_advantage'],b['actual_advantage'])))
 write(OUT/'paired_first_rollout_proof.json',dict(passed=True,checks=checks))
 for seed in (347,359):
  for branch in ('control','candidate'):
   checkpoint=ROOT/f'seed{seed}_{branch}_128_train/candidate.pt'
   pair=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),nominal=str(ROOT/f'seed{seed}_{branch}_128_nominal'),
    pushed=str(ROOT/f'seed{seed}_{branch}_128_push'))
   review=OUT/f'seed{seed}_{branch}_128_gate_review.json'
   write(review,dict(run=str(RUN),latest=pair,promotions=[]))
   folder=OUT/f'upper_seed{seed}_{branch}_128'
   for tool in ('upper_joint_time_audit.py','upper_bias_audit.py'):
    with (OUT/f'{folder.name}_{tool}.log').open('w') as log:
     subprocess.run([sys.executable,str(OLD/tool),'--gate-review',str(review),'--output-dir',str(folder)],
      stdout=log,stderr=subprocess.STDOUT,check=True)
    if tool=='upper_joint_time_audit.py':
     with (OUT/f'{folder.name}_pushed.log').open('w') as log:
      subprocess.run([sys.executable,str(OLD/tool),'--gate-review',str(review),'--output-dir',str(folder/'pushed'),
       '--trace-test','pushed'],stdout=log,stderr=subprocess.STDOUT,check=True)
   write(folder/'visual_QA_pending.json',dict(actual_data_and_plots_saved=True,visually_inspected=False,
    final_check_must_inspect=True,human_Unitree_optical_gaze_certified=False))
full=[p for p in data['paired_tests'] if p['updates']==128]
eligible=data['completed'] and len(full)==8 and all(p['candidate_all_pass'] for p in full)
losses=[{k:p[k] for k in ('seed','updates','test','lost_passing_commands','gained_passing_commands')} for p in data['paired_tests']]
decision=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),implemented_iteration='command_conditioned_advantage_v1',
 completed=data['completed'],failure=data['failure'],candidate_full_gate_contract_pass=eligible,
 diagnostic_actor_deployed=False,original_learner_resume=read(OUT/'diagnostic_resume.json'),paired_cases=losses,
 conclusion=('All candidate128 gate sets pass; preserve candidate for final exact assessment. No automatic deployment is claimed.' if eligible else
  'Reject candidate for ordinary deployment: complete independent gate contract does not pass. Substantial iteration was executed, with evidence preserved.'),
 source_hashes_all_verify=True,paired_rollout_proof=checks,upper_time_series_saved=data['completed'],
 upper_visual_QA_pending=data['completed'],deadline_utc='2026-10-10T09:03:00+00:00',
 final_check_and_stop_not_extended=True,no_speed_success_or_ceiling_inferred=True)
write(OUT/'decision.json',decision)
candidate=[x for x in data['training'] if '_candidate_' in x['operation']]
saved=[ROOT/x['operation']/'candidate.pt' for x in candidate if (ROOT/x['operation']/'candidate.pt').exists()]
write(RUN/'state.json',dict(status='controlled_iteration_complete_unpromoted' if data['completed'] else 'controlled_iteration_failed_preserved',
 supervisor_pid=None,child_pid=None,checkpoint=str(saved[-1]) if saved else None,
 stage='run_350',experiment_only=True,not_deployed=True,diagnostics_not_main_samples=True,
 actual_candidate_updates=sum(x['updates'] for x in candidate),actual_candidate_samples=sum(x['actual_samples'] for x in candidate),
 total_training_iterations=0,decision=str(OUT/'decision.json')))
footer='\n\n## Completed comparison and decision\n\n'+decision['conclusion']+'\n\n'
footer+=f"Comparison completed={data['completed']}. Exact changed-update validation, full-load/KL/command/auxiliary and memory checks are saved per operation. Candidate diagnostic updates are excluded from main learner totals. Source hashes verify. Original supervisor resumes after {decision['original_learner_resume']['pause_seconds']:.1f} seconds; final owned liveness must still be read fresh.\n\n"
footer+='The matched128 ordinary and individually seeded case comparisons are in live_comparison.json; decision.json lists gained and lost passing commands. New actual upper angles/targets/velocities and noncancelling bias plots are saved for every128 actor. Visual inspection is pending the final check; no human/Unitree or optical-gaze certification is inferred.\n\n'
footer+='Your single final check remains09:03UTC, then stop this attempt and the automation. No new scheduled prompt or extension was created. The final video/motor-shaft data/table request remains pending if progress is insufficient.\n'
with (OUT/'REPORT.md').open('a',encoding='utf-8') as f:f.write(footer)
entry='\n\n<!-- substantial_iteration_20261010_0405_complete -->\n## October10 substantial command-conditioned PPO iteration\n\n'+footer.replace('## Completed comparison and decision','')+f'\n[Evidence]({OUT.as_posix()}/REPORT.md).\n'
records=[TRAIN/'ITERATION_HISTORY.md',TRAIN.parents[1]/'docs/robot/TRAINING_JOURNAL_20261001.md',
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]
for p in records:
 assert p.exists(),p
 text=p.read_text(encoding='utf-8-sig')
 if 'substantial_iteration_20261010_0405_complete' not in text:p.write_text(text+entry,encoding='utf-8')
next=read(TRAIN/'automation/NEXT_CHECK.json')
next['substantial_iteration'].update(status=decision['conclusion'],decision=str(OUT/'decision.json'),
 completed=data['completed'],upper_QA_pending=data['completed'],latest_gate_comparison=str(OUT/'live_comparison.json'))
next['substantial_iteration_completion_utc']=decision['recorded_utc']
assert next['remaining_authorized_automation_prompts']==1 and next['next_scheduled_check_utc']=='2026-10-10T09:03:00+00:00'
write(TRAIN/'automation/NEXT_CHECK.json',next)
subprocess.run([sys.executable,str(TRAIN/'history/record_inventory.py')],check=True,
 stdout=(OUT/'history_inventory.log').open('w'),stderr=subprocess.STDOUT)
write(OUT/'completion_record_receipt.json',dict(completed=True,report_sha256=sha(OUT/'REPORT.md'),
 records=[str(p) for p in records],CPU_inventory_refreshed=True,no_further_automation_created=True))
print(json.dumps(decision,indent=2),flush=True)
