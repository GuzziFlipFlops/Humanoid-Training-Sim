"""Publish verified closeout evidence and prepare the same native heartbeat update."""
import hashlib,json,tomllib
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
MARKER='adaptive_check_20261009_2048_record'
AUTO=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
PREFIX_SHA='46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
 Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
assert all(p.is_file() for p in paths)
a=tomllib.loads(AUTO.read_text());prefix=a['prompt'].split('Latest dated five-hour check',1)[0]
assert hashlib.sha256(prefix.encode()).hexdigest()==PREFIX_SHA
assert a['rrule']=='FREQ=HOURLY;INTERVAL=5' and a['status']=='ACTIVE'
assert a['target_thread_id']=='01a08b12-e85a-7f73-8677-315c708cd9f7'
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
v=read(OUT/'final_learning_verification.json');pair=read(OUT/'closing_gate_review.json')['latest']
matched=read(OUT/'matched_row_route_review.json');validation=read(OUT/'experiment_validation.json')
upper=read(OUT/'final_upper/upper_joint_time_audit.json');um=read(OUT/'upper_matched_comparison.json')
fixed_response=read(OUT/'fixed_state_learning_response.json')
older_preflight=read(OUT/'older_branch_CPU_preflight.json')
assert matched['all_expected_components_complete'] and validation['passed'] and um['all_six128_actors_complete']
assert fixed_response['all_12_checkpoints_complete'],'Complete the fixed-state review of all twelve learned checkpoints'
assert older_preflight['all_CPU_preflight_checks_pass'],'Resolve earlier full-state preparation before publishing the next focused test'
assert not matched['all_candidate_gates_pass'],'Passing candidate requires a fresh frozen provenance/deployment review'
assert read(OUT/'final_upper/audit_set.json')['all_plots_visually_inspected']
assert all(read(OUT/f'seed{s}_{b}_upper/visual_inspection.json')['visually_inspected'] for s in [251,263] for b in ['control','shared','rows'])
assert v['supervisor_pid']==39000 and len(v['actual_worker_pids'])==1 and v['frozen_files_verified']==55 and v['protected_authored_worker_files_equal']==53
assert v['hash_proven_recovered_main_updates']==50 and v['inherited_actual_main_updates']==27507
resume=read(OUT/'diagnostic_resume.json');assert resume['same_supervisor']
assert any(b['full_load_proof']['source_sha256']==resume['checkpoint_sha256'] and b['updates']>=64 for b in v['blocks']),'Wait for genuine resumed main receiver'
previous=read(OUT/'previous_end_verification.json')
known={b['folder'] for b in previous['blocks']};new=[b for b in v['blocks'] if b['folder'] not in known]
delta=dict(retained_main_updates=sum(b['updates'] for b in new),actual_main_samples=sum(b['samples'] for b in new),
 active_collection_PPO_seconds=sum(b['active_seconds'] for b in new),worker_seconds=sum(b['worker_seconds'] for b in new),
 elapsed_endpoint_seconds=v['whole_chain_measurement_endpoint_unix']-previous['whole_chain_measurement_endpoint_unix'],
 scope='Completed and hash-received main blocks after the exact previous18:34 owned snapshot; diagnostics and partial updates excluded.')
delta['documented_pause_seconds']=v['documented_diagnostic_pause_seconds_excluded']-previous['documented_diagnostic_pause_seconds_excluded']
delta['chain_seconds_excluding_documented_pauses']=delta['elapsed_endpoint_seconds']-delta['documented_pause_seconds']
for label,seconds in [('active',delta['active_collection_PPO_seconds']),('worker_effective',delta['worker_seconds']),('whole_chain',delta['chain_seconds_excluding_documented_pauses'])]:
 assert seconds>0
 delta[label+'_samples_s']=delta['actual_main_samples']/seconds
def case(test,command):return next(c for c in pair[test+'_cases'] if c['command']==command)
def mcase(r,test,command):return next(c for c in r['tests'][test]['cases'] if c['command']==command)
fast=[case(t,3.5) for t in ['nominal','pushed']]
fasttext='/'.join(f"{c['signed_speed']:.3f}" for c in fast)
lattext='/'.join(f"{c['lateral_speed']:.3f}" for c in fast)
failures={t:[dict(command=c['command'],failures=c['failures']) for c in pair[t+'_cases'] if not c['passed']] for t in ['nominal','pushed']}
u=next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
stamp=datetime.now(timezone.utc).isoformat();actor=Path(pair['checkpoint']).parent.name
priority=read(OUT/'next_priorities_draft.json')['priorities']
decision=dict(recorded_utc=stamp,production_continuation='Same exact original full learner resumed; no experimental actor or gradient deployed',
 experiment='Final21-output-row derivative versus identical shared target and unchanged full-state control',
 all_expected_components_complete=True,branch_all_gates_pass=matched['branch_all_gates_pass'],
 deployment='Rejected for immediate deployment; preserve all evidence and useful passing actors',
 production_source_reward_gain_cap_actor_optimizer_reset=False,endpoint_established=False,next_priorities=priority)
write(OUT/'decision.json',decision)
evidence=dict(recorded_utc=stamp,exact_pair=pair,owned=read(OUT/'final_cpu/live_handoff.json'),learning=v,
 retained_main_totals=dict(current_updates=v['actual_retained_main_updates'],inherited_updates=27507,
 current_run_actual_samples=sum(b['samples'] for b in v['blocks']),inherited_sample_count_not_inferred_from_updates=True),
 measured_since_previous=delta,matched_validation=validation,physical=read(OUT/'neutral_teacher_response_analysis.json'),
 upper_motion=um,fixed_state_weight_response=fixed_response,
 primary_upper_reference_review=read(OUT/'reference_review.json'),
 earlier_complete_state_CPU_preparation=older_preflight,
 saved_checkpoint_trends=read(OUT/'saved_gate_trends.json'),
 scope='Saved gates/upper QA bind exact actors; newer partial learning is separately unqualified.')
write(OUT/'evidence.json',evidence)
gate_table='| Command m/s | Nominal failures | Pushed failures | Signed N/P m/s | True flight N/P % |\n|---|---|---|---|---|\n'
for n,p in zip(pair['nominal_cases'],pair['pushed_cases']):
 gate_table+=f"|{n['command']:g}|{', '.join(n['failures']) or 'Pass'}|{', '.join(p['failures']) or 'Pass'}|{n['signed_speed']:.3f}/{p['signed_speed']:.3f}|{100*n['flight']:.3f}/{100*p['flight']:.3f}|\n"
table='| Seed | Route | Updates | 3.5 lateral N/P m/s | 1.5 true flight N/P % | All gates N/P |\n|---|---|---|---|---|---|\n'
for r in matched['records']:
 n=mcase(r,'nominal',3.5);p=mcase(r,'push',3.5);fn=mcase(r,'nominal',1.5);fp=mcase(r,'push',1.5)
 table+=f"|{r['seed']}|{r['branch']}|{r['updates']}|{n['lateral_speed']:.3f}/{p['lateral_speed']:.3f}|{100*fn['true_flight']:.3f}/{100*fp['true_flight']:.3f}|{r['tests']['nominal']['passed']}/{r['tests']['push']['passed']}|\n"
summary=(f"The same R40 full learner and supervisor 39000 have one actual CUDA worker, "
 f"55 verified frozen hashes and 53 unchanged protected files. "
 f"{v['completed_main_updates']:,} completed + 50 recovered = {v['actual_retained_main_updates']:,} "
 f"retained current-run main updates, plus 27,507 inherited; "
 f"{v['observed_partial_main_updates']} partial updates remain separately unqualified. "
 f"Exact actor {actor}, SHA {pair['sha256']}, reaches {fasttext} m/s signed forward "
 f"and {lattext} m/s lateral at command 3.5. Current nominal/pushed passes: "
 f"{[c['passed'] for c in fast]}. Saved failures: {failures}. "
 "No complete 3.5, 4 or 5 m/s stage or causal native-limit endpoint is established.")
from closeout_report import render
report=render(OUT,v,pair,matched,delta,resume,upper,gate_table,table,priority,evidence)
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
handoff=(f"Latest dated five-hour check October 9, 20:48 UTC. Owned evidence {v['owned_snapshot_utc']}: "
 f"{summary}\n\n"
 "The final 21-output-row restriction and matched shared-feature route are rejected after exact "
 "64/128-update controls, two learning seeds and independently generated initial states. "
 "No diagnostic actor, target, reward, gain, cap or optimizer reset is deployed. "
 "Mandatory all-command upper timelines are inspected for the exact closing actor and all six "
 "128-update actors; held bias and asymmetry persist. Human/Unitree encoder likeness and "
 "optical gaze remain uncertified. "
 f"The same supervisor 39000 resumes after {resume['pause_seconds']:.1f} seconds. The genuine main "
 "receiver loads exact 1112 actor, critic, Adam and exploration with zero pre-update KL/command "
 "mismatch and all three auxiliaries at 20 batches per update. "
 f"Active/worker-effective/whole-chain rates are {v['active_PPO_samples_s']/1000:.2f}k/"
 f"{v['worker_effective_samples_s']/1000:.2f}k/{v['whole_chain_samples_s']/1000:.2f}k samples/s; "
 "documented suspensions are excluded only through each measurement endpoint. Original 35/1 gains, "
 "truthful 400/23 inference, native CAD caps, local ground and iteration-4 execution are retained.\n\n"
 f"Read diagnostics/{OUT.name}/REPORT.md, decision.json, matched_row_route_review.json, "
 "experiment_validation.json, current_row_response.json, fixed_state_learning_response.json, "
 "neutral_teacher_response_analysis.json, older_branch_CPU_preflight.json, final_upper, upper_matched_comparison.json, "
 "closing_gate_review.json and final_learning_verification.json; then read fresh ACTIVE_RUN/state.\n\n"
 "Next focused tests:\n"+'\n'.join(f'{i}. {p}' for i,p in enumerate(priority,1))+
 "\n\nKeep the same ACTIVE five-hour heartbeat, thread and notification preference. Only qualified "
 "5 m/s, or qualified 4 m/s with causal native limits blocking 5, ends training. At that verified "
 "endpoint save, pause both, prevent restart and brief with the computer on. Preserve all learned "
 "state, forward head, closed Fusion queued work and completed videos.")
nextcheck=read(TRAIN/'automation/NEXT_CHECK.json')
nextcheck.update(updated_utc=stamp,current_run=v['run'],current_checkpoint=pair['checkpoint'],diagnostic_report=str(OUT/'REPORT.md'),current_check=str(OUT),latest_evidence=summary,
 latest_gate_pair=pair,measured_since_previous=delta,current_retained_updates=v['actual_retained_main_updates'],inherited_updates=27507,
 throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},next_priorities=priority,latest_focus=handoff)
from prepare_next_handoff import refresh_canonical
refresh_canonical(nextcheck,OUT,TRAIN,v,pair,summary,priority,stamp)
write(TRAIN/'automation/NEXT_CHECK.json',nextcheck)
entry=f"\n\n<!-- {MARKER} -->\n## October9 20:48UTC five-hour check\n\n{summary}\n\nFinal-row coupled target tested with exact full-state64/128 controls and shared-route control in two seeds; deployment rejected. Current true-feedback physics still supports qualified267 body/arms with current head at1.5. Fixed-state head preservation does not protect fast outputs. No production reward/gain/cap/optimizer reset. All mandatory upper timelines and independent gates inspected; human/Unitree/gaze unverified. Main resumed after{resume['pause_seconds']:.1f}s with exact full state. Since previous snapshot:{delta['actual_main_samples']:,} genuine samples/{delta['retained_main_updates']} updates. [Report]({OUT.as_posix()}/REPORT.md).\n\n"+'\n'.join(f'{i}. {p}' for i,p in enumerate(priority,1))+'\n'
paths=[TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
 Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
a=tomllib.loads(AUTO.read_text());prefix=a['prompt'].split('Latest dated five-hour check',1)[0]
assert hashlib.sha256(prefix.encode()).hexdigest()==PREFIX_SHA and a['rrule']=='FREQ=HOURLY;INTERVAL=5' and a['status']=='ACTIVE'
args=dict(mode='update',id=a['id'],kind=a['kind'],name=a['name'],prompt=prefix+handoff,rrule=a['rrule'],status=a['status'],targetThreadId=a['target_thread_id'])
if 'notification_policy' in a:args['notificationPolicy']=a['notification_policy']
originals=[p.read_text(encoding='utf-8-sig') for p in paths]
for i,(p,old) in enumerate(zip(paths,originals)):
 if MARKER in old:continue
 assert p.read_text(encoding='utf-8-sig')==old,'User record changed during publication'
 p.write_text(old+entry if i<3 else entry+'\n'+old,encoding='utf-8')
write(OUT/'automation_update_args.json',args)
write(OUT/'records_receipt.json',dict(recorded_utc=stamp,records=[dict(path=str(p),sha256=sha(p)) for p in paths],report_sha256=sha(OUT/'REPORT.md'),preserved_durable_prefix_sha256=PREFIX_SHA,
 same_automation_id=a['id'],same_target_thread=a['target_thread_id'],same_five_hour_cadence=True,native_update_pending=True,decision=decision))
print(summary)
