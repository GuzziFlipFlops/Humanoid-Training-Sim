"""Evidence-backed original-learner handoff after a completed diagnostic decision."""
import json,hashlib,tomllib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
live=read(OUT/'final_cpu/live_handoff.json');learning=read(OUT/'final_learning_verification.json')
gates=read(OUT/'closing_gate_review.json');pair=gates['latest'];decision=read(OUT/'decision.json');resume=read(OUT/'diagnostic_resume.json')
matched=read(OUT/'matched_heading_review.json');physical=read(OUT/'heading_analysis.json')
assert not decision['production_change'] and resume['same_supervisor'] and not resume['diagnostic_actors_deployed']
assert learning['supervisor_pid']==39000 and learning['frozen_files_verified']==55 and len(live['actual_worker_pids'])==1
assert live['state']['status']=='running' and all(live['frozen_source_integrity'].values())
qa=read(OUT/'plot_qa.json')
for name in ['upper_joint_angles.png','upper_joint_velocities.png']:
 f=OUT/'final_upper'/name;digest=hashlib.sha256(f.read_bytes()).hexdigest()
 assert any(x['path']==str(f) and x['sha256']==digest and x['visually_inspected'] for x in qa['files'])
upper=read(OUT/'final_upper/upper_joint_time_audit.json')
bias=read(OUT/'final_upper/upper_bias_audit.json')
bias25=next(r for r in bias['records'] if r['test']=='nominal' and r['speed_m_s']==2.5)
bs=[bias25['joints'][side+'_shoulder_pitch'] for side in ['left','right']]
bias_text=f"Latest exact actor at2.5m/s has median shoulder pitch p95-p05 total ranges{bs[0]['total_p95_p05_deg']:.2f}/{bs[1]['total_p95_p05_deg']:.2f}deg,slow-bias ranges{bs[0]['slow_bias_p95_p05_deg']:.2f}/{bs[1]['slow_bias_p95_p05_deg']:.2f}deg and faster residual ranges{bs[0]['cycle_residual_p95_p05_deg']:.2f}/{bs[1]['cycle_residual_p95_p05_deg']:.2f}deg."
write(OUT/'final_upper_summary.json',[r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s'] in (1.,2.5,3.,3.5)])
rates=[learning[k]/1000 for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']]
cp=Path(pair['checkpoint']);actor=cp.parent.name;cur=[pair[k][-1] for k in ['nominal_cases','pushed_cases']]
fail={k:[dict(command=c['command'],failures=c['failures']) for c in pair[k] if not c['passed']] for k in ['nominal_cases','pushed_cases']}
last=learning['blocks'][-1];recovered=learning['hash_proven_recovered_main_updates']
summary=f"The same R40 learner and supervisor 39000 are running, with one actual worker and 55 verified frozen files. It retains {learning['completed_main_updates']} completed + {recovered} hash-proven recovered = {learning['actual_retained_main_updates']} current-run updates, plus 27,507 inherited. Exact actor {actor} reaches {cur[0]['signed_speed']:.3f}/{cur[1]['signed_speed']:.3f} m/s in matching nominal/pushed 3.5-command tests, with {cur[0]['lateral_speed']:.3f}/{cur[1]['lateral_speed']:.3f} m/s lateral drift. The stage remains unqualified. Saved gates never qualify newer partial actors. The 2.5 m/s cases recovered in both tests; 0.15 net travel, 1.5 geometric flight and 3.0 lateral direction still fail, and nominal 1.0 support fails."
table=[]
for r in matched['comparisons']:
 vals=[r[k]['cases'][-1]['statistics']['reset_frame_lateral_speed_m_s'] for k in ['nominal','pushed']]
 failures='; '.join(k+': '+','.join(f"{c['command']:g}({'/'.join(c['failures'])})" for c in r[k]['cases'] if not c['passed']) for k in ['nominal','pushed'])
 table.append(f"|{r['branch']} {r['updates']}|{vals[0]:.3f}/{vals[1]:.3f}|{failures}|")
report=f'''# Five-hour humanoid check — October 7, 13:44 UTC

{summary}

{decision['summary']}

## Current training and ownership

Owned observation: {live['recorded_utc']}. Exact saved checkpoint SHA`{pair['sha256']}`. Matching tests contain320 robots over20 seconds,32 trials per command. Current/prior failures: `{json.dumps(fail)}`. Original nominal resets are cloned; this is not final-endpoint initial-state diversity. Separately certified3m actor349 and all-prior actor267 remain preserved. No3.5,4,5m or native-limit endpoint is established.

The same supervisor resumes after{resume['pause_seconds']:.1f}s of diagnostic suspension, including{resume['worker_drain_seconds']:.1f}s worker drain. Exact actor/critic/Adam/exploration loads, both prior teachers and upper20batches/update remain. All pre-update KL and observed/action-command differences are zero. No production source/reward/gain/cap/optimizer/actor change or diagnostic actor is deployed. Original local ground,400-input23-output inference, teacher context, gains35/1 and iteration4 remain.

Latest completed main uses{last['environments']} environments. Minimum free GPU{last['minimum_free_gpu_MiB']}MiB,host{last['minimum_host_available_MiB']}MiB,private growth{last['private_memory_growth_MiB']}MiB,Torch allocation growth{last['torch_allocation_growth_MiB']}MiB. Rates{rates[0]:.2f}k active collection+PPO/{rates[1]:.2f}k worker-effective/{rates[2]:.2f}k whole-chain samples/s have different denominators. Exact successful diagnostic suspensions are excluded through the measurement endpoint; earlier untimestamped brief bookkeeping pause remains conservatively included. Nominal255 wrapper includes this pause. Capacity,smoke and diagnostic learning never count as main. Current recovered updates are{recovered},not the dated41; receiver hashes/full-state equality are verified.32 interrupted unsaved R39 updates remain excluded. Target8192,reserve300MiB,host fallback and180 continuous healthy TRAINING seconds remain.

## Controlled heading and course response

The exact254 actor SHA`5a96798e87070c606610055860150fe5a8dca52b46b5b8e3c7475b088d378e96` is tested in384 robots,12 cohorts of32,20 seconds with truthful commands/free base/gravity/native CAD limits. No motor pulse,support,observation spoof or hybrid is used. All survive without resets. Command mismatch0; reward term sums reconstruct within9.54e-7. Every native5ms driving-speed clip statistic is zero in this experiment. This is actor-specific evidence,not proof5m is feasible or of a motor ceiling.

At3.5, original half-gain heading feedback gives0.386m/s instantaneous fixed-frame lateral speed after5s warmup. Heading gain2 gives0.158. Adding unit-gain course feedback to the original half-gain heading term gives0.128m/s,3.574m/s forward,40.77% genuine flight and+0.207 full mean reward rate. Fixed left/right desired yaw commands produce large opposite turns,confirming actual yaw-channel sensitivity. The physical instantaneous-velocity averages differ from the independent verifier's sampled-pose finite differences and do not qualify a stage.

The local candidate adds wrapped velocity-course feedback only forx>3m/s, using measured world planar velocity,zero course term below0.2m/s and the existing+/-0.2rad/s yaw-command bound. Earlier commands through3m/s stay bitwise unchanged in60 CPU fixtures; tick idempotence/held commands are tested.34 earlier heading/wrap/bound fixtures pass. This requires a velocity measurement/estimate in command generation; it is not merely a training-only loss or a copied Unitree formula. Original checkpoint signatures explicitly DO NOT certify the diagnostic override.

## Full-state learning comparison

Both4096-env branches begin the exact254 actor/critic/Adam/exploration. Actual64-env4-update PPO and reload pass. Complete64/128-update tests use unchanged320-robot20s nominal/pushed gates,extra push seed1901 and nominal-physics seed2301 with32 independently generated small q/qd initial states per command. Arrays,seeds,unique counts and hashes are saved. These bounded resets give limited diagnostic diversity; they do not qualify a future endpoint actor. All comparisons retain truthful student inputs,original teachers,upper refinement,native physics/caps/actions,PPO and classifiers. Every observed KL/command mismatch is zero. Diagnostic actors are preserved outside production.

|Branch/update|3.5 lateral nominal/pushed(m/s)|Failed cases|
|---|---:|---|
{chr(10).join(table)}

Decision: {decision['summary']} Detailed evidence,extra seeds and exact hashes are in[matched_heading_review.json](matched_heading_review.json),[decision.json](decision.json),[heading_analysis.json](heading_analysis.json) and[physical response plot](heading_response.png).

The harness completed every requested result before its final overall-time assertion:2581.892s exceeded the2500s harness budget. Its exit1 is preserved and is not claimed as a successful harness exit or a production failure. Independent completion,lineage,full-state loads,KL/command and diverse-reset validation pass in[comparison_validation.json](comparison_validation.json). A diagnostic validator initially read the completion flag from the wrong JSON file; the failed source and correction are retained. Final status helper invocations also initially doubled a relative path; corrected runs exit0. These are diagnostic accounting/invocation issues,not reproduced runtime faults. See[diagnostic_accounting_notes.json](diagnostic_accounting_notes.json).

## Actual upper-body motion

Head yaw/pitch,shoulder pitch/roll and elbow angles,physical targets and velocities are analyzed by speed. Initial251,completed matched branches and latest production{actor} plots are visually inspected; hashes/notes are in[plot_qa.json](plot_qa.json). Population32/cmd10Hz and16 selected representative robots/gate50Hz are distinguished and reset-segmented. Full per-speed coordination/amplitude/symmetry/elbow/cadence/smoothness/saturation data are retained in[final_upper](final_upper/upper_joint_time_audit.json).

Opposite shoulder biases,unequal swing and elbow coordination remain. At2.5m/s shoulders can hold large pitch biases; at3m/s bias drifts over seconds. Fast right pitch swing is weaker than left and target excursions exceed measured response. Head motion remains gait-linked. Existing bounded posture/symmetry auxiliary updates only eight upper output rows20batches/update; no new gain or imposed arm trajectory is selected. Head native zero is torso-relative. Prior Isaac-head/FK parity is known; original CAD optical axis remains uncalibrated. These are not natural-arm,human/Unitree encoder or world optical-gaze certification.

New CPU[upper_bias_audit.py](upper_bias_audit.py) separates slow bias with a zero-phase0.5Hz Butterworth filter from faster residual motion. Four synthetic constant/slow/fast/mixture fixtures pass. It uses actual10Hz population angles,first continuous episode,correct post-warmup2.02s time origin and3s edge exclusion; it never filters across resets. {bias_text} In the matched original128/2.5 cohort,total ranges39.82/42.12deg contained slow ranges36.81/34.85deg and only10.18/12.83deg faster residual. Full angle range therefore overstates periodic swing. The residual also includes faster non-cycle motion; this decomposition is descriptive,not actuator identification or human-motion certification. Final population data and inspected plot are in[upper_bias_audit.json](final_upper/upper_bias_audit.json) and[upper_bias_decomposition.png](final_upper/upper_bias_decomposition.png). Earlier1s-edge/zero-origin attempts are preserved separately.

Primary references:[official Isaac Lab heading command](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py),[Mirelman human arm-swing features](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043),[official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab). The course term is our local adaptation. No verified speed-matched G1/H1 encoder trajectory was obtained and no reference curves were fabricated.

## Next check

{chr(10).join(str(i+1)+'. '+s for i,s in enumerate(decision['next_focused_tests']))}

Same ACTIVE five-hour heartbeat/thread/notification preference remain. Endpoint is qualified5m/s>=4.95,or qualified4m/s>=3.96 with repeated controlled native-limit evidence proving5blocked; all unchanged nominal/pushed/stopped/prior gates and32 independently seeded initial-state trials per command are required. Save,pause training+automation,brief and leave computer on only at that endpoint. Completed videos and unsaved Fusion work are preserved. Original pause scope text had a stale diagnostic label;[heading_diagnosis_scope.json](heading_diagnosis_scope.json) corrects the description without overwriting pause evidence.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
write(OUT/'evidence.json',dict(recorded_utc=now,live=str(OUT/'final_cpu/live_handoff.json'),learning=str(OUT/'final_learning_verification.json'),pair=pair,decision=decision,resume=resume,rates_k_samples_s=rates,no_production_change=True))
marker='adaptive_check_20261007_1344_record'
entry=f"\n<!-- {marker} -->\n## {now} — Course-feedback diagnosis and full-state comparison\n\n{summary}\n\n{decision['summary']} Physical384robot20s heading/course probe confirms yaw response and0.128m/s course-feedback lateral speed,with zero5ms driving clips,all survivors and exact term sums. It is not independent qualification.60 command/tick fixtures and34 heading fixtures pass. Exact4096env64/128 controls,extra push and32 independent small q/qd resets/cmd complete; all experimental actors stay outside production. Mandatory actual upper angle-target-velocity plots inspected; asymmetry,bias and head oscillation remain,not human/Unitree certification. Same39000 resumes after{resume['pause_seconds']:.1f}s with full state/teachers/upper20.55hashes/oneworker,completed+{recovered} recovered updates,27507 inherited and32 unsaved excluded. Rates{rates[0]:.2f}kactive/{rates[1]:.2f}kworker/{rates[2]:.2f}kchain,successful pauses excluded.8192target/reserve300/hostfallback/180healthyTRAINING ramp. ACTIVE5h and verified5 or verified4+causal limits endpoint unchanged; computeron.\n\nReport:{OUT/'REPORT.md'}. Decision:{OUT/'decision.json'}. Comparison:{OUT/'matched_heading_review.json'}. Upper:{OUT/'final_upper/upper_joint_time_audit.json'}. Resume:{OUT/'final_learning_verification.json'}. Earlier records preserved.\n<!-- /{marker} -->\n"
for f in [TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
 assert marker not in f.read_text(encoding='utf-8-sig')
 with f.open('a',encoding='utf-8') as stream:stream.write(entry)
handoff=f"<!-- {marker} -->\nLatest check{now}:{summary} {decision['summary']} Read{OUT/'REPORT.md'},decision,matched_heading_review,heading_analysis,final_upper,final_learning_verification and freshACTIVE_RUN/state. Mandatory upper timelines remain asymmetric; no exact human/Unitree/optical-gaze certification. Same full learner resumed after{resume['pause_seconds']:.1f}s,55hashes,oneworker,upper/priors20. Original signatures do not certify diagnostic course overrides. Current recovered count{recovered}.8192target,reserve300,180healthyTRAINING ramp,ACTIVE5h verified5 or qualified4+causal native-limit endpoint;save,pauseboth,brief,computeron. Earlier entries below are historical.\n<!-- /{marker} -->\n\n"
for f in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
 old=f.read_text(encoding='utf-8-sig');assert marker not in old;f.write_text(handoff+old,encoding='utf-8')
npath=TRAINING/'automation/NEXT_CHECK.json';n=read(npath)
n.update(recorded_utc=live['recorded_utc'],active_run=str(live['active_run']['run']),report=str(OUT/'REPORT.md'),latest_observation=str(OUT/'final_cpu/live_handoff.json'),new_main_updates=learning['completed_main_updates'],recovered_main_updates=recovered,actual_retained_current_run_main_updates=learning['actual_retained_main_updates'],inherited_main_updates_preserved=27507,selected_actor=dict(checkpoint=pair['checkpoint'],sha256=pair['sha256'],matching_gate_pair=str(OUT/'closing_gate_review.json'),current_stage_qualified=False,scope='Exact complete pair and upper timelines; newer partial actors unqualified.'),highest_fully_qualified_stage='run_300',highest_fully_qualified_command_m_s=3.,current_stage=live['state']['stage'],current_stage_qualified=False,running_qualified=False,velocity_ceiling_proven=False,focused_tests=decision['next_focused_tests'],latest_repair_decision=decision,throughput={k:learning[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},diagnostic_pause=resume,mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),mandatory_upper_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'),mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'),independent_initial_states_diagnostic=str(OUT/'matched_heading_review.json'),production_change_this_check=False,newer_matching_pair=str(OUT/'closing_gate_review.json'),latest_saved_actor_unqualified=dict(checkpoint=pair['checkpoint'],scope='Current complete pair; no qualification of newer partial learning.'),newer_gate_trend=summary)
write(npath,n)
native_path=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml');native=tomllib.loads(native_path.read_text(encoding='utf-8'))
prefix=native['prompt'].split('Latest dated five-hour check',1)[0]
dated=f"Latest dated five-hour check, evidence {live['recorded_utc']}: {summary} Read diagnostics/adaptive_check_20261007_1344/REPORT.md, decision.json, matched_heading_review.json, comparison_validation.json, heading_analysis.json, heading_response.png, final_upper plots/data, final_learning_verification.json and final_cpu/live_handoff.json, then fresh ACTIVE_RUN/state. {decision['summary']} No production source/reward/gain/cap/optimizer/actor change or diagnostic actor deployed. Same supervisor 39000 resumes after {resume['pause_seconds']:.1f}s with full actor/critic/Adam/exploration, both teachers and upper20batches/update. Rates {rates[0]:.2f}k active/{rates[1]:.2f}k worker/{rates[2]:.2f}k chain; successful pauses excluded. Physical384robots20s course response reduces instantaneous lateral speed0.386to0.128 with all survivors,zero5ms driving clips and exact terms; this is not independent qualification or a native ceiling. The unit-gain course term is a local velocity-dependent command adaptation, not copied Unitree.60 command/tick and34heading fixtures pass; prior commands<=3 are unchanged. The comparison harness completed all results but exited on its final overall-time guard (2581.9s versus2500s); independent completion/full-state/diversity validation passes. Do not call that a production fault or successful harness exit. All diagnostic actors/overrides remain outside original source certification. Mandatory upper actual angles/targets/velocities and new slow-bias decomposition inspected. {bias_text} Fast asymmetry, slow bias and head oscillation remain; no human/Unitree encoder or calibrated optical-gaze certification. Existing gains35/1 and upper20continue.\n\nNext focused tests:\n"+'\n'.join(f"{i+1}. {s}" for i,s in enumerate(decision['next_focused_tests']))+"\n\nPreserve the full neutral-head learner, actor267 all-prior pair and separately certified3m actor349. Keep rejected candidates rejected absent new evidence.8192target,reserve300,hostfallback,180continuoushealthyTRAINING seconds; one worker,localground,full-state recovery and actual retained sample accounting. Ordinary nominal clones do not establish final initial-state diversity. Every endpoint actor needs32 independently seeded nominal-physics states/cmd and all unchanged gates. Same ACTIVEfive-hour cadence/thread/notifications and verified5 or verified4+causal native limits blocking5 endpoint;save,pause training+automation,brief,computeron. Videos fulfilled,unsavedFusion protected."
auto=dict(id=native['id'],mode='update',kind='heartbeat',name=native['name'],status='ACTIVE',rrule=native['rrule'],targetThreadId=native['target_thread_id'],prompt=prefix+dated)
if 'notification_policy' in native:auto['notificationPolicy']=native['notification_policy']
write(OUT/'automation_update_args.json',auto)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),same_run=True,retained_updates=learning['actual_retained_main_updates'],recovered_updates=recovered,native_args_ready=True)))
