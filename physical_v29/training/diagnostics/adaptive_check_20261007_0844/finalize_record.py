"""Write evidence-backed local reports/handoffs and structured native automation args."""
import json,hashlib,tomllib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
live=read(OUT/'final_cpu/live_handoff.json');learning=read(OUT/'final_learning_verification.json')
gates=read(OUT/'closing_gate_review.json');pair=gates['latest'];decision=read(OUT/'decision.json');resume=read(OUT/'diagnostic_resume.json')
assert learning['supervisor_pid']==39000 and learning['frozen_files_verified']==55 and len(live['actual_worker_pids'])==1
assert live['state']['status']=='running' and all(live['frozen_source_integrity'].values())
assert resume['same_supervisor'] and not resume['diagnostic_actors_deployed']
qa=read(OUT/'plot_qa.json')
for name in ['upper_joint_angles.png','upper_joint_velocities.png']:
    p=OUT/'final_upper'/name;digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert any(x['path']==str(p) and x['sha256']==digest and x['visually_inspected'] for x in qa['files']),'Final images need visual QA before handoff'
cur=[pair[k][-1] for k in ['nominal_cases','pushed_cases']]
fail={k:[{'command':c['command'],'failures':c['failures']} for c in pair[k] if not c['passed']] for k in ['nominal_cases','pushed_cases']}
upper=read(OUT/'final_upper/upper_joint_time_audit.json')
upper_rows=[r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s'] in (1.,3.5)]
brief_upper=[]
for r in upper_rows:
    brief_upper.append({'command':r['command_m_s'],'head_rms_deg':r['head_rms_deg'],'coordination':r['coordination'],'representative_50hz':r.get('representative_50hz')})
write(OUT/'final_upper_summary.json',brief_upper)
rates=[learning[k]/1000 for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']]
checkpoint=Path(pair['checkpoint']);actor=checkpoint.parent.name
failed_text='; '.join(f"{label}: "+', '.join(f"{c['command']:g} ({'/'.join(c['failures'])})" for c in fail[label+'_cases']) for label in ['nominal','pushed'])
last_main=learning['blocks'][-1];main_count=last_main['environments']
summary=f"Same R40 learner/supervisor39000 is running; latest completed main block uses{main_count} environments, with one actual worker and55 verified frozen files. The owned snapshot operation is{Path(live['state']['operation_path']).name} at{live['state']['environment_count']} robots; evaluation count is not PPO capacity. {learning['completed_main_updates']} completed +41 hash-proven recovered = {learning['actual_retained_main_updates']} retained current-run updates, plus27507 inherited. Exact {actor} matching nominal/pushed signed3.5 speeds {cur[0]['signed_speed']:.3f}/{cur[1]['signed_speed']:.3f}m/s, lateral {cur[0]['lateral_speed']:.3f}/{cur[1]['lateral_speed']:.3f}; current stage remains unqualified. Gates bind this saved actor, never newer partial learning."
report=f'''# Five-hour humanoid check — October 7, 08:44 UTC

{summary}

No production source, reward, shoulder gain, motor cap, actor or optimizer reset was deployed. The exact original learner resumed after a {resume['pause_seconds']:.1f}-second controlled suspension, including {resume['worker_drain_seconds']:.1f} seconds of worker drain. Current upper refinement and both prior auxiliaries remain at20 batches/update. Latest completed main minimum free GPU is{last_main['minimum_free_gpu_MiB']}MiB,host available{last_main['minimum_host_available_MiB']}MiB,private memory growth{last_main['private_memory_growth_MiB']}MiB and Torch allocation growth{last_main['torch_allocation_growth_MiB']}MiB. Highest separately certified stage remains3m/s on preserved teacher349; original267's all-prior pair is preserved. There is no qualified3.5,4,5m/s or relevant motor-ceiling endpoint.

## Current evidence

Owned snapshot: {live['recorded_utc']}. Saved pair: [{actor}]({checkpoint.as_posix()}), SHA`{pair['sha256']}`. Each matching test has320 robots,20 seconds,32 trials per command. Current/prior failures: {failed_text}. Ordinary nominal resets remain cloned; trial count alone does not establish initial-state diversity.

Retained-main throughput is {rates[0]:.2f}k active collection+PPO / {rates[1]:.2f}k worker-effective / {rates[2]:.2f}k whole-chain samples/s. Startup, capacity and baseline gates enter the whole-chain denominator; the documented successful pause is excluded when before the endpoint. The first brief failed bookkeeping pause lacks exact timestamps and remains included, making this rate slightly conservative. Diagnostic, smoke and capacity actors never count as production learning. Exact full actor/critic/Adam/exploration loads,zero pre-update KL and action-command mismatch are verified. All55 frozen hashes and53 protected authored worker files match. Target8192,reserve300MiB,host guards,fallback and180 continuous healthy training seconds before upward ramps remain.

## Tested teacher-label hypothesis — rejected

The certified neutral-head349 teacher still receives head-history labels fitted to an older tilted-head walker. Four actual current-state PPO minibatches reproduce lower effective-label differences0.112–0.200 action units. Running PPO norms17.54–38.31 versus prior1.99–2.20 have cosine–0.126 to+0.296; this is not consistently opposed. The candidate removes legacy head context only for exact teacher349, with truthful student observations and unchanged0.25MSE/23-output denominator, stopped teacher, upper refinement, PPO, physics, actions, native caps and gates.

Ten CPU ownership/gradient command fixtures, actual64-env4-update PPO/full-state reload and exact4096-env64/128 controls complete. Both branches start exact139 SHA`8fa38ecdb9abc20d4cf6465b9cee254b8387477cf857e29844ce220a7d552527` with actor/critic/Adam/exploration equal. Every pre-update KL/command mismatch is zero. Fresh truthful349 nominal/pushed288-robot tests pass all earlier cases. Experimental overrides are explicitly outside original signature certification; no experimental actor is deployable under that signature.

|128-update result|Control|Candidate|
|---|---:|---:|
|3.5 lateral, nominal/pushed (m/s)|0.383/0.353|0.362/0.343|
|0.15 gait/net travel|Fails|Passes|
|1.5 case, nominal/pushed|Passes|Fails lateral|
|2.5 case, nominal/pushed|Passes|Fails slip|
|3.5 direction|Fails|Fails|

An additional pushed seed1901 reproduces the tradeoff. A separate seed2301 nominal-physics test uses32 independently generated small q/qd reset states per command, with saved arrays/seeds/hashes and unchanged classifiers. It reproduces candidate1.5 lateral and2.5 slip failures;3.5 lateral is0.379 control versus0.365 candidate. All survive. This establishes limited reset robustness of the comparison, not qualification of production or the final endpoint. Candidate deployment is rejected; useful full state and all diagnostic checkpoints are preserved.

## Actual upper-body timelines

Actual head yaw/pitch, shoulder pitch/roll and elbows were saved as angles,physical targets and velocities for initial133, both128 diagnostic actors and latest production{actor}; every angle/velocity plot was visually inspected. Population sampling is32 robots/command at10Hz; detailed traces are16 representative robots/gate at50Hz, reset-segmented. At initial133/1m shoulders range39.37/30.78deg with hip phases–159.8/–166.2deg. At3.5 range14.20/7.49deg, phases140.9/149.5deg and mean biases–13.63/+4.16deg. Slow bias drift at2.5/3 is distinguished from periodic swing. Latest data and coordination metrics are in [final_upper_summary.json](final_upper_summary.json).

Asymmetric fast swing,roll/elbow coordination and gait-linked head oscillation remain. Both matched actors show target excursions larger than measured fast shoulder response. These diagnostics do not certify human-like arms,exact Unitree similarity or world optical gaze. Prior Isaac head/FK parity is known; original CAD face/camera optical axis remains uncalibrated. Shoulder stiffness70,uniform lateral exposure and previous failed kernels remain rejected.

## Direction objective lead

A108-case CPU audit plus gravity-alignment fixture confirms the pinned Unitree tracker is yaw-aligned,not a full-body-frame port error. A synthetic3.5m/s path/body yaw of8deg gives0.487m/s world lateral velocity while retaining4.853/5 of the tracking-plus-heading rate. Other terms are fixed; this is not a complete-return or dynamics proof. Exact139 representative50Hz states show strong tracking/heading credit despite excessive drift. Only one representative3.5 robot per test enters this objective reconstruction; population gate statistics are separate. No reward or command change is deployed. Next measure truthful normal-limit heading-channel response,actual target/heading timing and complete-objective tradeoffs before selecting a narrowly versioned64/128 continuation.

## Resume, failures and next work

The first pause bookkeeping attempt hit a str/Path hashing error after draining an evaluation; exception cleanup resumed the same supervisor. Corrected bookkeeping repeated the saved-boundary pause. Failed state and diagnostic summary are preserved; no policy update was lost. The separate stopped-hook CPU subtraction remains unresolved diagnostic accounting uncertainty,not a demonstrated production defect.

The same39000 supervisor resumed from exact139 after every diagnostic worker exited. Fresh actual main learning/full-state evidence is in [final_learning_verification.json](final_learning_verification.json); [live ownership](final_cpu/live_handoff.json),[matching gates](closing_gate_review.json),[decision](decision.json),[comparison](matched_context_review.json),[upper angles](final_upper/upper_joint_angles.png),[velocities](final_upper/upper_joint_velocities.png) and [objective audit](direction_objective_audit.json) retain exact provenance. No new actuator-ceiling evidence is claimed from this check. Videos are already fulfilled; unsaved Fusion work is untouched.

Next focus: controlled heading-response/full-objective diagnosis; mandatory per-speed upper timelines and effective targets; exact full-state/memory/sample accounting and independently seeded endpoint tests. Same ACTIVE five-hour heartbeat and notification preference remain. Stop only at qualified5m/s or qualified4m/s with causal native limits blocking5; then save,pause training+automation,brief and leave the computer on.

Primary references reviewed: [Ross et al., state-distribution-aware imitation learning](https://proceedings.mlr.press/v15/ross11a.html),[Mirelman et al., arm swing features](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043),[official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab). No speed-matched Unitree encoder trajectory was obtained and no reference curves were invented.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
write(OUT/'evidence.json',{'recorded_utc':now,'owned_observation':str(OUT/'final_cpu/live_handoff.json'),'learning':str(OUT/'final_learning_verification.json'),'pair':pair,'decision':str(OUT/'decision.json'),'upper_QA':str(OUT/'plot_qa.json'),'resume':resume,'rates_k_samples_s':rates,'no_production_change':True})
marker='adaptive_check_20261007_0844_record'
entry=f'''\n<!-- {marker} -->
## {now} — Teacher-label comparison, independent reset tests and direction objective audit

{summary} Full-state teacher-context removal tested against unchanged4096-env64/128 control is REJECTED: candidate recovers0.15 but adds1.5 lateral and2.5 slip failures;3.5 remains unqualified. Extra push seed and32 independently generated q/qd reset states per command reproduce tradeoff. Truthful teacher349 requalifies all earlier cases. Four actual PPO minibatches and10 CPU ownership checks pass; no deployed source/reward/gain/cap/optimizer reset or diagnostic actor. Original signature does not certify overrides.

Required actual upper angle-target-velocity plots inspected at133,both128 diagnostic actors and{actor}; fast asymmetry,slow shoulder bias drift and head oscillation remain. No human/Unitree encoder or optical gaze certification.108 synthetic direction fixtures reproduce high yaw-aligned reward for diagonal travel; no port error or complete-return proof. Same supervisor resumes after{resume['pause_seconds']:.1f}s; exact actor/critic/Adam/exploration and both teachers,zeroKL/mismatch,upper20batches preserved. Rates{rates[0]:.2f}k active/{rates[1]:.2f}k worker/{rates[2]:.2f}k chain with documented pause excluded. Initial failed pause bookkeeping recovered safely and is recorded separately. No5/4/ceiling endpoint. Target8192/reserve300/hostfallback/180healthyTRAINING ramp; ACTIVE5h,save/pause/brief/computeron endpoint unchanged.

Report: {OUT/'REPORT.md'}. Gates: {OUT/'closing_gate_review.json'}. Decision: {OUT/'decision.json'}. Upper: {OUT/'final_upper/upper_joint_time_audit.json'}. Resume: {OUT/'final_learning_verification.json'}. Next focused tests recorded in NEXT_CHECK; all earlier history preserved.
<!-- /{marker} -->\n'''
for p in [TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    existing=p.read_text(encoding='utf-8-sig')
    assert marker not in existing,'Already recorded; append a separately timestamped addendum instead of overwriting history'
    with p.open('a',encoding='utf-8') as f:f.write(entry)
handoff=f'''<!-- {marker} -->
Latest check {now}: {summary} Read {OUT/'REPORT.md'},decision,matched_context_review,final_upper,final_learning_verification,NEXT_CHECK and freshACTIVE_RUN/state. Teacher-context removal REJECTED after64/128 full-state controls and32 independent small-reset states per command: new1.5 lateral/2.5 slip failures. No production changes or diagnostic actor. Same39000 resumed after{resume['pause_seconds']:.1f}s with full state/upper20. Direction objective audit is a lead for truthful heading-response/full-objective tests,not a port fault or deployed reward. Mandatory timelines show asymmetry/bias/head oscillation; no human/Unitree/optical-gaze certification.55hashes/oneworker/8192target/reserve300/180healthyTRAININGramp. Same ACTIVE5h endpoint qualified5 or qualified4+causal native limits;save,pauseboth,brief,computeron. Earlier entries below are historical.
<!-- /{marker} -->\n\n'''
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    existing=p.read_text(encoding='utf-8-sig');assert marker not in existing;p.write_text(handoff+existing,encoding='utf-8')
next_path=TRAINING/'automation/NEXT_CHECK.json';n=read(next_path)
n.update(recorded_utc=live['recorded_utc'],active_run=str(live['active_run']['run']),report=str(OUT/'REPORT.md'),latest_observation=str(OUT/'final_cpu/live_handoff.json'),
    new_main_updates=learning['completed_main_updates'],recovered_main_updates=41,actual_retained_current_run_main_updates=learning['actual_retained_main_updates'],
    inherited_main_updates_preserved=27507,selected_actor={'checkpoint':pair['checkpoint'],'sha256':pair['sha256'],'matching_gate_pair':str(OUT/'closing_gate_review.json'),'current_stage_qualified':False,'scope':'Exact completed pair; newer partial learner unqualified. Latest upper timelines bind same exact pair.'},
    highest_fully_qualified_stage='run_300',highest_fully_qualified_command_m_s=3.,current_stage=live['state']['stage'],running_qualified=False,velocity_ceiling_proven=False,
    focused_tests=decision['next_focused_tests'],latest_repair_decision=decision,throughput={'active_PPO_samples_s':learning['active_PPO_samples_s'],'worker_effective_samples_s':learning['worker_effective_samples_s'],'whole_chain_samples_s':learning['whole_chain_samples_s'],'scope':learning['whole_chain_scope']},
    diagnostic_pause=resume,mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),independent_initial_states_diagnostic=str(OUT/'matched_context/diversity_completion.json'))
write(next_path,n)
native_path=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml');native=tomllib.loads(native_path.read_text(encoding='utf-8'))
prefix=native['prompt'].split('Latest dated five-hour check',1)[0]
dated=f'''Latest dated five-hour check, evidence {live['recorded_utc']}: {summary} Read diagnostics/adaptive_check_20261007_0844/REPORT.md,decision.json,matched_context_review.json,direction_objective_audit.json,current_gradient_analysis.json,final_upper angle/velocity plots/data,final_learning_verification.json and final_cpu/live_handoff.json,then freshACTIVE_RUN/state. No production source/reward/gain/cap/optimizer/actor reset or diagnostic actor deployed. Existing upper and both priors20batches/update continue.53 authored worker files match,55hashes,oneworker. Rates{rates[0]:.2f}kactive/{rates[1]:.2f}kworker/{rates[2]:.2f}kchain; documented{resume['pause_seconds']:.1f}s suspension excluded when before endpoint,first brief failed-bookkeeping pause remains conservatively included.32 interrupted unsaved R39 updates excluded.

Teacher-context removal is REJECTED after exact4096-env64/128 controls. At128 candidate3.5 lateral0.362/0.343 versuscontrol0.383/0.353 remains failed; candidate adds1.5 lateral and2.5 slip failures whilecontrol passes them.0.15 recovers in candidate; this tradeoff is not a safe repair. Extra pushedseed1901 and nominalphysicsseed2301 with32 independently generated smallq/qd states per command reproduce failures. Original signatures do not certify experimental overrides. All diagnostic actors remain outside production. Truthful neutral-head349 freshly passes every earlier case. Four actual PPO minibatches reproduce changed labels0.112–0.200 units but mixed gradient alignment; not a proven production fault. No new DAgger or estimator deployed.

Every required actual upper angle/target/velocity timeline was inspected at133,both128 diagnostic actors and latest saved{actor}. Fast shoulder asymmetry,slow2.5/3 bias drift,roll/elbow coordination and gait-locked head oscillation remain.32/cmd10Hz population versus16/gate50Hz representatives segmented by resets; no natural/human/Unitree encoder certification. Isaachead/FK parity known,original opticalaxis uncalibrated. No new gain or imposed trajectory.108 synthetic objective fixtures show yaw-aligned tracking+heading can score diagonal3.5 travel highly (8deg gives0.487m/s lateral and4.853/5 partialrate),not a fullbodyframeport error or complete-return proof. One representative per3.5 test used in reconstruction,not fullpopulation objective statistics. No reward change.

Next focused tests:\n'''+'\n'.join(f'{i+1}. {t}' for i,t in enumerate(decision['next_focused_tests']))+'''

Read newer exact evidence and preserve passing actors267/349/current full neutral-head state. Do not spend next check only waiting or repeat rejected teacher-context removal,stiffness70,fixed-frame kernel,lateral exposure or other failed variants without new evidence. Keep same ACTIVEfive-hour cadence/thread/notification preference and qualified5 or qualified4+causal native-limit endpoint;save,pause training+automation,brief,computeron. Videos fulfilled,unsavedFusion protected.
'''
args={'id':native['id'],'mode':'update','kind':'heartbeat','name':native['name'],'status':'ACTIVE','rrule':native['rrule'],'targetThreadId':native['target_thread_id'],'prompt':prefix+dated}
if 'notification_policy' in native:args['notificationPolicy']=native['notification_policy']
write(OUT/'automation_update_args.json',args)
write(OUT/'recording_receipt.json',{'recorded_utc':now,'same_automation_id':native['id'],'durable_prompt_prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),'report_written':True,'journals_appended':3,'handoffs_prepended':2,'native_update_pending':True})
print(json.dumps({'report':str(OUT/'REPORT.md'),'actor':actor,'current_signed_lateral':[(c['signed_speed'],c['lateral_speed']) for c in cur],'retained_main_updates':learning['actual_retained_main_updates'],'same_automation_args_ready':True},indent=2))
