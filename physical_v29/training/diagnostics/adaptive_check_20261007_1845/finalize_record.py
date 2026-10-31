"""Save evidence and adaptive priorities; retain the original full learner."""
import json,hashlib,tomllib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
live=read(OUT/'final_cpu/live_handoff.json');learning=read(OUT/'final_learning_verification.json')
review=read(OUT/'final_gate_review.json');pair=review['latest'];resume=read(OUT/'diagnostic_resume.json')
earlier_resume=read(OUT/'attempt01_initial_cache_failure/diagnostic_resume.json')
physical=read(OUT/'heading_response/result.json');response=read(OUT/'saved_actor_response_analysis.json')
grad=read(OUT/'current_gradient_analysis.json');upper=read(OUT/'final_upper/upper_joint_time_audit.json')
bias=read(OUT/'final_upper/upper_bias_audit.json');angular=read(OUT/'angular_heading_audit.json')
assert live['state']['status']=='running' and len(live['actual_worker_pids'])==1
assert learning['supervisor_pid']==39000 and learning['frozen_files_verified']==55
assert all(live['frozen_source_integrity'].values()) and resume['same_supervisor']
assert read(OUT/'diagnostic_completion.json')['completed'] and read(OUT/'passive_capture_cpu_tests.json')['passed']
assert physical['completed'] and physical['maximum_command_difference']==0
assert not response['actual_native_5ms_driving_clips']
assert all(r['checkpoint']==pair['checkpoint'] for r in upper['records'] if r['actor_tag']=='latest')
for f,h in read(OUT/'protected_source_copy.json')['protected_hashes'].items():
    assert hashlib.sha256((OUT/'control_source/physical_v29'/f).read_bytes()).hexdigest()==h
files=[]
for folder,names in [('initial_upper',['upper_joint_angles.png','upper_joint_velocities.png','upper_bias_decomposition.png']),
                      ('final_upper',['upper_joint_angles.png','upper_joint_velocities.png','upper_bias_decomposition.png']),
                      ('',['saved_actor_response.png'])]:
    for name in names:
        f=OUT/folder/name
        files.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),visually_inspected=True))
write(OUT/'plot_qa.json',dict(recorded_utc=now,files=files,initial_checkpoint=read(OUT/'initial_gate_review.json')['latest']['checkpoint'],
    final_checkpoint=pair['checkpoint'],observations='Unequal arm swing and opposite held biases persist.2.5/3m/s include slow bias drift. Targets exceed fast measured pitch excursions; neck motion is gait-linked. Response plot shows greater residual drift of course-trained128 versus original/current under the same command law. No natural-motion,world optical gaze or Unitree encoder certification.'))
cur=[pair[k][-1] for k in ['nominal_cases','pushed_cases']]
failed={k:[dict(command=c['command'],failures=c['failures']) for c in pair[k] if not c['passed']] for k in ['nominal_cases','pushed_cases']}
rates=[learning[k]/1000 for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']]
last=learning['blocks'][-1];count=learning['actual_retained_main_updates'];actor=Path(pair['checkpoint']).parent.name
summary=f"Same R40 learner/supervisor39000,one actual worker,55 frozen hashes. {learning['completed_main_updates']} completed+{learning['hash_proven_recovered_main_updates']} hash-proven recovered={count} retained new main updates,plus27507 inherited. Exact reviewed {actor} reaches3.5-command signed{cur[0]['signed_speed']:.3f}/{cur[1]['signed_speed']:.3f}m/s with{cur[0]['lateral_speed']:.3f}/{cur[1]['lateral_speed']:.3f}m/s lateral;both current direction gates fail. Both0.15 net travel and1.5 geometric flight fail;other stopped/prior cases including recovered3m pass in this pair. Qualification never extends to newer partial actors."
next_tests=[
 "Preserve current neutral-head full learner and compare fresh exact3.5 direction,0.15 path efficiency and1.5 geometric flight plus every prior gate. The paired384-robot response reproduces useful course feedback but course-trained128 worsens residual drift even under its own law,without yaw-command saturation or5ms driving clips. Actual desired-yaw variation is gait-linked near2.66Hz. Before another learning candidate, verify5ms quaternion-heading/world-gyro/action-command timing and phase-resolved response;50Hz heading differences and gyro kinematics disagree and do not establish a production clock defect. Then test bounded cycle-averaged course/heading feedback or another evidence-based timing candidate in truthful normal-limit physics,complete return and separate seeds. Require fresh versioned provenance,exact-current-full-state64/128 controls and all gates before deploying. Do not merely raise gains,caps or blindly repeat course1.",
 "Mandatory actual head/shoulder pitch+roll/elbow angle-target-velocity timelines,arm-leg phase and slow-bias decomposition. Current fast arm swing remains unequal and2.5/3m slow bias drifts. Four actual minibatches show upper versus PPO alignment of both signs; no universal opposition. Measure per-command slow target bias versus measured joint bias and phase-conditioned effective targets,then test a narrow training-only bias refinement using the current full state against64/128 control/all gates. Preserve original35/1 gains,forward head,legs,teachers and useful motion;no imposed reference curves,spoofs,hybrids or from-scratch restart. Human/Unitree encoder likeness and optical gaze remain unverified.",
 "Verify same supervisor/oneworker55 hashes,local ground,full actor/critic/Adam/exploration and both teachers. Account27507 inherited plus completed and50 recovered;32 unsaved R39 remain excluded. Exclude both current diagnostic pauses370.7/310.7s once before the timing endpoint,plus prior documented pauses. Fusion is already closed;do not reopen it. Actual6144 steady freeGPU about1942MiB,host about9GiB;current scheduler needs2531MiB free for8192,so retain300MiB reserve,host guards and180 continuous healthy TRAINING seconds rather than forcing the ramp. Endpoint still needs every gate and32 independently seeded initial states/cmd;only qualified5 or qualified4 with causal native limits blocking5 ends training/automation."
]
decision=dict(recorded_utc=now,production_change=False,diagnostic_actors_deployed=False,parameter_change_deployed=False,
    summary='Preserve and resume the exact original full-state learner. Saved-actor response and passive actual gradients narrow the next diagnosis; no new candidate has independent learning/retention evidence sufficient for deployment. Previous rejected candidates remain rejected.',next_focused_tests=next_tests)
write(OUT/'decision.json',decision)
gtable=['|Command m/s|Signed nominal/pushed|Lateral nominal/pushed|Failed nominal/pushed checks|','|---:|---:|---:|---|']
for a,b in zip(pair['nominal_cases'],pair['pushed_cases']):
    gtable.append(f"|{a['command']:g}|{a['signed_speed']:.3f}/{b['signed_speed']:.3f}|{a['lateral_speed']:.3f}/{b['lateral_speed']:.3f}|{'/'.join(a['failures']) or 'pass'}; {'/'.join(b['failures']) or 'pass'}|")
ptable=['|Saved actor|3.5 baseline/course lateral m/s|Complete reward-rate change|','|---|---:|---:|']
for c in response['course_comparisons']:
    ptable.append(f"|{c['actor']}|{c['baseline_lateral']:.3f}/{c['course_lateral']:.3f}|{c['complete_reward_rate_change']:+.3f}|")
norm=lambda k:[float(b[k]) for b in grad['batches']]
u=next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
bs=next(r for r in bias['records'] if r['test']=='nominal' and r['speed_m_s']==3.5)['joints']
upper_text=f"At3.5 nominal,head RMS{u['head_rms_deg']:.2f}deg;physical pitch ranges{u['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{u['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}deg. Slow pitch means{bs['left_shoulder_pitch']['slow_bias_mean_deg']:.2f}/{bs['right_shoulder_pitch']['slow_bias_mean_deg']:.2f}deg and faster residual ranges{bs['left_shoulder_pitch']['cycle_residual_p95_p05_deg']:.2f}/{bs['right_shoulder_pitch']['cycle_residual_p95_p05_deg']:.2f}deg. Arm/ipsilateral-hip phases{u['coordination']['arms'][0]['phase_at_hip_cycle_deg']:.1f}/{u['coordination']['arms'][1]['phase_at_hip_cycle_deg']:.1f}deg at{u['coordination']['hip_cycle_hz']:.2f}Hz;phase and amplitude are both diagnostic."
pause_total=resume['pause_seconds']+earlier_resume['pause_seconds']
report=f'''# Five-hour check — October7,18:45 UTC

{summary}

{decision['summary']}

## Exact saved gates and progress

Owned observation:{live['recorded_utc']}. Checkpoint SHA`{pair['sha256']}`. Matching320-robot20-second nominal/pushed tests contain32 trials per command. All survive. Nominal cloned resets do not establish final independently seeded initial-state diversity. Separately certified original349 at3m and267 all-prior remain preserved.

{chr(10).join(gtable)}

Compared with the initial332 pair,338 recovers3m hopping/direction and retains1m support,while current3.5 drift decreases. Retention remains nonmonotonic:0.15 efficiency and1.5 flight still fail. Running1.5 geometric flight is{pair['nominal_cases'][5]['flight']*100:.4f}%/{pair['pushed_cases'][5]['flight']*100:.4f}%,below unchanged1.5% minimum. Neither a3.5 command nor these signed speeds qualify the whole stage. No4/5m or native-limit endpoint is established.

## Bounded saved-actor physical comparison

Exact boundary338,original254,unchanged-control128 and rejected-course128 actors are replayed in384 free-base robots for20 seconds. Each of16 cohorts uses24 independently drawn small q/qd states,identical across actor/law branches. Root,gravity,self-collision,geometry,35/1 shoulder gains and all native motor caps remain. This is a diagnostic24-state protocol,not the final32-state qualification. Actor hashes,initial states,commands and synchronized joints/targets/forces/root/actions are saved in[heading_response](heading_response/result.json). All actors load exactly,all survive without resets,observed/action command difference is0 and full reward sums reconstruct within{physical['maximum_reward_term_sum_error']:.2g}. No pulses,root support,input rewrite or hybrid is deployed.

{chr(10).join(ptable)}

Statistics above are instantaneous world velocity after5s warmup,not the verifier's sampled-pose direction statistics. Current338 still responds physically to course feedback;the learned course128 actor has more drift under the same law than original254/current338. Its0.15 travel and1.5 lateral/flight behavior also weakens. After warmup no group's yaw command saturates the+/-0.2rad/s bound. Every sampled native5ms CAD driving-speed clip statistic is zero over4000 substeps. These results reject command-bound saturation as the explanation in this experiment;they do not prove5m feasible or establish a high-speed motor ceiling. All diagnostic actors and command laws remain outside production certification.

The[response plot](saved_actor_response.png) was inspected. Desired-yaw variation is dominated near2.66Hz,with substantial within-stride oscillation. The[angular audit](angular_heading_audit.json) distinguishes world angular-z from quaternion heading differences. Its analytic gyro/heading-rate reconstruction and50Hz forward differences disagree substantially;sampling/coordinate/phase attribution remains unresolved. No production clock fault or smoothing benefit is inferred. Establish5ms parity and controlled phase-response before a feedback-timing candidate. The course term is a local command adaptation,not copied Unitree numerical guidance.

## Actual PPO,teacher and upper gradients

Five CPU hook-order fixtures pass. Four genuine4096-env diagnostic PPO updates load exact338 actor/critic/Adam/exploration and both teachers,with all KL/command checks0. Its learned actor is discarded. Four actual minibatches capture exact advantages,means,Gaussian parameters,teacher labels and incoming PPO mean derivatives before either retention hook. The new passive before-standing capture resolves subtraction uncertainty on these fresh samples:CPU standing derivative differs by at most{max(norm('CPU_standing_subtraction_max_error')):.2g};moving analytic-surrogate cosine is1. Earlier failed accounting evidence remains preserved and is not a production defect.

Current-command PPO mean-network parameter norms range{min(norm('running_PPO_norm')):.2f}–{max(norm('running_PPO_norm')):.2f},prior norms{min(norm('prior_parameter_norm')):.2f}–{max(norm('prior_parameter_norm')):.2f},upper auxiliary norms{min(norm('upper_auxiliary_parameter_norm')):.2f}–{max(norm('upper_auxiliary_parameter_norm')):.2f}. Running/prior cosine ranges{min(norm('running_teacher_cosine')):.3f}–{max(norm('running_teacher_cosine')):.3f};upper/PPO cosine{min(norm('upper_auxiliary_PPO_cosine')):.3f}–{max(norm('upper_auxiliary_PPO_cosine')):.3f}. Alignment has both signs. Per-command advantages,coverage,effective clipped errors and contribution norms are saved in[current_gradient_analysis.json](current_gradient_analysis.json). These are before global clipping/Adam,mean-network only;four minibatches do not identify a universal conflict or justify an arbitrary weight change.

## Required upper-body time series

Both initial332 and reviewed338 head yaw/pitch,shoulder pitch/roll,elbow angles,physical targets and measured velocities were visually inspected,with[plot hashes/QA](plot_qa.json). Population32/cmd10Hz and16 representative robots/gate50Hz are separated and reset-segmented. {upper_text} At2.5/3m,large angle span includes slow shoulder bias drift. Fast target excursions exceed actual shoulder swing;roll/elbow imbalance and gait-linked neck oscillation remain. No new gain or imposed trajectory is selected. Existing upper-only posture/symmetry refinement continues20 batches/update and does not redefine physical proportions or certify natural motion.

The[angle plot](final_upper/upper_joint_angles.png),[velocity plot](final_upper/upper_joint_velocities.png),[slow-bias plot](final_upper/upper_bias_decomposition.png) andCSV/JSON data remain nearby. The0.5Hz filter,3s edges and post-warmup2.02s origin are descriptive and never cross resets;four decomposition fixtures pass. Native neckzero is torso-relative. Isaac-head/FK parity is known;original CAD face/camera optical axis remains uncalibrated.

Primary references:[official Isaac Lab heading generator](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py),[primary human arm-swing features](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043),[official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab). No mapped speed-matched G1/H1 encoder trajectories were obtained;no reference curve,human likeness or optical world gaze is fabricated or certified.

## Resume,resources and failed attempts

Same supervisor39000 resumed after two bounded suspensions:{earlier_resume['pause_seconds']:.1f}s first attempt and{resume['pause_seconds']:.1f}s successful diagnostic,including respective worker drains{earlier_resume['worker_drain_seconds']:.1f}/{resume['worker_drain_seconds']:.1f}s. Total{pause_total:.1f}s. The first diagnostic found a stale initial command/history cache after its own perturbed-state write,before rollout. Its source,traceback,receipts and owned failed-Kit cleanup are preserved in[attempt01](attempt01_initial_cache_failure/physical_probe_failure.json). Production resumed automatically. Diagnostic-only reset-history/cache rebuilding fixed the setup;live sources never changed. A first CPU fixture incorrectly demanded bitwise equality to reordered algebra;the corrected comparison uses actual unhooked autograd and passes. A premature CPU analysis invocation before result creation is also preserved. None is a reproduced training fault.

Completed-main rates{rates[0]:.2f}k active collection+PPO/{rates[1]:.2f}k worker-effective/{rates[2]:.2f}k whole-chain samples/s use distinct denominators and exclude documented pauses only when before their measurement endpoint. Main capacity/smoke/diagnostic samples are excluded. Latest completed{last['environments']}-env block has minimumGPU{last['minimum_free_gpu_MiB']}MiB,host{last['minimum_host_available_MiB']}MiB,private growth{last['private_memory_growth_MiB']}MiB,Torch allocation growth{last['torch_allocation_growth_MiB']}MiB. Resumed full-state load/KL/teacher/upper evidence is in[final learning verification](final_learning_verification.json). Original local ground and55 source hashes verify;no optimizer/actor reset,reward/physics/limit change or new production parameter was deployed.

Fusion is already closed;queued work stays preserved and it should not be reopened for routine checks. Stable6144 training now has about1.9GiB free VRAM and9GiB host RAM. Current empirical ramp asks2531MiB free for8192;1942MiB does not meet it. Preserve8192 target,300MiB reserve,host guards/fallback and180 continuous healthy TRAINING seconds. Evaluation/startup free memory is not steady capacity.32 unsaved R39 updates remain excluded;50 recovered R40 updates are hash-proven.

## Next focused tests

{chr(10).join(str(i+1)+'. '+s for i,s in enumerate(next_tests))}

Keep the same ACTIVE five-hour heartbeat,thread,notification preference and verified5/conditional4 endpoint. Final success requires>=4.95m/s or>=3.96m/s with causal native limits proving5 blocked,every unchanged gate and32 independently seeded initial states per command. Save,pause training and automation,and brief with computer on only then. Completed videos remain fulfilled.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
write(OUT/'evidence.json',dict(recorded_utc=now,owned=live['recorded_utc'],checkpoint=pair['checkpoint'],sha256=pair['sha256'],failures=failed,
    decision=decision,resume=resume,first_attempt_resume=earlier_resume,actual_retained_current_run_updates=count,
    inherited_main_updates=27507,throughput_k_samples_s=rates,physical=physical,upper_summary=upper_text,no_production_change=True))
write(OUT/'diagnostic_accounting_notes.json',dict(failed_attempt_preserved='attempt01_initial_cache_failure',
    first_pause_seconds=earlier_resume['pause_seconds'],successful_pause_seconds=resume['pause_seconds'],
    CPU_formula_fixture='First fixture used reordered analytic operations;preserved first source. Actual unhooked-autograd comparison now passes5 fixtures.',
    premature_analysis='Missing result at first invocation;log preserved. Re-run after completed result exits0.',
    fresh_stopped_hook_accounting='Exact before-standing capture;CPU subtraction agrees within4.5e-10 on fresh actual minibatches. Earlier failures not erased.',production_fault_inferred=False))
marker='adaptive_check_20261007_1845_record'
entry=f"\n<!-- {marker} -->\n## {now} — Saved-actor response and passive gradient diagnosis\n\n{summary}\n\n{decision['summary']} Paired384robots20s at0.15/1.5/3.5 show current course response0.368to0.129m/s lateral,versus rejected course1280.634to0.185;all survive,zero command mismatch and5ms driving clips. This is not qualification. Five passive-hook CPU fixtures and exact4-update4096-env PPO pass;before-standing capture resolves fresh subtraction within4.5e-10,alignment mixed. Upper actual angle-target-velocity/bias plots inspected:{upper_text} Same39000 resumes after total{pause_total:.1f}s across failed/successful diagnostics;no production change.55hashes/oneworker,6144,minGPU{last['minimum_free_gpu_MiB']}MiB,host{last['minimum_host_available_MiB']}MiB,rates{rates[0]:.2f}kactive/{rates[1]:.2f}kworker/{rates[2]:.2f}kchain.8192target/reserve300/180healthyTRAINING ramp and verified5 or conditional4 endpoint remain. Report:{OUT/'REPORT.md'}. All failed/rejected evidence preserved.\n<!-- /{marker} -->\n"
for f in [TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    assert marker not in f.read_text(encoding='utf-8-sig')
    with f.open('a',encoding='utf-8') as stream:stream.write(entry)
handoff=f"<!-- {marker} -->\nLatest{now}:{summary} {decision['summary']} Read{OUT/'REPORT.md'},saved_actor_response_analysis,angular_heading_audit,current_gradient_analysis,final_upper,final_learning_verification and freshACTIVE_RUN/state. Same39000,55hashes/oneworker,current20-batch auxiliaries. Two diagnostic suspensions total{pause_total:.1f}s;no diagnostic actor/source deployed. Upper asymmetry/slow bias persist,no human/Unitree/optical certification. Fusion already closed,queued work preserved. ACTIVE5h,8192target/reserve300/180healthyTRAINING ramp,verified5 or qualified4+causal limits blocking5;save,pauseboth,brief,computeron at endpoint. Earlier entries below are historical.\n<!-- /{marker} -->\n\n"
for f in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    old=f.read_text(encoding='utf-8-sig');assert marker not in old;f.write_text(handoff+old,encoding='utf-8')
npath=TRAINING/'automation/NEXT_CHECK.json';n=read(npath)
n.update(recorded_utc=live['recorded_utc'],active_run=live['active_run']['run'],report=str(OUT/'REPORT.md'),latest_observation=str(OUT/'final_cpu/live_handoff.json'),
    new_main_updates=learning['completed_main_updates'],recovered_main_updates=50,actual_retained_current_run_main_updates=count,inherited_main_updates_preserved=27507,
    selected_actor=dict(checkpoint=pair['checkpoint'],sha256=pair['sha256'],matching_gate_pair=str(OUT/'final_gate_review.json'),current_stage_qualified=False),
    highest_fully_qualified_stage='run_300',highest_fully_qualified_command_m_s=3.,current_stage=live['state']['stage'],current_stage_qualified=False,
    running_qualified=False,velocity_ceiling_proven=False,focused_tests=next_tests,latest_repair_decision=decision,
    throughput={k:learning[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},diagnostic_pause=resume,
    diagnostic_pauses_this_check=[earlier_resume,resume],mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),
    mandatory_upper_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'),mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'),
    production_change_this_check=False,newer_matching_pair=str(OUT/'final_gate_review.json'),newer_gate_trend=summary,
    latest_saved_actor_unqualified=dict(checkpoint=pair['checkpoint'],scope='Exact reviewed complete pair;newer partial actors unqualified'))
write(npath,n)
native=tomllib.loads((OUT/'automation_before.toml').read_text());prefix=native['prompt'].split('Latest dated five-hour check',1)[0]
dated=f"Latest dated five-hour check,evidence{live['recorded_utc']}: {summary} Read diagnostics/adaptive_check_20261007_1845/REPORT.md,decision.json,saved_actor_response_analysis.json,saved_actor_response.png,angular_heading_audit.json,current_gradient_analysis.json,final_upper plots/data,final_learning_verification.json and final_cpu/live_handoff.json,then freshACTIVE_RUN/state. {decision['summary']} No production reward/gain/cap/actor/optimizer/source reset. All teachers and upper20 continue. Current384-robot20s paired-state course response reduces lateral0.368to0.129,but course128 retains0.185 versus original2540.129;all survive,zero5ms driving clips and yaw-command saturation. This is not qualification. Desired-yaw variation near2.66Hz and50Hz heading/gyro mismatch require5ms attribution before a timing repair,not a clock-fault claim. Four actual PPO minibatches capture exact incoming derivatives before standing/moving hooks;fresh CPU standing subtraction within4.5e-10,alignment both signs. Five CPU hook fixtures and4 genuine diagnostic PPO/full-load checks pass;diagnostic actor discarded. Required actual upper angles/targets/velocities and bias plots inspected:{upper_text} Natural motion and mapped Unitree encoder/optical gaze remain uncertified. Same39000 resumes after370.7s failed-setup and310.7s successful diagnosis;both exclude from throughput when before its endpoint. Rates{rates[0]:.2f}kactive/{rates[1]:.2f}kworker/{rates[2]:.2f}kchain. Fusion is closed;6144 steadyGPU~1942MiB/host~9GiB,current8192 rampneeds2531MiB. Do not reopenFusion or force ramp.\n\nNext focused tests:\n"+'\n'.join(f"{i+1}. {s}" for i,s in enumerate(next_tests))+"\n\nPreserve full neutral-head learner,267 all-prior and original349 certified3m. Same ACTIVE5h thread/notifications/scope;8192target,reserve300,hostguards/fallback,180healthyTRAINING seconds;oneworker55hashes and actual retained accounting. Final5 or conditional4 requires all unchanged gates,32 independently seeded initial states/cmd and causal native limits for4. At verified endpoint save,pause training+automation,brief,computeron. Videos fulfilled,Fusionqueuedwork preserved."
args=dict(id=native['id'],mode='update',kind='heartbeat',name=native['name'],status='ACTIVE',rrule=native['rrule'],targetThreadId=native['target_thread_id'],prompt=prefix+dated)
if 'notification_policy' in native:args['notificationPolicy']=native['notification_policy']
write(OUT/'automation_update_args.json',args)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),same_run=True,retained_current_updates=count,automation_args_ready=True)))
