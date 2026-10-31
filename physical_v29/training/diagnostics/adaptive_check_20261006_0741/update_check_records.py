"""Evidence-backed heartbeat/handoff records, preserving durable user scope."""
import argparse,hashlib,json,tomllib
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];WORKSPACE=HERE.parents[1]
AUTO=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def prepare():
 live=read(OUT/'closing_cpu/live_handoff.json');s=live['state'];auto=tomllib.loads(AUTO.read_text())
 assert auto['id']=='humanoid-training-research-and-repair-every-four-hours'
 assert auto['kind']=='heartbeat' and auto['status']=='ACTIVE' and auto['rrule']=='FREQ=HOURLY;INTERVAL=5'
 assert s['stage']=='run_350' and s['status']=='running'
 evidence=read(OUT/'new_main_evidence.json')
 assert s['total_training_iterations']>0 or any(x['observed_updates']>=8 for x in evidence['partial_main_blocks'])
 assert len(live['actual_worker_pids'])==1 and all(live['frozen_source_integrity'].values())
 assert read(Path(live['active_run']['run'])/'termination_requalification.json')['all_prior_gates_fresh']
 now=live['recorded_utc'];proof=read(OUT/'termination_continuation_proof_retry.json')
 focus=f'''Latest dated five-hour check, evidence {now}: R39_commanded_travel_teacher_certificate continues the EXACT full R37checkpoint459 actor,critic,Adam and trained exploration, SHA75b0a3ea651e5bb9872404b9df8ac0686eb38a24f04e3cf639ad90425ad92a08. Preserved16936 actual earlier main updates include9575 R37(9517 completed+58 verified recovery) and7361 inherited. Read diagnostics/adaptive_check_20261006_0741/REPORT.md, travel_guard_attribution.json, physical_validation.json, physical_parity.json, termination_continuation_proof.json, migration_cpu_tests.json, deployment.json, handoff_completion.json and closing_cpu/live_handoff.json, then fresh ACTIVE_RUN/state. Source-frozen55 hashes verify; one actual worker. Current stage3.5 remains unqualified. New completed main updates at observation:{s['total_training_iterations']}; read newer progress. Capacity/preflight/diagnostic actors were discarded.

R37 newly qualified3.0 at349, matching350/351: signed2.9979/2.9982m/s and every nominal/pushed/stopped/prior gate passed. The later apparent zero-survival3.5 plateau reproduced an unconditional60m absolute-travel termination: all24 sampled events across12 pairs were ONLY excessive_drift, near18s upright. Corrected source integrates actual planar action-command travel per environment across command switches and permits original60m excess margin beyond that budget. Reset clears only selected rows; all other fault predicates remain. Reward bodies/weights, verifier thresholds, CAD physics/caps, observations/actions/PPO/teachers/head module and iteration4 unchanged. A termination source-signature change is declared; strict proof binds only original459 and trusted349, inverse literal patches, protected source hashes and physical/load evidence. No generic signature bypass or live frozen edits.79 guard CPU checks,15 rejection fixtures, real PPO/reload, full-state equality and117-array same-actor physics comparisons pass. Current459 before each old false terminal and full teacher34920s tests match bit for bit. Fresh trusted349 all-prior3m tests also pass in R39; current learner must requalify every gate.

Corrected baseline459 nominal/pushed at3.5:100% survival,3.530/3.535 signedm/s,38.74/38.60% true geometric flight; world lateral0.401/0.362m/s FAILS. Both also fail retained0.15 forward-net-travel,1.5 genuine flight(1.323/1.427%, required1.5%) and2.5 foot slip. Other cases pass. Do not call all these prior skills retained on the latest learner, or interpret old artificial truncations as actual falls. Fresh gates bind exact actors;5/4 endpoint remains unproven.

Mandatory upper time-series completed on corrected459 and fresh349 pairs:32 robots/command at10Hz,16 representative50Hz traces, reset/terminal rows excluded. Inspect corrected_upper/upper_joint_angles.png, upper_joint_velocities.png, CSV and JSON. At459/1m/s shoulder ranges37.51/27.68deg, arm/ipsilateral-hip phase177/153deg; at3.0 phases89/-177deg show unequal coordination, and at3.5 small ranges9.41/7.21deg have phases102/111deg. HeadRMS at3.5 is3.78deg, with gait-locked yaw oscillation; native zero is torso-relative. No natural-arm or exact human/Unitree certification: primary human comparison is feature-based; no mapped speed-matched Unitree encoder curve obtained. CPU motor mapping on all320 robots at10Hz finds occasional ankle cap-speed crossings, not a sustained driving ceiling; left ankle motor2 p99~0.803cap, at/above-cap0.45%. Substep driving/torque attribution is still needed before a high-speed limit claim.

Old-to-new-supervisor saved-boundary gap1029.1s is documented, old supervisor safely retired and no unsaved user work touched. R37 pre-pause rates44.17k activePPO/38.23k worker-effective/25.62k whole-chain exclude679.7s prior pauses. New-run startup/capacity and current main rates have separate clocks; never count discarded benchmarks as learning or use evaluation free memory as steady capacity. Target8192/reserve300/host fallback/180healthy-training-second ramps remain.

Next focused tests:(1) preserve current full-state learner; compare newer matching3.5 direction AND all retained commands against corrected459, particularly0.15 net travel,1.5 flight and2.5 slip, preserving certified349. Verify commanded travel budget resets/tick consistency in fresh traces; investigate real heading/lateral objective and per-command teacher coverage if progress stalls, rather than mistaking the repaired travel guard for solved gait.(2) Required current upper angle/target/velocity timelines, bilateral amplitude/elbow coordination and shoulder/leg phase; quantify actual effective upper targets, physical response and auxiliary/PPO gradients before a narrow matched-control full-state refinement. Preserve neutral head and leg learning; do not spoof observations or deploy hybrids. Existing teacher-context mismatch remains undeployed unless controlled evidence warrants it.(3) verify one worker,55 hashes, exact full-state memory recovery, moving6144/8192 capacity and180continuous healthy-training-second ramps. If a real high-speed plateau remains, measure native5ms motor speed/available torque and sustained DRIVING saturation through exact CAD reductions/Jacobians before attributing a limit. Keep all caps and rejected candidates unchanged absent evidence. Continue this same FIVE-hour heartbeat until verified5 or verified4 plus causal native-limit evidence, then save,pause training+automation,brief,computer on.'''
 partial_count=sum(x['observed_updates'] for x in evidence['partial_main_blocks'])
 focus=focus.replace('Capacity/preflight/diagnostic actors were discarded.',f'Capacity/preflight/diagnostic actors were discarded. Genuine current partial main PPO updates observed:{partial_count}; these are separate from completed blocks. Read new_main_evidence.json for exact load,zero pre-updateKL/command mismatch and memory.')
 focus=focus.replace('termination_continuation_proof.json, migration_cpu_tests.json','termination_continuation_proof_retry.json, migration_cpu_tests_retry.json, certificate_filename_cpu_tests.json')
 focus=focus.replace('deployment.json, handoff_completion.json','deployment.json, certificate_retry_deployment.json, handoff_completion.json')
 timing=read(OUT/'continuation_timing.json')
 focus=focus.replace('Old-to-new-supervisor saved-boundary gap1029.1s is documented, old supervisor safely retired',f'R38 startup gap1029.1s and original supervisor retirement are documented. R38 subsequently stopped before main learning because fresh teacher filenames lacked the certificate lookup `_gate_` marker; no new main learning was lost. R39 corrects only the supervisor label, repeats actual startup/teacher gates, and is genuinely learning. Saved-boundary-to-R39-supervisor gap{timing["saved_boundary_to_R39_supervisor_seconds"]:.1f}s and to-first-reported-real-PPO gap{timing["saved_boundary_to_first_reported_R39_PPO_seconds"]:.1f}s include diagnosis and repeated startup/capacity; no silent pause remains')
 marker='Latest dated user schedule/upper-motion update'
 # Later repetitions replace only the dated handoff, preserving durable scope.
 if marker not in auto['prompt']:marker='Latest dated five-hour check'
 prefix=auto['prompt'].split(marker)[0];assert 'Every check MUST analyze upper-body joint angles' in prefix
 prompt=prefix+focus
 payload=dict(mode='update',id=auto['id'],kind='heartbeat',name=auto['name'],status='ACTIVE',rrule=auto['rrule'],
  targetThreadId=auto['target_thread_id'],prompt=prompt)
 if 'notification_policy' in auto:payload['notificationPolicy']=auto['notification_policy']
 write(OUT/'automation_payload.json',payload)
 (OUT/'automation_focus.txt').write_text(focus,encoding='utf-8')
 print(json.dumps({'payload':str(OUT/'automation_payload.json'),'recorded':now,'main_updates':s['total_training_iterations']}))

def finish():
 payload=read(OUT/'automation_payload.json');auto=tomllib.loads(AUTO.read_text())
 for key,k in [('id','id'),('kind','kind'),('name','name'),('rrule','rrule'),('status','status'),('target_thread_id','targetThreadId')]:assert auto[key]==payload[k]
 assert auto['prompt'].rstrip('\n')==payload['prompt'].rstrip('\n')
 before=tomllib.loads((OUT/'automation_before.toml').read_text())
 assert auto.get('notification_policy')==before.get('notification_policy')
 (OUT/'automation_after.toml').write_bytes(AUTO.read_bytes())
 verification=dict(passed=True,recorded_utc=datetime.now(timezone.utc).isoformat(),same_id_thread_kind_notification=True,
  five_hour_cadence_verified=True,status='ACTIVE',durable_scope_endpoint_upper_motion_preserved=True,
  prompt_sha256=hashlib.sha256(auto['prompt'].encode()).hexdigest())
 write(OUT/'automation_verification.json',verification)
 live=read(OUT/'closing_cpu/live_handoff.json');s=live['state'];now=live['recorded_utc'];run=Path(live['active_run']['run'])
 corrected=read(OUT/'physical_validation.json');upper=read(OUT/'corrected_upper/upper_joint_time_audit.json');counts=read(OUT/'throughput_and_loads.json')
 row=lambda speed:next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==speed)
 decision=dict(recorded_utc=now,action='Deploy proved commanded-travel termination correction; preserve exact original full learner.',
  current_stage='run_350',current_stage_qualified=False,highest_fresh_teacher_qualified_command_m_s=3,
  numerical_rewards_or_weights_changed=False,independent_classifier_changed=False,physics_motor_caps_changed=False,
  termination_predicate_changed=True,source_signature_migration_explicit=True,actor_critic_Adam_exploration_retained=True,
  smoke_actors_deployed=False,human_like_arms_certified=False,velocity_ceiling_proven=False,
  actual5m_s_proven=False,actual4m_s_fallback_proven=False,automation_active_every5hours=True)
 write(OUT/'decision.json',decision)
 report=f'''# Five-hour training check — {now}

R37 independently qualified **3 m/s**, then encountered an artificial episode termination at3.5. The saved full learner is continued as **R39_commanded_travel_teacher_certificate**; no policy or optimizer is started over. Current3.5m/s remains unqualified because of direction and prior-skill failures. Training is healthy at the owned observation; read fresh ACTIVE_RUN/state for later progress.

## Independent progress and reproduced failure

Exact R37checkpoint349, matching350/351, passed every nominal/pushed stopped/prior gate through3m/s with signed2.9979/2.9982m/s. Later3.5 tests had zero end-of-rollout survival. `travel_guard_attribution.json` reproduces24 representative terminal events over12 exact gate pairs: all were ONLY `excessive_drift`, while upright near18s. Frozen environment terminated total displacement above60m regardless of command.3.5m/s for20s exceeds that cap. These events are **false physical-fall classifications**, not evidence of an actuator ceiling. The earlier user-directed schedule review remains historical.

The official [Unitree G1 velocity configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py) uses timeout, root-height and orientation termination. Our60m absolute-travel guard was a local addition; copied numerical reward bodies alone did not prevent this fault. The correction retains60m of excess travel margin beyond integrated actual planar action-command travel. The budget accumulates across command switches, is idempotent per control tick and resets only selected environments. Low height, tilt, invalid differential, nonfinite and nonfoot fault predicates remain. Every independent direction, speed, flight, contact and prior gate remains unchanged.

Corrected current459 uses320 robots,32 trials per command and20s, matching nominal/pushed. At3.5 all survive, signed3.530/3.535m/s, true geometric flight38.743/38.597%; **world lateral speed0.401/0.362m/s still fails**. Retained0.15 fails forward net travel,1.5 flight is1.323/1.427% below1.5%,2.5 fails foot slip. Stopped0.35/0.65/1.0/2.0/3.0 cases pass. `physical_validation.json` saves every case. These are baseline459 findings, never qualifications of later actors. Teacher349 freshly passes every prior gate through3 on corrected source.

## Repair, validation and continuation

79 CPU guard fixtures pass, covering commanded speeds through5m/s, switches, stops, row resets, idempotence and original physical faults. Fifteen negative migration fixtures reject wrong schema/checkpoints/hash, unprotected reward/gate/geometry changes, fabricated qualification and lost recovery. `physical_parity.json` records117 bitwise array comparisons: current459 is identical before each old false termination; nominal and pushed teacher349 are identical for their full20s. No trajectories, observation rewrites, head overrides or hybrids are deployed. A matched learning experiment is unnecessary for this deterministic fault predicate: unchanged-actor physical counterfactual/parity directly reproduces the error and isolates its correction; future behavioral refinements still require matched learning control.

Source-signature changes are explicitly declared. Only baseenvironment and worker loading provenance change inside the runtime tree; the supervisor gains a strict proof verifier and fresh teacher requalification.51 other runtime files are identical, including all CAD geometry/mass/contact, native caps, numerical reward kernels/weights, verifier,400-input23-output observations/actions, PPO, teachers/head refinement and iteration4 execution. The proof restricts old-signature loads to original459 and trusted349, verifies literal inverse patches and protected hashes, and binds CPU/physics/reload evidence. No generic source mismatch is accepted. Original459 SHA75b0a3ea651e5bb9872404b9df8ac0686eb38a24f04e3cf639ad90425ad92a08 supplies the exact actor,critic,Adam and trained exploration.16936 actual earlier main updates are preserved:9517 completed R37+58 verified recovered+7361 inherited. Recovery069→070 provenance is retained. Smoke/capacity actors are discarded.

The first simulator attempt rejected changed source signatures; a second PPO smoke correctly rejected the old-signature moving teacher after fresh physical requalification. Both failures/logs/verified owned cleanup are preserved. Narrow explicit checkpoint and teacher migration was then tested:4 genuine64-env PPO updates, zero pre-updateKL/command mismatch, all actor/critic/Adam/exploration tensors equal before training, and reload passes. The4s reload is a load smoke, **not** a qualifying gate. A diagnostic manifest-reader initially used the wrong supervisor path; corrected mapping matches the existing CPU recorder. No training evidence was overwritten.

R38 fresh3m teacher tests passed, but certificate lookup required a _gate_ filename and stopped before main learning. The failed frozen run is preserved. R39 changes only that supervisor filename label; a CPU regression reproduces the old lookup error and validates the new pattern and real diagnostic certificate. Fresh physics/full-load/capacity/teacher gates are repeated before production learning. No diagnostic actor is deployed.\n\nR37 was suspended at the saved459 boundary and drained, then retired after healthy R39 source freeze/physics/full-load/reload. Old learning-to-new supervisor gap1029.1s and suspension-until-retirement1230.1s are separately saved in `handoff_completion.json`; new startup/capacity clocks are separate. No unsaved user work was touched. At the closing observation, new completed main updates={s['total_training_iterations']}, stage={s['stage']}, operation={Path(s['operation_path']).name}, operation count={s['environment_count']}, selected next count={s.get('next_training_environment_count')}. One actual CUDA worker and55 frozen hashes verify. Read `new_main_evidence.json` for actual update/load/memory rates; startup/capacity is not qualification. R39 freshly requalifies trusted349; the actual learning actor must still pass its own current and prior gates.

## Required upper joint-angle timelines

`corrected_upper` contains actual head yaw/pitch, shoulder pitch/**roll**, elbows and hips, physical target plots, separate velocity plots, SVG, CSV and JSON. Both corrected459 and fresh349 exact pairs are analyzed:32 robots per command at10Hz and16 selected representatives per gate at50Hz. Terminal/reset rows are excluded. Four metric fixtures pass. Both plots were visually inspected. An inefficient repeated compressed-NPZ read during diagnostic CSV export was replaced by one immutable array read; cached rerun finishes in seconds. This changes no training runtime.

At459/1m/s shoulder pitch ranges are37.51/27.68deg and ipsilateral phases177/153deg. At3m/s phases89/-177deg are markedly unequal; at3.5 ranges9.41/7.21deg and phases102/111deg show weak counter-swing despite alternating hips. Representative left shoulder is biased back relative to the right, with smaller pitch swings and synchronous roll tendencies at fast speeds. Head RMS is4.06deg at1,2.56 at2.5,2.88 at3 and3.78 at3.5; native yaw shows gait-linked oscillation. This is **not natural-arm certification**. Head native zero describes torso-relative alignment, not world gaze. Physical target/joint RMS upper error at3.5 is7.69deg; raw upper actions outside the clamp are0% in the selected trace, so raw clipping alone does not explain those arms. Target/slew and physical response still matter. Velocity spikes and smoothness statistics are descriptive50Hz sample diagnostics, not imported human thresholds.10Hz population statistics cannot diagnose faster jitter.

The [Mirelman et al. human study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) measured wrist/lower-back IMUs, arm amplitude, symmetry and jerk under different walking conditions. Our comparison is feature-based, with mapped native signs/canted hip axes and different morphology. No verified comparable-speed Unitree G1/H1 encoder trajectories were obtained; rest/reward configs and dance motions are not substituted. Exact human/Unitree angle-curve similarity remains unmeasured. Preserve learned head/legs and test any later arm refinement against matched control.

## Resources and native motor evidence

Reaudited R37 retained-main rates:44.17k active collection+PPO,38.23k worker-effective,25.62k whole-chain samples/s.679.7s of earlier diagnostic pauses are excluded; current measurement ends at suspension. These denominators differ. `throughput_and_loads.json` binds all completed/recovered samples and exact receiver proof; all sampled main pre-updateKL and action-command mismatches are zero. New R39 rates are separately saved, with discarded capacity trials excluded from learned progress.

`motor_speed_audit.json` maps all320 robots' saved10Hz physical q/qd through exact pose-dependent CAD Jacobians/reductions; directional finite-difference checks pass. At3.5 left ankle motor2 p99 is0.803cap, sampled at/above-cap fraction0.45%, maximum1.75cap; right ankle also has sparse crossings. These may include passive/inertial motion and **do not prove sustained driving clipping, torque limitation or a velocity ceiling**. No fresh5ms ceiling attribution was performed this check. A real high-speed plateau requires phase-specific native substep speed/torque evidence before the4m/s fallback. Target8192,300MiB reserve,host fallback,one worker,no stereo and180continuous healthy **training** seconds before ramps remain. Evaluation/startup free memory is not steady PPO capacity.

## Next focused work and automation

1. Compare newer exact nominal/pushed3.5 direction and all prior gates against corrected459. Preserve certified349; investigate real lateral/heading behavior and retained0.15 net travel,1.5 flight and2.5 slip if they stall. Do not classify the repaired displacement guard as solved locomotion.
2. Repeat/reset-segment upper actual-angle/target/velocity, bilateral coordination and arm-leg phase audits. Quantify effective upper targets, response and auxiliary/PPO gradients before a narrow matched-control full-state refinement; no head rollback, observation spoof or policy reset.
3. Verify55 hashes,one owned worker,full-state fallback/sample accounting,moving6144/8192 memory and healthy ramps. If native limits appear, attribute actual5ms motor speed/torque driving saturation rather than sparse spikes. No rejected candidate is revived without evidence.

The same heartbeat remains ACTIVE every FIVE hours, same thread/notification/GPT-6.1 Sol max instruction, with dated focus refreshed through the native update tool. [Official schedule guidance](https://learn.chatgpt.com/docs/automations) supports updating an existing scheduled task. Endpoint stays verified5m/s, or verified4m/s only with proven native limits blocking5, every matching nominal/pushed/stopped/prior gate and32 independent trials per command. At endpoint save,pause owned training and automation,brief; **computer stays on**. No endpoint is proven now.
'''
 report=report.replace('## Repair, validation and continuation',
  'The corrected termination/done semantics intentionally change episode endings and any downstream terminal-penalty/return effects after the former60m boundary. Numerical reward bodies and weights are preserved; the complete training trajectory is not asserted identical after that boundary.\n\n## Repair, validation and continuation')
 main=read(OUT/'new_main_evidence.json');partial_count=sum(x['observed_updates'] for x in main['partial_main_blocks'])
 report+=f'\nClosing main-learning accounting: {s["total_training_iterations"]} completed updates plus {partial_count} observed current partial PPO updates. Partial active rates/memory/load proof are in new_main_evidence.json; no partial block is called completed or qualified.\n'
 (OUT/'REPORT.md').write_text(report,encoding='utf-8')
 nxt=read(HERE/'automation/NEXT_CHECK.json');nxt['previous_status_report']=nxt.get('report')
 nxt.update(recorded_utc=now,active_run=str(run),report=str(OUT/'REPORT.md'),latest_observation=str(OUT/'closing_cpu/live_handoff.json'),
  new_main_updates=s['total_training_iterations'],recovered_main_updates=0,actual_retained_current_run_main_updates=s['total_training_iterations'],
  inherited_main_updates_preserved=16936,current_stage=s['stage'],highest_fully_qualified_stage='run_300',highest_fully_qualified_command_m_s=3,
  running_qualified=False,velocity_ceiling_proven=False,latest_verified_state=s,latest_gate_trends=str(OUT/'initial_gate_review.json'),
  latest_closing_gate_pair=str(OUT/'corrected_gate_review.json'),latest_repair_decision=decision,
  throughput=read(OUT/'new_main_evidence.json'),throughput_evidence=str(OUT/'new_main_evidence.json'),
  prior_run_throughput=counts,final_liveness=str(OUT/'closing_cpu/live_handoff.json'),automation_verified_active=True,
  latest_schedule_verification=str(OUT/'automation_verification.json'),cadence_hours=5)
 nxt['observed_partial_main_updates']=partial_count
 nxt['main_count_scope']='New R39 genuine main completed and observed partial updates are separate;16936 earlier actual main updates preserved, including58 recovered R37. Discarded smoke/capacity excluded.'
 nxt['actual_retained_count_status']='Verified original R37 recovered samples/receiver; current R39 completed-main/partial progress separately bound.'
 nxt['selected_actor']=dict(checkpoint=corrected['current_checkpoint'],sha256=corrected['checkpoint_sha256'],matching_gate_pair=str(OUT/'corrected_gate_review.json'),
  current_stage_qualified=False,scope='Exact original459 corrected baseline, not attribution of later learned actor.')
 nxt['certified_run_300_checkpoint']=read(OUT/'state_before_diagnosis.json')['promotions'][-1]['checkpoint']
 nxt['promoted_policy_audit']=str(run/'termination_requalification.json')
 nxt['focused_tests']=(OUT/'automation_focus.txt').read_text().split('Next focused tests:')[1].split('Continue this same')[0].strip()
 nxt['focused_tests']=[
  'Preserve full learner. Compare newer matched3.5 direction plus all prior cases against corrected459; especially0.15 net travel,1.5 flight,2.5 slip. Keep freshly requalified349. Corrected travel budget is not a successful gait or limit proof.',
  'Required per-speed actual head/shoulder pitch+roll/elbow angle-target-velocity timelines and bilateral/arm-leg phase. Track asymmetry and gait-locked head yaw; measure effective upper targets/physical response/auxiliary-PPO gradients before narrow matched-control full-state refinement. No reset,spoof,hybrid or posture rollback.',
  'Verify55 hashes,one worker,full-state recovery/accounting,moving6144/8192 memory,reserve300/host guards/180healthy-training-second ramps. Real high-speed limit needs native5ms speed/torque sustained driving attribution; sparse ankle cap crossings do not certify4m fallback.']
 nxt['upper_body_analysis']=dict(required_every_check=True,script=str(OUT/'upper_joint_time_audit.py'),latest_json=str(OUT/'corrected_upper/upper_joint_time_audit.json'),
  angle_plot=str(OUT/'corrected_upper/upper_joint_angles.png'),velocity_plot=str(OUT/'corrected_upper/upper_joint_velocities.png'),csv=str(OUT/'corrected_upper/upper_joint_angles.csv'),
  reset_segmentation=True,exact_Unitree_joint_curve_similarity_measured=False,human_like_motion_certified=False)
 write(HERE/'automation/NEXT_CHECK.json',nxt)
 workflow=HERE/'automation/FIVE_HOUR_CHECK.md';content=workflow.read_text(encoding='utf-8-sig')
 content=content.replace('diagnostics/automation_five_hour_update_20261006/upper_joint_time_audit.py','diagnostics/adaptive_check_20261006_0741/upper_joint_time_audit.py')
 content=content.replace('It writes actual angle/target plots, CSV and per-command population/coordination statistics.','It caches immutable NPZ arrays once, segments first episodes before terminal/reset ticks, and writes actual head/shoulder pitch+roll/elbow angle-target plots, separate velocity plots, CSV and per-command population/coordination statistics.')
 workflow.write_text(content,encoding='utf-8')
 entry=f'''\n\n## {now} — Corrected commanded60m termination, exact full-state R39 continuation\n\nR37 newly qualified3m/s at349. All24 sampled later3.5 terminal events over12 pairs were solely excessive_drift near18s,upright: a60m absolute-travel guard falsely classified commanded motion. Repair integrates actual planar action-command travel and retains original60m excess margin; other physical faults/all independent classifiers unchanged.79 CPU checks,15 migration-rejection fixtures,117 bitwise array comparisons,real4-update PPO/full-load/reload pass. Failed signature/teacher-signature attempts and owned cleanup are preserved. Only environment/worker provenance+strict supervisor change;51 other runtime files identical,physics/CAD/native caps/reward/PPO/upper modules unchanged.\n\nR39 preserves459 original actor/critic/Adam/exploration and16936 earlier actual updates including58 recovered; all55 hashes verify. Old supervisor retired after healthy startup; gap1029.1s recorded. Fresh349 priors through3 pass. Current459 baseline3.5 survives100%,3.530/3.535signedm/s,38.74/38.60%flight but fails lateral0.401/0.362; prior0.15net travel,1.5flight,2.5slip also fail. New completed main updates at observation{s['total_training_iterations']}; no3.5/5/4 qualification or ceiling claim.\n\nMandatory reset-segmented per-speed upper angles/targets/velocities inspected: at1 shoulder ranges37.51/27.68deg,phase177/153; at3 unequal89/-177; at3.5small9.41/7.21deg ranges andheadRMS3.78deg/gait oscillation. Human comparison remains morphology-limited,Unitree speed-matched encoder curves unavailable. Sparse10Hz CAD ankle cap crossings are not sustained driving clipping;5ms/torque attribution needed before fallback. Report/evidence/decision: {OUT/'REPORT.md'}. Existing same-thread heartbeat stays ACTIVE5h,verified5 orverified4+causal native limit,save/pause/brief,computer on.\n'''
 attempts=[]
 for folder,label,explanation in [
  ('001_current_run_350_gate_nominal','Guard diagnostic1:source signature rejection','Changed baseenvironment source signature correctly rejected the old checkpoint. No learned state deployed. Added a strict checkpoint-SHA/source-inverse-patch bridge; no generic bypass.'),
  ('104_full_state_ppo_smoke','Guard diagnostic2:moving teacher signature rejection','Fresh3m physical teacher gates passed; PPO correctly rejected its original source signature. Added the same bounded explicit teacher migration and reran exact full-state PPO/reload. Failed actor was never deployed.')]:
  identity=read(OUT/folder/'owned_pid.json');stamp=datetime.fromtimestamp(identity['started'],timezone.utc).isoformat()
  attempts.append(f'\n\n## {stamp} — {label}\n\n{explanation} Evidence: {OUT/folder}. Verified owned failed-worker cleanup is preserved; numerical rewards, classifiers, physics and caps unchanged.\n')
 failed=Path(read(OUT/'deployment.json')['new_run']);bad=read(failed/'state.json')
 stamp=datetime.fromtimestamp(bad['updated_unix'],timezone.utc).isoformat()
 attempts.append(f'\n\n## {stamp} — Guard continuation R38 startup:teacher certificate lookup failure\n\nFresh teacher nominal/pushed pass and current459 gates run, but qualification lookup expects `_gate_` filenames. Supervisor stops before main learning;0 new main updates. Frozen run/logs preserved: {failed}. R39 uses the same original full459 learner and runtime bytes with a corrected supervisor gate label. Lookup regression and fifteen negative migration fixtures pass; fresh production startup/teacher/PPO repeats are required. No lost main learning or from-scratch actor.\n')
 for path in [HERE/'ITERATION_HISTORY.md',WORKSPACE/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
  with path.open('a',encoding='utf-8') as f:
   for attempt in attempts:f.write(attempt)
   f.write(entry)
 head=f'''<!-- adaptive_check_20261006_0741 -->\nCurrent handoff {now}: R39_commanded_travel_teacher_certificate preserves exact459 actor/critic/Adam/exploration and16936 earlier actual updates. Read {OUT/'REPORT.md'}, {HERE/'automation/FIVE_HOUR_CHECK.md'}, NEXT_CHECK and fresh ACTIVE_RUN/state. R37 qualified3 at349; later3.5zero survival was a reproduced60m commanded-travel truncation, not sampled physical falls. Corrected459 survives but direction/retention fail.55 hashes verify,one actual worker,new main updates{s['total_training_iterations']}. Mandatory upper angle-target-velocity timelines show asymmetry; no exact Unitree/human likeness or ceiling proof. Same5h heartbeat/5 orverified4+native-limit endpoint,save/pause/brief,computer on. Earlier notes below are historical.\n<!-- /adaptive_check_20261006_0741 -->\n\n'''
 for path in [Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
  path.write_text(head+path.read_text(encoding='utf-8-sig'),encoding='utf-8')
 print(json.dumps({'records_saved':True,'automation_verified':verification,'report':str(OUT/'REPORT.md')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','finish']);a=p.parse_args()
 prepare() if a.mode=='prepare' else finish()
