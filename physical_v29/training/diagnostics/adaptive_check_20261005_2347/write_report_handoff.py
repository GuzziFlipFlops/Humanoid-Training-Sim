"""Save evidence-backed check records and prepare one native heartbeat update."""
import json,tomllib,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
e=read(OUT/'evidence.json');d=e['decision'];g=e['latest_pair'];q=e['qualified_1_5'];s=read(OUT/'closing_owned_observation.json')['state']
rates=read(OUT/'throughput_and_loads.json');audit=read(OUT/'promoted_policy_audit.json');now=e['recorded_utc'];run=Path(e['run'])
marker='adaptive_check_20261005_2347'
case=lambda p,k,cmd:next(c for c in p[k] if abs(c['command']-cmd)<1e-6)
n,p=case(g,'nominal_cases',2.),case(g,'pushed_cases',2.);rn,rp=case(g,'nominal_cases',1.5),case(g,'pushed_cases',1.5)
qn,qp=case(q,'nominal_cases',1.5),case(q,'pushed_cases',1.5)
head=[next(c['head_RMS_deg'] for c in e['latest_head'][k] if c['command']==2.) for k in ['nominal','pushed']]
report=f'''# Adaptive training check — October 5, 23:47 UTC

Evidence consolidated at {now}. **Running at the 1.5 m/s command has independently qualified, and the same R37 learner has advanced to 2.0 m/s.** The exact qualified actor is `199_run_150_train_064/candidate.pt`, SHA `{q['sha256']}`, with full matching nominal200/pushed201 gates. Actual signed speed is {qn['signed_speed']:.3f}/{qp['signed_speed']:.3f} m/s and geometric flight is {100*qn['flight']:.4f}%/{100*qp['flight']:.4f}%. Every stopped and earlier-speed case passes those same tests. This is a real qualification, rather than a command, reward or isolated video claim.

**The current 2.0 stage remains unqualified because 1.5 m/s flight retention fails.** Exact newer actor `{Path(g['checkpoint']).parent.name}` passes both 2.0 cases at {n['signed_speed']:.3f}/{p['signed_speed']:.3f} m/s and {100*n['flight']:.4f}%/{100*p['flight']:.4f}% true flight; all stopped/walking commands through1.0 also pass. At retained1.5, actual signed speed is {rn['signed_speed']:.3f}/{rp['signed_speed']:.3f}, but flight is {100*rn['flight']:.4f}%/{100*rp['flight']:.4f}% against the unchanged1.5% minimum. These are improved over the initial near-zero retained-flight blocks, but improvement is nonmonotonic. All224 robots survive20 seconds,32 per command. Newer unsaved/partial updates are separately unqualified.

The decision is to preserve the improving full-state learner and its certified checkpoints. **No production source, reward, weight, physics, CAD motor cap, optimizer, actor or exploration was reset or changed; training was not paused this check.** No5m/s or relevant velocity-ceiling endpoint is established. The same supervisor32492, one actual worker and55 frozen authored hashes verify. Rewardv8/verifierv6, truthful400-input/23-output inference, existing teacher/posture module, original cached ground and iteration4 execution remain. [Decision](decision.json), [fresh owned snapshot](closing_owned_observation.json), [qualification](qualified_run_150.json), [latest exact pair](closing_matching_gates.json).

## Exact current independent results

Checkpoint SHA `{g['sha256']}`; nominal `{Path(g['nominal']).name}`, pushed `{Path(g['pushed']).name}`. Each case includes32 trials for20 seconds, retaining startup falls in the survival decision. These results bind this saved actor.

| Command m/s | Nominal/pushed signed speed | Nominal/pushed true flight % | Pass both | Failure |
|---:|---:|---:|---|---|
'''
for a,b in zip(g['nominal_cases'],g['pushed_cases']):
    report+=f"| {a['command']:.2f} | {a['signed_speed']:.3f}/{b['signed_speed']:.3f} | {100*a['flight']:.4f}/{100*b['flight']:.4f} | {a['passed'] and b['passed']} | {', '.join(sorted(set(a['failures']+b['failures']))) or 'None'} |\n"
report+=f'''
The original1.5 qualification remains attached to199. It must not be assigned to later2.0-stage actors that lose1.5 flight. New2.0 cases repeatedly pass direction, actual speed, alternating steps, cadence, clearance and geometric flight. The declining2.0 flight fraction remains above the existing gate minimum; monitor it together with retained1.5 rather than assuming monotonic progress. [All exact pairs](closing_gate_review.json), [inspected trend chart](running_retention_trends.png).

## New teacher-provenance diagnosis

The source now loads certified neutral-head199 as its locomotion teacher, covering commands up to1.5. However, `LocomotionOnlyRetention` still applies the unchanged head-context table originally collected for the tilted-head R34 walking teacher. Its context table records physical source traces, but the runtime does not bind its applicability to the current teacher checkpoint SHA. This was found by reading the actual frozen source and current qualified-prior certificate, not by assuming a dated teacher remained active.

CPU reconstruction from synchronized50Hz saved q/qd/root/action/commands reproduces **11,713 actual actor inputs for each of199 and222**, with maximum action discrepancies {audit['records'][0]['actor_reconstruction_max_error']:.2e}/{audit['records'][1]['actor_reconstruction_max_error']:.2e}. Actor observation normalization is disabled in the saved configuration. The replay checks native float32 phase, five-frame history, prior applied actions, exact observed/action commands, and excludes unavailable reset histories; no robot resets occur in these traces.

On199's own qualifying truthful walking states, self-teacher lower-output discrepancy without the old context is approximately1e-6. Applying that context creates0.231–0.258 RMS discrepancy in clipped action units, with maximum changes above1.2. Thus the deployed auxiliary no longer has the certified actor as a fixed point on those states. On222's truthful current walking states, context changes the same teacher's lower labels by0.236–0.279 RMS; the current student is closer to those conditioned labels than to the truthful teacher. This is a reproduced label/fixed-point mismatch and a useful next experiment, rather than proof that context removal will improve physical gait.

**The old table changes no1.5 labels.** It cannot directly explain the1.5 flight regression; indirect shared-parameter/trajectory effects remain hypotheses. These CPU samples are noiseless independent evaluation observations, not actual noisy PPO minibatch advantages or measured PPO/prior gradient interference. Any correction needs a current full-state matched-control64/128-update comparison, truthful student input, physical/current/prior/head/direction/flight checks, original-teacher parity and checkpoint-load verification before deployment. Preserve conditioning for the original teacher whose dependence was physically validated. A later neutral-head teacher requires its own justified context decision. No candidate or diagnostic observation change was deployed. [Replay, labels and context provenance](promoted_policy_audit.json).

## Posture, contacts and native motor audit

At the current2.0 command, full-population head RMS is {head[0]:.2f}/{head[1]:.2f} degrees; retained1.5 is about2.2 degrees. These are diagnostic posture measurements, not natural-arm certification. The nearby CPU upper-body audit covers148 saved gates/906 cases at its own observation time, including head bands and shoulder/elbow positions and velocities. Historical upper-body progress is preserved. No new GUI capture was requested; completed full-CAD videos remain intact. [Upper audit](upper_audit.json).

The bounded16-robot50Hz contact audit compares the certified1.5 actor and222. Both actual sole clearance and forces are used for flight. New2.0 trace flight consists of sampled20–40ms events, with raw-output action clipping about18.4% in that command's trace. These bounded traces are timing diagnostics; the full224-robot independent gate remains the qualification evidence, and sampled event widths do not certify substep contact pressure.

The current full-population10Hz motor audit uses each physical pose's exact CAD differential Jacobian and the declared external reductions.64 directional finite-difference checks per differential agree within1e-5rad/s. At2.0, ankle-motor p99/cap ratios are approximately0.88 right and0.84 left. Some saved samples cross caps, with right-ankle crossings about0.764%; these do not identify positive driving-torque clipping, duration, a speed plateau or a persistent relevant ceiling. The earlier200Hz no-driving-clip probe concerns an older actor and does not certify this newer running actor. Retain native caps and perform actor-specific5ms attribution if high-speed progress plateaus. [Exact motor mapping](promoted_policy_audit.json).

Mass/inertia estimates and simplified collision envelopes remain the existing physical-v29 contract; this check does not certify the newer CAD release or real hardware. Stand-only support margins are not running-flight gates.

## Training health and sample accounting

The owned snapshot has {s['total_training_iterations']:,} completed-operation main updates plus58 preserved guard-recovery updates, totaling {s['total_training_iterations']+58:,} retained R37 updates, with7,361 inherited updates declared separately. This is {e['main_updates_since_previous_owned_check']:,} additional completed updates since the previous closing owned observation. The069 memory-recovery SHA and subsequent exact070 actor/critic/Adam/exploration load are reverified; capacity/smoke/diagnostic updates do not enter main counts. Every completed main block across1.5 and2.0 loads full state exactly and has zero maximum pre-update KL and action-command mismatch.

All-retained rates are **{rates['active_collection_PPO_samples_s']/1000:.2f}k active collection+PPO, {rates['main_worker_effective_samples_s']/1000:.2f}k main-worker-effective, {rates['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s**. The first denominator excludes startup/exit/gates; the second includes main-worker startup/exit; the third includes R37 capacity/gates, excluding both documented historical R37 diagnostic pauses totaling {rates['excluded_pause_seconds']:.1f}s. This check introduces no pause. These rates are not directly comparable with older-model or public Unitree collection-only benchmarks.

Current production is6,144 environments with8,192 retained as the target. Moving training freeGPU is around1.5–1.6GiB and host available around6GiB. The180-second upward ramp also needs enough memory for the added tier: approximately2,091MiB freeGPU for6,144→8,192, plus the300MiB reserve and host guards. Startup/evaluation GPU headroom cannot satisfy that training condition. The existing fallback and recovery preserve full learned state; no environment count was forced upward. [Per-block rates/memory/full loads](throughput_and_loads.json).

Two initial CPU gate-parser failures are preserved: a nonexistent lateral-statistic key, then an overstrict unique-checkpoint assumption when the same199 actor was evaluated in both1.5 and2.0 stages. The diagnostic now uses the actual statistic key and stage-specific promotion matching, and passes. No training artifact was altered to address either parser failure. [First failure](initial_gate_first_failure.log), [second failure](initial_gate_second_failure.log).

## Next focused checks

'''
for i,test in enumerate(d['next_tests'],1):report+=f'{i}. {test}\n\n'
report+='''Previously rejected timing advances, box-smoothed force feedback, running-row transfer,44% exposure and teacher-clipping-equivalence remain rejected absent new evidence. CAPG remains an undeployed research lead; the new independent running progress provides no reason to replace PPO now. The same adaptive three-hour heartbeat stays active. Endpoint remains verified actual5m/s with all matching gates, or a causally demonstrated relevant actuator velocity ceiling; save,pause,brief and leave the computer on. Preserve unrelated unsaved Fusion work.

The heartbeat update uses the native automation API and retains its existing thread/cadence/notification preference/durable scope; [official scheduled-task documentation](https://learn.chatgpt.com/docs/automations) was checked. No duplicate automation is created.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
entry=f'''\n\n<!-- {marker} -->
## {now} — independently qualified running1.5; improving2.0 and prior-teacher provenance audit

[Report]({(OUT/'REPORT.md').as_posix()}). Same R37 supervisor32492 and full-state learner continue unchanged. Exact199 SHA{q['sha256']} passes200/201 full nominal/pushed/stopped/prior gates: signed1.456/1.458m/s, true flight1.681/1.865%. The curriculum advances to2.0. Latest exact{Path(g['checkpoint']).parent.name} passes2.0 at{n['signed_speed']:.3f}/{p['signed_speed']:.3f}m/s, flight{100*n['flight']:.3f}/{100*p['flight']:.3f}%, all stopped/walking cases pass; retained1.5 flight{100*rn['flight']:.3f}/{100*rp['flight']:.3f}% still fails1.5% minimum. All224 survive20s;32 per command. Stage2 remains unqualified; no5m/s or ceiling. Head2.0 RMS{head[0]:.2f}/{head[1]:.2f}deg; natural arms uncertified.

New CPU exact-history replay11713 states each for199/222 agrees within1e-5action units. The old R34 head-context table is still applied after the prior teacher becomes neutral-head199. Conditioning creates self-teacher clipped lower-label RMS0.231–0.258 on certified199 states; truthful self-teacher error~1e-6. Current222 label changes0.236–0.279; student closer to conditioned labels. This identifies target/fixed-point mismatch, not physical causal repair;1.5 labels are unaffected. No actual PPO advantage capture, matched64/128 candidate or deployment is claimed. Preserve improving state; next test teacher-specific context provenance if retention stalls, with actual gradients and full-state matched physical/learning gates. Never spoof production inputs or remove the old original teacher's validated context indiscriminately.

Exact current CAD motor mapping uses full-population10Hz q/qd and64 finite-difference poses per pair; checks pass.2.0 ankle p99cap~0.88/0.84, sparse crossings do not prove driving clipping or a ceiling. Upper audit and inspected trend plot saved. CPU parser failures and repairs preserved; frozen runtime untouched.55hashes,one actual worker,{s['total_training_iterations']}completed+58recovered R37 updates,7361inherited separately; rates{rates['active_collection_PPO_samples_s']/1000:.2f}k active/{rates['main_worker_effective_samples_s']/1000:.2f}k worker/{rates['whole_chain_samples_s']/1000:.2f}k whole chain, historical679.7s pauses excluded. No pause/source/reward/geometry/cap/optimizer/actor reset this check.6144production,target8192,reserve300,host guards/180healthy-training-second ramp retained. Same3h automation remains active with fresh priorities. Videos fulfilled; unsaved Fusion preserved; save/pause/brief computer-on endpoint remains.
'''
for path in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    if '<!-- '+marker+' -->' not in path.read_text(encoding='utf-8-sig'):
        with path.open('a',encoding='utf-8') as f:f.write(entry)
brief=f'''<!-- {marker}_current -->
## Current adaptive check: {now}

Read [the latest report]({(OUT/'REPORT.md').as_posix()}) and fresh ACTIVE_RUN/state. Same R37 full-state learner has independently qualified running1.5 on199 with exact200/201 full gates and is now learning2.0. Current saved actor{Path(g['checkpoint']).parent.name} passes actual2.0 cases and every stopped/walking case, but fails1.5 geometric-flight retention; current2.0 stage remains unqualified. Preserve improving learner, neutral head and certified199. New CPU replay finds old R34 head-context labels remain active after teacher promotion to neutral199; this creates a reproduced self-teacher target mismatch for walking, not1.5 labels or proof of physical cause. Only test a teacher-specific context candidate with actual gradients/current full-state64/128 control and all gates if retention stalls; no candidate deployed. No runtime/reward/physics/cap/optimizer/actor reset or training pause this check.55hashes/one worker;6144production,target8192/reserve300/hostguards/180healthy ramp. New exact10Hz CAD motor audit does not establish a ceiling. Same3h active; endpoint save/pause/brief with computer on.

'''
for path in [HERE/'automation/THREE_HOUR_CHECK.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
    old=path.read_text(encoding='utf-8-sig')
    if '<!-- '+marker+'_current -->' not in old:path.write_text(brief+old,encoding='utf-8')
nextpath=HERE/'automation/NEXT_CHECK.json';old=read(nextpath)
if not (OUT/'next_check_before.json').exists():write(OUT/'next_check_before.json',old)
new=dict(recorded_utc=now,automation_id=old['automation_id'],active_run=str(run),report=str(OUT/'REPORT.md'),
    latest_observation=str(OUT/'closing_owned_observation.json'),new_main_updates=s['total_training_iterations'],recovered_main_updates=58,
    actual_retained_current_run_main_updates=s['total_training_iterations']+58,inherited_main_updates_preserved=7361,
    selected_actor=dict(checkpoint=g['checkpoint'],sha256=g['sha256'],matching_gate_pair=str(OUT/'closing_matching_gates.json'),
        full_actor_critic_Adam_exploration=True,standing_and_walking_cases_through_1_m_s_pass=all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in g[k] if c['command']<=1),
        current_stage_qualified=d['current_run_200_qualified']),qualified_run_150=str(OUT/'qualified_run_150.json'),
    highest_fully_qualified_stage='run_150',highest_fully_qualified_command_m_s=1.5,current_stage='run_200',
    running_qualified=d['current_run_200_qualified'],velocity_ceiling_proven=False,focused_tests=d['next_tests'],latest_repair_decision=d,
    throughput={k:v for k,v in rates.items() if k!='blocks'},throughput_evidence=str(OUT/'throughput_and_loads.json'),
    latest_verified_state=dict(recorded_utc=now,owned_snapshot_utc=e['owned_snapshot_utc'],status=s['status'],stage=s['stage'],operation=s['operation_path'],
        actual_workers=e['actual_workers'],frozen_hashes_verified=55,gpu=e['gpu'],host=e['host'],operation_environments=s['environment_count'],learning_environments=s.get('next_training_environment_count')),
    endpoint=old['endpoint'],recording_request=old['recording_request'],latest_gate_trends=str(OUT/'closing_gate_review.json'),promoted_policy_audit=str(OUT/'promoted_policy_audit.json'))
write(nextpath,new)
a=read(HERE/'ACTIVE_RUN.json');a.update(last_verified_utc=now,review_report=str(OUT/'REPORT.md'),final_owned_observation=str(OUT/'closing_owned_observation.json'));write(HERE/'ACTIVE_RUN.json',a)
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text(encoding='utf-8'))
assert cfg['id']==old['automation_id'] and cfg['kind']=='heartbeat' and cfg['rrule']=='FREQ=HOURLY;INTERVAL=3'
write(OUT/'automation_current_before_update.json',cfg)
prefix=cfg['prompt'].split('Latest dated focus',1)[0].rstrip()
focus=f'''Latest dated focus, check beginning October5 at23:47UTC, evidence {now}: R37_offline_ground_full_state is still the same supervisor32492 and full-state learner. IMPORTANT: the dated1.5 plateau is superseded. Exact199_run_150_train_064 SHA{q['sha256']} independently passed full200 nominal/201 pushed, stopped and every prior-speed gate. At1.5 its signed speed is1.456/1.458m/s and genuine geometric flight1.681/1.865%. Running1.5 is qualified for that exact actor. The current stage is run_200. Read diagnostics/adaptive_check_20261005_2347/REPORT.md, qualified_run_150.json, closing_matching_gates.json, promoted_policy_audit.json, throughput_and_loads.json and fresh ACTIVE_RUN/state.

Current saved actor{Path(g['checkpoint']).parent.name} has matching224-robot20s nominal/pushed tests,32 trials per command and100% survival. Stopped and walking0.15/0.35/0.65/1.0 pass both. The2.0 case also passes all speed/direction/gait/flight checks, actual signed {n['signed_speed']:.3f}/{p['signed_speed']:.3f}m/s and true flight {100*n['flight']:.4f}%/{100*p['flight']:.4f}%. Retained1.5 reaches {rn['signed_speed']:.3f}/{rp['signed_speed']:.3f}m/s but has only {100*rn['flight']:.4f}%/{100*rp['flight']:.4f}% geometric flight, below1.5%, so the full2.0 stage remains unqualified. This retention is improved from near-zero early2.0 blocks but nonmonotonic. Preserve improving learner and certified199; do not restart merely because an older handoff said stalled. Head2.0 RMS{head[0]:.2f}/{head[1]:.2f}deg; natural arms uncertified. Qualification binds exact saved actors, not newer partial updates. No5m/s or motor-ceiling endpoint proven.

New CPU exact-input reconstruction reproduces11713 saved actor samples for each199 and222 within1e-5action units. Teacher promotion changed the qualified prior to neutral-head199, but the old R34 phase-conditioned head table is still applied for walking commands. It creates artificial self-teacher clipped lower-label RMS0.231-0.258 on199's qualifying truthful states versus~1e-6 without the old context; on222 truthful states it changes labels0.236-0.279RMS and the student is closer to conditioned labels. The table applies to original walking commands only;1.5 targets are unchanged. This reproduces a label/fixed-point mismatch, not proof of physical cause or benefit from disabling conditioning. No actual noisy PPO advantage/gradient audit, physical candidate or64/128 comparison was performed this check. No context change is deployed. Production observations remain truthful, and the original tilted-head teacher's physically validated context must be retained when that original teacher is used.

New full-population10Hz actual CAD Jacobian/reduction motor mapping passes64 directional finite-difference checks per pair. At2.0 ankle p99cap ratios are about0.88 right/0.84 left with sparse sampled cap crossings; this is not driving-torque clipping, a sustained relevant ceiling or substep evidence. Bounded50Hz traces include real20-40ms sampled flight events; these traces are not full-population qualification or contact-pressure certification. Earlier200Hz no-driving-clip evidence is actor-specific. Native limits stay unchanged.

No production source, reward, physical geometry, cap, optimizer, actor or exploration reset/change and no training pause this check. Original local ground, rewardv8/verifierv6, teachers, head module, truthful400-input/23-output history, PPO and iteration4 remain. All55 authored hashes verify and one actual worker is owned. Saved main count {s['total_training_iterations']} completed plus58 proven retained recovery updates,7,361 inherited declared separately. All-retained rates {rates['active_collection_PPO_samples_s']/1000:.2f}k active collection+PPO/{rates['main_worker_effective_samples_s']/1000:.2f}k worker-effective/{rates['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s, excluding both historical R37 pauses totaling679.7s; no new pause. Current production6144, target8192/reserve300/hostguards/180healthyTRAINING-second ramp remain. About2091MiB freeGPU is needed for the added tier, not merely300; startup/eval memory cannot qualify. Full-body videos already fulfilled; preserve unsaved Fusion work.

Next focused tests:
1. Compare fresh complete nominal/pushed stopped, all prior walking/running and2.0 gates. Preserve current progress; only promote after every same-actor case passes. If retained1.5 flight continues improving, finish matching verification without reward or estimator redesign. Monitor2.0 flight/direction/head together and inspect the2.5 frontier only after qualification. Do not assign199's qualification to newer failing actors.
2. If prior retention stalls, prioritize teacher-specific head-context provenance and actual per-command advantages/teacher-PPO gradients. Test a narrow context binding that preserves the original R34 teacher's validated labels and uses truthful labels for newer certified neutral-head teachers. Require original-teacher parity, current full-state actor/critic/Adam/exploration continuation, physical/PPO/load checks, and unchanged-control versus candidate64/128 learning with every prior/current/head/direction/flight gate before deployment. The old table leaves1.5 labels untouched; do not assert direct causation. CPU sensitivity alone does not authorize replacing the actor. Keep CAPG undeployed and lower priority while genuine running progresses; no arbitrary weights, pulses, hybrids, observation spoofs, posture rollback or raised caps.
3. Verify one worker,55 hashes, native caps, exact full-state fallback,6144/8192 memory, actual retained samples/time and180continuous healthy-training seconds. Count58 recovery updates separately and exclude only the documented679.7s R37 pauses. If high-speed progress plateaus, add actor-specific5ms positive-driving clip/torque attribution and controlled evidence; sampled spikes are not a ceiling. Rejected timing advances, box smoothing, running-row transfer,44% exposure and teacher-clipping-equivalence stay rejected absent new evidence.

Keep this same adaptive three-hour heartbeat active until verified actual5m/s with every gate or a causally established persistent relevant actuator velocity ceiling. At that endpoint save exact evidence, pause owned training and this automation, brief the user and leave the computer on. Read fresh ACTIVE_RUN/state rather than this dated observation.'''
args=dict(mode='update',id=cfg['id'],kind=cfg['kind'],name=cfg['name'],status='ACTIVE',rrule=cfg['rrule'],targetThreadId=cfg['target_thread_id'],prompt=prefix+'\n\n'+focus)
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
write(OUT/'automation_update_args.json',args)
write(OUT/'diagnostic_source_hashes.json',{p.name:sha(p) for p in OUT.glob('*.py')})
print(json.dumps(dict(report=str(OUT/'REPORT.md'),same_heartbeat=cfg['id'],prepared_update=True,latest_actor=Path(g['checkpoint']).parent.name)))
