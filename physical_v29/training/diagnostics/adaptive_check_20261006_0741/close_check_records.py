"""Close this check using saved exact gates, without touching simulator sources."""
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAINING = OUT.parents[1]
WORKSPACE = TRAINING.parents[1]
AUTO = Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def prepare():
    main = read(OUT/'new_main_evidence.json')
    live = read(OUT/'closing_cpu/live_handoff.json')
    review = read(OUT/'final_gate_review.json')
    upper = read(OUT/'final_upper/upper_joint_time_audit.json')
    timing = read(OUT/'continuation_timing.json')
    state = live['state']
    gate = review['latest']
    auto = tomllib.loads(AUTO.read_text(encoding='utf-8'))
    assert main['passed'] and main['new_completed_main_updates'] == 128
    assert len(live['actual_worker_pids']) == 1
    assert len(live['frozen_source_integrity']) == 55 and all(live['frozen_source_integrity'].values())
    assert state['status'] == 'running' and state['stage'] == 'run_350'
    assert Path(gate['checkpoint']).parent.name == '014_run_350_train_002'
    assert upper['metric_fixtures_passed']
    assert auto['kind'] == 'heartbeat' and auto['status'] == 'ACTIVE' and auto['rrule'] == 'FREQ=HOURLY;INTERVAL=5'
    rows = [x for x in upper['records'] if x['actor_tag'] == 'latest' and x['test'] == 'nominal']
    fast = next(x for x in rows if x['command_m_s'] == 3.5)
    nom = {x['command']:x for x in gate['nominal_cases']}
    push = {x['command']:x for x in gate['pushed_cases']}
    assert all(x['trials'] == 32 and x['survival'] == 1 for x in gate['nominal_cases'] + gate['pushed_cases'])
    cases = [{k:x[k] for k in ['command','passed','failures','trials','survival','signed_speed','lateral_speed','flight']} for x in gate['nominal_cases']]
    cases_push = [{k:x[k] for k in ['command','passed','failures','trials','survival','signed_speed','lateral_speed','flight']} for x in gate['pushed_cases']]
    partial = sum(x['observed_updates'] for x in main['partial_main_blocks'])
    now = main['recorded_utc']
    focus = f'''Latest dated five-hour check, evidence {now}: R39_commanded_travel_teacher_certificate is the exact full-state continuation of original R37 checkpoint459, SHA75b0a3ea651e5bb9872404b9df8ac0686eb38a24f04e3cf639ad90425ad92a08. Actor, critic, Adam and trained exploration were equal on the first genuine block; 16936 earlier actual main updates are preserved, including58 verified recovered. R39 has128 completed new main updates, plus{partial} observed partial updates at8192 in block017. Read diagnostics/adaptive_check_20261006_0741/REPORT.md, final_gate_review.json, final_upper/upper_joint_time_audit.json and its angle/velocity plots, new_main_evidence.json, continuation_timing.json and closing_cpu/live_handoff.json, then fresh ACTIVE_RUN/state. One actual worker and55 frozen hashes verify. Never attribute saved gates to newer partial actors.

R37 independently qualified3.0 at349, matching350/351, with signed2.9979/2.9982m/s and every stopped/prior nominal+pushed gate. Later3.5 zero survival reproduced an unconditional60m total-displacement termination:24 sampled events across12 pairs were ONLY excessive_drift near18s while upright. The proved correction accumulates actual planar action-command travel across switches and keeps60m excess margin beyond that budget. Selected resets clear budget and updates are idempotent per tick. Other physical fault predicates and all independent classifiers remain unchanged. Numerical rewards/weights are unchanged, but corrected done/terminal-penalty/return semantics after the old boundary intentionally differ. Read travel_guard_attribution.json, physical_validation.json, physical_parity.json, termination_continuation_proof_retry.json and migration_cpu_tests_retry.json.79 guard CPU fixtures,15 rejection fixtures,117 bitwise physical-array comparisons and real PPO/full-load/reload pass. Strict migration binds only original459 and trusted349, literal inverse patches and protected hashes; no generic signature bypass.51 other runtime files, CAD/mass/contact/native caps,400-input23-output inference, teachers/head module, PPO and iteration4 are identical. Fresh trusted349 nominal/pushed all-prior requalification passed in R39.

The preserved R38 startup passed physical/load/teacher tests but failed certificate filename lookup before any main update. R39 fixes only the supervisor gate label and repeats fresh startup/capacity/requalification. Failed signature/teacher-signature attempts and R38 logs are preserved. The original supervisor was retired after healthy R38 startup, before its later bookkeeping failure; corrected-source R39 then started. Saved-boundary-to-R38-supervisor gap1029.1s, retirement1230.1s, to-R39-supervisor{timing['saved_boundary_to_R39_supervisor_seconds']:.1f}s, to-first-reported-genuine-R39-PPO{timing['saved_boundary_to_first_reported_R39_PPO_seconds']:.1f}s are separate. No silent training pause remains or unsaved user work was touched.

Post64 actor011 nominal012/pushed013 all survive but3.5 lateral worsens to0.481/0.444m/s; retained0.15 net travel,1.5 flight and2.5 hopping fail. New exact128-update actor014 matching015/016 still all survives:3.5 signed{nom[3.5]['signed_speed']:.4f}/{push[3.5]['signed_speed']:.4f}m/s, lateral{nom[3.5]['lateral_speed']:.4f}/{push[3.5]['lateral_speed']:.4f}, true flight{100*nom[3.5]['flight']:.3f}/{100*push[3.5]['flight']:.3f}%. Both3.5 tests fail lateral/world-lateral. Both retained0.15 fail net travel at{nom[.15]['signed_speed']:.3f}/{push[.15]['signed_speed']:.3f}; nominal1.0 fails walking_support; both retained3.0 fail world lateral. Stopped,0.35,0.65,1.5,2.0,2.5 pass both;1.5 genuine flight recovers to{100*nom[1.5]['flight']:.3f}/{100*push[1.5]['flight']:.3f}%. Retention is nonmonotonic and direction remains unresolved. This does not qualify3.5 or establish5/4 or a ceiling. Preserve separately certified349 and current full state.

Mandatory upper angle/target/velocity time-series were inspected at baseline459,64-update011 and128-update014, plus fresh349:32 robots per command at10Hz,16 representative robots per gate at50Hz; resets/terminal rows segmented. final_upper contains latest PNG/SVG/CSV/JSON and both plots were visually inspected. At014/3.5 headRMS{fast['head_rms_deg']:.3f}deg; shoulder pitch ranges{fast['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{fast['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}deg and ipsilateral phases{fast['coordination']['arms'][0]['phase_at_hip_cycle_deg']:.1f}/{fast['coordination']['arms'][1]['phase_at_hip_cycle_deg']:.1f}deg. Small fast pitch swing, unequal shoulder bias, roll motion and gait-linked head oscillation remain. Physical upper target/joint RMS{fast['representative_50hz']['target_joint_rms_error_deg']:.2f}deg; selected raw upper outputs outside clamp0%. Native head zero is torso-relative, not world gaze. No natural-arm/human/Unitree certification: primary human comparison is feature-based and morphology-limited; mapped speed-matched official Unitree encoder curves were not obtained.10Hz population sampling cannot certify fast jitter.

Two completed8192 main blocks have exact full-state loads, zero pre-updateKL/command mismatches and stable memory; rates{main['rates']['completed_main_active_collection_PPO_samples_s']/1000:.2f}k active collection+PPO/{main['rates']['completed_main_worker_effective_samples_s']/1000:.2f}k worker-effective/{main['rates']['completed_main_whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. Whole-chain includes new startup/capacity/fresh gates; discarded capacity actors never count as learned samples. Main minimumGPU402MiB, host about5GiB; target8192/reserve300/host fallback/180continuous healthy-training-second ramps unchanged. Baseline all320-robot10Hz CAD motor audit has sparse ankle cap crossings, not sustained driving saturation; left ankle motor2 p99~0.803cap, at/above-cap0.45%. No fresh5ms driving/torque attribution or proven high-speed limit.

Next one to three focused tests: (1) Direction and retained0.15 net travel,1.0 support and3.0 lateral are now primary. Compare further exact nominal/pushed pairs against baseline459,011 and014; retain improving actors and certified349. Audit per-command current/prior teacher coverage/targets and actual objective/PPO gradients versus heading/lateral response. If failures persist, run controlled normal-limit heading/lateral interventions and a narrow full-state matched-control continuation before deployment; do not repeat reward-identical waiting or change arbitrary weights. The repaired travel guard is not solved gait. (2) Required current head/shoulder pitch+roll/elbow angle-target-velocity, bilateral amplitude/coordination and arm-leg phase; measure effective upper targets, physical response and auxiliary/PPO gradients before refining. Preserve neutral head, torso/legs/full learner; no observation spoof, hybrid or from-scratch policy. Teacher-context provenance hypothesis stays undeployed without evidence. (3) Verify one worker,55 hashes, local ground, exact full-state recovery and retained samples, moving6144/8192 reserves and180healthy-training-second ramps. If a real high-speed plateau appears, measure native5ms speed/available torque and sustained DRIVING clipping through CAD reductions/Jacobians before declaring4m fallback. All caps/gates and rejected candidates remain. Continue this same FIVE-hour heartbeat until verified5m/s or verified4m/s with proven native limits blocking5; save,pause training+automation,brief,computer on.'''
    prefix = auto['prompt'].split('Latest dated five-hour check')[0]
    assert 'Every check MUST analyze upper-body joint angles' in prefix
    assert 'at least 3.96' in prefix and 'at least 4.95' in prefix
    payload = dict(mode='update',id=auto['id'],kind='heartbeat',name=auto['name'],status='ACTIVE',rrule=auto['rrule'],targetThreadId=auto['target_thread_id'],prompt=prefix+focus)
    if 'notification_policy' in auto: payload['notificationPolicy'] = auto['notification_policy']
    write(OUT/'automation_payload.json',payload)
    (OUT/'automation_focus.txt').write_text(focus,encoding='utf-8')
    write(OUT/'closing_verification.json',dict(recorded_utc=now,completed_main=128,partial_main=partial,checkpoint=gate['checkpoint'],sha256=gate['sha256'],nominal=gate['nominal'],pushed=gate['pushed'],cases=cases,pushed_cases=cases_push,upper=fast,plots_inspected=True,run_350_qualified=False,endpoint_qualified=False))
    print(json.dumps({'prepared':True,'main':128,'partial':partial,'nominal':gate['nominal'],'pushed':gate['pushed']}))

def finish():
    payload=read(OUT/'automation_payload.json')
    auto=tomllib.loads(AUTO.read_text())
    before=tomllib.loads((OUT/'automation_before.toml').read_text())
    for a,b in [('id','id'),('kind','kind'),('name','name'),('rrule','rrule'),('status','status'),('target_thread_id','targetThreadId')]: assert auto[a]==payload[b]
    assert auto['prompt'].rstrip('\n')==payload['prompt'].rstrip('\n')
    assert auto.get('notification_policy')==before.get('notification_policy')
    (OUT/'automation_after.toml').write_bytes(AUTO.read_bytes())
    write(OUT/'automation_verification.json',dict(passed=True,recorded_utc=datetime.now(timezone.utc).isoformat(),same_id_thread_kind_notification=True,five_hour_cadence_verified=True,status='ACTIVE',durable_scope_endpoint_upper_motion_preserved=True,prompt_sha256=hashlib.sha256(auto['prompt'].encode()).hexdigest()))
    close=read(OUT/'closing_verification.json');main=read(OUT/'new_main_evidence.json');live=read(OUT/'closing_cpu/live_handoff.json')
    gate=read(OUT/'final_gate_review.json')['latest']; now=close['recorded_utc'];state=live['state']
    table='| Command m/s | Nominal signed | Pushed signed | Nominal failure | Pushed failure |\n|---:|---:|---:|---|---|\n'
    for n,p in zip(close['cases'],close['pushed_cases']):
        assert n['command']==p['command']
        table+=f"| {n['command']:g} | {n['signed_speed']:.3f} | {p['signed_speed']:.3f} | {', '.join(n['failures']) or 'Pass'} | {', '.join(p['failures']) or 'Pass'} |\n"
    addendum=f'''\n\n## Closing verification — {now}\n\nThe saved actor014 after128 new main updates is now the latest reviewed actor, with exact matching015 nominal/016 pushed tests. All320 robots survive20s,32 trials per command. The repaired termination removes a false failure; direction and retention remain unresolved.3.5 signed speed is3.575/3.582m/s, but lateral motion still fails. Prior1.5 flight and2.5 hopping recover;0.15 net travel, nominal1.0 support and both3.0 lateral fail. Qualification is not inferred from speed or survival.\n\n{table}\n\nThe first64-update results remain in `closing_gate_review.json` and `closing_upper`. The newest128-update pair and mandatory per-speed actual head/arm angle-target-velocity audit are in `final_gate_review.json` and `final_upper`. Both latest plots were visually inspected: asymmetric shoulder pitch bias, small fast pitch swing, larger roll motion, bent elbows and gait-linked head oscillation remain alongside alternating hips. HeadRMS at3.5 is{close['upper']['head_rms_deg']:.3f}deg; this does not certify human-like arms or world gaze. No comparable-speed mapped Unitree encoder reference was obtained.\n\nLatest liveness: sameR39 supervisor35688, one actual worker,55 hashes verified; block017 trains at8192.128 completed plus{close['partial_main']} observed partial updates are separate. Two completed blocks load exact actor/critic/Adam/exploration, with zero pre-updateKL/action-command mismatch. Rates49.05k active/42.65k worker-effective/16.86k whole-chain samples/s; startup/capacity/fresh gates lower the separate whole-chain rate. Minimum main GPU402MiB, host roughly5GiB; memory stable. Target8192/reserve300/host guards/fallback/180healthy-training-second ramp retained.\n\nTimeline correction to the earlier handoff: R37 was retired after healthy **R38** startup, before R38's later pre-main certificate filename failure. R39 then repeated startup and teacher checks. Saved-boundary-to-R39 supervisor1742.3s and first-reported genuine PPO2347.5s include diagnosis and repeated startup/capacity. There is no current silent pause. The CPU final-review command initially duplicated its relative output root; retry with absolute paths succeeds. This reporting failure changed no simulator source or training state.\n\nThe final same-thread native automation update verifies ACTIVE/FIVE-hour cadence and identical notification preference/durable endpoint. Next focus is reproduced lateral direction and prior retention, then upper target/phase diagnostics and native substep attribution if needed. The current full learner remains running; neither5m/s nor the conditional4m/s finish is proven.\n'''
    report=OUT/'REPORT.md';text=report.read_text()
    text=text.replace('then retired after healthy R39 source freeze/physics/full-load/reload.','then retired after healthy R38 source freeze/physics/full-load/reload, before its later pre-main certificate lookup failure. R39 subsequently repeated startup and requalification.')
    text=text.replace('At the closing observation, new completed main updates=0,','At the earlier 08:29 observation, new completed main updates=0,')
    text=text.replace('Closing main-learning accounting: 0 completed updates plus 18 observed current partial PPO updates.','Earlier 08:29 main-learning accounting: 0 completed updates plus 18 observed current partial PPO updates.')
    assert '## Closing verification —' not in text
    report.write_text(text+addendum,encoding='utf-8')
    nxt=read(TRAINING/'automation/NEXT_CHECK.json')
    nxt.update(recorded_utc=now,new_main_updates=128,actual_retained_current_run_main_updates=128,observed_partial_main_updates=close['partial_main'],latest_verified_state=state,throughput=main,latest_gate_trends=str(OUT/'final_gate_review.json'),latest_closing_gate_pair=str(OUT/'final_gate_review.json'),latest_schedule_verification=str(OUT/'automation_verification.json'),latest_observation=str(OUT/'closing_cpu/live_handoff.json'))
    nxt['selected_actor']=dict(checkpoint=gate['checkpoint'],sha256=gate['sha256'],matching_gate_pair=str(OUT/'final_gate_review.json'),current_stage_qualified=False,scope='Exact128-update actor014, nominal015/pushed016. Current partial actor not qualified.')
    nxt['focused_tests']=[
        'Prioritize current3.5 lateral and retained0.15 net travel,1.0 nominal support,3.0 lateral. Compare further exact pairs to459,011,014; preserve certified349 and full current learner. Audit conditioned-prior coverage/targets and real PPO/objective gradients; if persistent, controlled heading/lateral response then narrow matched-control full-state continuation. No arbitrary weights or reward-identical waiting.',
        'Required per-speed reset-segmented head/shoulder pitch+roll/elbow angle-target-velocity and bilateral/arm-leg phase. Latest128-update plots still show asymmetric shoulder bias/small fast pitch swing/gait-linked head yaw. Measure effective upper targets/physical response/auxiliary-PPO gradients before refinement; preserve head and legs, no reset/spoof/hybrid.',
        'Verify55 hashes,one worker,local ground,full-state recovery/sample accounting,6144/8192/reserve300/host guards/180healthy-training-second ramps. Native5ms speed/torque sustained driving attribution required for conditional4m fallback; sparse cap crossings do not prove ceiling.']
    nxt['upper_body_analysis'].update(latest_json=str(OUT/'final_upper/upper_joint_time_audit.json'),angle_plot=str(OUT/'final_upper/upper_joint_angles.png'),velocity_plot=str(OUT/'final_upper/upper_joint_velocities.png'),csv=str(OUT/'final_upper/upper_joint_angles.csv'))
    write(TRAINING/'automation/NEXT_CHECK.json',nxt)
    note=f'''\n\n## {now} — Closing R39 verification after128 main updates\n\nRead {report} closing addendum and {OUT/'final_gate_review.json'}. Exact actor014 nominal015/pushed016 survives320/320 but3.5 fails lateral; earlier0.15 net travel,nominal1.0 support,both3.0 lateral fail.1.5 flight and2.5 recover,so progress is nonmonotonic. Same full learner continues8192;128completed plus{close['partial_main']} observed partial,55 hashes,oneworker,zero KL/command mismatch,stable memory. Latest actual upper angles/targets/velocities inspected in final_upper; asymmetric fast arms and gait-linked head motion remain,not human-like certification. Timeline correction: original supervisor retired afterR38 healthy startup,notR39; R38 subsequently failed before any main learning,thenR39 repeated startup. Gaps1742.3s toR39supervisor/2347.5s tofirstreported PPO documented. Native automation verifiedsameID/thread/notifications,ACTIVE5h.5/conditional4 endpoints unproven;computeron. No further runtime or policy reset since deployed guard repair.\n'''
    for p in [TRAINING/'ITERATION_HISTORY.md',WORKSPACE/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    handoff=f'''<!-- closing_adaptive_check_20261006_0741 -->\nClosing handoff {now}: R39 same full learner has128 genuine main updates plus{close['partial_main']} observed partial at8192. Read {report} closing addendum, final_gate_review.json, final_upper angle/velocity plots, NEXT_CHECK and freshACTIVE_RUN/state. Actor0143.5direction and0.15/1.0/3.0retention failures remain; all320survive. Separately certified349through3 is preserved.55hashes/oneworker; ACTIVE5h/verified5 orconditionalverified4+native-limit endpoint. Earlier observation counts below are historical.\n<!-- /closing_adaptive_check_20261006_0741 -->\n\n'''
    for p in [Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
        p.write_text(handoff+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps({'saved':True,'automation_verified':True,'report':str(report),'upper':str(OUT/'final_upper/upper_joint_angles.png')}))

if __name__=='__main__':
    import sys
    prepare() if sys.argv[1]=='prepare' else finish()
