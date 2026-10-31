"""Publish this check's measured decision, journals and adaptive handoff."""
import json, hashlib, tomllib
from pathlib import Path
from datetime import datetime, timezone

OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
utc=datetime.now(timezone.utc).isoformat()
def prose(text):
    # Only named prose tokens; never alter paths, hashes or the durable prompt.
    replacements={
        'All55':'All 55','All53':'All 53','all53':'all 53','all55':'all 55',
        'its18-hour':'its 18-hour','an18-hour':'an 18-hour','The600':'The 600',
        'preserves27507':'preserves 27507','preserving27507':'preserving 27507',
        'retained10538':'retained 10538','plus33':'plus 33','with16936':'with 16936',
        'stiffness70':'stiffness 70','stiffnesses35':'stiffnesses 35','damping1':'damping 1',
        'gains35/1':'gains 35/1','original35/1':'original 35/1','gain70':'gain 70',
        'Both3.5':'Both 3.5','both3.5':'both 3.5','current3.5':'current 3.5','exact3.5':'exact 3.5',
        'introduces1.5':'introduces 1.5','has1.5':'has 1.5','nominal2.5':'nominal 2.5',
        'by128':'by 128','at128':'at 128','at64':'at 64','branch64':'branch 64',
        'Candidate64':'Candidate 64','Candidate128':'Candidate 128','candidate128':'candidate 128',
        'control128':'control 128','control2.448':'control 2.448','has0.795':'has 0.795',
        'below1.5':'below 1.5','control retains2.448':'control retains 2.448',
        'exact498':'exact 498','original498':'original 498','baseline498':'baseline 498',
        'operation501':'operation 501','interrupted501':'interrupted 501','interrupted healthy501':'interrupted healthy 501',
        'saved498':'saved 498','exact499/500':'exact 499/500','full498':'full 498',
        'original267':'original 267','trusted349':'trusted 349','certified349':'certified 349',
        '349certified3':'349 certified 3','267all-prior':'267 all-prior','20batches':'20 batches',
        'upper20':'upper 20','both teacher / upper20':'both teacher/upper 20',
        '320-robot20-second':'320-robot, 20-second','320-robot20s':'320-robot, 20-second',
        '4096-env64/128':'4096-env, 64/128','4096 environments':'4096 environments',
        '64-env4-update':'64-env, 4-update','real4-update64-env':'real 4-update, 64-env',
        'real32-env':'real 32-env','Real32-env':'Real 32-env','with53':'with 53',
        '13CPU':'13 CPU','14CPU':'14 CPU','14CPU/real':'14 CPU / real',
        'rewardv8':'reward v8','verifierv6':'verifier v6','iteration4':'iteration 4',
        'at10Hz':'at 10 Hz','at50Hz':'at 50 Hz','representative50Hz':'representative 50 Hz',
        '32robots':'32 robots','16representative':'16 representative','from16':'from 16',
        'no3.5':'no 3.5','No3.5':'No 3.5','No4/5':'No 4/5','no4/5':'no 4/5',
        'does not establish32':'does not establish 32','not32':'not 32',
        'seed1901':'seed 1901','mainfirst':'main first','Firstmain':'First main',
        'Read newer':'Read newer','fullstate':'full state','fulllearner':'full learner',
        'oneactualworker':'one actual worker','55hashes':'55 hashes','oneworker':'one worker',
        'allpreupdateKL':'all pre-update KL','commandmismatch0':'command mismatch 0',
        'target8192':'target 8192','reserve300':'reserve 300','max_hours0':'max_hours=0',
        'explicitmax_hours':'explicit max_hours','explicitgoal':'explicit goal',
        '600s':'600 s','180healthy':'180 healthy','nominalclone':'nominal clone',
        'sameactive5h':'same active 5-hour','Sameactive5h':'Same active 5-hour',
        'same ACTIVE5h':'same ACTIVE 5-hour','save,pauseboth,brief,computeron':'save, pause both, brief, computer on',
        'unsavedFusion':'unsaved Fusion','full-state/teacher/context':'full-state, teacher and context',
        'no pendingvideo':'no pending video','gates bindexactactor':'gates bind the exact actor',
        'newerpartialupdates':'newer partial updates','newstartup':'new startup','Currentfailures':'Current failures',
    }
    for before,after in replacements.items():text=text.replace(before,after)
    return text
live=read(OUT/'closing_cpu/live_handoff.json');state=live['state'];run=Path(live['active_run']['run'])
proof=read(OUT/'resumed_learning_verification.json');gates=read(OUT/'closing_gate_review.json')
old=read(OUT/'initial_gate_review.json');gain=read(OUT/'matched_gain_review.json');decision=read(OUT/'decision.json')
accounting=read(OUT/'throughput_and_loads.json');parity=read(OUT/'baseline_physical_parity.json')
assert proof['completed_main_updates']>=64 and proof['blocks'] and parity['all_equal']
assert all(live['frozen_source_integrity'].values()) and len(live['actual_worker_pids'])==1
latest=gates['latest'];current={label:latest[label+'_cases'][-1] for label in ['nominal','pushed']}
assert Path(latest['checkpoint']).resolve()==(Path(proof['blocks'][-1]['folder'])/'candidate.pt').resolve(), 'Wait for both post-learning gates'
prior_fail={label:[dict(command=c['command'],failures=c['failures']) for c in latest[label+'_cases'][:-1] if not c['passed']] for label in ['nominal','pushed']}
focus=[
    'Compare fresh exact3.5 direction and all stopped/prior gates against original498 and the first recovered R40 pair; preserve independently improving actors,267 all-prior and separately certified349. Diagnose recurring 0.15 net-travel and newly failed 2.5 hopping/lateral gates, alongside current 3.5 drift using actual per-command advantages, teacher coverage/effective targets and heading/yaw response. Review exact 2.5 startup landing events and prior/PPO gradient contributions without relaxing the no-hopping gate. The inherited old walking-head context on a newer neutral-head teacher is a measured target-provenance hypothesis, not a proven cause. If retention/direction still stalls, test one narrowly versioned training-only teacher/gradient or direction continuation on the CURRENT full state against unchanged64/128 controls and independently seeded replication before deployment. The prior fixed-frame kernel remains rejected for paired2.5 startup touchdowns; do not blindly repeat it, relax gates or change arbitrary weights.',
    'Every check inspect actual head/shoulder pitch+roll/elbow angle-target-velocity timelines by speed, bilateral amplitude, arm/leg phase, slow bias drift versus cycle motion, smoothness and saturation. The completed stiffness70 versus35/damping1 exact-full-state4096-env64/128 test is REJECTED: early drift reduction fades, both3.5 direction gates still fail, and1.5 geometric flight falls below 1.5% in both tests while control passes. Larger slow swings and persistent fast bias/asymmetry are not natural-motion certification. Preserve existing upper20-batch refinement and truthful inference/head/gait/full learner. Before another gain or posture change, identify effective target/auxiliary/PPO and physical coupling tradeoffs; require matched full-state controls/all gates. Isaac-head/FK parity is known but CAD face/camera optical axis remains uncalibrated; no fabricated speed-matched Unitree curves, spoof, hybrid or tilted-head rollback.',
    'Verify R40 goal-budget full-state recovery, exact actor/critic/Adam/exploration and both teachers, local ground, one actual worker and55 hashes. It inherits27507 verified actual main updates, including33 recovered R39 updates; interrupted unsaved32 updates are excluded. Explicitmax_hours0 removes the expired18h stop, retaining600s no-progress and all memory/native-limit guards; finite budgets finish at saved boundaries. Keep8192 target,300MiB reserve,host fallback and180 continuous healthy TRAINING-second upward ramp. Compare retained main samples/time separately from capacity/smoke/diagnostic and downtime. Establish32 independently seeded nominal-physics initial states per command before final5 or conditional4 endpoint; repeated nominal clones and isolated cap spikes are insufficient.'
]
focus=[prose(x) for x in focus]
decision['next_focused_tests']=focus;decision['latest_matching_pair']=latest
(OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8')
table=[]
for row in gain['rows']:
    n=row['nominal_cases'][-1];p=row['pushed_cases'][-1]
    f=[next(c for c in row[label+'_cases'] if c['command']==1.5)['flight']*100 for label in ['nominal','pushed']]
    table.append(f"| {row['branch']} {row['updates']} | {n['lateral_speed']:.3f} / {p['lateral_speed']:.3f} | {f[0]:.3f}% / {f[1]:.3f}% |")
upper=[]
for branch in ['initial','control','candidate','final']:
    path=OUT/(branch+'_upper')/'upper_joint_time_audit.json'
    if not path.exists():continue
    data=read(path)
    for r in data['records']:
        if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s'] in [1.,3.5]:
            a=[r['joints'][s+'_shoulder_pitch']['median_per_robot_p95_p05_range_deg'] for s in ['left','right']]
            upper.append(f"| {branch} | {r['command_m_s']:.1f} | {a[0]:.2f} / {a[1]:.2f} | {r['head_rms_deg']:.2f} |")
blocks=proof['blocks'];last=blocks[-1]
failure=[f"{label}: "+', '.join(f"{c['command']:g} ({'/'.join(c['failures'])})" for c in prior_fail[label]) for label in ['nominal','pushed']]
report=f"""# Five-hour training check — October 7, 03:44 UTC

Observed {utc}. **R40 is a full-state recovery of R39, with original shoulder gains.** The shoulder stiffness experiment is rejected. Current 3.5 m/s direction and some prior-skill gates remain incomplete; the separately certified 3 m/s actor remains preserved. No 4/5 m/s or actuator-limit endpoint is established.

## What stopped the previous run

R39's 18-hour wall-time guard interrupted healthy operation501 before this heartbeat. The failure is `CHAIN_TIME_LIMIT`, not qualified completion, physical instability or a motor ceiling. All55 original frozen hashes and the latest saved498 full checkpoint remain intact: SHA `c04ae2f26310d5514905ce1a2fec9159594ec8ce8d5a19f67910291f497e2ae2`. Its exact499/500 pair survives completely but fails3.5 world lateral speed at signed3.414/3.420 m/s and lateral0.582/0.555 m/s. Preserve original267's all-prior pair and separately trusted349's3 m/s certification.

Thirty-two genuine updates in interrupted501 have no saved learned checkpoint; they are excluded. R39 retained10538 completed-operation updates plus33 hash-proven recovered updates, with16936 inherited. R40 therefore preserves **27507 actual prior main updates**. At failure retained rates were44.77k active collection+PPO,38.81k worker-effective and26.21k whole-chain samples/s, excluding3590.8 seconds of documented pauses. Stopped downtime before this check was5160.8 seconds. The current diagnostic interval and recovery startup are separately recorded; discarded capacity, smoke and test actors never count as main progress. See [accounting](throughput_and_loads.json).

## Tested recovery maintenance

Only the supervisor changes. Explicit `max_hours=0` runs to the authorized goal instead of an18-hour cutoff. Finite budgets now finish the active worker at a saved boundary. The600-second no-progress guard, GPU/host guards, fallback,300MiB reserve,8192 target and180 continuous healthy training-second ramp remain. Strict resume restores full actor, critic, Adam, trained exploration, standing teacher, qualified moving teacher and the narrowly validated travel-guard bridge. All53 authored worker/physics/PPO/teacher files must match. Native gains35/1, CAD geometry/mass/contact/motor caps, rewardv8, verifierv6,400-input23-output inference, truthful history and iteration4 are unchanged.

Fourteen CPU tests pass, including real Chain snapshot construction, exact teacher/recovery context, rejection of changed or live parents and a standard-library mock worker demonstrating that an expired finite budget no longer interrupts fresh progress. The CPU mock is not a policy test. Real32-env physical preflight,64-env4-update full-state PPO and checkpoint reload pass; every smoke pre-update KL and command mismatch is zero, both teacher gradients and upper refinement are present. Smoke actors are discarded. The entire320-robot20-second same-actor nominal/pushed baseline repeats all saved arrays and cases **bit for bit**. See [CPU tests](resume_cpu_tests.json), [resume context](resume_context_cpu.json), [baseline parity](baseline_physical_parity.json), [deployment](deployment.json).

## Shoulder gain comparison and decision

Both diagnostic branches start exact498 actor/critic/Adam/exploration at 4096 environments. They run64 and128 updates with unchanged320-robot20-second nominal/pushed gates. Candidate changes only both shoulder-pitch stiffnesses35→70, with damping1 and native caps. All four blocks load exactly and have zero pre-update KL/command mismatch. Original source/reward signatures explicitly **do not certify the executed physical override**; no experimental actor or gain enters production.

| Branch / updates | 3.5 lateral m/s nominal / pushed | 1.5 true flight nominal / pushed |
| --- | --- | --- |
{chr(10).join(table)}

Early direction improvement fades by128 updates. Both current direction tests still fail. Candidate128 introduces1.5 geometric-flight failures in both tests, while control retains2.448%/2.503% flight; candidate has0.795%/0.899%, below 1.5%. Candidate64 also adds a nominal2.5 foot-slip failure. **Reject stiffness70 for deployment.** The additional seed1901 nominal invocation is recorded, but fixed-reset clones do not establish32 independently seeded initial-state trials. See [matched review](matched_gain_review.json), [decision](decision.json). All diagnostic learned actors and failed attempts are preserved outside production.

## Actual upper-body motion over time

Actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets and velocities are saved for baseline498, control128, candidate128 and the new saved R40 pair. Six original/diagnostic rendered plots plus the fresh R40 plots have been visually inspected. Whole-population32 robots/command at10Hz is separated from16 representative50Hz traces/gate; resets are segmented. Fast shoulder bias/asymmetry and attenuated pitch remain; stronger gains enlarge slow swings without establishing natural coordination. The new 2.5 and 3 m/s representative traces also show slow shoulder-bias reversal while the hips keep cycling. At 2.5, a large total angle range includes this drift and must not be described as normal rhythmic swing. The native neck angles remain gait-linked and do not alone measure world gaze.

| Actor cohort | Command m/s | Shoulder pitch ranges deg left / right | Head RMS deg |
| --- | --- | --- | --- |
{chr(10).join(upper)}

See [baseline angles](initial_upper/upper_joint_angles.png), [control angles](control_upper/upper_joint_angles.png), [candidate angles](candidate_upper/upper_joint_angles.png), [current angles](final_upper/upper_joint_angles.png), [current velocities](final_upper/upper_joint_velocities.png), [all current data](final_upper/upper_joint_time_audit.json), [visual QA](plot_qa.json). Angles, phase and weak swing amplitudes must be interpreted together;10Hz data cannot certify faster jitter. Prior actual Isaac head-link/URDF parity remains validated, but CAD face/camera optical axis is uncalibrated. No exact speed-matched Unitree encoder trajectories were obtained; human comparison remains feature-based and morphology-limited. Natural/human/Unitree motion is **not certified**.

Primary references: [Isaac Lab actuator gains and limits](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html), [official Unitree G1 velocity configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py), [human arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043). These provide definitions and qualitative comparison, not mapped speed-matched encoder curves or convergence assurance.

## Resumed production and qualification

R40 supervisor {proof['supervisor_pid']}, one actual CUDA worker,55 frozen hashes verified. It has {proof['completed_main_updates']} completed-operation updates plus {proof['hash_proven_recovered_main_updates']} hash-proven recovered updates: **{proof['actual_retained_main_updates']} retained new main updates**, with {proof['observed_partial_main_updates']} further observed partial updates. The 8192 block hit the reserve after 41 updates and saved `memory_recovery.pt`; the 6144 receiver loads it exactly, including critic/Adam/exploration. All 41 updates are preserved. The first production block loads exact498, including critic/Adam/exploration and both teacher contexts; all completed main pre-update KL and command mismatches are zero. Teacher/upper auxiliaries remain20 batches/update. No test actor is deployed.

Completed main rates: {proof['active_PPO_samples_s']/1000:.2f}k active collection+PPO / {proof['worker_effective_samples_s']/1000:.2f}k worker-effective / {proof['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. Whole-chain includes R40 startup, capacity and baseline gates; its endpoint is the latest retained PPO progress timestamp, before final save/exit. Last completed block uses {last['environments']} environments, minimum free GPU {last['minimum_free_gpu_MiB']}MiB, host available{last['minimum_host_available_MiB']}MiB, private growth {last['private_memory_growth_MiB']}MiB, Torch allocation growth {last['torch_allocation_growth_MiB']}MiB. Read [resume verification](resumed_learning_verification.json) and [owned observation](closing_cpu/live_handoff.json), not capacity or evaluation memory as steady training capacity.

Exact saved actor `{Path(latest['checkpoint']).parent.name}`: signed3.5 speed{current['nominal']['signed_speed']:.3f}/{current['pushed']['signed_speed']:.3f} m/s, lateral{current['nominal']['lateral_speed']:.3f}/{current['pushed']['lateral_speed']:.3f} m/s. Current failures: {current['nominal']['failures']} / {current['pushed']['failures']}. Prior failures: {'; '.join(failure)}. Saved gates bind this actor only; newer partial updates remain independently unqualified. See [exact pair](closing_gate_review.json). No new high-speed substep ceiling certification was performed; prior isolated elbow/ankle clips do not establish a relevant limit.

## Preserved diagnostic limitations and next tests

The initial running-only CPU gate reader rejected the genuinely failed run; its read-only status scope was corrected without changing classifiers. A stopped-process cleanup probe found that its launcher had already exited and took no action. The matched harness completed normally; no diagnostic CUDA worker was force-killed. A future-only harness timeout edit occurred after this running harness loaded its original code; original launch artifacts remain preserved. Minor shell quoting and process-list display probes failed without changing training. The older stopped-hook CPU subtraction discrepancy remains unresolved diagnostic accounting, not a reproduced production defect.

{chr(10).join(str(i+1)+'. '+x for i,x in enumerate(focus))}

Keep the same active five-hour automation, thread and notification preference. Verified5m/s or qualified4m/s with causally demonstrated native limits blocking5 remains the endpoint; save, pause learner/automation and brief with computer on only then. Completed videos and unrelated unsaved Fusion work are preserved.
"""
(OUT/'REPORT.md').write_text(prose(report),encoding='utf-8')
evidence=dict(recorded_utc=utc,run=str(run),latest_owned=str(OUT/'closing_cpu/live_handoff.json'),
    latest_matching_pair=str(OUT/'closing_gate_review.json'),resume_verification=str(OUT/'resumed_learning_verification.json'),
    gain_decision=str(OUT/'decision.json'),baseline_parity=str(OUT/'baseline_physical_parity.json'),
    required_upper=str(OUT/'final_upper/upper_joint_time_audit.json'),visual_QA=str(OUT/'plot_qa.json'),
    report=str(OUT/'REPORT.md'),all_current_classifiers_unchanged=True,current_stage_qualified=False,
    no_diagnostic_gain_or_actor_deployed=True,inherited_actual_main_updates=27507,actual_retained_current_run_main_updates=proof['actual_retained_main_updates'])
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
next_check=read(TRAIN/'automation/NEXT_CHECK.json')
next_check.update(recorded_utc=utc,active_run=str(run),report=str(OUT/'REPORT.md'),latest_observation=evidence['latest_owned'],
    new_main_updates=proof['completed_main_updates'],recovered_main_updates=proof['hash_proven_recovered_main_updates'],
    actual_retained_current_run_main_updates=proof['actual_retained_main_updates'],inherited_main_updates_preserved=27507,
    selected_actor=dict(checkpoint=latest['checkpoint'],sha256=latest['sha256'],matching_gate_pair=evidence['latest_matching_pair'],current_stage_qualified=False,scope='Exact saved pair, not newer partial learner.'),
    highest_fully_qualified_stage='run_300',highest_fully_qualified_command_m_s=3.,current_stage='run_350',running_qualified=False,
    velocity_ceiling_proven=False,focused_tests=focus,latest_repair_decision=decision,throughput=proof,
    throughput_evidence=evidence['resume_verification'],latest_verified_state=state,
    mandatory_upper_time_series=evidence['required_upper'],failed_unsaved_previous_updates_excluded=32)
(TRAIN/'automation/NEXT_CHECK.json').write_text(json.dumps(next_check,indent=2)+'\n',encoding='utf-8')
tag='adaptive_check_20261007_0344_record'
entry=f"""<!-- {tag} -->
## {utc} — Goal-budget recovery and rejected shoulder-gain continuation

R39 failed before the heartbeat when its18-hour guard interrupted501, not a physical failure/completion. Last full498 SHA c04ae2f26310d5514905ce1a2fec9159594ec8ce8d5a19f67910291f497e2ae2 retained.32 unsaved updates excluded;10538 completed+33 recovered+16936 inherited=27507 actual prior main updates preserved. Supervisor-only R40 recovery adds explicitgoal budget0 and finite saved-boundary budget handling, restoring exact actor/critic/Adam/exploration and both teacher/proof contexts. All53 authored worker/physics/PPO/teacher files unchanged;55 frozen hashes verify.14CPU checks, real4-update64-env PPO/full reload, both320-robot20s baseline arrays/cases bitwise identical. Existing no-progress/memory/native-limit guards retained.

Completed4096-env64/128 shoulder-pitch stiffness70 versus35,damping1 exact-full-state control experiment REJECTED: early drift improvement fades; both3.5 direction gates fail; candidate128 has1.5 true flight0.795/0.899% versuscontrol2.448/2.503%, introducing both retention failures. Diagnostic gain override not certified by original signature; all experimental actors remain outside production. Upper actual angle-target-velocity/arm-leg phase plots inspected for baseline/control/candidate and fresh saved R40 pair. Fast bias/asymmetry and gait-linked neck oscillation remain; no human/Unitree encoder likeness or world-gaze certification. Failed probes/accounting uncertainties preserved.

R40 supervisor {proof['supervisor_pid']} has {proof['completed_main_updates']} completed plus {proof['hash_proven_recovered_main_updates']} recovered = {proof['actual_retained_main_updates']} retained new main updates, with {proof['observed_partial_main_updates']} observed partial, one worker,55 hashes. Original498 exact full main load, zero KL/command mismatch, standing/moving/upper20 batches. Last block {last['environments']} envs; rates{proof['active_PPO_samples_s']/1000:.2f}k active/{proof['worker_effective_samples_s']/1000:.2f}k worker/{proof['whole_chain_samples_s']/1000:.2f}k chain incl startup/gates. MinimumGPU {last['minimum_free_gpu_MiB']}MiB/host {last['minimum_host_available_MiB']}MiB. Exact{Path(latest['checkpoint']).parent.name}3.5 signed{current['nominal']['signed_speed']:.3f}/{current['pushed']['signed_speed']:.3f}, lateral{current['nominal']['lateral_speed']:.3f}/{current['pushed']['lateral_speed']:.3f}; no3.5/4/5/ceiling qualification.267all-prior/349certified3 retained separately. Target8192/reserve300/host fallback/180healthy TRAINING-second ramp remain. Same active5h/verified5 orqualified4+native-limit endpoint; save,pauseboth,brief,computeron. Videos fulfilled,unsavedFusion untouched.

Report: {OUT/'REPORT.md'}. Decision: {OUT/'decision.json'}. Exact gates: {OUT/'closing_gate_review.json'}. Recovery: {OUT/'resumed_learning_verification.json'}. Upper: {OUT/'final_upper/upper_joint_time_audit.json'}. Next focused tests in NEXT_CHECK; full history preserved.
<!-- /{tag} -->
"""
for p in [TRAIN/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    text=p.read_text(encoding='utf-8-sig');assert tag not in text
    p.write_text(text+'\n\n'+prose(entry),encoding='utf-8')
handoff=f"""<!-- {tag} -->
Current handoff {utc}: R40_goal_budget_full_state supervisor {proof['supervisor_pid']} continues exact R39498 full actor/critic/Adam/exploration and teachers after18-hour budget interruption, preserving27507 prior actual main updates;32 unsaved excluded. No worker/reward/physics/gain/native-cap change. Read {OUT/'REPORT.md'}, decision, baseline_physical_parity, resumed_learning_verification, final_upper/upper_joint_time_audit, NEXT_CHECK and fresh ACTIVE_RUN/state.14CPU/real PPO/reload and full same-actor320-robot20s baseline parity pass. Completed shoulderK70 exact4096-env64/128 gain continuation REJECTED: early drift gains fade and1.5 geometric flight retention fails both. All diagnostic actors excluded,original35/1 kept. Mandatory upper timelines inspected; fast asymmetry/bias and head oscillation remain, no exact human/Unitree curve certification. R40 {proof['completed_main_updates']} completed plus {proof['hash_proven_recovered_main_updates']} recovered = {proof['actual_retained_main_updates']} retained new main,oneworker55hashes,exactfull load/zeroKL/commands/bothteachers+upper20batches. Current3.5/retention gates incomplete; separately preserve267 all-prior and349certified3. Goal-drivenmax_hours0 retains no-progress/memoryguards,target8192/reserve300/hostfallback/180healthy TRAININGramp. Sameactive5h/thread/notifications/verified5 orqualified4+native limitsblocking5 endpoint;save,pauseboth,brief,computeron. Nominalclone diversity insufficient beforeendpoint. Earlier entries below historical.
<!-- /{tag} -->

"""
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    text=p.read_text(encoding='utf-8-sig');assert tag not in text
    p.write_text(prose(handoff)+text,encoding='utf-8')
native=tomllib.loads((OUT/'automation_before.toml').read_text(encoding='utf-8-sig'))
prefix=native['prompt'].split('Latest dated five-hour check',1)[0]
dated=f"""Latest dated five-hour check, evidence {utc}: R40_goal_budget_full_state is the exact full-state recovery of R39498 SHA c04ae2f26310d5514905ce1a2fec9159594ec8ce8d5a19f67910291f497e2ae2. Read diagnostics/adaptive_check_20261007_0344/REPORT.md, decision.json, matched_gain_review.json, baseline_physical_parity.json, resumed_learning_verification.json, closing_gate_review.json, final_upper angle/velocity plots/data and closing_cpu/live_handoff.json, then fresh ACTIVE_RUN/state. R39's18-hour guard interrupted healthy501 before this heartbeat;32 unsaved updates excluded. R40 preserves27507 actual prior main updates, including33 recovered R39. Only supervisor changed: explicitmax_hours0 goal-driven budget, finite budgets finish saved boundaries, strict stopped full-state/teacher/context recovery. All53 authored worker/physics/PPO/teacher files byte-identical,55 hashes verify.14CPU tests, real4-update64-env PPO/full reload and every array/case of 320-robot20s baseline nominal/pushed replay pass. Actor/critic/Adam/exploration and both teachers retained exactly, original35/1 gains/rewardv8/verifierv6/history/inference/nativecaps/iteration4 retained. Smoke/capacity actors discarded.

Completed stiffness70 versus35 shoulder-pitch full-state4096-env64/128 controls are REJECTED. Early direction gains fade by128; both3.5 direction tests still fail. Candidate128 introduces1.5 geometric-flight failures at0.795/0.899%, versuscontrol2.448/2.503%, plus candidate64 nominal2.5 slip. Original signatures do not certify diagnostic physical overrides. No experimental actor/gain deployed. Mandatory actual upper angles/targets/velocities and bilateral/arm-leg phase inspected for baseline/control/candidate/current. Larger slow swings and fast bias/asymmetry remain; no natural/Unitree certification. Isaac-head/FK parity known, CAD face/camera optical axis uncalibrated. Repeated nominal seed invocation still uses fixed resets and is not32 independently seeded initial-state diversity. Old stopped-hook subtraction remains diagnostic accounting uncertainty, not proven runtimefault.

Fresh R40 main {proof['completed_main_updates']} completed plus {proof['hash_proven_recovered_main_updates']} recovered = {proof['actual_retained_main_updates']} retained, with {proof['observed_partial_main_updates']} observed partial; supervisor {proof['supervisor_pid']},oneactualworker55hashes. First main loads original498 full state exactly. The 8192 reserve guard saves41 updates and the 6144 receiver loads the exact recovery checkpoint; allpreupdateKL/commandmismatch0,both teacher / upper20batches. Rates{proof['active_PPO_samples_s']/1000:.2f}kactive/{proof['worker_effective_samples_s']/1000:.2f}kworker/{proof['whole_chain_samples_s']/1000:.2f}kchain includingnewstartup/capacity/gates; lastblock {last['environments']} minGPU {last['minimum_free_gpu_MiB']}MiB,host {last['minimum_host_available_MiB']}MiB,stablememory. Exact{Path(latest['checkpoint']).parent.name} signed3.5 {current['nominal']['signed_speed']:.3f}/{current['pushed']['signed_speed']:.3f},lateral{current['nominal']['lateral_speed']:.3f}/{current['pushed']['lateral_speed']:.3f}; current failures{current['nominal']['failures']}/{current['pushed']['failures']}. Prior failures{prior_fail}. Savedgates bindexactactor,not newerpartialupdates. Separatelypreserve267all-prior and349certified3. No3.5/4/5 or relevantmotorceilingendpoint.

Next focused tests:
{chr(10).join(str(i+1)+'. '+x for i,x in enumerate(focus))}

Keep this same ACTIVE five-hour cadence/thread/notification preference and verified5 or qualified4+causal native-limit endpoint. Leave healthy training running while incomplete and computer on. No pending video request; unsavedFusion protected. Read newer actual evidence and do not spend the next check solely waiting.
"""
args=dict(id=native['id'],mode='update',kind='heartbeat',name=native['name'],status=native['status'],rrule=native['rrule'],targetThreadId=native['target_thread_id'],prompt=prefix+prose(dated))
if 'notification_policy' in native:args['notificationPolicy']=native['notification_policy']
(OUT/'automation_update_args.json').write_text(json.dumps(args,indent=2)+'\n',encoding='utf-8')
(OUT/'documentation_receipt.json').write_text(json.dumps(dict(recorded_utc=utc,tag=tag,report=str(OUT/'REPORT.md'),native_update_pending=True,focus=focus),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'REPORT.md'),journals_and_handoffs_written=True,native_update_pending=True,completed_main=proof['completed_main_updates'])))
