"""Prepare native heartbeat arguments, then record a confirmed schedule update.

This changes documentation only. No simulator, policy, frozen runtime, or process
control is imported or invoked.
"""
import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAINING = OUT.parents[1]
AUTOMATION = Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
write = lambda p, z: Path(p).write_text(json.dumps(z,indent=2)+'\n', encoding='utf-8')


def prepare():
    cfg = tomllib.loads((OUT/'automation_before.toml').read_text(encoding='utf-8-sig'))
    old = cfg['prompt'].split('Latest dated focus',1)[0]
    old = old.replace('every three hours','every five hours').replace('three-hour cadence','five-hour cadence').replace('THREE_HOUR_CHECK.md','FIVE_HOUR_CHECK.md')
    start = old.index('Stop only when actual fixed-frame')
    end = old.index('After every check',start)
    endpoint = '''Stop after actual fixed-frame 5 m/s locomotion passes matching independent nominal/pushed/stopped/all-prior-speed gates: at least 4.95 m/s signed forward under the existing declared 1% measurement tolerance, with at least 32 independent trials per command. If native locomotion-relevant actuator limits prevent reaching 5, the user instead authorizes a verified 4 m/s finish: at least 3.96 m/s signed forward under the same declared tolerance, every matching gate, and repeated higher-speed tests plus controlled CAD motor-speed/torque and sustained driving-saturation evidence proving the limit. A command, reward, isolated cap spike, fall, turning, policy fault, expired budget or crash never qualifies. Do not raise caps or stop successfully at an unqualified lower speed. At either verified endpoint save exact full checkpoints, hashes, gates, telemetry and report, safely pause owned training and prevent supervisor restart, pause this same automation and brief the user. Leave the computer on; this supersedes historical shutdown and generic any-speed ceiling endpoints.

Every check MUST analyze upper-body joint angles over time, not only posture RMS or video stills. Save and inspect actual head yaw/pitch, shoulder pitch/roll and elbow angle/target/velocity plots and data by speed, with full-population statistics, bilateral amplitude/symmetry, shoulder-versus-ipsilateral-leg phase, elbow coordination, cadence, smoothness and saturation. Distinguish high-rate representative traces from population sampling and segment resets. Compare primary human and official Unitree motion references where axes, signs, rest poses, speed and morphology are comparable; explicitly record unavailable speed-matched joint trajectories and never fabricate reference curves or certify human-like motion from bounds/reward. Use the reusable CPU upper_joint_time_audit.py identified in FIVE_HOUR_CHECK.md with current exact gate pairs and a new dated output folder. Preserve learned forward head, useful arms, torso/legs and full learner state; if refinement is justified, test a narrow continuation against matched control without starting the whole policy again or spoofing production observations. Native-zero head angles are torso-relative, not proof of world gaze. These diagnostics do not replace or relax locomotion gates.

'''
    prompt = old[:start]+endpoint+old[end:]
    prompt += '''Latest dated user schedule/upper-motion update, evidence 2026-10-06T02:29:30Z: same R37_offline_ground_full_state supervisor32492, 55 frozen hashes verified, no runtime/reward/physics/cap/optimizer/actor change or training pause. The new exact-pair CPU review independently verifies promotions at 2.0 (checkpoint240, actual1.911/1.912m/s) and2.5 (checkpoint293, actual2.488/2.487m/s), including all matching stopped/prior gates. Current stage is run_300; command3 alone is not qualification. Exact301 nominal/pushed have3.009/3.009 signedm/s and25.65/25.56% genuine flight but fail world lateral speed; retained1.5 geometric flight also fails. Stopped/walking through1.0 and2.0/2.5 cases pass in that pair. Read diagnostics/automation_five_hour_update_20261006/REPORT.md, initial_gate_review.json, upper_joint_time_audit.json, upper_joint_angles.png, and fresh ACTIVE_RUN/state.

The required joint-time-series audit ran now on both exact301 tests and certified293 tests: 32 robots per command at10Hz, plus16 representative robots at50Hz, no resets in those detailed traces. It finds actual arm asymmetry and imperfect leg coordination. At1.0,301 median shoulder pitch ranges are38.8deg left/24.9deg right; ipsilateral correlations-0.896/-0.304 and phases164/108deg. At3.0, phases113/-176deg are asymmetric; head RMS4.93deg with gait-locked oscillation. Certified293 at2.5 has headRMS3.24deg; newer301 at2.5 worsens to4.27deg. These are diagnostic findings, not natural-arm certification or reasons to erase progressing legs. Plot physical targets versus measured motion; no exact speed-matched Unitree encoder reference was obtained, so exact Unitree curve similarity remains unmeasured. The primary human comparison is feature-based and morphology-limited.

Next one to three priorities: (1) preserve the improving same full-state learner and complete matching3.0 current, direction and all prior running gates before promotion; compare retained1.5 flight, keeping separately certified293. (2) rerun and inspect per-speed joint-angle/target/velocity timelines and arm/leg phase, asymmetry and head oscillation; if problems persist, quantify effective upper targets and auxiliary/PPO gradients, and test a narrow full-state upper refinement against control while retaining gait/posture. The previously reproduced teacher-context provenance mismatch remains a controlled candidate only if retention stalls, never a deployed fix. (3) verify one owned worker,55 hashes, native caps, actual recovery accounting and6144/8192 memory/reserves/180healthy-training-second ramps; use actor-specific substep motor velocity/torque driving-clipping attribution if a high-speed plateau appears. Read fresh evidence; the dated evaluation's free memory is not steady training capacity. Keep this same adaptive FIVE-hour heartbeat active until verified5m/s or the verified4m/s actuator-limited fallback. At that endpoint save,pause training and automation,brief,computer on.
'''
    args = dict(mode='update',id=cfg['id'],kind='heartbeat',name='Humanoid training - adaptive five-hour check',
                status='ACTIVE',rrule='FREQ=HOURLY;INTERVAL=5',targetThreadId=cfg['target_thread_id'],prompt=prompt)
    if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
    write(OUT/'automation_update_args.json',args)
    print(json.dumps({'prepared':True,'id':args['id'],'interval_hours':5,'prompt_characters':len(prompt)}))


def finish():
    args=read(OUT/'automation_update_args.json')
    cfg=tomllib.loads(AUTOMATION.read_text(encoding='utf-8-sig'))
    old=tomllib.loads((OUT/'automation_before.toml').read_text(encoding='utf-8-sig'))
    for key in ['id','name','kind','status','rrule']:assert cfg[key]==args[key],key
    # The app trims the single terminal newline; no interior content differs.
    assert cfg['prompt']==args['prompt'].rstrip('\n')
    assert cfg['target_thread_id']==old['target_thread_id']==args['targetThreadId']
    assert cfg.get('notification_policy')==old.get('notification_policy')
    now=datetime.now(timezone.utc).isoformat();review=read(OUT/'initial_gate_review.json');live=read(OUT/'initial_owned_observation.json')
    audit=read(OUT/'upper_joint_time_audit.json');latest=review['latest']
    assert len(audit['records'])==34 and audit['metric_fixtures_passed']==4
    (OUT/'automation_after.toml').write_text(AUTOMATION.read_text(encoding='utf-8-sig'),encoding='utf-8')
    proof=dict(recorded_utc=now,id=cfg['id'],native_update_verified=True,kind=cfg['kind'],name=cfg['name'],status=cfg['status'],
               rrule=cfg['rrule'],target_thread_id=cfg['target_thread_id'],same_thread=True,
               notification_preference_preserved=True,model_reasoning_scope_preserved='GPT-6.1 Sol max',
               prompt_body_exact=True,app_terminal_newline_trimmed=True,
               runtime_or_policy_changed=False,training_paused=False)
    write(OUT/'automation_verification.json',proof)
    report=f'''# Five-hour automation and upper-body joint-time audit

Recorded {now}. User request dated October5 NewYork; files use October6 UTC. The existing native heartbeat is verified **ACTIVE every five hours**, with the same ID/thread/notification preference and GPT-6.1 Sol max instruction. No duplicate was created. Full verification is in `automation_verification.json`; before/after TOML and exact tool arguments are preserved.

The first strict verification detected the app's trimming of one terminal newline. Its failed comparison is preserved in `native_update_check_failure.json`; line-by-line content was identical. Verification now explicitly permits that terminal newline normalization and checks every interior prompt character and all schedule fields.

## New stopping rule

Finish at verified5m/s (signed forward>=4.95m/s under the existing declared1% tolerance) with matching nominal/pushed/stopped/all-prior gates and>=32 trials per command. If controlled higher-speed tests prove native locomotion-relevant actuator limits block5, finish at verified4m/s (>=3.96m/s under that same tolerance), with every matching gate and measured limit attribution. A generic ceiling at an arbitrary speed is no longer successful completion. Preserve native caps and diagnose sustained actual motor speed/torque driving saturation through CAD Jacobians/reductions. At a verified endpoint save exact full state/evidence, safely pause owned training and the same automation, brief the user and leave the computer on. No live frozen supervisor source was edited to implement this; the heartbeat applies the fallback at an owned saved boundary.

`automation/FIVE_HOUR_CHECK.md` is now authoritative; the older three-hour workflow remains historical. Native app updates are executed using the existing automation, consistent with [OpenAI automation documentation](https://learn.chatgpt.com/docs/automations).

## Exact independent progress

The CPU review verifies all matching nominal/pushed stopped/prior gates on checkpoint240 at2.0 and checkpoint293 at2.5. Signed speeds are1.911/1.912 and2.488/2.487m/s respectively. These qualifications belong to those exact actors. Current301 at3.0 gives3.009/3.009m/s, but both tests fail lateral speed and retained1.5 geometric flight; it is not fully qualified. See `initial_gate_review.json` for hashes and every case. The owned snapshot is dated {live['recorded_utc']}; read fresh state for later progress. Its288-environment evaluation count is not PPO capacity; selected next training count is6144. All55 frozen hashes verify in that snapshot. This turn made no source/reward/physics/motor-cap/optimizer/actor change, no training pause and no simulator launch.

## Joint angles over time: findings

The new reusable `upper_joint_time_audit.py` analyzes both exact301 gates and both certified293 gates:34 command/test records,32 robots per command at10Hz and16 representative50Hz traces per gate. Actual joints, physical targets and velocities are separate. No resets occur in the high-rate input traces. Four numerical metric fixtures pass; finite angles/velocities, CAD axes and exact checkpoint binding are checked. The PNG was visually inspected, including multiple cycles at1,1.5,2.5 and3m/s. Machine-readable JSON and CSV and an exportable SVG are saved beside it.

At301/1m/s, median shoulder-pitch ranges are38.8deg left and24.9deg right. Left/right ipsilateral shoulder/hip correlations are-0.896/-0.304; phase at the measured hip cycle is164/108deg. This is substantial arm asymmetry, with weaker right counter-swing. At3m/s ranges are14.0/14.3deg, but phases are113/-176deg: similar amplitude does not mean similar leg coordination. Elbows also change coordination with speed. These are motion features, not proof of anatomically natural motion. Head RMS is3.59deg at1,4.27deg at2.5 and4.93deg at3; the curves show gait-locked head oscillation. Certified293/2.5 has3.24deg head RMS, so the newer actor worsens this metric despite faster-stage progress. Prior gates and useful learned state remain protected.

Actual upper targets and joints differ by about8.45deg RMS at1m/s and9.36deg at3 in the selected50Hz traces. This is a descriptive lag/response observation, not proof of an actuator ceiling or a need to change gains. Range, mean posture, velocity, elbow coordination and selected high-rate smoothness are in the JSON. Quantiles/FFT phase are native-angle diagnostics; axes are dominant positiveY for both shoulder and hip pitch but the hip axes are canted. Correlation does not equal anatomical shoulder extension. Head nativezero is torso-forward, not global gaze. The5deg reporting band is diagnostic only. FFT power/finite-difference acceleration are not imported human smoothness thresholds, and high-frequency claims cannot be drawn from the10Hz population traces.

The primary human reference supports speed-dependent amplitude/coordination/smoothness comparison, but its wrist IMUs are not these shoulder/elbow encoders: [Mirelman et al., PLOS ONE2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043). The [official Unitree repository](https://github.com/unitreerobotics/unitree_rl_mjlab) documents trained-policy playback and motion workflows, but no verified speed-matched G1/H1 encoder trajectory was obtained in this check. Exact Unitree angle-curve similarity remains **unmeasured**; a reward config/rest pose/dance dataset is not substituted. A PMC alternative human paper was blocked by access verification and a Unitree motion-tree web read returned an error; neither failure is silently presented as reviewed data. Future checks must record reference availability, mapping and speed before claiming a numerical comparison.

## Next checks

Every check now must save and inspect upper-body angle/target/velocity timelines, shoulder/elbow bilateral coordination and arm-to-leg phase alongside the exact locomotion gates. Preserve neutral head, torso/leg learning and certified293. Continue the improving3m/s stage until every direction and prior-flight gate passes. If unnatural upper motion persists, measure effective clipped targets/physical response and auxiliary/PPO gradients before a narrow matched-control full-state refinement. The teacher-context mismatch remains a prior reproduced, undeployed lead if retention stalls. Use actor-specific native substep speed/torque attribution if higher-speed progress plateaus. Existing execution optimizations, reserve/fallback and healthy-training ramps remain.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    workflow=TRAINING/'automation/FIVE_HOUR_CHECK.md'
    oldwork=TRAINING/'automation/THREE_HOUR_CHECK.md'
    marker='<!-- superseded_by_five_hour_update_20261006 -->'
    if marker not in oldwork.read_text(encoding='utf-8-sig'):
        oldwork.write_text(marker+'\n**Superseded cadence and endpoint:** read [FIVE_HOUR_CHECK.md](FIVE_HOUR_CHECK.md). The user changed checks to five hours, completion to verified5m/s or verified4m/s when native actuator limits prevent5, and required angle-over-time upper-body analysis in every check. The records below are historical.\n\n'+oldwork.read_text(encoding='utf-8-sig'),encoding='utf-8')
    nxt=read(TRAINING/'automation/NEXT_CHECK.json')
    nxt['previous_status_report']=nxt.get('report');nxt['recorded_utc']=now
    nxt['report']=str(OUT/'REPORT.md');nxt['workflow']=str(workflow);nxt['cadence_hours']=5
    nxt['current_stage']=live['state']['stage'];nxt['latest_observation']=str(OUT/'initial_owned_observation.json')
    nxt['new_main_updates']=live['state']['total_training_iterations']
    nxt['inherited_main_updates_preserved']=live['state']['prior_training_iterations']
    nxt['main_count_scope']='Supervisor completed-operation count at owned snapshot. Historical recovered updates and old throughput retain their separately dated evidence.'
    nxt['actual_retained_current_run_main_updates']=None
    nxt['actual_retained_count_status']='Not reaudited during this schedule/upper-body turn; add verified recovered updates only from their exact provenance.'
    nxt['latest_verified_state']=live['state'];nxt['highest_fully_qualified_stage']='run_250';nxt['highest_fully_qualified_command_m_s']=2.5
    nxt['running_qualified']=False;nxt['velocity_ceiling_proven']=False
    nxt['selected_actor']={'checkpoint':latest['checkpoint'],'sha256':latest['sha256'],'matching_gate_pair':str(OUT/'initial_gate_review.json'),
                           'current_stage_qualified':False,'standing_and_walking_cases_through_1_m_s_pass':True,
                           'scope':'Exact completed301 tests; not a deployed replacement or attribution of293 qualification'}
    nxt['certified_run_250_checkpoint']=[p['checkpoint'] for p in review['promotions'] if p['stage']=='run_250'][0]
    nxt['endpoint']='Verified5m/s with all exact matching gates; or verified4m/s with all gates and controlled proof native actuator limits prevent5. Save,pause owned training and same automation,brief;computer on.'
    nxt['endpoint_policy']={'primary_command_m_s':5,'primary_minimum_signed_speed_m_s':4.95,'fallback_command_m_s':4,'fallback_minimum_signed_speed_m_s':3.96,
                            'measurement_tolerance_fraction':.01,'minimum_independent_trials_per_command':32,'all_nominal_pushed_stopped_prior_gates_required':True,
                            'fallback_requires_causally_demonstrated_native_actuator_limit_before_5':True,'raise_caps':False,'at_verified_endpoint':'save,pause training and automation,brief; leave computer on'}
    nxt['upper_body_analysis']={'required_every_check':True,'script':str(OUT/'upper_joint_time_audit.py'),'latest_json':str(OUT/'upper_joint_time_audit.json'),
                                'angle_plot':str(OUT/'upper_joint_angles.png'),'csv':str(OUT/'upper_joint_angles.csv'),
                                'scope':'Actual per-command joint-angle/target/velocity timelines, amplitude, bilateral and arm-leg phase, head alignment and smoothness; exact Unitree similarity unmeasured until mapped speed-matched trajectories available'}
    nxt['throughput']['measurement_status']='Historical prior check; not remeasured during this CPU-only schedule/upper-body turn'
    nxt['focused_tests']=[
        'Preserve improving same R37 full-state learner. Complete matching3.0 direction and all stopped/prior gates, especially retained1.5 genuine flight; keep exact certified293/2.5 actor. Latest301 reaches3.009 but fails lateral and1.5 flight. Read newer exact pairs before deciding to repair.',
        'Required: save/inspect current nominal+pushed head/shoulder/elbow actual-angle, physical-target and velocity time series, bilateral amplitude, elbow coordination and shoulder/ipsilateral-leg phase. Track current asymmetry and gait-locked head oscillation versus293. Compare mapped primary human/official Unitree references with explicit missing-data limitations. If persistent, quantify physical target response and auxiliary/PPO derivatives before a narrow unchanged-control full-state continuation; no whole-policy reset or posture rollback. Teacher-context mismatch remains undeployed if retention stalls.',
        'Verify one actual worker,55 hashes, exact learned-state recovery,6144/8192 memory,300MiB reserve/host guards and180healthy-training-second ramps. Audit actual sample accounting and actor-specific native5ms driving speed/torque saturation if high-speed progress plateaus. Preserve caps; only finish at verified5 or verified4 plus causal actuator-limit evidence, then save,pause both and brief with computer on.'
    ]
    nxt['latest_repair_decision']={'recorded_utc':now,'action':'User cadence/endpoint update and mandatory angle-time analysis; preserve improving same learner',
                                  'runtime_reward_physics_caps_optimizer_actor_changed':False,'training_paused':False,'current_run_300_qualified':False,
                                  'run_250_historically_qualified':True,'upper_motion_human_like_certified':False,'velocity_ceiling_proven':False,
                                  'actual_5_m_s_proven':False,'native_schedule_verified':True}
    nxt['latest_gate_trends']=str(OUT/'initial_gate_review.json');nxt['latest_closing_gate_pair']=str(OUT/'initial_gate_review.json')
    nxt['automation_verified_active']=True
    write(TRAINING/'automation/NEXT_CHECK.json',nxt)
    entry=f'''\n\n## {now} — User five-hour cadence,5/4 endpoint and upper joint-angle time series\n\nUpdated existing native heartbeat to ACTIVE every5hours, same ID/thread/notification/GPT-6.1 Sol max. Verified5m/s remains primary; verified4m/s is fallback only with controlled proof native actuator limits block5; all exact nominal/pushed/stopped/prior gates remain. At endpoint save,pause training+automation,brief,computer on. Canonical workflow FIVE_HOUR_CHECK.md supersedes historical3h policy. No live training source/reward/physics/cap/optimizer/actor reset, pause or simulator launch.\n\nCPU exact-pair review verifies R37 promotions2.0 and2.5; actor301 at3.0 is still unqualified (lateral and prior1.5 flight). New mandatory per-check angle-time diagnostic analyzes34 command/test records across301/293, actual head/shoulder/elbow angles, targets, velocities and arm/leg phase. Plot QA confirms asymmetry (1m/s left/right shoulder ranges38.8/24.9deg), gait-locked head oscillation and unequal3m/s arm/leg phase113/-176deg. Human reference is feature-based, exact speed-matched Unitree encoder curves unavailable; no human-like certification. Four metric fixtures pass. Sources/limitations, native update before/after and current exact gates: {OUT/'REPORT.md'}. Preserve progressing full learner and independently certified actors.\n'''
    for file in [TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
        with file.open('a',encoding='utf-8') as f:f.write(entry)
    head=f'''<!-- automation_five_hour_update_20261006 -->\nCurrent user workflow update {now}: checks every FIVE hours, verified5m/s endpoint or verified4m/s with controlled native actuator-limit evidence; save,pause both,brief,computer on. Read {workflow} and {OUT/'REPORT.md'}, then fresh ACTIVE_RUN/state. Required every-check upper joint-angle/target/velocity timelines and bilateral/arm-leg phase; do not claim exact human/Unitree similarity without comparable motion data. Same R37 progressing; certified2.5 actor293 and current unqualified3.0 actor301 are separately bound to exact tests. No live runtime/state reset or training pause this turn. Earlier handoffs below are historical.\n<!-- /automation_five_hour_update_20261006 -->\n\n'''
    for file in [Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
        file.write_text(head+file.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','finish']);a=p.parse_args()
    prepare() if a.mode=='prepare' else finish()
