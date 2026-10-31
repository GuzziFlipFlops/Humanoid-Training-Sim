"""Save check evidence, append journals, and prepare same-heartbeat update arguments."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,tomllib

OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1];AUTO=TRAINING/'automation'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')

owned=read(TRAINING/'diagnostics/walk_regression_20261003/live_handoff.json')
state=owned['state'];run=Path(owned['active_run']['run'])
assert state['status']=='running' and len(owned['actual_worker_pids'])==1
assert len(owned['frozen_source_integrity'])==46 and all(owned['frozen_source_integrity'].values())
resume=read(OUT/'video_resume.json');verified=read(OUT/'video_verification.json')
assert resume['same_supervisor'] and resume['verified_videos_complete'] and verified['passed']
assert Path(resume['run'])==run
write(OUT/'resumed_owned_observation.json',owned)
qa=read(OUT/'frame_visual_review.json')
qa['additional_flight_frame_review']={'frames':[34,35,36,277,278,279,489,490,491],
    'sheet':str(Path(verified['manifest']).parent/'locomotion_150_flight_frames.jpg'),
    'finding':'All parts remain attached in takeoff/landing neighbors. Flight is brief, low-clearance and mainly established by synchronized sole/force telemetry; this single video does not certify robust or natural-looking running.'}
write(OUT/'frame_visual_review.json',qa)
evidence=read(OUT/'evidence.json');latest=evidence['gates'][-2:]
blocks=evidence['blocks'][-2:]
comparison=read(OUT/'phase_target_comparison.json')
main=owned.get('worker_progress') or {}
write(OUT/'resumed_main_evidence.json',{'recorded_utc':owned['recorded_utc'],'state':state,'worker_progress':main,
    'worker_pids':owned['actual_worker_pids'],'source_hashes':owned['frozen_source_integrity'],'gpu':owned['gpu'],
    'system_memory':owned['system_memory'],'pause_excluded_seconds':resume['pause_seconds'],
    'scope':'Same supervisor/checkpoint resumes; partial block samples are not completed-main throughput or new qualification.'})
videos=verified['clips'];now=datetime.now(timezone.utc).isoformat()

lines=['# October4 23:40UTC adaptive check and full-body GUI recordings','',
    'The requested five full-body GUI videos are complete, checked and linked to the user. The exact R34 supervisor resumed after a temporary saved-boundary pause. No training rewards, optimizer, policy, joint, collision, motor limit or frozen source changed in this check.','',
    '## Recordings','',
    'All140 compatible original physical_v29 CAD meshes and source RGBA materials are rendered on their matching articulated bodies. The separate playback URDF preserves every frozen nonvisual link element and joint exactly. Current CADv32 was not substituted. Diagnostic visual-instance overrides and material binding affect only render subtrees. Actual GUI import/neutral preview and moving frames were inspected; initial washed-out lighting/black band and an instance-proxy material binding failure were preserved and fixed before final capture.','',
    'All2500 encoded H.264 frames decode, each video has500 distinct physical and rendered frames at50FPS,1280x720, exactly10 simulated seconds in10 real playback seconds. Slow capture took114–123wallseconds per clip, excluding settling. Full-frame and closeup sheets at0/2.5/5/7.5/10seconds per speed plus nine takeoff/landing-neighbor frames were visually inspected. Original head, torso, waist/motor assemblies, arms/hands/fingers, thighs, shins, ankles and feet remain attached and framed. Zero falls.','',
    '| Command m/s | Actual signed forward m/s | Video |','|---|---|---|']
for v in videos:lines.append(f"| {v['command_m_s']:.2f} | {v['mean_signed_world_forward_speed_m_s']:.3f} | [{Path(v['video']).name}]({Path(v['video']).as_posix()}) |")
lines += ['',
    'The four walking clips use the exact freshly qualified baseline actor;1.5 uses preserved checkpoint038, which passed the running case nominal/pushed but failed slow retention. No higher speed is actually demonstrated. The running video measures2% geometric flight, ten brief control ticks. This is a recording, not whole-stage qualification. Tilted head, crouched knees and asymmetric arm posture are real policy output; cosmetic pose correction was not applied. Full-CAD collision clearance/hardware dynamics remain unqualified because training collision proxies are unchanged.','',
    '## Pause and resume','',
    f"Owned supervisor PID{resume['supervisor_pid']} was suspended at checkpoint065 after1025 completed new main updates, with evaluation067 drained before the GUI. Exact checkpointSHA:{resume['checkpoint_sha256']}. The GUI exited before the same supervisor resumed at{resume['recorded_utc']}. Pause:{resume['pause_seconds']:.1f}seconds ({resume['pause_seconds']/60:.2f}minutes), including rendering repairs/capture/encoding/frame QA. Evaluation067 wrapper and whole-chain clocks include this pause; exclude it from throughput comparisons. No critic/Adam/exploration reset or training restart. Verified one owned training worker and all46 frozen hashes after resume; see resumed_owned_observation.json and resumed_main_evidence.json.",
    '', '## Behavioral status and focused diagnosis','',
    'R34 checkpoint038 at449 new updates passed the1.5 running case:1.476/1.477m/s signed travel and1.8854%/1.7569% geometric flight,100% survival. Earlier0.15/0.35 failed speed. Later actors recover prior walking but lose flight. At checkpoint065/1025 updates, matching066nominal/067pushed retain every stopped and prior walking command; running signed1.457/1.460m/s,100% survival,29/30 alternating landings, but geometric flight0% in both. Complete run_150 remains unqualified. The successful baseline walking actor and flight actor are preserved. No5m/s or motor-ceiling endpoint.','',
    'New CPU phase/action comparison reconstructs each of three saved actors on its own observed histories within4.77e-6 max action error, then compares them at identical running states. Latest intended-flight windows still have a loaded leg100% of the time, with about366N on the loaded side; geometric clearance is zero. All observed/action commands agree. Latest all-output raw clipping is37.17%; pitch ankles clip about69%/64% and roll ankles76%/78% in the bounded reconstructed samples. The successful flight actor also clips heavily, so clipping alone does not establish the cause. Knees are not clipped in those samples. Cross-state clipped responses and pre-CAD target changes are saved; these are not applied targets or closed-loop counterfactuals. Exact sampled CAD motor velocities show isolated ankle overshoot and elbow saturation, not a persistent locomotion-relevant velocity ceiling. No motor cap changed.','',
    'The earlier bounded knee-retraction physics test gained flight but lost total objective through velocity-tracking, acceleration and height costs; this remains a stronger next lead than another arbitrary force weight. This check prioritizes the user capture and adds matched-history/phase diagnostics. It leaves useful retained-walking learning intact. At the next owned boundary, if zero flight persists, collect multiple actual PPO/prior parameter-gradient samples and a phase-timed normal-limit effective-target/full-objective physical comparison using the preserved flight and grounded actors. Measure cost/advantage and clipped/CAD-projected leg target response; distinguish command1.2–1.5 walking-class exposure from exact1.5 running rows. Do not assume retention-gradient conflict, raise action/motor caps, erase walking, or spend another check only waiting. A source repair must follow reproduced evidence and CPU/physics/PPO/load tests in a fresh freeze.','',
    '## Resources','',
    f"Last two completed6144 blocks: active collection+PPO {blocks[0]['active_samples_per_second']:.0f}/{blocks[1]['active_samples_per_second']:.0f} samples/s; worker-effective {blocks[0]['worker_effective_samples_per_second']:.0f}/{blocks[1]['worker_effective_samples_per_second']:.0f}. Last minimumGPU {blocks[-1]['minimum_free_gpu_mb']}MiB, host {blocks[-1]['minimum_host_mb']}MiB. Main pre-update KL/action-command mismatch remain zero. Current resumed block068 is6144. Target8192,300MiB VRAM reserve, host guards, full-state fallback and180healthy live-training seconds before upward ramps remain. Eval/GUI idle memory does not qualify an upward ramp. Separate denominators and exclude logged pauses; do not present partial postresume samples as completed throughput.",
    '', 'Evidence: capture_config.json; video_verification.json; frame_visual_review.json; video_pause.json/video_resume.json; evidence.json; running_audit.json; phase_target_comparison.json; resumed_main_evidence.json. The same adaptive three-hour heartbeat stays active, with only the fulfilled one-time capture priority removed. Leave the computer on.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

handoff=read(AUTO/'NEXT_CHECK.json')
request=handoff['recording_request'];assert request['status']=='pending'
request.update(status='complete',completed_utc=now,delivered_in_user_commentary=True,
    deliverables=[{'path':v['video'],'sha256':v['sha256'],'command_m_s':v['command_m_s'],'duration_s':10} for v in videos],
    frame_qa=str(OUT/'video_verification.json'),resume_evidence=str(OUT/'video_resume.json'),
    higher_speed_scope='No independent demonstrated speed above1.5m/s; none fabricated')
handoff.update(recorded_utc=now,report=str(OUT/'REPORT.md'),latest_observation=str(OUT/'resumed_owned_observation.json'),
    latest_verified_state={'recorded_utc':owned['recorded_utc'],'run':str(run),'status':state['status'],'stage':state['stage'],
        'checkpoint':state['checkpoint'],'main_updates':state['total_training_iterations'],
        'current_operation':state.get('operation_path'),'current_environments':state.get('environment_count'),
        'actual_worker_pids':owned['actual_worker_pids'],'frozen_hashes_verified':46},
    running_qualified=False,full_body_gui_recording_complete=True,
    latest_completed_throughput=blocks,logged_user_capture_pause_seconds=resume['pause_seconds'],
    preserved_flight_actor={'checkpoint':videos[-1]['checkpoint'],'sha256':videos[-1]['checkpoint_sha256'],
        'qualification':'Running case nominal/pushed only; whole stage fails slow retention'},
    focused_tests=[
        'If newer matched pairs still lose flight, controlled normal-limit phase-target/full-objective tradeoff and multiple actual PPO/prior gradients; distinguish exact-running versus near-frontier walking samples before a narrow repair',
        'Require simultaneous genuine flight AND all stopped/prior speed nominal+pushed gates; preserve checkpoint038 flight actor and qualified walking baseline; no higher-speed promotion from video',
        'Moving6144/8192 memory, fallback/full-state recovery,180healthy live-training ramp, samples/time excluding1458.4s capture pause; exact CAD motor speeds without cap inflation'])
write(AUTO/'NEXT_CHECK.json',handoff)
workflow=AUTO/'THREE_HOUR_CHECK.md';text=workflow.read_text()
start=text.index('<!-- next_gui_recording_20261004 -->');end=text.index('<!-- /next_gui_recording_20261004 -->')+len('<!-- /next_gui_recording_20261004 -->')
text=text[:start]+'<!-- full_body_recording_completed_20261005 -->\n## One-time full-body recording completed\n\nFive checked ten-second GUI clips at0.15/0.35/0.65/1.0/1.5m/s are delivered; all140 original compatible CAD visuals are present. The same R34 supervisor/checkpoint resumed after1458.4seconds. Read diagnostics/adaptive_check_20261004_2340/REPORT.md and NEXT_CHECK.json recording_request completion receipts. Do not repeat this fulfilled request without a new user request. Global three-hour diagnosis/qualification/pause-and-brief scope remains.\n<!-- /full_body_recording_completed_20261005 -->'+text[end:]
workflow.write_text(text,encoding='utf-8')
with (AUTO/'NEXT_GUI_RECORDING.md').open('a',encoding='utf-8') as f:
    f.write('\n\nCompleted October5 00:07UTC. Five full-body MP4s delivered with decoded/visual QA; same supervisor resumed. Completion receipts in diagnostics/adaptive_check_20261004_2340. This historical workflow does not authorize repeating the fulfilled capture.\n')

note='''\n\n## October4 23:40UTC check - full CAD GUI capture and exact resume\n\nFive actual GUI ten-second normal-time clips at0.15/0.35/0.65/1.0/1.5m/s are complete and delivered. All140 frozen-v29 original CAD visual meshes/source colors restored on matching links with unchanged physics/joints/caps. All2500 distinct encoded frames decode;25 timeline and9 takeoff-neighbor frames inspected, full body remains attached,0falls. Walking baseline and preserved checkpoint038 running-case actor are explicitly separate. Tilted head/crouched/asymmetric pose remains actual policy output; full-CAD collision fidelity is still unqualified. Rejected white/band and instance-binding previews preserved. Owned R34 supervisor14864 drained atcheckpoint065/1025updates, resumed exact state after1458.4s; one CUDA worker and46hashes verify. Latest gates066/067 retain all prior walks but running flight0%, signed1.457/1.460. New CPU three-actor phase/clipped-target comparison reconstructs actions within4.77e-6; loaded leg persists in intended flight. Heavy clipping also exists in successful actor, so no causal diagnosis or cap change. No training runtime/reward/policy update. Next focused normal-limit full-objective/phase-target and actual PPO/prior gradient experiments if plateau continues; no another waiting-only check. Same adaptive three-hour heartbeat active, fulfilled video priority removed; no5m/s/ceiling endpoint. Report/QA/checkpoint/resume/CPU evidence: E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_2340/REPORT.md. Videos: training/videos/full_body_gui_20261004_2340.\n'''
for path in [TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),
             Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
             Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    with path.open('a',encoding='utf-8') as f:f.write(note)
reference=Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')
text=reference.read_text(encoding='utf-8-sig')
reference.write_text('<!-- full_body_gui_20261004_2340_current -->\n## Latest check: full CAD GUI clips delivered; R34 resumes\n\nOctober4 23:40UTC check, capture completed October5 00:07UTC. Read E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_2340/REPORT.md and fresh ACTIVE_RUN/state. Five ten-second full-body GUI clips restore all140 compatiblev29 visuals with unchanged physics. Same R34 supervisor/checkpoint resumes after1458.4s. At1025updates prior walking passes,1.5running loses flight; preserve earlier038 flight actor. New exact-history phase/clipped-target diagnostics inform next controlled objective/gradient test; no runtime/reward/limit change. Videos do not qualify full stage or CAD collision fidelity. Same adaptive3h heartbeat remains, one-time capture request complete. Earlier sections below are dated evidence.\n\n'+text,encoding='utf-8')

automation_path=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
current=tomllib.loads(automation_path.read_text());write(OUT/'automation_before.json',current)
prompt=current['prompt'];prefix=prompt[:prompt.index('Continue this humanoid training thread every three hours')]
assert prefix.startswith('ONE-TIME PRIORITY')
durable=prompt[prompt.index('Continue this humanoid training thread every three hours'):prompt.index('Latest dated focus,')]
focus='''Latest dated focus, October4 23:40UTC check, full-body GUI capture completed October5 00:07UTC: fulfilled user recording_request is complete. Five ten-second50FPS H.264 GUI videos at0.15/0.35/0.65/1.0/1.5m/s restore all140 original compatiblev29 CAD meshes/materials; all2500 distinct frames decode and full-body/takeoff frames pass visual review. Actual mean signed speeds0.223/0.435/0.728/1.043/1.483m/s,0falls. Slow clips use freshly qualified baseline;1.5uses saved R34checkpoint038, whose running case passes039nominal/040pushed but slow retention fails. No higher speed is demonstrated. Read diagnostics/adaptive_check_20261004_2340/REPORT.md,video_verification.json,frame_visual_review.json,video_pause.json/video_resume.json and videos/full_body_gui_20261004_2340. Do not repeat this completed one-time recording request. Visual-only asset preserves training physics/joints/motor caps; no silent CADv32 migration, full-CAD collision fidelity remains unqualified. Head tilt/crouched/asymmetric pose is actual policy behavior. Rejected previews preserved. Same owned R34supervisor14864 drained atcheckpoint065/1025mainupdates and resumed exact state after1458.4seconds; one actual CUDA worker/all46 frozen hashes verify. Exclude capture pause from evaluation067 wrapper/full-chain clocks; idleGUI/eval memory does not qualify an upward ramp.

Latest matched066nominal/067pushed gates retain every stopped/prior walking command,100% survival, but1.5running signed1.457/1.460m/s has0% geometric flight and29/30 alternating landings. Earliercheckpoint038 briefly passed runningcase with1.8854%/1.7569% flight; whole stage never qualified. Preserve both useful actors. No5m/s or relevantvelocity-ceiling endpoint. New CPU running_audit.json and phase_target_comparison.json reconstruct2240 observed actions per actor within4.77e-6 and compare three actors at identical states. Latest intendedflight windows always retain one loaded leg,about366N; no genuine clearance. Rawrunning action clipping37.17%; anklepitch clips~69%/64%,ankleroll~76%/78%,knees0 in bounded samples. Successfulflight actor also clips heavily,so clipping alone or behavioral tradeoff cannot establish causal failure or retention-gradient opposition. Observed/action commands and pre-update KL stayzero. Exact sampledCAD motor speeds include isolated ankleovershoot/elbow saturation,not a locomotion-relevantceiling; substep attribution unexcluded.

At next owned saved boundary, if fresh matched flight still plateaus, do not spend another check solely waiting or changing arbitraryweights: perform controlled normal-limit phase-timed effectiveclipped/CAD-projected leg-target and full-objective/cost physical comparison using preservedflight andgrounded actors; collect multiple actual PPO/prior parameter-gradient samples and phase-conditioned advantages. The prior takeoffpulse gainedflight butlosttracking,acceleration,height reward; testthis tradeoff and exact1.5running versusnear-frontier1.2–1.5walking-class exposure before choosing a focusedrepair. Retain qualifiedwalkingteacher; teacher-conflictrequires actualgradient evidence. No helperpulse/rootsupport,capsincrease orgate relaxation deployed. Sourcechanges require CPUcounterexamples,physical/PPO/full-loadsmoke andfresh frozenrun; preservelearned compatible state. Simultaneous nominal,pushed,stopped/prior gates remainmandatory. Last completed6144 blocks give44.7k/44.6kactive collection-plus-PPO and38.5k/38.4kworker-effective samples/s,minGPU669MiB,host7556MiB. Same resumedblock068 runs6144; read freshACTIVE_RUN/state/progress ratherthan this datedbaseline. Target8192,reserve300,hostguards,fallback/full-state recovery,180continuoushealthy live-training seconds remain. Update sameadaptive3h heartbeat aftercheck. Leavecomputeron; pause-and-brief endpointunchanged.'''
args={'id':current['id'],'mode':'update','kind':current['kind'],'name':current['name'],
    'prompt':durable+focus,'rrule':current['rrule'],'status':current['status'],'destination':'thread','targetThreadId':current['target_thread_id']}
if 'notification_policy' in current:args['notificationPolicy']=current['notification_policy']
elif 'notificationPolicy' in current:args['notificationPolicy']=current['notificationPolicy']
write(OUT/'automation_update_args.json',args)
print(json.dumps({'report':str(OUT/'REPORT.md'),'videos_delivered':5,'same_supervisor_resumed':True,
    'current_worker_iteration':main.get('iteration'),'recording_request':request['status'],'automation_args':str(OUT/'automation_update_args.json')}))
