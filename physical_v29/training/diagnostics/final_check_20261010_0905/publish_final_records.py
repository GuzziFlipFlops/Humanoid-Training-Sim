"""Publish verified final exports and durable stopped handoff. No simulator imports."""
import csv,gzip,hashlib,json,os,shutil,tomllib,re
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;T=OUT.parents[1];R=T/'runs/locomotion_20261007_r40_goal_budget_full_state'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readable(text):
 # Space prose/units while preserving exact URLs, hashes, IDs and source paths.
 replacements={'CADvideos':'CAD videos','CADmeshes':'CAD meshes','GPU/host':'GPU/host','Unitreeencoder':'Unitree encoder','Fusionwork':'Fusion work','Fusionqueuedwork':'Fusion queued work','QAreceipt':'QA receipt','QApass':'QA pass','KL/command':'KL/command','PPOiteration':'PPO iteration','PPOcapacity':'PPO capacity','FKparity':'FK parity','NPZmotor':'NPZ motor','CSVdata':'CSV data','CSVtables':'CSV tables','PTSs':'PTS','PTSstarts':'PTS starts','Hzmotor':'Hz motor','Hzrobots':'Hz robots','gait-linked':'gait-linked','seededstates':'seeded states','outputsand':'outputs and','state64':'state 64','fullstate':'full state','measured3':'measured 3','native200':'native 200','original35':'original 35','R 40':'R40','R 41':'R41','R 39':'R39','SHA 256':'SHA256'}
 lines=[]
 for line in text.splitlines():
  if line.startswith('<!--'):lines.append(line);continue
  parts=re.split(r'(`[^`]*`|\]\([^)]*\)|\b\d{4}-\d{2}-\d{2}T[^ ,;]+)',line)
  for i in range(0,len(parts),2):
   s=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',parts[i])
   for a,b in replacements.items():s=s.replace(a,b)
   s=re.sub(r',(?=[A-Za-z])',', ',s);parts[i]=s
  lines.append(''.join(parts))
 return '\n'.join(lines)+'\n'
now=datetime.now(timezone.utc).isoformat();V=Path(read(OUT/'capture_config.json')['destination'])
manifest=read(V/'manifest.json');decode=read(OUT/'decode_verification.json');visual=read(OUT/'video_visual_QA.json')
assert decode['passed'] and len(decode['clips'])==10 and visual['all_ten_clips_inspected']
assert not (V/'failure.json').exists()
assert read(R/'state.json')['status']=='stopped_by_user_budget'
for p in psutil.process_iter(['pid','cmdline'],ad_value=None):
 cmd=p.info.get('cmdline') or []
 assert not any(Path(a).name in ('worker_unitree.py','eval_entry.py','record_motor_gui.py') for a in cmd),p.info
 assert not any('supervisor_unitree.py' in a and str(R).replace('\\','/') in ' '.join(cmd).replace('\\','/') for a in cmd),p.info
automation_file=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
automation=tomllib.loads(automation_file.read_text());assert automation['status']=='PAUSED'
gates=read(OUT/'gate_assessment.json');timing=read(OUT/'final_timing.json');state_save=read(OUT/'full_state_save_receipt.json')
finalhash=sha(state_save['saved_copy']);assert finalhash==state_save['sha256']
inventory=read(T/'history/inventory.json') if (T/'history/inventory.json').exists() else None
rows=[];episode_summaries=[]
for clip in manifest['clips']:
 label=f"locomotion_{round(clip['speed']*100):03d}";folder=V/label
 assert (folder/'motor_population_or_episode_summary.json').exists()
 assert clip['falls']==0 and clip['maximum_command_difference']<1e-6
 alignment=read(folder/'frame_alignment.json');assert len(alignment)==500
 with (folder/'frame_alignment.csv').open('w',encoding='utf-8',newline='') as f:
  fields=['frame','video_time_s','post_step_sim_time_s','motor_first_sample','motor_last_sample','render_state_after_motor_interval_end_s','terminated','alive']
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(alignment)
 summary=read(folder/'motor_population_or_episode_summary.json')
 key=next(k for k in summary['summaries'] if abs(float(k)-clip['speed'])<1e-6)
 for motor in summary['summaries'][key]:episode_summaries.append(dict(command_m_s=clip['speed'],episode_measured_signed_speed_m_s=clip['mean_signed_world_forward_speed_m_s'],actor_sha256=clip['sha256'],scope='single exact video episode; not 32-trial population',**motor))
 rows.append(f"| {clip['speed']:.2f} | {clip['mean_signed_world_forward_speed_m_s']:.4f} | [video]({label}.mp4) | [CSV]({label}/motor_samples_{round(clip['speed']*100):03d}.csv.gz), [NPZ]({label}/native_motor_200hz.npz) | [frame/sample map]({label}/frame_alignment.csv) |")
write(V/'video_episode_motor_tables.json',episode_summaries)
with (V/'video_episode_motor_tables.csv').open('w',encoding='utf-8',newline='') as f:
 fields=[k for k in episode_summaries[0] if k!='shaft'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(episode_summaries)
readme='''# Final real-time videos and synchronized installed-motor data

Training and the native automation are stopped at the user's final-check budget. These are fresh evaluation recordings; no learner was resumed. All ten clips use preserved R40 actor1064, SHA256 `0ad6a1cc1c018951b82d05c67d02ca7a2663eb93845030e255a04503c60a61d0`. Native motor caps, gravity, free base, truthful deterministic inference, local ground and all 140 CAD visual meshes are retained. Each clip contains 10 simulated seconds at 50fps/500frames and plays in 10seconds; wall rendering is slower and is recorded separately in manifest.json.

The 3.5m/s command passes its current-speed nominal/pushed cases, including fresh32 independently seeded initial states per command, at3.4945/3.4921m/s signed fixed-frame travel. This actor's full stage remains unqualified:0.15path efficiency,1.5geometric flight and2.5foot slip fail. The grounded1.5m/s episode is not qualified running. Captions include actual episode speed, command, actor hash and the command-specific fresh nominal gate. Episode averages below are separate from20-second population gate statistics.

| Command m/s | Exact video episode signed speed m/s | Normal-time video | Synchronized motor samples | Alignment |
|---:|---:|---|---|---|
'''+ '\n'.join(rows)+'''

Installed motor shafts are measured before external transmissions. Individual ankle/waist crank shafts use exact CADtheta/J(q), requested/applied motor torque and velocity; they are not pitch/roll pivot values. DJI hips use bare motor shafts before15:1; knees16:1; shoulder pitch uses factory-geared output before4:1belt; other factory motors/servos use installed output. Motor_mapping.json in each episode records all references and modeled efficiencies. Inaccessible internal rotor torque is unavailable; angle zeros are relative/uncalibrated. Power is signed mechanical P=tau*omega at the same shaft/time, not battery/electrical or hardware sensor power.

Each episode NPZ retains2400 native5ms samples including400settling samples; measured video corresponds to2000 samples. Video frame i maps to samples400+4i through403+4i. Motor state/torque is at each pre-application interval start; rendering is after all four intervals, at2.02+0.02i simulated seconds, while videoPTSstarts0. The raw arrays include motor angle/rad/s/RPM/requested+applied torque/power, joint q/qd/targets, CADstates, action/command, contact, clearance, COM/root/body states, validity and terminal/alive masks. CSV retains excluded settling rows and reasons. No falls/resets occurred in these recordings. Every encoded frame was decoded and timestamp/sample alignment checked; five frames per clip were visually inspected with the full CAD body visible.

The [single-episode motor table](video_episode_motor_tables.csv) is distinct from the [32-trial population torque/RPM/power tables](../../diagnostics/final_check_20261010_0905/MOTOR_TABLES.md). Population [CSV](../../diagnostics/final_check_20261010_0905/motor_tables.csv), [NPZ](../../diagnostics/final_check_20261010_0905/selected_diverse_nominal/native_motor_200hz.npz), [1m/s rawCSV](../../diagnostics/final_check_20261010_0905/selected_diverse_nominal/motor_samples_100.csv.gz) and [3.5m/s rawCSV](../../diagnostics/final_check_20261010_0905/selected_diverse_nominal/motor_samples_350.csv.gz) retain independently seeded initial-state statistics. Torque and RPM peaks can occur at different times; the power export gives simultaneous torque/RPM at peak absolute power.

[Final report](../../diagnostics/final_check_20261010_0905/REPORT.md), [manifest](manifest.json), [decode/alignment verification](../../diagnostics/final_check_20261010_0905/decode_verification.json), [visual QA](../../diagnostics/final_check_20261010_0905/video_visual_QA.json). All earlier evidence/videos and failed capture attempts are preserved. Computer on; Fusion remains closed with queued work preserved.
'''
(V/'README.md').write_text(readable(readme),encoding='utf-8')
failure_lines=[]
for comparison in gates['comparisons']:
 fail='; '.join(f"{c['command']:g}: {', '.join(c['failures'])}" for c in comparison['cases'] if c['failures'])
 c=next(c for c in comparison['cases'] if c['command']==3.5)
 failure_lines.append(f"| {comparison['scope']} / {comparison['test']} | {c['signed_speed']:.4f} | {c['world_lateral']:.4f} | {fail} |")
report=f'''# Final training attempt: stopped, no reliable speed breakthrough

At{now}, this attempt is complete under the user's final-check budget. The same native automation is PAUSED with zero remaining authorized scheduled prompts. The verified R40supervisor39000 was retired after a saved full-state boundary and owned evaluation drain at2026-10-10T09:09:47.424133Z. All later Isaac work was evaluation/capture only. No learning is authorized after this report, no automatic restart is allowed, and the computer remains on. A budget stop is not locomotion success or an actuator ceiling.

## Result and why advancement stalled

The final saved actor1226 reaches3.5339/3.5410m/s signed fixed-frame speed at command3.5, but lateral drift0.2022/0.1840m/s fails both direction gates. Fresh independently seeded nominal/pushed tests reproduce this. Retained0.35tracking and1.5genuine/geometric flight also fail; pushed0.15path efficiency fails. Relative to reviewed1157's3.420/3.434m/s, the final actor has a smaller measured speed gain, but it does not preserve the full locomotion contract and is not a breakthrough. Earlier selected actor1064 passes current3.5cases at3.4938/3.5016m/s, freshly reproduced at3.4945/3.4921 with32 unique independently seeded states/command. Its0.15path,1.5geometric flight and2.5slip failures prevent complete3.5qualification. The curriculum blocks promotion when retention/direction fail; more command exposure or elapsed time cannot certify advancement. The highest separately certified complete earlier milestone remains3.0m/s. Earlier4/4.5/5extrapolation failed gait/direction. No causally established native motor ceiling, qualified4fallback or5endpoint exists.

The substantial command_conditioned_advantage_v1 iteration was executed and rejected: two learning seeds347/359, exact full-state64/128control comparisons and independently seeded initial states did not pass the full gate contract. Paired starting observations/actions/rewards/dones/returns/raw advantages were identical; only the declared normalized advantages changed. Reward/GAE/critic targets, physical/native caps, teachers, upper20batch refinement and inference stayed original. This local optimizer experiment does not establish unbiasedness for the original mixed objective or a locomotion guarantee. No experimental actor was deployed. All20 matched128upper plots were inspected during this final turn; the earlier pending flag is now closed by its QAreceipt. The previously tested reward/teacher/arm/gain branches remain preserved, without pretending they solved the plateau.

| Exact actor / independent test | Signed3.5speed m/s | Lateral m/s | Failed commands and unchanged classifiers |
|---|---:|---:|---|
'''+ '\n'.join(failure_lines)+f'''

[Full gate assessment](gate_assessment.json), [32-state diversity](selected_diverse_nominal/initial_diversity.json), [R41decision](../substantial_iteration_20261010_0405/decision.json), [recent exact pairs](recent_gate_pairs.json). All320robots in each20-second gate survived; survival alone is insufficient. Passing cases belong to the exact hashed actor. No nominal-clone invocation is described as independently seeded diversity.

## State preservation and actual training work

Final full actor/critic/Adam/exploration checkpoint: [saved copy](final_full_state_checkpoint.pt), SHA256`{finalhash}`. [Save receipt](full_state_save_receipt.json) and [copied standing/qualified349teachers and selected1064actor](preserved_actors_and_teachers/receipt.json) preserve both teachers and the video actor with exact hashes. The freshly requalified neutral-head267, certified349 and independently useful earlier fast actors remain in their original immutable paths. All55frozen source hashes and53protected authored worker files verify. Original35/1shoulder gains,400-input/23-output inference, native CAD caps, rewardv8/verifierv6, local ground and iteration4execution remain unchanged.

The stopped run contains26,023completed main updates plus50exactly received memory-recovery updates=26,073current-run retained;27,507inherited updates bring the retained chain to53,580. Actual current-run samples are{timing['completed_main_samples']:,}completed+{timing['recovered_samples']:,}recovered={timing['total_current_run_retained_samples']:,}. R39's32unsaved updates and all capacity/smoke/diagnostic learning are excluded. Through{timing['measurement_endpoint_utc']}, rates are{timing['active_PPO_samples_s']/1000:.3f}kactive collection+PPO /{timing['worker_effective_samples_s']/1000:.3f}kworker-effective /{timing['whole_chain_samples_s']/1000:.3f}kwhole-chain samples/s. Denominators differ;{timing['documented_pause_seconds_excluded']:.3f}s of documented suspensions through that endpoint are excluded. Evaluation/capture time after stop is outside training throughput. An earlier unmeasured brief bookkeeping pause remains included conservatively. [Exact accounting](final_timing.json), [full load/hash checks](final_learning_evidence.json).

Latest real main block used8,192environments with exact actor/critic/Adam/exploration load, zero pre-updateKL/command mismatch and all three auxiliaries20batches/update. Minimum GPU/host availability was{timing['last_main']['minimum_free_gpu_MiB']}/{timing['last_main']['minimum_host_available_MiB']}MiB. Reserve300MiB, host guards/fallback and180continuous healthy-training-second ramps remained. Evaluations'320robots do not establish PPOcapacity. Those capacity settings are preserved as historical configuration; they do not authorize another run.

## Fresh synchronized recordings and motor use

Ten NEW normal-time full-CAD clips cover stopped,0.15,0.35,0.65,1,1.5,2,2.5,3and3.5m/s commands. Each10sclip has500frames at50fps and2400native200Hzmotor samples including2ssettling. Frame/sample timing, original actor SHA, actual measured episode speed and failed gates are saved/burned into captions. Every video fully decodes with exact timestamps,500distinct physical/image frames and no resets/falls. All140CADmeshes retain source colors; visual material instance repair is confined to visual scopes and preserves mass/inertia/joints/collisions. [All clips/data](../../videos/final_motor_capture_20261010_0905/README.md), [1m/s](../../videos/final_motor_capture_20261010_0905/locomotion_100.mp4), [maximum stable running](../../videos/final_motor_capture_20261010_0905/locomotion_350.mp4), [decode/sample audit](decode_verification.json), [actual decoded visual QA](video_visual_QA.json).

[Per-motor torque/RPMtables](MOTOR_TABLES.md) contain23individual installed shafts at1m/s and measured3.4945m/s, with32independent trials per command and115,200valid post-settle samples/motor. [Machine-readable torque/RPM/power](motor_tables.csv) includes RMS/absolutep95/peak torque, absolutep95/peakRPM, signed/positive/braking mechanical power and the simultaneous torque/RPMpair at peak power. The population summary is separate from each exact video episode. [Motor mapping](selected_diverse_nominal/motor_mapping.json), [raw populationNPZ](selected_diverse_nominal/native_motor_200hz.npz), [1m/sCSV](selected_diverse_nominal/motor_samples_100.csv.gz), [maximumCSV](selected_diverse_nominal/motor_samples_350.csv.gz). Differential motors use individual CADtheta/J(q)/requested+applied torque states. DJIhips/knees use bare shafts before15:1/16:1; shoulder pitch uses factory output before4:1belt; others use installed factory/servo output. Relative zeros are uncalibrated and internal rotor torque for geared motors/servos is unavailable. SignedP=tau*omega is simulated mechanical shaft power, not measured battery/electrical/thermal power. Independent torque/RPM peaks are never combined as one operating point.

The passive sampling wrapper reproduces every original physical replay array and evaluation case bitwise on latest1226: [parity receipt](passive_capture_parity.json). Original actuator execution is unchanged. Capture setup failures (visual instance binding, inference-mode reset and integer done-mask casting) are archived with logs and ownership cleanup. The done-mask fix converts the wrapper's integermask tobool only; it does not change simulator actions/dones. One prematurely launched failed evaluation briefly overlapped a draining context before ownership cleanup; its data were discarded. Successful final evaluations/captures ran sequentially with one actual CUDAworker. No capture/export failure resumed learning.

## Upper-body joint-time audit and limits of interpretation

All-command nominal/pushed actual headyaw/pitch, shoulderpitch/roll and elbow angle/physical target/velocity timelines were inspected for latest1226and selected1064, with full32-per-command10Hzpopulation statistics and16representative50Hzrobots per gate, reset-segmented. Latest3.5neckRMS is3.059/3.062degrees; shoulder mean pitch remains approximately-20.7/-1.1degrees, ranges13.18/8.67degrees nominal and arm/ipsilateral-hip phase123/137degrees at2.667Hz. Selected1064has unequal held biases and swing/roll/elbow coordination too. Slow2.5/3m/s bias reversals and gait-linked neck oscillation persist. [Latest angle/target plots](latest_upper/nominal/fast/upper_joint_angles.png), [velocities](latest_upper/nominal/fast/upper_joint_velocities.png), [bias](latest_upper/nominal/bias/upper_bias_decomposition.png), [QAreceipt](upper_visual_QA.json). Biashelper images in pushedfolders duplicate nominaldata and are explicitly labeled nominal; actual pushed angle/velocity/statistics were independently checked. Small native neckangles are torso-relative, not proof of world-forward optical gaze. Isaac-head/FKparity is known; face/camera optical calibration remains unavailable. Primary human comparisons remain feature-based/morphology-limited; no mapped speed-matched official Unitreeencoder curves were obtained or fabricated. Human-like motion is not certified from bounds/reward/videos.

## Final receipts

[Stop/owned process receipt](stop_receipt.json), [no restart marker](../../runs/locomotion_20261007_r40_goal_budget_full_state/NO_AUTOMATIC_RESTART.json), [final process/source/automation verification](final_status_receipt.json), [artifact hashes](artifact_hashes.json). Local iteration history, journals, skill handoff, file locations andNEXT_CHECKare marked final/stopped; CPU-only history inventory is refreshed. Zero further scheduled prompts and no automatic continuation are authorized. All original evidence, videos, queued Fusionwork and learned head/gait state are preserved. Computer stays on.
'''
(OUT/'REPORT.md').write_text(readable(report),encoding='utf-8')
decision=dict(recorded_utc=now,status='complete_user_budget_stop',reliable_speed_breakthrough=False,qualified_5_or_4=False,causal_actuator_ceiling=False,automation_status='PAUSED',remaining_scheduled_prompts=0,learning_authorized=False,computer_on=True,final_checkpoint=str(state_save['saved_copy']),final_checkpoint_sha256=finalhash,selected_video_checkpoint_sha256=manifest['clips'][0]['sha256'],video_count=10,motor_shafts=23,population_trials_per_command=32,native_sample_Hz=200,report=str(OUT/'REPORT.md'))
write(OUT/'decision.json',decision)
marker='<!-- final_check_20261010_0905_completed -->'
entry=f'''\n\n{marker}\n## October10 final authorized check — training and automation stopped\n\nAt{now}, the single final check is complete under the user budget. R40supervisor39000retired after full saved boundary1226at09:09:47Z; no learning resumed during exports. Same native automationPAUSED, zero remaining prompts, no automatic restart. Final actor3.5339/3.5410m/s still fails direction and prior gates; preserved1064fresh32-state currentcase3.4945/3.4921passes but0.15path/1.5flight/2.5slip retention fails. No reliable breakthrough,4/5qualification or causal ceiling. Substantial command-conditioned PPOiteration(two64/128learning seeds)executed/rejected; all20upperplots now visually inspected. Full actor/critic/Adam/exploration, both teachers and all55hashes/53authoredfiles preserved. 26,023completed+50recovered=26,073current retained,+27,507inherited=53,580;4,293,328,896current samples. Rates36.063kactive/29.807kworker/19.201kchain through09:06:06Z;53,218.311sdocumented suspensions excluded.\n\nNEW10normal-time full-CADvideos and synchronized native200Hz23installed-motor angle/velocity/requested+appliedtorque/signed mechanicalpower exports are complete.32independent states per1/3.5command provide populationtables; separate single-video episodes have exactframe/samplemaps. Full decode, visible140meshes,500frames/10s, timing/power and actualupperangle/target/velocity/biasQApass; heldbias/asymmetry remain, human/Unitree/opticalgaze uncertified. Archived setupfailures fixed without learning; discarded evaluation overlap explicitly recorded. All earlier evidence/videos/Fusionqueuedwork preserved,computeron.\n\n[Final report]({OUT.as_posix()}/REPORT.md), [new videos]({V.as_posix()}/README.md), [installed-shaft table]({OUT.as_posix()}/MOTOR_TABLES.md), [full checkpoint]({OUT.as_posix()}/final_full_state_checkpoint.pt). No next training tests or scheduled continuation are authorized; later work requires new explicit authorization.\n'''
entry=readable(entry)
for p in [T/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
 text=p.read_text(encoding='utf-8-sig')
 if marker not in text:p.write_text(text+entry,encoding='utf-8')
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
 text=p.read_text(encoding='utf-8-sig')
 if marker not in text:p.write_text(entry+'\n---\n\n'+text,encoding='utf-8')
handoff=dict(recorded_utc=now,status='final_attempt_stopped_by_user_budget',automation_id=automation['id'],automation_status='PAUSED',target_thread_id=automation['target_thread_id'],remaining_scheduled_prompts=0,learning_authorized=False,resume_requires_new_user_authorization=True,run=str(R),checkpoint=state_save['source'],saved_checkpoint=state_save['saved_copy'],checkpoint_sha256=finalhash,retained_updates=timing['current_retained_main_updates'],inherited_updates=27507,throughput=timing,gate_assessment=str(OUT/'gate_assessment.json'),final_report=str(OUT/'REPORT.md'),videos=str(V/'README.md'),motor_tables=str(OUT/'MOTOR_TABLES.md'),next_focused_tests=[],historical_goal='5m/s or actuator-limited4 was not achieved; replaced by final user budget stop',computer_on=True)
handoff['throughput']={k:v for k,v in timing.items() if k not in ('blocks','intervals')}
write(T/'automation/NEXT_CHECK.json',handoff)
request=read(T/'automation/FINAL_CHECK_REQUEST.json');request.update(final_check_status='completed_user_budget_stop',completed_utc=now,remaining_scheduled_prompts=0,final_check_started=True,learning_stopped=True,automation_paused=True,report=str(OUT/'REPORT.md'),videos=str(V/'README.md'),motor_tables=str(OUT/'MOTOR_TABLES.md'),checkpoint_sha256=finalhash)
tmp=T/'automation/FINAL_CHECK_REQUEST.json.tmp';write(tmp,request);os.replace(tmp,T/'automation/FINAL_CHECK_REQUEST.json')
print(json.dumps(decision,indent=2))
