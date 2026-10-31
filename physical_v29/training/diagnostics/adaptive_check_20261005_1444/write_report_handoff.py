"""Persist the completed check and prepare an update of the existing heartbeat."""
import json,tomllib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
e=read(OUT/'evidence.json');live=read(OUT/'final_owned_observation.json');d=e['decision'];g=e['gates'];rates=e['rates'];matched=read(OUT/'matched_comparison.json')
run=Path(d['run']);s=live['state'];now=e['recorded_utc'];head=[g[k][-1]['head_combined_RMS_deg'] for k in ['nominal_cases','pushed_cases']]
report=f'''# October 5, 14:44 UTC adaptive check — startup recovery and rejected gradient candidate

Training is running in **R37_offline_ground_full_state**, continuing the exact R36 checkpoint 080 actor, critic, Adam and exploration. It preserves **7,361 earlier genuine main updates**, including 1,536 from R36. At {now}, R37 has {s['total_training_iterations']} completed new main updates. The newest complete independent pair belongs to `{Path(g['checkpoint']).parent.name}`; stopped/prior cases all pass: **{g['all_prior_pass']}**. Running case passes: **{g['running_pass']}**. Full running-stage qualification: **{d['running_qualified']}**. No 5 m/s or actuator-ceiling endpoint is established.

## Failure, exact recovery and startup validation

R36 stopped at 13:07 UTC during evaluation081 because Isaac's online NVIDIA ground USD could not be opened. Its supervisor and worker were absent when this check began. The last full candidate080 and all49 frozen hashes were intact. The active pointer still said running; the check corrected that stale metadata and preserved the failed logs/state. This was an asset startup failure, not evidence that PPO finished or a policy crash erased learning. [Initial evidence](initial_evidence.json), [original state](state_preserved_before_supersede.json).

The repair caches the **original, byte-exact** NVIDIA ground USD and its three texture references locally, verifies their hashes and changes only the spawner's `usd_path`. Ground size, collision, transforms, friction, restitution, material and all other spawner arguments are retained. The root SHA256 is `8a21c317d638d33a4e6c20c958a7ed8f4d8c5195efd7e20f1b85178b4a15638a`. Bundled OmniPBR MDL still resolves through Isaac's installed material search path; a relative S3 lookup returned404 and is preserved, not substituted with a new physical asset. This matches the documented ground-spawner interface in [Isaac Lab's primary documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.spawners.html).

The entire192-robot,20-second,32-trial-per-command nominal replay of checkpoint080 using online versus local ground is **bitwise identical for all51 saved arrays and all cases**. Ten CPU spawner/provenance checks, eleven strict migration acceptance/rejection checks and real4096-environment four-update PPO/full-state load pass. The supervisor's fresh32-robot physics preflight,64-environment PPO/load and reload checks also complete. Diagnostic smoke/capacity/A-B actors are discarded. [Physical parity](ground_physical_parity.json), [CPU tests](offline_ground_cpu_tests.json), [migration tests](offline_migration_cpu_tests.json), [full-load smoke](offline_full_state_smoke.json).

R37 starts the exact checkpoint SHA `e59b721540593883223ccc07d19aaca1d69cdd0b129f21b9aaecb37ff3a528ec`. One existing worker startup file changes; a local resolver module and five ground/provenance assets are added. Forty-six other authored frozen source files are byte-identical. The supervisor adds a narrowly hashed, evidence-pinned startup migration and inventories these new files. All55 authored hashes verify. Rewardv8,verifierv6,truthful400-input/23-output inference,teacher context,upper refinement,commands/history/actions,iteration4 execution,PPO settings,geometry,mass/contact and native CAD motor caps are unchanged. No optimizer reset, actor-only transfer or head rollback. [Pinned continuation proof](offline_ground_continuation_proof.json), [deployment](deployment.json).

## Current walking, head and running evidence

The earlier059 speed regression was real but no longer describes the newest actor. The fresh checkpoint080 baseline passes stopped and every prior walking command in nominal and pushed tests; retention remains nonmonotonic over training. Qualification always binds a particular saved actor and pair, not a newer actor.

Newest full matching R37 pair:

| Command m/s | Nominal signed speed | Pushed signed speed | Nominal/pushed case pass | True flight % nominal/pushed |
|---:|---:|---:|:---:|---:|
'''
for a,b in zip(g['nominal_cases'],g['pushed_cases']):
 report+=f"| {a['command']:.2f} | {a['signed_speed_m_s']:.3f} | {b['signed_speed_m_s']:.3f} | {a['passed']}/{b['passed']} | {100*a['geometric_flight_fraction']:.4f}/{100*b['geometric_flight_fraction']:.4f} |\n"
report+=f'''
Each pair is192 robots,20 seconds and32 independent trials per command, with the unchanged nominal/pushed/stopped/prior/direction/contact/gait gates. At1.5, combined head yaw/pitch RMS is {head[0]:.2f}/{head[1]:.2f} degrees. This retains the forward-head improvement from the original51.7-degree posture; natural arms are not certified. [Exact matching gates](closing_matching_gates.json). Independent walking success does not turn a grounded1.5 m/s gait into running.

## Teacher labels, objective gradients and bounded A/B rejection

Four actual4096-environment PPO minibatches passively capture current conditioned-teacher labels, truthful student histories and pre-auxiliary derivatives. Per command approximately52–79 teacher samples are selected; conditioning eligibility is approximately68–85% under the unchanged all-five-history-command/phase/head guard. Do not weaken that guard just to increase coverage. The moving-prior derivative spends3.76–5.39% of its sampled squared energy pulling raw student means toward a clipped teacher endpoint even when both execute the same clipped action. At1.0, this can account for6–10.7% of that command's error. Current raw running action clipping is around15%, not the earlier actor's38%.

Prior gradient norms are about0.40–0.49 versus running PPO5.97–8.61. Running/teacher cosine ranges−0.146 to+0.226, not consistently opposed; earlier-command PPO/teacher opposition is also not universal. CPU actor reconstruction differs by less than1.1e-5. These finite minibatches describe a numerical supervision mismatch, not the proven cause of running failure. [Actual gradients and coverage](current_gradient_analysis.json).

The [Clipped Action Policy Gradient paper](https://arxiv.org/abs/1802.07564) studies bounded Gaussian actions and equivalent clipped outcomes. Our staged candidate only zeros the existing walking-teacher derivative for already equivalent clipped endpoints. It is **not CAPG**, changes no likelihood/PPO estimator and carries no convergence guarantee. Sixteen targeted CPU fixtures pass, including exact unchanged stopped-command and in-range derivatives.

Both arms start the identical full checkpoint/Adam/exploration at 4,096 environments and seed 71. Each performs 64 then 128 cumulative genuine updates, with full 192-robot, 20-second nominal/pushed evaluations at each boundary. Full load equality and zero pre-update KL/command mismatch hold. Both preserve all walking cases; both produce **zero true running flight** at both boundaries. The candidate's 1.5 speed at 128 is 1.393/1.396 versus control 1.432/1.435 m/s, with head 3.28 versus control 3.62/3.63 degrees. It does not meet the declared retain-priors/improve-flight/full-running-case criterion and is **REJECTED**. No diagnostic learned actor or changed auxiliary reaches production. [CPU candidate checks](alias_cpu_tests.json), [64/128 matched comparison](matched_comparison.json).

## Target timing and new5ms motor audit

The baseline50Hz trace requests a42.4ms double-swing window at1.65Hz and0.43 duty. Pre-action and post-step reward phase differ by0.033 cycles, exactly one20ms control tick. Hip target/joint correlation lags are60ms; knees40ms. Ankle correlations are weak/negative and do not identify a useful lag. Loaded-foot fraction during intended flight remains100%, about316N total load. These descriptive periodic correlations suggest a timing test, not identified causality; walking also shows delays. [Phase audit](current_phase_audit.json).

New normal-limit physical cohorts use six **truthful fixed commands**,32 robots each,20 seconds, with free base/gravity/self-collision and no pulse, root support, pose/phase/observation rewrite or new actor. A passive post-actuator-compute hook records all4000 native5ms substeps through the exact CAD Jacobians/reductions and ordinary native output-joint caps. Each speed is diagnostic; the actor was trained only through1.5.

| Command m/s | Actual reset-frame forward speed | Survival | True flight % after5s |
|---:|---:|---:|---:|
'''
for row in e['probe']['groups']:
 report+=f"| {row['command']:.2f} | {row['mean_forward_world_speed']:.3f} | {100*row['survival']:.0f}% | {100*row['unloaded_geometric_flight_fraction']:.4f} |\n"
report+='''
There is **zero driving-speed clipping in all motors at0.65,1.0,1.5,1.75 and2.0** in this experiment. At1.5 the ankle p99 cap ratios are about0.70/0.69 for the faster ankle outputs; an isolated1.03 cap crossing is braking, not removed driving torque. At2.5 every robot eventually falls. Its isolated5–10ms driving clips include an ankle and elbow and do not establish a sustained locomotion ceiling. Post-fall cap spikes must not qualify an endpoint. These observations also do not rule out torque-speed droop, torque limits, posture, target timing or policy faults. The brief2.0 flight and pre-fall2.5 samples are not qualifications. [Physical speed response](physical_speed_response/result.json), [200Hz motor attribution](physical_speed_response/substep_motor_audit.json).

Coarse fall attribution finds 19 isolated driving-clip samples in alive control ticks and 19 in failed ticks at 2.5. The post-step alive mask cannot locate fall onset inside its terminal 20 ms tick. A peak cap crossing while still flagged alive is therefore also insufficient to identify a persistent plateau. [Fall attribution and its temporal limit](physical_speed_response/fall_attribution.json).

The primary timing lead remains [Periodic Reward Composition](https://arxiv.org/html/2011.01387v2), which uses phase-dependent force/velocity composition and smooth interval boundaries. A feasible complete-return trajectory or a matched learning improvement must precede any production timing/reward repair. No paper's numerical weights were copied this check.

## Genuine resumed learning and resources

'''
for b in rates['blocks']:
 report+=f"`{Path(b['folder']).name}`: {b['updates']} genuine main updates at{b['environments']:,}, exact actor/critic/Adam/exploration load, zero KL/command difference. Minimum free GPU{b['min_GPU_MiB']}MiB, host{b['min_host_MiB']}MiB, private growth{b['private_growth_MiB']}MiB and Torch allocation growth{b['Torch_allocation_growth_MiB']}MiB.\n\n"
report+=f'''Completed-main rates are {rates['active_collection_PPO_samples_s']/1000:.2f}k active collection-plus-PPO, {rates['main_worker_effective_samples_s']/1000:.2f}k main-worker-effective and {rates['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. Denominators differ. Whole chain includes R37 startup/capacity/gates; the preceding R36 asset failure and diagnostic downtime are not counted as R37 training. One actual worker; source hashes55. Current operation `{Path(s['operation_path']).name}` at{s['environment_count']} environments may be an evaluation count, not learning capacity. Target8192,300MiB reserve,host guards/fallback and180 continuous healthy training seconds before upward ramps remain. [Resumed full-state learning](resumed_learning_verification.json), [fresh owned observation](final_owned_observation.json).

## Preserved failed diagnostics and decision

The initial ownership recorder rejected an absent supervisor; its CPU-only failed-state branch now records a non-running checkpoint without claiming ownership. No runtime inference changed. An initial CPU analysis head-index shape error is preserved and corrected only in diagnostics. The first128 control launch found no convenience qualified-teacher certificate beside a direct-worker checkpoint before spawning CUDA; the empty operation/logs are retained. The corrected diagnostic runner reads the unchanged embedded certificate, verifies the exact qualified teacher path/hash, and lets the original strict worker validate it. It certifies the teacher, never the diagnostic student. The local ground's missing relative MDL lookup and one nonfatal graphics-interface cache log are preserved; all completed physics/PPO/recorded checks remain explicit.

**Decision:** continue the exact useful learner with verified offline startup maintenance. Reject clipping-equivalence fine-tuning for deployment because it does not improve flight. Preserve checkpoint080, all prior passing walking actors, earlier flight actor038, teachers, failed candidates, logs/hashes and unrelated unsaved Fusion work. Full-body GUI recordings are already fulfilled; no repeat is pending.

Next priorities:

'''
for i,p in enumerate(d['next_tests'],1):report+=f'{i}. {p}\n\n'
report+='Keep the same adaptive three-hour heartbeat active until independently verified actual5m/s or a causally demonstrated relevant velocity ceiling. At that endpoint save,pause training and this automation,brief the user and leave the computer on.\n'
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
marker='adaptive_check_20261005_1444'
brief=f'''<!-- {marker}_current -->
## Latest startup recovery and matched rejection: October5,{now}

Read {str(OUT/'REPORT.md').replace(chr(92),'/')} and fresh ACTIVE_RUN/state. R37_offline_ground_full_state retains exact R36checkpoint080 actor/critic/Adam/exploration,7,361 earlier main updates, now{s['total_training_iterations']} new updates. R36 failed opening the online ground; verified original local ground reproduces all51 physical arrays/cases bit for bit. Startup-only migration:55 hashes, unchanged rewards/physics/CAD caps/teacher/posture/PPO. Latest exact matching pair all stopped/prior through1.0 pass:{g['all_prior_pass']}; running passes:{g['running_pass']}. Head at1.5 about{head[0]:.2f}/{head[1]:.2f}deg; natural arms uncertified. Do not attribute saved gates to a newer actor.

Actual teacher-gradient endpoint aliases account for~4–5% sampled derivative energy, but a full-state4096-env64/128 A/B candidate gives zero flight just like control and is rejected. New passive200Hz actuator audit finds no driving-speed clipping at trained1.5 or truthful diagnostic commands through2.0; extrapolated2.5 falls and brief spikes do not prove a ceiling. Running phase/target timing remains a research lead. Next bounded constraint-aware complete-return trajectory search and matched learning/gates; preserve useful state and rejected variants. Main rates{rates['active_collection_PPO_samples_s']/1000:.1f}k active/{rates['main_worker_effective_samples_s']/1000:.1f}k worker-effective/{rates['whole_chain_samples_s']/1000:.1f}k whole-chain since R37 startup. Target8192/reserve300/host fallback/180healthy-second ramp, same adaptive3h heartbeat/computer-on endpoint remain. No new GUI request.

'''
for path in [HERE/'automation/THREE_HOUR_CHECK.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
 old=path.read_text(encoding='utf-8-sig')
 if f'<!-- {marker}_current -->' not in old:path.write_text(brief+old,encoding='utf-8')
entry=f'''\n\n<!-- {marker} -->
## October5,{now} — exact R37 offline startup recovery; bounded clipping candidate rejected

[Report]({(OUT/'REPORT.md').as_posix()}). R36 online ground startup failure at081,1,536 main+5,825 inherited, no live supervisor;49 hashes and exact checkpoint080 preserved. Cached original NVIDIA ground and3textures, path-only spawner replacement, all51 arrays/cases in192-robot20s replay bitwise identical.10 CPU asset tests,11 strict migration checks,4 genuine4096-env PPO smoke updates, fresh physical/PPO/reload validation pass. Exact full actor/critic/Adam/exploration cpSHA{d['source_sha256']} continues R37 with7,361 prior main updates. Startup-only runtime:worker+local module+5asset/provenance files;46 protected authored source files identical;55 hashes. No gradient/reward/cap/PPO/head rollback.

Four actual PPO minibatches diagnose3.76–5.39% walking-prior derivative energy on identical clipped endpoints; no consistent running/teacher opposition.16 candidate CPU fixtures; matched4096-env64/128 full-state A/B and192-robot20s nominal/pushed tests retain walking but yield zero flight in both. Candidate rejected, all diagnostic actors discarded. Primary leads CAPG https://arxiv.org/abs/1802.07564 (not implemented) and periodic force/velocity composition https://arxiv.org/html/2011.01387v2. Phase requests42ms double swing; hip/knee target correlations40–60ms, descriptive only. New192-robot20s truthful speed probe saves4000 native5ms CAD motor substeps: no driving-speed clipping through2.0,1.5 grounded;2.5 falls and5–10ms clips do not prove a ceiling. No pulses/input rewrites/root support/hybrids deployed.

R37{s['total_training_iterations']} completed main updates; latest exact pair prior passes{g['all_prior_pass']}, running passes{g['running_pass']}, head{head[0]:.2f}/{head[1]:.2f}deg. Rates{rates['active_collection_PPO_samples_s']/1000:.2f}k active/{rates['main_worker_effective_samples_s']/1000:.2f}k worker-effective/{rates['whole_chain_samples_s']/1000:.2f}k whole-chain; denominators and earlier failure/downtime exclusion documented. Real resumed block exactload/zeroKL/commandmismatch, oneworker/55hashes. Preserve failed certificate launch/indexing/asset diagnostics and CPU failed-status recorder repair. Next constraint-aware full-return/timing search plus matched transfer, fresh prior/head/flight gates and memory/full-state recovery. Same3h automation,8192target/300MiBreserve/180healthy ramp; save,pause,brief endpoint with computeron. Existing videos/unsaved Fusion/earlier history retained.
'''
for path in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
 if f'<!-- {marker} -->' not in path.read_text(encoding='utf-8-sig'):
  with path.open('a',encoding='utf-8') as f:f.write(entry)
nextpath=HERE/'automation/NEXT_CHECK.json';old=read(nextpath)
if not (OUT/'next_check_before.json').exists():write(OUT/'next_check_before.json',old)
nextcheck=dict(recorded_utc=now,automation_id=old['automation_id'],active_run=str(run),report=str(OUT/'REPORT.md'),
 latest_observation=str(OUT/'final_owned_observation.json'),new_main_updates=s['total_training_iterations'],inherited_main_updates_preserved=7361,
 selected_actor=dict(checkpoint=g['checkpoint'],sha256=g['checkpoint_sha256'],matching_gate_pair=str(OUT/'closing_matching_gates.json'),
 full_actor_critic_Adam_exploration=True,standing_and_walking_cases_through_1_m_s_pass=g['all_prior_pass'],running_qualified=d['running_qualified']),
 running_qualified=d['running_qualified'],velocity_ceiling_proven=False,focused_tests=d['next_tests'],latest_repair_decision=d,
 throughput={k:v for k,v in rates.items() if k!='blocks'},resumed_learning_verification=str(OUT/'resumed_learning_verification.json'),
 latest_verified_state=dict(recorded_utc=now,status=s['status'],stage=s['stage'],operation=s['operation_path'],actual_workers=live['actual_worker_pids'],
 frozen_hashes_verified=55,gpu=live['gpu'],host=live['system_memory'],environments=s['environment_count'],learning_environments=s.get('next_training_environment_count')),
 endpoint=old['endpoint'],recording_request=old['recording_request'],latest_gate_trends=str(OUT/'closing_matching_gates.json'),
 physical_speed_response=str(OUT/'physical_speed_response/result.json'),substep_motor_audit=str(OUT/'physical_speed_response/substep_motor_audit.json'))
write(nextpath,nextcheck)
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text(encoding='utf-8'))
assert cfg['id']==old['automation_id'] and cfg['kind']=='heartbeat' and cfg['rrule']=='FREQ=HOURLY;INTERVAL=3'
write(OUT/'automation_current_before_update.json',cfg)
prefix=cfg['prompt'].split('Latest dated focus',1)[0].rstrip()
focus=f'''Latest dated focus, October5,{now}: R37_offline_ground_full_state continues exact R36checkpoint080 SHA{d['source_sha256']} actor,critic,Adam and exploration after7,361 earlier main updates. R36 failed at081 on an online NVIDIA ground USD startup load; all49 original hashes and saved learning were intact. Verified original local ground reproduces every51-array/case192-robot20s replay bit for bit. Startup-only maintenance passes10asset/11migration CPUchecks, real PPO/load and fresh supervisor preflight/PPO/reload.46 protected authored source files identical;55 hashes verify. Rewards v8/verifier v6,teacher context,posture module,PPO,commands/history/actions,geometry/contact/CAD caps and iteration4 remain. No diagnostic actor,gradient change or optimizer reset deployed. Read diagnostics/adaptive_check_20261005_1444/REPORT.md,evidence.json,offline_ground_continuation_proof.json,deployment.json,matched_comparison.json,current_gradient_analysis.json,current_phase_audit.json,physical_speed_response/result.json,physical_speed_response/substep_motor_audit.json,resumed_learning_verification.json and closing_matching_gates.json. Read fresh ACTIVE_RUN/state rather than this dated observation.

Fresh R37{s['total_training_iterations']} main updates; latest exact pair {Path(g['checkpoint']).parent.name} stopped/prior through1.0 all pass:{g['all_prior_pass']}; running case pass:{g['running_pass']}. Running signed speed{g['nominal_cases'][-1]['signed_speed_m_s']:.3f}/{g['pushed_cases'][-1]['signed_speed_m_s']:.3f}, true flight{100*g['nominal_cases'][-1]['geometric_flight_fraction']:.4f}%/{100*g['pushed_cases'][-1]['geometric_flight_fraction']:.4f}%, headRMS{head[0]:.2f}/{head[1]:.2f}deg. Natural arms uncertified. Gates bind that exact actor, not newer updates. No full running-stage or5m/s qualification inferred. Newest R36checkpoint080 had recovered all prior cases; old059 regression is dated, retention remains nonmonotonic.

Four actual PPO minibatches reveal~4–5% teacher derivative energy on raw means with identical clipped endpoints, conditioned-label eligibility~68–85%, no consistently opposed running/teacher parameter gradient.16 CPUfixtures and matched full-state4096env64/128-update control/candidate pairs retain walking but both yield zero flight. Candidate rejected and all diagnostic learned actors discarded. Do not repeat it as a running repair, nor rejected running-row transfer or44% exposure without new evidence. Truthful current student observations and exact existing teachers remain; no CAPG estimator is deployed.

New192-robot20s physical truthful-command probe and4000 native5ms motor substeps find zero driving-speed clipping at0.65,1.0,1.5,1.75,2.0. At1.5 the actual speed is1.378m/s,zero flight,ankle p99 cap ratios about.70/.69. At2.0 untrained extrapolation reaches1.671m/s and.154% flight; no qualification. At2.5 every robot eventually falls; isolated5–10ms ankle/elbow driving clips and post-fall cap spikes do not prove a relevant ceiling. No pulses,root support,pose/phase/observation rewrite or hybrid deployed. Current intended flight window is42ms; descriptive hip/knee target-to-joint lags40–60ms and one20ms action/reward clock offset require correct timing interpretation, not immediate weights/phase edits. This closes the prior substep gap for this actor/experiment but is not high-speed ceiling proof.

Main rates{rates['active_collection_PPO_samples_s']/1000:.2f}k active/{rates['main_worker_effective_samples_s']/1000:.2f}k worker-effective/{rates['whole_chain_samples_s']/1000:.2f}k whole-chain since R37 startup. Completed resumed blocks retain exact full state,zero KL/command mismatch. One actual worker and55 hashes; target8192/reserve300/host guards/fallback/180continuous healthy training-second ramp remain. Capacity/smoke updates are discarded, not main progress. R36 failure/downtime/diagnostic tests precede R37 and are excluded from R37 clocks. Current operation{Path(s['operation_path']).name} count{s['environment_count']} may be evaluation rather than learning. Full-body videos already fulfilled; no repeat pending. Preserve unrelated unsaved Fusion work.

Next priorities: (1) {d['next_tests'][0]} (2) {d['next_tests'][1]} (3) {d['next_tests'][2]} Preserve all learned posture/leg/walking state, no from-scratch restart or tilted-head rollback. Do not spend the next check only waiting. Keep this same adaptive three-hour heartbeat active until verified actual5m/s or causally established relevant velocity ceiling, then save,pause training and this automation,brief with computeron.'''
args=dict(mode='update',id=cfg['id'],kind='heartbeat',name=cfg['name'],status='ACTIVE',rrule=cfg['rrule'],targetThreadId=cfg['target_thread_id'],prompt=prefix+'\n\n'+focus)
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
write(OUT/'automation_update_args.json',args)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),new_main=s['total_training_iterations'],prior_pass=g['all_prior_pass'],running_pass=g['running_pass'],automation_args=str(OUT/'automation_update_args.json')),indent=2))
