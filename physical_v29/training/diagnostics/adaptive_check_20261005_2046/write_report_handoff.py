"""Write evidence-backed local history and prepare the existing heartbeat update."""
import json,tomllib,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
e=read(OUT/'evidence.json');d=e['decision'];a=e['trajectory_analysis'];s=read(OUT/'final_owned_observation.json')['state']
g=e['gates'];rates=e['rates'];now=e['recorded_utc'];run=Path(e['run']) if 'run' in e else Path(d['run'])
score=read(OUT/'clipped_score_audit.json');smooth=read(OUT/'current_smoothed_counterfactual.json')
head=[g[k][-1]['head_combined_RMS_deg'] for k in ['nominal_cases','pushed_cases']]
report=f'''# Three-hour training check — October 5, 20:46 UTC

Evidence consolidated at {now}. R37 remains the same full-state learner and supervisor {s['supervisor_pid']}. **Running is unqualified.** Current exact checkpoint `{Path(g['checkpoint']).parent.name}` has matching nominal/pushed tests; stopped, 0.15, 0.65 and 1.0 m/s pass, while 0.35 overspeeds again. At the 1.5 command, signed speed is {g['nominal_cases'][-1]['signed_speed_m_s']:.3f}/{g['pushed_cases'][-1]['signed_speed_m_s']:.3f} m/s and genuine flight is {100*g['nominal_cases'][-1]['geometric_flight_fraction']:.4f}%/{100*g['pushed_cases'][-1]['geometric_flight_fraction']:.4f}%, below the unchanged 1.5% minimum. All robots survive. Head RMS is {head[0]:.2f}/{head[1]:.2f} degrees; natural arm motion is not certified. No 5 m/s or actuator-ceiling endpoint is established.

The tested fixed timing advances and box-smoothed force candidate are **rejected**. They supply no justified deployment. Production reward v8, verifier v6, truthful 400-input/23-output history, teacher context, upper-body module, native joint/CAD/motor caps, geometry/mass/contact, PPO and iteration 4 execution are unchanged. The full actor, critic, Adam and exploration are preserved; no optimizer reset or diagnostic actor is deployed. All 55 frozen hashes verify. [Decision](decision.json), [owned evidence](final_owned_observation.json).

## Exact independent qualification

Checkpoint SHA `{g['checkpoint_sha256']}`. Each test has 192 robots for 20 seconds, 32 trials per command. Results bind this saved actor; later learner updates are separately unqualified.

| Command m/s | Nominal/pushed signed speed | Nominal/pushed pass | Nominal/pushed true flight % | Failure |
|---:|---:|---|---:|---|
'''
for n,p in zip(g['nominal_cases'],g['pushed_cases']):
    report+=f"| {n['command']:.2f} | {n['signed_speed_m_s']:.3f}/{p['signed_speed_m_s']:.3f} | {n['passed']}/{p['passed']} | {100*n['geometric_flight_fraction']:.4f}/{100*p['geometric_flight_fraction']:.4f} | {', '.join(sorted(set(n['failures']+p['failures']))) or 'None'} |\n"
report+='''
Recent checkpoints 130 and 133 passed all earlier commands and had small 1.5-command flight fractions (133: 0.2361% nominal/0.3646% pushed). This is seed-sensitive and nonmonotonic: current 139 loses 0.35 speed retention and almost all flight. Preserve the independently passing actors, walking teacher and earlier 038 flight actor. Do not claim the newest actor retains all speeds or call short flight running. [Matching gates](closing_matching_gates.json), [gate trends](gate_trends.json), [last prior-passing actor](last_all_prior_passing_actor.json), [upper-body diagnostic history](upper_audit.json).

## Controlled physical timing test

The supervisor was suspended at saved checkpoint 136; its existing pushed evaluation was allowed to finish, then its worker drained before diagnostics. The exact actor is tested at seed 91, six fixed cohorts of 32 robots, each 20 seconds: zero offsets, previous best-flight offsets, the same offsets 20/40/60 ms earlier, and previous best-return offsets. This independently tests prior search ideas on the CURRENT actor. It does not pair identical initial states across cohorts or qualify a new policy.

The leg offsets use pre-action phase and fade in after 3 seconds. Amplitudes/widths are held fixed for the timing interventions; center bounds are explicitly expanded to [0,0.5), rather than silently treating new centers as the earlier search bounds. All offsets pass through the existing +/-3 action clamp, soft joint limits, slew limiter and CAD feasibility checks. Base/gravity/self collision are active; observations are truthful. No head override, root support, pose/phase write or hybrid is used. Every offset remains diagnostic only. Seven CPU-check categories pass; actor tensors load exactly, observed/action commands match and full reward sums reconstruct within 2e-6. [Timing selection](timing_selection.json), [CPU checks](timing_cpu_checks.json), [physical results](timing_probe/result.json), [telemetry](timing_probe/round_00.npz).

| Cohort | Signed speed m/s | True flight % after 5s | Complete 20s return difference | Head RMS degrees |
|---|---:|---:|---:|---:|
'''
for r in a['validation']:
    report+=f"| {r['label']} | {r['signed_forward_m_s']:.3f} | {100*r['geometric_flight_fraction']:.4f} | {r['return_delta']:+.3f} | {r['head_RMS_deg']:.2f} |\n"
report+='''
All 192 survive. Each earlier timing cohort is grounded and loses complete return. Original flight offsets achieve only 0.479% flight and lose 4.274 return units. The best-return offsets gain 1.958, but have only 0.133% flight. None meets the prerequisite flight and return together. The original flight offset now has a few late-in-window flight samples and mostly just-after-wrap samples; this is not the older finding that ALL flight was after wrap. Fixed 20/40/60 ms advances eliminate flight, so the descriptive delay does not justify a universal timing shift or declare the reward clock wrong. 50 Hz samples do not certify substep flight duration or contact pressure. These bounded failures do not prove physical running impossible. [Term/timing analysis](trajectory_analysis.json), [inspected plot](timing_tradeoff.png).

## Phase-smoothing counterfactual rejected before learning

[Periodic Reward Composition](https://arxiv.org/html/2011.01387v2) uses expected phase intervals to smooth force/velocity costs. Its Von Mises formulation and Cassie results do not supply numerical weights or a convergence guarantee for this robot. The new CPU candidate is explicitly a LOCAL box convolution of the existing summed-swing-force term, using full boundary widths 20/40/80 ms. The existing coefficient 0.5 and quarter-body-weight scale are held fixed. It is not the paper's exact algorithm and changes no dynamics.

Sixteen synthetic check categories cover finite/bounded/circular probabilities, preserved average swing fraction, monotone force response and zero-force bounds. Recomputed original term agrees with the actual saved reward. A first periodicity fixture failed at the discontinuous binary boundary from floating-point wrap; its log is preserved. Midpoint samples fix that diagnostic fixture; no runtime change is involved. Force-only tests cannot reject hopping, sliding or early exit, so this candidate is not represented as a fully validated reward.

On current recorded states, the original-flight trajectory's relative complete return remains -4.20/-4.17/-4.06 under the three smooth widths, versus -4.274 before smoothing. It remains below control. Near smoothed double-swing edges the two foot coefficients differ; redistributing the same total load changes quality by up to about 0.22 in these records. This reintroduces an edge incentive needing resolution. The candidate is rejected BEFORE a learning comparison. No 64/128 PPO candidate was run or claimed successful. [CPU checks](smoothed_phase_cpu_tests.json), [current counterfactual](current_smoothed_counterfactual.json), [earlier independent records](old_smoothed_counterfactual.json), [failed fixture](smoothed_first_failure.log).

## New action-bound diagnostic

Current unchanged-control running telemetry supplies 24,032 truthful states after 5 seconds and exact trained diagonal-Gaussian standard deviations. Raw actor means exceed the outer +/-3 clamp on 14.53% of joint outputs; predicted Gaussian probability outside it averages 14.93%. Per-joint predicted clipping is 57.8%/63.3% for left/right ankle pitch and 39.0%/73.2% for left/right ankle roll. Native joint/slew/CAD constraints can remove further target variation.

[Clipped Action Policy Gradient](https://proceedings.mlr.press/v80/fujita18a.html) gives an unbiased lower-variance estimator under its clipping assumptions. Six CPU synthetic cases check zero expected constant-return score and clipped-objective mean/std derivatives against finite differences at means -4,0,4. Analytic constant-return mean-score second moments on the saved states decrease by about 49%/56% at ankle pitch when tail scores are integrated. These are synthetic score moments, NOT measured actual advantage-weighted PPO parameter gradients or proof that clipping causes grounded locomotion. No CAPG estimator, new entropy objective, action transform or raised cap is deployed. PPO remains unchanged. The earlier teacher-clipping-equivalence experiment remains rejected; it is a different algorithm. [CPU audit and limits](clipped_score_audit.json), [runnable audit](clipped_score_audit.py).

The next test should passively compare real per-command advantages, teacher coverage/labels, effective targets and ordinary versus CDF tail scores before attempting a learning continuation. A candidate must retain inference/physics and pass current-full-state 64/128 matched control, every prior/head/direction gate and independent flight. State-dependent native constraints and PPO importance ratios must be handled explicitly; synthetic variance reduction is not deployment authorization by itself.

## Exact resume, recovery accounting and memory

'''
for b in rates['resumed_blocks']:
    report+=f"Resumed `{Path(b['folder']).name}` completes {b['updates']} genuine updates at {b['environments']:,}. Actor/critic/Adam/exploration load equality is exact; maximum pre-update KL and observed/action-command difference are zero. Minimum free GPU {b['min_GPU_MiB']} MiB, available host {b['min_host_MiB']} MiB, private growth {b['private_growth_MiB']} MiB and Torch allocation growth {b['Torch_allocation_growth_MiB']} MiB.\n\n"
report+=f'''The same supervisor resumes after this **{e['resume']['pause_seconds']:.1f}-second** pause. Both R37 diagnostic pauses total {rates['excluded_diagnostic_pause_seconds']:.1f} seconds and are excluded from whole-chain time. Pushed evaluation 138's wrapper includes this pause, so it is not a physics-throughput measurement. Current selected training capacity is 6,144; target 8,192 and the 300 MiB reserve remain. The live ramp requires about 2,091 MiB free GPU for the additional tier, not simply more than 300 MiB. It also requires healthy host RAM and 180 continuous healthy TRAINING seconds; startup/evaluation memory never qualifies.

An earlier 069 VRAM reserve breach saved `memory_recovery.pt` after 58 logged updates; 070's full-load proof verifies exact actor/critic/Adam/exploration from that checkpoint. Those 58 preserved updates are omitted from the supervisor's completed-operation counter. This is a reporting distinction, not lost training. Current completed-operation count is {s['total_training_iterations']}; this run actually retains {rates['actual_retained_main_updates']} main updates including that recovery, plus 7,361 declared inherited updates. The inherited ancestry's incomplete-operation count is not newly audited here. No frozen supervisor counter is edited merely to change the display.

Completed-operation rates remain {rates['active_collection_PPO_samples_s']/1000:.2f}k active collection-plus-PPO / {rates['main_worker_effective_samples_s']/1000:.2f}k worker-effective / {rates['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. Including the exact retained failed-operation samples gives **{rates['all_retained_active_samples_s']/1000:.2f}k active / {rates['all_retained_worker_effective_samples_s']/1000:.2f}k worker-effective / {rates['all_retained_whole_chain_samples_s']/1000:.2f}k whole-chain**, with {rates['actual_retained_main_samples']:,} actual retained main samples. Denominators are separate; no capacity/smoke/diagnostic frames are main progress. [Pause](diagnostic_pause.json), [resume](diagnostic_resume.json), [full-state/rates/recovery evidence](resumed_learning_verification.json).

One actual Isaac/CUDA worker and all 55 authored hashes verify. The previous actor-specific 200 Hz motor probe found no driving-speed clips through 2.0 m/s; brief clips after extrapolated 2.5 m/s falls do not prove the endpoint. This check adds no new motor-ceiling certification. Full-body videos are already complete; no new recording is pending. Unrelated unsaved Fusion work is preserved.

## Next focused checks

'''
for i,t in enumerate(d['next_tests'],1):report+=f'{i}. {t}\n\n'
report+='Keep the same adaptive three-hour heartbeat active. At verified actual 5 m/s or a causally established relevant velocity ceiling, save, pause training and this automation, brief the user and leave the computer on.\n'
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
marker='adaptive_check_20261005_2046'
brief=f'''<!-- {marker}_current -->
## Latest timing/score diagnosis — {now}

Read {(OUT/'REPORT.md').as_posix()} and fresh ACTIVE_RUN/state. Same R37 full-state learner, {s['total_training_iterations']} completed-operation main plus 58 recovered main updates; 7,361 declared inherited. Latest exact 139 pair loses 0.35 speed retention (0.464/0.459 m/s), while stopped,0.15,0.65,1.0 pass. Running signed {g['nominal_cases'][-1]['signed_speed_m_s']:.3f}/{g['pushed_cases'][-1]['signed_speed_m_s']:.3f}, true flight {100*g['nominal_cases'][-1]['geometric_flight_fraction']:.4f}%/{100*g['pushed_cases'][-1]['geometric_flight_fraction']:.4f}%; head {head[0]:.2f}/{head[1]:.2f} degrees. Natural arms uncertified. Preserve 133 and other passing actors; progress is nonmonotonic.

192-robot current-actor timing probe rejects 20/40/60 ms advances (zero flight, lower full return). Original offset:0.479% flight/-4.274 return; best-return offset:0.133%/+1.958. Diagnostic box smoothing does not repair full return and introduces edge load-redistribution credit; rejected before learning. New CPU clipped-Gaussian score audit is a research lead, not deployed or causal:14.53% raw means outside +/-3, ankle pitch predicted clipping57.8/63.3%. Actual advantage/PPO/teacher gradients require controlled tests before a matched64/128 continuation. No runtime/reward/physics/cap/optimizer/actor reset;55 hashes,one worker. Same supervisor resumes after337.6s; genuine64-update6144 block exact full-state load/zero KL/command mismatch. All retained main rates{rates['all_retained_active_samples_s']/1000:.1f}k active/{rates['all_retained_worker_effective_samples_s']/1000:.1f}k worker-effective/{rates['all_retained_whole_chain_samples_s']/1000:.1f}k whole-chain,excluding both R37 pauses. Target8192/reserve300/180healthy ramp retained;current6144. Same3h automation and computer-on endpoint remain.

'''
for p in [HERE/'automation/THREE_HOUR_CHECK.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
    old=p.read_text(encoding='utf-8-sig')
    if f'<!-- {marker}_current -->' not in old:p.write_text(brief+old,encoding='utf-8')
entry=f'''\n\n<!-- {marker} -->
## {now} — current-actor timing and smoothed force candidate rejected; bound-score diagnosis added

[Report]({(OUT/'REPORT.md').as_posix()}). Same R37 paused at136 SHA{d['paused_checkpoint_sha256']};worker drained. Six fixed cohorts,192 robots,seed91,20s,truthful current actor,normal limits. Seven CPU categories;full reward sum reconstruction and commands pass.20/40/60ms earlier offsets givezero flight/lower return. Original offsets0.479%flight/-4.274return,bestreturn0.133%/+1.958. Reject;notproof impossible running. Sixteen box-smoothing CPU categories and exact old-term reconstruction pass,but full-return deficit persists and smoothed edge redistribution gainscredit;reject BEFORE64/128learning. Firstbinaryperiodicity fixturefails atfloatingboundary;midpointfixturefixandfailedlog preserved. Six synthetic clipped-Gaussian score cases/24,032 truthful running states,14.53%meansoutsideclamp,anklepitch predicted57.8/63.3%;researchleadnotactualPPOvariance/causalproof/deployedCAPG. Primarysources linkedinreport.

Same supervisor resumes after337.6s,full actor/critic/Adam/exploration exact;64 actual6144updates zero KL/command mismatch.55 frozenhashes,noproduction source/reward/physics/cap/optimizer/actor reset. Exact139 gates:0.35failsboth,signed0.464/0.459;otherpriorspass;1.5flightfails,head{head[0]:.2f}/{head[1]:.2f}deg,natural armsuncertified. Preserve133/qualifiedteacher/038. Maincompleted{s['total_training_iterations']} plus58recoveredfrom069VRAMbreachandexact070reload;inherited7361declared. Actual retained rates{rates['all_retained_active_samples_s']/1000:.2f}k/{rates['all_retained_worker_effective_samples_s']/1000:.2f}k/{rates['all_retained_whole_chain_samples_s']/1000:.2f}k active/worker/wholechain,bothR37pauses excluded. Current6144,target8192/reserve300/180healthytrainingseconds. Nextactualper-commandadvantage/teacher/effectivetarget/CDFtailgradienttestsandmatched64/128beforecandidate;do notdeployoffsets/hybrids/spoofs/tiltedheador raise caps. Same3hactive,save/pause/briefcomputer-onendpoint;no5m/s/ceiling. Videosfulfilled,unsavedFusionpreserved.
'''
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    if f'<!-- {marker} -->' not in p.read_text(encoding='utf-8-sig'):
        with p.open('a',encoding='utf-8') as f:f.write(entry)
nextpath=HERE/'automation/NEXT_CHECK.json';old=read(nextpath)
if not (OUT/'next_check_before.json').exists():write(OUT/'next_check_before.json',old)
new=dict(recorded_utc=now,automation_id=old['automation_id'],active_run=str(run),report=str(OUT/'REPORT.md'),
    latest_observation=str(OUT/'final_owned_observation.json'),new_main_updates=s['total_training_iterations'],
    recovered_main_updates=58,actual_retained_current_run_main_updates=rates['actual_retained_main_updates'],inherited_main_updates_preserved=7361,
    selected_actor=dict(checkpoint=g['checkpoint'],sha256=g['checkpoint_sha256'],matching_gate_pair=str(OUT/'closing_matching_gates.json'),
        full_actor_critic_Adam_exploration=True,standing_and_walking_cases_through_1_m_s_pass=g['all_prior_pass'],running_qualified=d['running_qualified']),
    last_all_prior_passing_actor=str(OUT/'last_all_prior_passing_actor.json'),running_qualified=d['running_qualified'],velocity_ceiling_proven=False,
    focused_tests=d['next_tests'],latest_repair_decision=d,
    throughput={k:v for k,v in rates.items() if k not in ['blocks','resumed_blocks']},resumed_learning_verification=str(OUT/'resumed_learning_verification.json'),
    latest_verified_state=dict(recorded_utc=now,status=s['status'],stage=s['stage'],operation=s['operation_path'],
        actual_workers=e['workers'],frozen_hashes_verified=55,gpu=e['gpu'],host=e['host'],operation_environments=s['environment_count'],learning_environments=s.get('next_training_environment_count')),
    endpoint=old['endpoint'],recording_request=old['recording_request'],latest_gate_trends=str(OUT/'gate_trends.json'),
    trajectory_analysis=str(OUT/'trajectory_analysis.json'),clipped_score_audit=str(OUT/'clipped_score_audit.json'),smoothed_phase_counterfactual=str(OUT/'current_smoothed_counterfactual.json'))
write(nextpath,new)
cfg=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text(encoding='utf-8'))
assert cfg['id']==old['automation_id'] and cfg['kind']=='heartbeat' and cfg['rrule']=='FREQ=HOURLY;INTERVAL=3'
write(OUT/'automation_current_before_update.json',cfg)
prefix=cfg['prompt'].split('Latest dated focus',1)[0].rstrip()
focus=f'''Latest dated focus, October 5, {now}: R37_offline_ground_full_state remains the same full-state learner and supervisor {s['supervisor_pid']}. It has {s['total_training_iterations']} completed-operation main updates and 58 additional preserved updates from the 069 VRAM failure, plus 7,361 declared inherited updates. Read diagnostics/adaptive_check_20261005_2046/REPORT.md, evidence.json, trajectory_analysis.json, timing_probe/result.json, current_smoothed_counterfactual.json, clipped_score_audit.json, decision.json, closing_matching_gates.json, resumed_learning_verification.json and fresh ACTIVE_RUN/state. No production source, reward, physical geometry, CAD cap, optimizer or actor reset was deployed; 55 frozen hashes verify. The original local ground, truthful 400-input/23-output history, teacher context, neutral-head module, PPO and iteration 4 execution remain.

Latest exact 139 checkpoint matching 140 nominal/141 pushed tests have 192 robots,20 seconds,32 trials per command and 100% survival. Stopped,0.15,0.65,1.0 cases pass both. The 0.35 case fails speed retention at0.464/0.459 m/s. Running command1.5 achieves {g['nominal_cases'][-1]['signed_speed_m_s']:.3f}/{g['pushed_cases'][-1]['signed_speed_m_s']:.3f} signed m/s with {100*g['nominal_cases'][-1]['geometric_flight_fraction']:.4f}%/{100*g['pushed_cases'][-1]['geometric_flight_fraction']:.4f}% genuine flight,below1.5%. Head RMS {head[0]:.2f}/{head[1]:.2f} degrees; natural arms uncertified. Prior retention and small flight are nonmonotonic:133 passed all priors with0.236/0.365% flight,while139 regressed. Preserve133 and other passing actors,walking teacher and038 flight actor. Saved gates never qualify newer updates. No whole running stage,5m/s or actuator ceiling is proven.

A new current-actor normal-limit physical probe uses seed91,six fixed32-robot cohorts,20 seconds. All192 survive;exact actor load,truthful commands and full reward sums pass. Previous best-flight offsets give0.479% flight and-4.274 full return versus current control. Advancing exactly those offsets20/40/60ms giveszero flight and lower return. Previous best-return offsets give+1.958 return butonly0.133% flight. New flight samples include a few late-window samples and mostly just-after-wrap;do not repeat the older ALL-after-wrap claim. Diagnostic offsets are rejected;no pulses,root support,observation/phase/pose rewrites,head override orhybrid is deployed. These bounds do not prove running impossible,a wrong reward clock,or an optimal fixed latency shift.

The CPU box-smoothed summed-force candidate preserves weight0.5 and quarter-body-weight scale,but20/40/80ms boundaries leave the flight trajectory about4.20/4.17/4.06 complete-return units below control and introduce load-redistribution credit at unequal edge coefficients. It is rejected BEFORE learning. Sixteen synthetic categories and exact old-term reconstruction pass;first floating binary-boundary fixture failure and midpoint fix are preserved. No64/128 candidate is falsely claimed. Periodic Reward Composition uses expected VonMises force/velocity intervals;this local box candidate is not its exact algorithm or a convergence guarantee.

New CPU clipped-Gaussian score audit uses24,032 truthful running states and exact trained exploration. Raw means outside +/-3:14.53%;predicted Gaussian clipping14.93% overall,57.8/63.3% ankle pitch and39.0/73.2% ankle roll. Six synthetic score/unbiasedness/objective-derivative cases pass. Constant-return score second moments decrease, but actual advantage-weighted PPO gradients,entropy/ratios and causal locomotion effects are NOT measured. CAPG (https://proceedings.mlr.press/v80/fujita18a.html) is a primary-source research lead,not deployed. The earlier teacher-clipping-equivalence candidate is a different rejected experiment. Do not imply the ordinary estimator is biased or lower variance guarantees running.

Same supervisor resumes after{e['resume']['pause_seconds']:.1f}s;resumed139 completes64 genuine updates at6144 with exact full actor/critic/Adam/exploration load andzero KL/command mismatch. Both R37 diagnostic pauses total{rates['excluded_diagnostic_pause_seconds']:.1f}s and are excluded from whole-chain clocks. Pushed138 wrapper includes this pause. Completed-operation counters omit58 saved069 updates;070 loaded the full memory-recovery checkpoint exactly. Do not lose these samples in reporting or treat them as lost learning. All retained main rates are{rates['all_retained_active_samples_s']/1000:.2f}k active/{rates['all_retained_worker_effective_samples_s']/1000:.2f}k worker-effective/{rates['all_retained_whole_chain_samples_s']/1000:.2f}k whole-chain. Minimum resumed free GPU/host and growth are in resumed_learning_verification.json. Current selected6144,target8192/reserve300,host guards/fallback/180healthyTRAINING-second ramp remain. Current ramp requires about2091MiB GPU for the added tier;startup/eval memory does not qualify. Earlier actor-specific200Hz tests had no driving clips through2.0;fallen2.5 spikes do not prove a ceiling. Full-body videos fulfilled;unsavedFusion work preserved.

Next focused tests:
1. {d['next_tests'][0]}
2. {d['next_tests'][1]}
3. {d['next_tests'][2]}

Keep this same adaptive three-hour heartbeat active until actual5m/s or a causally established relevant velocity ceiling;at the endpoint save,pause training and this automation,brief,and leave the computer on. Preserve neutral-head,leg/walking/full optimizer state. Do not spend the next check solely waiting,repeat fixed phase shifts/box smoothing,deploy rejected labels,or arbitrarily change weights.'''
args=dict(mode='update',id=cfg['id'],kind='heartbeat',name=cfg['name'],status='ACTIVE',rrule=cfg['rrule'],targetThreadId=cfg['target_thread_id'],prompt=prefix+'\n\n'+focus)
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
write(OUT/'automation_update_args.json',args)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')}
write(OUT/'diagnostic_source_hashes.json',manifest)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),main=s['total_training_iterations'],actual_retained=rates['actual_retained_main_updates'],latest_prior_pass=g['all_prior_pass'],running_pass=g['running_pass'],automation_args=str(OUT/'automation_update_args.json')),indent=2))
