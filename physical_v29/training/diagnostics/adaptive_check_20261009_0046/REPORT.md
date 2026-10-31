# Five-hour training check — October 9, 2026, 00:46 UTC

Same R40 full learner/supervisor 39000 is running at 2026-10-09T01:50:08.566848+00:00, with one actual worker and 55 verified frozen hashes. It retains 17,831 completed + 50 recovered = 17,881 current-run main updates, plus 27,507 inherited; 0 partial updates are separate. Exact 839_run_350_train_277 at command 3.5 gives 3.410/3.414 signed and 0.221/0.205 lateral m/s; failures ['world_lateral_speed']/['world_lateral_speed']. Prior failures: {'nominal': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['flight', 'geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}], 'pushed': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['flight', 'geometric_flight']}]}. Saved gates bind that exact actor, not newer partial learning. No new production source,reward,gain,cap or optimizer change is justified by these controlled diagnostics. Existing upper fine-tuning continues at20 batches/update. The improved neutral head and learned gait are preserved; no from-scratch restart or diagnostic learned actor is deployed. Exact839:1088 robots20s,two seeds32 paired initial states/cohort. Controls,truthful command transitions and own-feedback shoulder offsets all survive. Added768-robot sham shows all zero-offset body-replay cohorts fall,so replay falls cannot identify mechanical arm coupling. True-feedback5deg shoulder shifts have command-dependent direction tradeoffs and alter lower outputs; no uniform correction is selected. Passive qualified349 target reconstruction gives about2.27/2.25deg lower-target RMS at0.15/1.5,25%/33% joint clipping,about4.4%1.5 slew limits and no sampled CAD rejection.4000 native5ms baseline samples have zero driving-speed clips. Four actual PPO captures have mixed prior/upper alignment; no universal opposition is inferred. No64/128 candidate learning comparison is claimed this check. Required actual head/shoulder/elbow angle, target, velocity and slow-bias timelines were inspected at initial 827 and closing 839_run_350_train_277 across every command in both tests. Head RMS 3.81 degrees; held arm biases and unequal motion remain. No human/Unitree encoder or optical-gaze certification. Two failed probe interfaces resumed safely after 275.0/291.8 seconds; main diagnosis and sham resumed the same supervisor after 450.7/246.8 seconds. First resumed block 842_run_350_train_278 completes 64 genuine updates at 8192 with exact actor/critic/Adam/exploration, zero KL/command mismatch and all three auxiliaries at 20 batches/update. Rates 35.26k active / 29.33k worker-effective / 18.59k whole-chain samples/s; documented pauses excluded through each endpoint. No production source, reward, gain, cap, actor or optimizer reset and no diagnostic controller deployed. Original 35/1 gains, truthful 400/23 inference, teachers, upper refinement and iteration 4 remain. Same active five-hour endpoint; computer on.

## Decision and exact gates

No new production source,reward,gain,cap or optimizer change is justified by these controlled diagnostics. Existing upper fine-tuning continues at20 batches/update. The improved neutral head and learned gait are preserved; no from-scratch restart or diagnostic learned actor is deployed.

Exact839:1088 robots20s,two seeds32 paired initial states/cohort. Controls,truthful command transitions and own-feedback shoulder offsets all survive. Added768-robot sham shows all zero-offset body-replay cohorts fall,so replay falls cannot identify mechanical arm coupling. True-feedback5deg shoulder shifts have command-dependent direction tradeoffs and alter lower outputs; no uniform correction is selected. Passive qualified349 target reconstruction gives about2.27/2.25deg lower-target RMS at0.15/1.5,25%/33% joint clipping,about4.4%1.5 slew limits and no sampled CAD rejection.4000 native5ms baseline samples have zero driving-speed clips. Four actual PPO captures have mixed prior/upper alignment; no universal opposition is inferred. No64/128 candidate learning comparison is claimed this check.

Matching pair: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\839_run_350_train_277\candidate.pt`, SHA256 `6234b8315c980dbda22ba3084cd534810b6b9c7831cb6ca6c977be1fab80baa1`. Existing nominal/pushed 320-robot, 20-second classifiers remain unchanged. The usual nominal clones do not establish final independent initial-state diversity. The diagnostic paired states below are separate physical tests and never replace gates.

|Command m/s|Nominal signed|Pushed signed|Nominal / pushed failures|
|---:|---:|---:|---|
|0|-0.0000|-0.0000|pass / pass|
|0.15|0.1262|0.1284|forward_net_travel / forward_net_travel|
|0.35|0.4071|0.4046|pass / pass|
|0.65|0.7427|0.7408|pass / pass|
|1|1.0185|1.0163|pass / pass|
|1.5|1.4950|1.4942|flight, geometric_flight / flight, geometric_flight|
|2|2.0662|2.0610|pass / pass|
|2.5|2.5317|2.5315|pass / pass|
|3|2.9117|2.9162|world_lateral_speed / pass|
|3.5|3.4100|3.4138|world_lateral_speed / world_lateral_speed|

This check's initial 827 pair failed 3 m/s direction; later 833 recovered that pair. Preserve each actor's evidence separately. Qualification is nonmonotonic. Preserve current full state, 605's separately passing 3.5 current-case evidence, 267's earlier all-prior pair and separately certified 349. No 3.5 stage, 4/5 endpoint or relevant actuator ceiling is inferred.

## Controlled arm/base response and truthful teacher histories

Exact diagnostic actor: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\839_run_350_train_277\candidate.pt`, SHA256 `6234b8315c980dbda22ba3084cd534810b6b9c7831cb6ca6c977be1fab80baa1`. A repaired 68-robot, 2-second smoke test precedes 1,088 robots for 20 seconds: two seeds, 32 individually seeded nominal-physics q/qd states per cohort, paired between branches. All initial-state hashes and truthful command-history transitions are checked. Gravity, free base, self-collision, original 35/1 gains and native CAD limits remain. Initial q/qd writes occur only at reset.

After 5 seconds, two opposite signs of smooth native-angle 5-degree shoulder-pitch offsets are compared with unchanged control. A prescribed nonshoulder action stream tests replay; a second cohort uses its own truthful feedback. These external diagnostic controllers are never deployed or certified by the original reward signature. They cannot establish an additive separation of mechanical and learned effects.

|Seed|Command|Mode|Survival|Signed m/s|Abs lateral m/s|True flight %|Full reward rate|
|---:|---:|---|---:|---:|---:|---:|---:|
|4321|0.15|control|100.0%|0.1282|0.0930|0.0000|13.9948|
|4321|1.5|control|100.0%|1.4996|0.0926|0.0446|14.2496|
|4321|3.5|control|100.0%|3.4312|0.2153|36.6530|13.8981|
|4321|0.15|transition|100.0%|0.1349|0.0930|0.0000|13.9987|
|4321|1.5|transition|100.0%|1.4836|0.0773|0.2444|14.3057|
|4331|0.15|control|100.0%|0.1290|0.0931|0.0000|13.9953|
|4331|1.5|control|100.0%|1.4997|0.0927|0.0312|14.2490|
|4331|3.5|control|100.0%|3.4304|0.2166|36.6218|13.8966|
|4331|0.15|transition|100.0%|0.1349|0.0930|0.0000|13.9993|
|4331|1.5|transition|100.0%|1.4835|0.0770|0.2028|14.3072|

Paired per-robot differences, 5th/95th percentiles and both seeds are saved in [arm response](arm_coupling_analysis.json). Original body-action replay differences are exactly zero; actual target response and lower-policy feedback are measured rather than assumed. The plotted curves are 32-robot means at seed 4321; individual statistics are also saved.

![Actual arm and world-lateral response](arm_coupling_response.png)

The added 768-robot,20-second sham uses the same exact839 actor and two seeds,32 paired states per command/mode. Every zero-offset replay cohort falls too; all original controls survive. Joint states already diverge before5 seconds,and some slow-command zero-offset falls precede the offsets. Prescribed body-action replay is therefore rejected as mechanical isolation. Its falls cannot show that the arms prevent natural motion or establish an actuator ceiling. All replay streams and reset evidence are preserved; post-fall joint data are excluded from the plotted paired-live errors. See [replay sham](replay_sham_analysis.json).

![Explicit zero-offset replay control](replay_sham_control.png)

Original teacher 349 is loaded exactly and independently certified through 3 m/s. Passive labels reconstruct the complete existing action clamp, joint bounds, slew and CAD rejection map on current truthful histories. Student observations are never substituted. The 3.5 teacher extrapolation is explicitly outside the production retention contract and cannot supply a safe new label. Mixed histories retain four older 3.5 command frames at the switch; existing slow-walking context conditioning is excluded until its exact eligibility conditions hold.

|Seed|Command|Window|Projected lower target RMS deg|Teacher joint/slew fraction|Context eligible|
|---:|---:|---|---:|---:|---:|
|4321|0.15|control: settled|2.270|25.0% / 0.0%|100.0%|
|4321|1.5|control: settled|2.255|32.9% / 4.4%|0.0%|
|4321|0.15|transition: settled|2.215|25.5% / 0.1%|100.0%|
|4321|0.15|transition: command_switch_first_second|3.286|30.0% / 8.9%|90.0%|
|4321|1.5|transition: settled|2.205|33.1% / 4.5%|0.0%|
|4321|1.5|transition: command_switch_first_second|2.858|32.4% / 12.6%|0.0%|
|4331|0.15|control: settled|2.268|25.0% / 0.0%|100.0%|
|4331|1.5|control: settled|2.255|32.9% / 4.4%|0.0%|
|4331|0.15|transition: settled|2.215|25.5% / 0.1%|100.0%|
|4331|0.15|transition: command_switch_first_second|3.290|30.0% / 9.0%|90.0%|
|4331|1.5|transition: settled|2.202|33.1% / 4.5%|0.0%|
|4331|1.5|transition: command_switch_first_second|2.864|32.4% / 12.6%|0.0%|

[Teacher effective targets](teacher_effective_target_analysis.json) contain 16 phase bins and actual next-20-ms sole forces/clearance. The 10 Hz label sample is descriptive and does not identify a fixed physical delay or prove a gradient repair. [Physical validation](physical_validation.json) passes 91 checks, including finite synchronous COM/foot/joint/action data, complete reward sums, target parity and strict resume.

## Native motor and actual PPO evidence

Unmodified control cohorts alone record 4,000 native 5 ms calls, requested/applied motor torque, CAD Jacobian/reduction motor speeds and driving-clip flags. [Current substep attribution](current_substep_motor_attribution.json) is bound to the diagnostic actor and commands 0.15/1.5/3.5. Overspeed while braking and brief clips do not establish that limits block 5 m/s. Later closing-actor 10 Hz motor audits remain separately attributed. No native limit is raised.

Four genuine 4096-environment PPO updates pass exact actor/critic/Adam/exploration load, zero pre-update KL and action-command mismatch, and all three 20-batch auxiliaries. The diagnostic learned actor is discarded. Actual incoming derivatives are captured before standing/moving hooks; CPU reproduction independently cross-checks them. Mean-network prior norms range 2.375–3.430, full PPO 26.970–54.899, upper 2.261–2.680. Running/prior cosine ranges -0.185 to 0.297. These are pre-global-clipping/Adam contributions, not full optimizer steps or proof of universal gradient opposition. See [actual gradient analysis](current_gradient_analysis.json).

## Required upper joint angles over time

All 14 initial-827 and all 14 closing-pair angle/target/velocity/bias plots were visually inspected with saved hashes. Head yaw/pitch, shoulder pitch/roll and elbow data exist at every command. Population statistics use 32 robots per command at 10 Hz; 16 selected representatives per gate use 50 Hz. Resets and terminal samples are segmented. Sparse population traces cannot certify high-frequency jitter.

Closing exact actor head RMS 3.806 degrees, p95 5.881. Median shoulder pitch p95-p05 ranges 7.81/6.30 degrees; slow means -18.14/4.51. Hip cadence 2.667 Hz; ipsilateral shoulder/hip phase 169.6/174.2 degrees. Representative physical target/joint RMS error 9.88 degrees. The head 5-degree band is diagnostic only.

![Actual joint angles and physical targets](final_upper/upper_joint_angles.png)

![Actual joint velocities](final_upper/upper_joint_velocities.png)

![Slow bias and faster residual](final_upper/upper_bias_decomposition.png)

Held shoulder bias, unequal swing/elbow/roll and gait-linked head oscillation remain. The 0.5 Hz bias split is descriptive, excludes filter edges and does not certify natural swing. Existing upper-row refinement continues; no imposed reference curve or gain change is deployed. The primary [human arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) uses wrist/trunk IMUs rather than mapped shoulder encoders. [Official Unitree code](https://github.com/unitreerobotics/unitree_rl_mjlab) is not a verified speed-matched joint trajectory for this CAD. Comparison remains feature-based and morphology-limited; no reference curve is fabricated. Native-zero head is torso-relative. Historical Isaac/FK parity does not calibrate the CAD face/camera optical axis or certify world gaze.

## Preserved failures, safe resume and real progress

Attempt 01 failed before rollout because RSL-RL consumes nested configuration keys; separate deep copies correct the external probe. Attempt 02 failed before the first step because raw observations are a dict, while wrapped observations are TensorDict; storage-owning cloning supports both. Failed executed sources, logs and original checkpoint identities are preserved in attempt01/attempt02. Both resumed the original supervisor after 274.97 and 291.84 seconds. The first automatic attempt-02 resume status assertion failed; fresh identity/status inspection and strict retry succeeded. No production fault or training change is inferred. Nineteen CPU boundary/constraint/storage-ownership tests and the physical smoke pass before the larger experiment.

The successful bounded diagnosis suspends the same supervisor at a saved boundary, drains its worker, runs one diagnostic CUDA worker at a time and resumes it after 450.7 seconds (drain 108.9). The separately bounded replay-sham test pauses at the same839 checkpoint and resumes after 246.8 seconds (drain 4.8). First resumed `842_run_350_train_278` completes 64 genuine main updates with exact full learned state and all KL/command checks zero. All later loads also verify. No silent pause remains.

Since the previous check: 960 retained main updates, 188,743,680 samples; 5047.1 active seconds, 6168.8 worker seconds and 9109.6 whole-chain seconds excluding newly documented suspensions. Interval rates 37.40k/30.60k/20.72k samples/s. Total current-run main samples 2,682,716,160; retained updates 17,881 plus 27,507 inherited. Capacity/smoke/diagnostic samples and 32 unsaved interrupted R39 updates are excluded. Partial updates are separate.

Current-run rates 35.26k active / 29.33k worker-effective / 18.59k whole-chain; denominators differ. 19927.0 seconds of documented suspensions are excluded only through each measurement endpoint; an older untimestamped brief bookkeeping pause remains conservatively included. Last actual main block minimum free GPU 1339 MiB, host 5365 MiB, private growth 10 MiB, Torch allocation growth 0 MiB. One actual worker, 55 frozen hashes and 53 protected authored files verify. Target 8192, reserve 300 MiB, host fallback and 180 continuous healthy training seconds remain; evaluation memory never qualifies capacity. Fusion remains closed and queued work preserved. Full-body videos remain fulfilled.

## Next focused tests

1. Preserve the current full learner and compare exact3.5 direction,0.15 net travel,1.5 genuine/geometric flight and all prior gates,keeping833 recovered3m,605 current-case evidence,267 and separately certified349. The present true-feedback arm cohorts are stable but neither offsets nor passive labels qualify a repair. Run a bounded normal-limit diagnostic at0.15/1.5 comparing current,qualified349,and teacher body with the current learned upper controller under truthful feedback; these diagnostic hybrids must never enter production. Test whether349 labels remain physically useful with current arm histories before any arm-context/dynamics-aware command-local training labels.3.5 teacher output is outside its certificate and must not supply a label. Only stage a justified narrow current-full-state continuation after physical validation,64/128 unchanged controls,two learning seeds and all independent prior/current/head/direction/flight gates. Do not repeat focused-prior bucket reweighting or other rejected variants without new evidence.
2. Mandatory actual head/shoulder pitch+roll/elbow angle,target,velocity,arm-leg coordination and noncancelling bias timelines remain.839 closed-loop5deg offsets change effective lower outputs about0.23-0.25RMS at3.5; one sign lowers3.5 drift but worsens1.5 drift,the other reverses that tradeoff. A uniform shoulder shift is not selected.768-robot replay sham shows every zero-offset replay cohort falls too,with pre-offset divergence; do not repeat body-action replay as mechanical isolation or infer an arm ceiling from it. Use truthful closed-loop,command-specific physical response and sequential per-robot target coverage before a narrow output-row upper refinement. Preserve forward head,35/1 gains,legs,teachers,full state and existingupper20 batches; require64/128 controls and every gate. No production observation substitution,hybrid,imposed encoder curve,cap increase or repeated rejected mean/stiffness changes.
3. Verify same39000,one actual worker,55 frozen hashes,local ground,exact actor/critic/Adam/exploration and both teachers. Count27507 inherited plus completed and50 recovered; exclude32 unsavedR39 and diagnostic/smoke samples. Exclude274.97/291.84s failed-probe pauses,450.65s main diagnosis and246.77s replay-sham pause only through each rate endpoint. Keep8192 target,reserve300,host guards/fallback and180 continuous healthy training seconds;Fusion stays closed.839 baseline native5ms requests/applications have zero driving-speed clips at0.15/1.5/3.5; ankle overspeed is braking and this is actor-specific. Later10Hz audits cannot inherit that attribution. Final5 or conditional4 still needs every unchanged gate and32 independently seeded initial states per command,with causal native limits blocking5 for the4 fallback.

Same ACTIVE five-hour heartbeat/thread/notification preference. Endpoint remains actual ≥4.95 m/s under the declared tolerance with every unchanged gate and 32 individually seeded states per command, or actual ≥3.96 m/s with every gate and controlled native-limit evidence proving 5 m/s blocked. At that endpoint save, pause training/prevent restart, pause automation and brief with the computer on.
