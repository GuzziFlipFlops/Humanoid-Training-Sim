# Five-hour humanoid training check — October 9, 2026, 10:47 UTC

Training qualification has stalled at 3.5 m/s. This check compares earlier qualified learning, tests a new narrow reward change against exact full-state controls and explicitly evaluates higher commands through5m/s. It preserves the existing learner and every gate instead of treating more elapsed training as success.

The new tolerance-width slow reward is REJECTED after exact full-state 64/128 tests, two learning seeds and diverse initial states. Both original 128-update control nominal/pushed pairs pass 3.5 direction; all four corresponding candidate tests fail, with lateral speeds about 0.26-0.33 m/s versus controls about 0.12-0.13. Some slow-command gains do not safely retain fast gait; 1.5 flight remains below the unchanged minimum. Fresh current-actor tests now cover 4/4.5/5 m/s with every original gate, plus 32 individually seeded states per command. No 4/5 qualification or causal actuator ceiling is established. Actual paired PPO objective captures and the current-head/body inverse arms-only physical test are preserved; neither supplies a production repair by itself. No diagnostic actor, reward, hybrid or optimizer reset is deployed.

## Exact current learner and gates

Initial reviewed992 passed both current3.5 cases, signed3.477/3.484m/s and lateral0.107/0.107. It failed retained0.15/0.35/0.65 tracking and1.5 geometric flight. Newer998 was pinned for the matched reward experiment; its direction had already regressed. These facts distinguish current-speed capability from whole-stage qualification.

Final reviewed actor `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\1004_run_350_train_332\candidate.pt`, SHA256 `dcbea16ade593a0dfefabb8ca3b9ea5c21272de556aa8e57e253f981e80d601d`. Both matching tests have320 robots,20seconds and32 trials per command. All classifiers and initial/startup events remain unchanged. Saved gates belong to this exact actor, not the snapshot's newer partial learning. Ordinary nominal clones remain distinct from the separately generated independent initial states.

|Command m/s|Nominal signed|Pushed signed|Genuine geometric flight N/P %|Nominal / pushed failures|
|---:|---:|---:|---:|---|
|0|-0.0000|-0.0000|0.000/0.000|pass / pass|
|0.15|0.1442|0.1485|0.000/0.000|forward_net_travel / forward_net_travel|
|0.35|0.3934|0.3922|0.000/0.000|pass / pass|
|0.65|0.7071|0.7067|0.000/0.000|pass / pass|
|1|1.0053|1.0015|0.000/0.017|pass / pass|
|1.5|1.4582|1.4568|0.000/0.007|geometric_flight / geometric_flight|
|2|2.0158|2.0068|4.253/4.417|pass / pass|
|2.5|2.4928|2.4928|12.313/12.337|pass / pass|
|3|3.0260|3.0224|24.819/25.160|pass / pass|
|3.5|3.4830|3.4866|38.743/38.927|world_lateral_speed / world_lateral_speed|

Preserve992's useful passing current case,605/908/917's separate passing evidence,267's earlier all-prior pair and freshly qualified349 through3m/s. Current-speed success alone does not promote the stage. [Gate trends](gate_trends.json), [synchronized COM/contact/action analysis](closing_com_contact_audit.json), and exact nominal/pushed motor audits retain actor scope.

## Earlier checkpoint and full physical objective

The controlled native-physics comparison pins current998, SHA256 `32c948f5cd9c391a5f814896b795cc08b0b7a814ae0a4ecfb4765e22828bdefa`, against exact freshly qualified349, SHA256 `18e19ec3b558922a5927136d2b1b51fc3b68dc44a22df252e5630fce84c00a3e`. There are832 robots, two initial-state seeds4491/4511,32 paired independent q/qd realizations per command/controller and20seconds. Every robot survives; actual input/action commands match and all physical reward terms reconstruct within9.54e-07. Free base,gravity,self collision,35/1 gains,local ground and native CAD motor caps remain. Controllers use their own truthful state/history; no hybrid,pose support,action pulse or learned diagnostic actor enters production.

At1.5m/s the current controller gives0.141/0.133% genuine unloaded geometric flight. The earlier349 gives1.835/1.818% and about0.132/0.133 higher complete mean reward rate. This reproduces retained-skill loss, not an actuator ceiling or a claim that all natural running is discouraged. [Physical evidence](motor_retention_response/result.json) includes synchronized actual joints/targets/actions/root/contact/COM and native5ms requests/applications.

The [phase-conditioned take-off comparison](takeoff_phase_comparison.json) resolves the force/clearance difference on those same physical states. Both seeds reproduce the earlier policy's narrow unloading/clearance event around reward phase0.46; the current policy's much weaker event occurs in the following bin. Current physical target/joint RMS is smaller for several leg joints despite worse flight. This is a closed-loop trajectory difference, not same-state teacher-label error, a proven clock defect or justification for repeating the rejected fixed timing advances. The200Hz target/joint harmonic response describes one representative per cohort and does not identify open-loop hardware dynamics.

![Physical force and genuine flight by phase](takeoff_phase_comparison.png)

## New reward hypothesis and controlled learning

The original yaw-frame exponential tracker has0.5m/s width at every speed. For0.1<command<=1.0, the candidate substitutes `max(0.10,0.12*command)` from the existing independent speed tolerance. It changes no weights,frame,stopped or faster reward, sampler,teacher,upper auxiliary,PPO,physics or cap. This is a tested hypothesis about retained slow tracking, not a proved numerical fault or a universal optimal width.

[112 numeric/gradient/parity fixtures plus6 counterexample categories](slow_reward_CPU_tests.json) pass. The same-actor physical replay has all51 arrays and all gate cases bitwise identical. Eight real smoke PPO updates, four reload updates and four passive original PPO captures load full actor/critic/Adam/exploration exactly with zero pre-update KL or command mismatch. Experimental reward receipts explicitly state that the original source signature does **not** certify the executed override; no experimental checkpoint is silently admitted to production.

The [recorded-state full-objective counterfactual](physical_reward_counterfactual.json) also reproduces an important risk: narrower instantaneous width penalizes normal stride variation. It changes the ranking of qualified slow motion at several commands. It stays finite and bounded, but those execution facts do not establish improvement. [Bias/variation decomposition](tracking_variation_analysis.json) verifies the second-moment identity and distinguishes mean error from temporal/population variation; no mean-only temporal reward is implemented.

![Tracking width hypothesis](tracking_width_candidate.png)

![Recorded tracking variation](tracking_variation.png)

The matched experiment starts exact current998 full state in every branch,4096 environments, independent learning seeds179/191,64 and128 real updates. It keeps both priors and upper auxiliary20batches/update. Each checkpoint has unchanged320-robot20second nominal/pushed gates;128 also has32 independently generated initial joint states per command at seeds4531/4551, paired between branches. All diagnostic learning is excluded from main progress. Work-only budget6600s excludes the saved-boundary wait/drain. Execution and selection are separate: [independent experiment validation](experiment_validation.json) passes70 checks, while the candidate decision is `REJECTED`.

|Learning seed|Branch|Updates|3.5 lateral N/P m/s|1.5 true flight N/P %|Prior failures N/P|
|---:|---|---:|---:|---:|---|
|179|control|64|0.0990/0.1083|0.000/0.003|1.5:geometric_flight; 3.0:world_lateral_speed / 1.5:geometric_flight; 3.0:world_lateral_speed|
|179|control|128|0.1274/0.1212|0.490/0.458|0.35:speed,world_forward_speed; 0.65:speed,world_forward_speed; 1.5:geometric_flight; 3.0:world_lateral_speed / 0.35:speed,world_forward_speed; 0.65:speed,world_forward_speed; 1.5:geometric_flight; 3.0:world_lateral_speed|
|179|candidate|64|0.2068/0.1878|0.000/0.000|0.15:forward_net_travel; 1.5:geometric_flight; 3.0:world_lateral_speed / 0.15:forward_net_travel; 1.5:geometric_flight; 3.0:world_lateral_speed|
|179|candidate|128|0.3273/0.3088|0.111/0.229|0.35:speed,world_forward_speed; 0.65:speed,world_forward_speed; 1.5:geometric_flight / 0.15:forward_net_travel; 0.35:speed,world_forward_speed; 0.65:speed,world_forward_speed; 1.5:geometric_flight; 3.0:world_lateral_speed|
|191|control|64|0.1865/0.1716|0.313/0.135|0.15:forward_net_travel; 0.65:speed,world_forward_speed; 1.5:geometric_flight; 3.0:world_lateral_speed / 0.15:forward_net_travel; 0.65:speed,world_forward_speed; 1.5:geometric_flight; 3.0:world_lateral_speed|
|191|control|128|0.1249/0.1217|0.392/0.243|0.35:speed,world_forward_speed; 1.5:geometric_flight / 0.15:forward_net_travel; 0.35:speed,world_forward_speed; 1.5:geometric_flight; 3.0:world_lateral_speed|
|191|candidate|64|0.2371/0.2139|0.069/0.056|0.15:forward_net_travel; 0.35:speed; 1.5:geometric_flight; 3.0:world_lateral_speed / 0.15:forward_net_travel; 1.5:geometric_flight; 3.0:world_lateral_speed|
|191|candidate|128|0.2797/0.2602|0.226/0.319|1.5:geometric_flight; 3.0:world_lateral_speed / 0.15:forward_net_travel; 1.5:geometric_flight; 3.0:world_lateral_speed|

[Exact checkpoint and diverse-state review](matched_reward_review.json) and [required upper comparisons](upper_matched_comparison.json) preserve all branches, seeds, failures and actor hashes. No diagnostic actor,reward override or optimizer reset is deployed.

## Explicit higher-speed tests

After safe main resume, the next fresh saved actor `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\1001_run_350_train_331\candidate.pt`, SHA256 `2bb2374623d1092c1fe2b5ff1be7b110b07410647ca3d10a7fbf63b12cd85f67`, is tested at commands through5m/s. Original nominal/pushed416-robot20second tests contain13 commands and32 trials per command. Additional nominal-physics/pushed tests generate32 distinct initial states per command at seeds4591/4611. These are untrained extrapolation tests; they do not bypass the retained-skill contract or assert that command5 is achieved.

|Test|Command|Signed m/s|Lateral m/s|Survival|Failures|
|---|---:|---:|---:|---:|---|
|nominal|3.5|3.4560|0.2126|1.000|world_lateral_speed|
|nominal|4|3.3884|1.3524|1.000|lateral_speed, bilateral_steps, bilateral_stance, gait_cadence, flight, geometric_flight, alternating_landings, fixed_heading, world_forward_speed, world_lateral_speed|
|nominal|4.5|1.8041|2.5217|1.000|speed, lateral_speed, bilateral_steps, bilateral_stance, no_hopping, gait_cadence, alternating_landings, fixed_heading, world_forward_speed, world_lateral_speed, forward_net_travel|
|nominal|5|0.3239|1.4026|1.000|foot_slip, speed, lateral_speed, bilateral_steps, bilateral_stance, no_hopping, gait_cadence, alternating_landings, actual_goal_speed, fixed_heading, world_forward_speed, world_lateral_speed, forward_net_travel|
|pushed|3.5|3.4524|0.2154|1.000|world_lateral_speed|
|pushed|4|3.4009|1.3225|1.000|lateral_speed, bilateral_steps, bilateral_stance, gait_cadence, flight, geometric_flight, alternating_landings, fixed_heading, world_forward_speed, world_lateral_speed|
|pushed|4.5|1.8261|2.4712|1.000|speed, lateral_speed, bilateral_steps, bilateral_stance, no_hopping, gait_cadence, alternating_landings, fixed_heading, world_forward_speed, world_lateral_speed, forward_net_travel|
|pushed|5|0.3110|1.4159|1.000|foot_slip, speed, lateral_speed, bilateral_steps, bilateral_stance, no_hopping, gait_cadence, alternating_landings, actual_goal_speed, fixed_heading, world_forward_speed, world_lateral_speed, forward_net_travel|
|diverse_nominal|3.5|3.4565|0.2136|1.000|world_lateral_speed|
|diverse_nominal|4|3.3959|1.3616|1.000|lateral_speed, bilateral_steps, bilateral_stance, gait_cadence, flight, geometric_flight, alternating_landings, fixed_heading, world_forward_speed, world_lateral_speed|
|diverse_nominal|4.5|1.6590|2.6225|1.000|speed, lateral_speed, bilateral_steps, bilateral_stance, no_hopping, gait_cadence, alternating_landings, fixed_heading, world_forward_speed, world_lateral_speed, forward_net_travel|
|diverse_nominal|5|0.3072|1.4200|1.000|foot_slip, speed, lateral_speed, bilateral_steps, bilateral_stance, no_hopping, gait_cadence, alternating_landings, actual_goal_speed, fixed_heading, world_forward_speed, world_lateral_speed, forward_net_travel|
|diverse_pushed|3.5|3.4595|0.2135|1.000|world_lateral_speed|
|diverse_pushed|4|3.4033|1.3119|1.000|lateral_speed, bilateral_steps, bilateral_stance, gait_cadence, flight, geometric_flight, alternating_landings, fixed_heading, world_forward_speed, world_lateral_speed|
|diverse_pushed|4.5|1.7836|2.5514|1.000|speed, lateral_speed, bilateral_steps, bilateral_stance, no_hopping, gait_cadence, alternating_landings, fixed_heading, world_forward_speed, world_lateral_speed, forward_net_travel|
|diverse_pushed|5|0.2677|1.4084|1.000|foot_slip, speed, lateral_speed, bilateral_steps, bilateral_stance, no_hopping, alternating_landings, actual_goal_speed, fixed_heading, world_forward_speed, world_lateral_speed, forward_net_travel|

A separate256-robot20second physical probe records3.5/4/4.5/5 under each of two seeds with native200Hz motor velocity, requested/applied torque and passive joint/target/phase data. [First-episode attribution](high_speed_extrapolation/analysis.json) excludes the terminal control tick and every later reset/post-fall motor sample. Failure reasons and times remain explicit. Falls,turning,untrained extrapolation or isolated cap spikes cannot prove a persistent locomotion-relevant velocity ceiling. No4/5 completion or fallback is claimed.

In this exact physical probe there are 0 first-episode failures and 0 native driving-speed clip samples across all eight cohorts, including startup. The higher-command failures therefore do not establish an actuator velocity ceiling. This does not prove that a future qualified 5 m/s gait is physically feasible, nor isolate torque limits from policy coordination.

Actual higher-command head/shoulder/elbow angles, physical targets and velocities were also plotted and visually inspected in [nominal](high_speed_extrapolation/upper_nominal/visual_QA.json) and [pushed](high_speed_extrapolation/upper_pushed/visual_QA.json) tests. One representative per plotted command shows positive shoulder-bias shifts and greater native neck-pitch bias at 4.5/5. Full-population distributions remain separate; the 2-8 second plotted interval does not represent the entire trial or certify natural motion.

![Higher-speed current-policy results](high_speed_extrapolation/higher_speed_results.png)

## Actual motor attribution and PPO contributions

The998 native motor probe records4000 actual5ms calls per cohort. Independent saved-data reconstruction matches the controller driving-clip masks exactly; all tested current and prior cohorts through3.5 have zero driving-speed clips. Ankle overspeed is braking demand. Elbow torque limits can be active, but those events do not establish a locomotion ceiling. Native output units and CAD differential Jacobians/reductions are preserved; declared flat-current/hard-speed BLDC and servo envelopes are assumptions, not measured voltage/current/thermal hardware identification. Later sparse10Hz actor audits cannot inherit998's substep attribution.

Four actual4096-environment PPO minibatches capture incoming gradients before standing/moving hooks. [Summary](passive_gradient_summary.json) gives prior parameter norms2.85–3.37, full mean-network PPO24.31–52.06 and upper1.86–2.22; alignment has both signs. Standing subtraction reproduces within6e-10 and moving mean-score cosine is1. Contributions are before global clipping/Adam and omit exploration parameters. Norm size, normalized advantages or mixed alignment do not prove globally opposed gradients or a sampling defect.

After the higher-speed physical tests, two additional4-update4096-environment passive captures start the same fresh exact full state, seed211, under the original and narrow objectives. Actor,critic,Adam and exploration match; all auxiliaries remain20batches/update and all KL/command checks pass. These8 diagnostic updates are discarded and excluded from main progress. [Actual paired advantage/gradient evidence](objective_advantage_comparison.json) records first-rollout observation/action/model equality rather than assuming it. Its first-minibatch paired attribution flag is `True`; normalized-advantage RMS difference is0.932724. The installed PPO normalizes advantages across the rollout, so changing only slow rewards can still change other commands' relative PPO signals. Later minibatches include earlier optimizer updates. Local derivative differences do not prove the cause of the long-term gait regressions or validate the rejected candidate.

In the exact first minibatch, the fast-command returns and retained critic values are identical. Global rollout advantage standard deviation changes from 0.081429 to 0.148334. The frontier's mean normalized advantage changes from -0.176626 to 0.170273; its actual pre-hook mean-network parameter gradient norm changes from 16.257891 to 8.301003, with cosine 0.844166. This directly reproduces a local cross-command learning-signal change from the objective intervention; it does not establish a biased policy-gradient estimator or a complete causal explanation of the 128-update direction loss.

## Current-body/current-head inverse arm diagnosis

The previously untested inverse arms-only comparison is completed on the same fresh saved actor as the higher-speed tests:384 robots,20seconds,two initial-state seeds4591/4611 and32 paired states per controller/command. It compares current,qualified349,and current head/body with only the six shoulder/elbow action rows supplied by349. Commands are limited to0.15/1.5 within the teacher certificate. Six CPU row-selection/rejection checks pass; every physical tick asserts exact current head/non-arm action preservation before the physical feedback response. Both policies receive truthful physical observations/history. The composite remains strictly diagnostic and outside original actor/signature certification.

[Inverse-arm results](inverse_arm_response_analysis.json) preserve speed,flight,survival,head/full reward and first-episode native motor attribution by seed. Its bounded physical-promising flag is `False`; this flag is not full qualification or a learning-tested repair. Physical and learned feedback effects remain coupled. No teacher label above3m/s,no production hybrid and no arm/controller rollback is deployed.

|Seed/controller|Command m/s|Signed m/s|Lateral m/s|Genuine flight %|Survival|Head RMS degrees|
|---|---:|---:|---:|---:|---:|---:|
|seed4591_current_0.15|0.15|0.2120|0.1089|0.0000|1.0000|4.8609|
|seed4591_current_1.5|1.5|1.4655|0.0724|0.2164|1.0000|3.1190|
|seed4591_qualified_prior_0.15|0.15|0.1839|0.0884|0.0000|1.0000|4.3272|
|seed4591_qualified_prior_1.5|1.5|1.4682|0.0779|1.8351|1.0000|4.2116|
|seed4591_current_body_teacher_arms_0.15|0.15|0.0798|0.0757|0.0000|1.0000|5.4919|
|seed4591_current_body_teacher_arms_1.5|1.5|1.4189|0.0888|0.4078|1.0000|2.8739|
|seed4611_current_0.15|0.15|0.2118|0.1091|0.0000|1.0000|4.8610|
|seed4611_current_1.5|1.5|1.4655|0.0726|0.2580|1.0000|3.1184|
|seed4611_qualified_prior_0.15|0.15|0.1839|0.0884|0.0000|1.0000|4.3272|
|seed4611_qualified_prior_1.5|1.5|1.4683|0.0773|1.8267|1.0000|4.2123|
|seed4611_current_body_teacher_arms_0.15|0.15|0.0796|0.0757|0.0000|1.0000|5.4913|
|seed4611_current_body_teacher_arms_1.5|1.5|1.4189|0.0889|0.3995|1.0000|2.8736|

![Inverse arms-only physical response](inverse_arm_response.png)

## Mandatory upper joint angles over time

All14 initial992 plots, four plots for each matched128 actor, and all14 final actor plots were visually inspected with hash-backed receipts. [Final data/plots](final_upper/upper_joint_time_audit.json) cover actual head yaw/pitch,shoulder pitch/roll and elbow angles, physical targets, velocities,bilateral amplitude,arm/ipsilateral-leg phase,cadence,smoothness and action saturation. Population32/cmd sampling is10Hz;16 representatives per gate are50Hz, with fewer representatives for some commands. First episodes and terminal/reset rows are segmented;10Hz sampling cannot certify faster jitter.

Final nominal3.5 head RMS3.101deg,p956.088deg; shoulder pitch ranges13.55/6.95deg. Slow means-26.87/14.28deg; hip cadence2.667Hz and shoulder phases-179.9/-155.2deg. Phase is interpreted with amplitude, not as natural-motion certification. Held bias,unequal roll/elbow motion and neck oscillation remain. The0.5Hz slow split excludes3second edges and is descriptive; slow reversals are not periodic arm swing.

![Actual upper angles and physical targets](final_upper/upper_joint_angles.png)

![Actual velocities](final_upper/upper_joint_velocities.png)

![Actual slow bias and faster residual](final_upper/upper_bias_decomposition.png)

![Physical target and measured bias](final_upper/upper_target_bias.png)

Primary [Collins/Adamczyk/Kuo2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/) Figure6 has human shoulder/elbow time plots. Its indexed primary text verifies10 adult subjects and1.25m/s walking; usable numerical curves,axes,rest poses and morphology mapping remain unverified. The official [Unitree motion assets](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/main/src/assets/motions) reviewed at commit1425b15f contain dance CSVs rather than matched locomotion encoders. Neither source supplies a verified numerical comparison here. No curve is fabricated, no human/Unitree likeness or calibrated optical gaze is claimed. Native neck zero is torso-relative; known Isaac/FK parity does not calibrate the original CAD face/camera axis. [Updated source metadata and limitations](updated_motion_reference_review.json) and [reward sources](research_sources.json) are preserved.

## Safe recovery and measured main progress

The same owned supervisor resumes after5449.8s reward comparison and1028.5s higher tests, including worker drains74.8/106.9s. Exact identity and all55 frozen files verify. No diagnostic worker remains. Each first main receiver loads its paused actor,critic,Adam and trained exploration exactly, with zero pre-update KL/mismatch and all three auxiliaries20batches/update. The original full learner,35/1 gains,teachers,upper module,400/23 truthful inference and iteration4 execution remain.

Relative to the previous check's actual06:54 endpoint: 1,664 retained main updates,327,155,712 samples,8688.0s active,10612.6s worker and16051.9s whole-chain excluding documented pauses. Interval rates37.66k/30.83k/20.38k samples/s. This check's diagnostic learning/smoke/capacity samples never count as main progress;32 unsaved interruptedR39 updates remain excluded.

Current R40 has21,287 completed+50 hash-proven received recovery updates=21,337, plus27507 inherited. Total main samples3,362,193,408. Rates35.70kactive/29.61kworker/18.93kchain, with27379.8s documented suspensions excluded only through each timing endpoint. Startup/capacity/gates remain in chain time; an older untimestamped brief bookkeeping pause remains conservatively included. Denominators differ.

Last real main block at8192 environments keeps minimumGPU1346MiB,host6184MiB; private+10MiB,Torch allocation+0MiB. Target8192,reserve300MiB,host guards/fallback and180 continuous healthy TRAINING seconds for upward ramps remain. Evaluation memory is not training capacity. Fusion stays closed; queued work and completed videos are preserved. Minor diagnostic plot-path/receipt-schema/monitor-order corrections do not change production and their failed outputs are preserved.

Crossing operation999 also exposed lexical ordering in CPU audit tools: an older999 pair could sort after1001. The [numeric-order repair](cpu_numeric_order_repair.json) preserves the three before-source files, reproduces the failure and passes five CPU boundary/actual-pair checks. Only the ownership recorder, history inventory and this check's reviewer changed. The original supervisor selects checkpoints from saved state/results; all55 frozen runtime hashes and53 authored worker files remain identical. This is a reporting repair, not an explanation for the gait plateau.

## Next focused tests

1. The tests now explicitly extend to 4, 4.5 and 5 m/s, but current-policy extrapolation does not qualify those speeds. Preserve the full R40 learner and separately passing 992/605/908/917 current-speed actors, 267 all-prior and certified 349. Compare fresh exact 3.5 direction and every retained speed, especially slow tracking and 1.5 genuine/geometric flight. Freshly requalify the earlier neutral-head all-prior actor 267 in the current runtime and compare its effective targets, full physical return and compatibility with the current upper controller against qualified 349 at 0.15/1.5. Neither earlier actor supplies labels above its independently reverified certificate. Use that previous-checkpoint evidence and the new exact paired reward/critic/global-advantage evidence to select a retention-preserving continuation. The narrow instantaneous slow kernel is rejected in both learning seeds; do not repeat it, reset the learner, relax gates or simply skip retention. Any new reward must have explicit versioned provenance, finite/counterexample checks, unchanged full-state 64/128 controls and independent seeds.
2. The inverse current-body/current-head with qualified-arm cohorts do not meet the bounded physical screen. Do not copy those arm rows into production. Resolve phase-specific leg targets, sole unloading and torso/arm feedback at 0.15/1.5 using the qualified teacher and current truthful states before a coupled command-local training target. Never use 349 labels above its 3 m/s certificate or deploy diagnostic composites. Every check still saves and inspects actual head, shoulder pitch/roll and elbow angles, targets, velocities, bilateral and arm-leg phase and noncancelling slow bias. Preserve forward head, both teachers, upper 20 batches/update and original 35/1 gains. No fabricated human/Unitree encoder curves, production observation substitution, imposed trajectory, hybrid or cap increase.
3. Verify the same owned supervisor 39000, one actual CUDA worker, all 55 frozen hashes/53 authored files, original local ground and exact actor/critic/Adam/exploration plus teachers. Count 27,507 inherited plus completed and 50 proven R40 recoveries; exclude diagnostic/smoke/capacity samples and 32 unsaved R39 updates. Exclude both current bounded pauses only through each timing endpoint. Keep target 8,192, reserve 300 MiB, host guards/fallback and 180 continuous healthy training seconds; Fusion stays closed. The higher-speed 5 ms motor attribution excludes terminal/reset samples. Turning, gait failure and brief clips are not a causal limit. Only qualified 5 m/s, or qualified 4 m/s with controlled native limits proving 5 blocked, all gates and 32 individually seeded states per command, ends training.

Same ACTIVE five-hour heartbeat/thread/notification preference. Only actual≥4.95m/s with every unchanged nominal/pushed/stopped/prior gate and32 independently generated initial-state trials per command, or actual≥3.96m/s under those same requirements plus controlled causal native limits proving5 blocked, ends training. At that endpoint save exact full state/hashes/gates/telemetry,pause owned training/prevent restart,pause this automation and brief with the computer on.

## Closing owned observation

Closing owned observation 2026-10-09T13:24:37.792646+00:00: the same R40 full-state learner and supervisor 39000 are running, with one actual worker, 55 verified frozen hashes and 53 unchanged authored worker files. It retains 21,351 completed plus 50 hash-proven recovered = 21,401 current-run main updates, plus 27,507 inherited; 7 observed partial updates remain separate. Exact actor 1007_run_350_train_333, SHA 4c44fbd01761e257a8f901c33054295fb0cbfaaa253e64d4696d2807fac67689, passes both current cases at command 3.5: nominal/pushed signed speed 3.490/3.497 m/s, lateral speed 0.137/0.133 m/s. The complete stage remains unqualified: 0.15 m/s path efficiency and 1.5 m/s geometric flight fail both matching tests. Every other stopped/prior case passes both. Saved gates apply only to this actor. Full all-command upper angle/target/velocity and bias QA remains explicitly bound to actor 1004; no newer posture or natural-motion certification is inferred. Its head RMS is 3.10 degrees, with held arm biases and unequal motion.

The tolerance-width slow reward is REJECTED after exact full-state 64/128-update comparisons, two learning seeds and independently generated initial states. Both ordinary 128-update control nominal/pushed pairs pass 3.5 m/s direction; all four candidate cases fail, with 0.26–0.33 m/s lateral speed versus 0.12–0.13 for controls. Explicit current-actor 4, 4.5 and 5 m/s tests fail gait/direction. The actor-specific 5 ms motor probe has zero driving-speed clips, so those failures do not establish a velocity ceiling or qualify the 4 m/s fallback. Paired PPO captures reproduce a local cross-command gradient change through global advantage normalization, even with identical fast-command returns. This is a research lead, not proof of the full cause of long-term regression. The inverse current-head/body plus qualified-arms physical test also fails the bounded screen. No diagnostic actor, composite, reward or optimizer reset entered production.

The original supervisor resumed after the 5,449.8-second reward comparison and 1,028.5-second higher-speed diagnosis. Genuine main blocks load exact actor, critic, Adam and exploration; both teachers and upper refinement remain at 20 batches per update. The latest completed 64-update block at 8,192 environments keeps at least 1348 MiB GPU and 6051 MiB host memory free. Retained active/worker-effective/whole-chain rates are 35.71k/29.61k/18.94k samples/s; the denominators differ and documented suspensions are excluded through each endpoint. Original 35/1 gains, truthful 400-input/23-output inference, native CAD caps and iteration-4 execution remain. The CPU numeric-order reporting repair passes five checks and does not alter frozen training sources. Keep the same active five-hour heartbeat and verified 5 m/s or qualified 4 m/s plus causal-limit endpoint. Fusion stays closed, queued work and videos are preserved, and the computer stays on.

[Exact later gate and ownership data](closeout_compact.json), [later full-state verification](closeout_learning_verification.json) and [earlier fully inspected upper plots](final_upper/upper_joint_time_audit.json) have separate actor scopes. The original report tables and motor experiments retain their pinned checkpoint identities.
