# Five-hour humanoid training check — October 8, 19:46 UTC

Recorded 2026-10-08T23:01:28.782701+00:00. Read current ACTIVE_RUN/state before reusing any observation.

The same R40 full learner and supervisor 39000 are running at 2026-10-08T22:56:16.427327+00:00, with one actual worker and 55 verified frozen hashes. It preserves 16,807 completed plus 50 hash-proven recovered = 16,857 current-run main updates, plus 27,507 inherited. The 60 partial updates are separate. Reviewed exact actor 794 reaches 3.465/3.472 m/s signed speed at command 3.5, with 0.212/0.200 m/s lateral drift. Both direction tests fail. Retained 0.15 net travel and 1.5 flight/geometric flight fail both tests; all other stopped/prior cases, including recovered 3 m/s direction, pass. All robots survive. Saved gates do not qualify newer partial learning. The focused prior-bucket rehearsal is rejected after two independent learning seeds, exact full-state 64/128 controls and 32 individually seeded initial states per command. It lowers 1.5 flight in both seeds at 128 updates and adds 1 m/s support failures; 0.15 recovery is seed-dependent. No diagnostic actor or sampler enters production. A separate 448-robot, 20-second normal-limit comparison on exact 773 gives the qualified 349 teacher 1.83–1.86% genuine 1.5 flight and higher complete mean reward, versus current 0%. All 4,000 native 5 ms request/application samples have zero driving-speed clips in every cohort. Ankle overspeed occurs while braking. This is actor-specific evidence, not a 4/5 m/s endpoint. Actual head/shoulder/elbow angles, targets, velocities, arm-leg coordination and slow bias were inspected at 767, all four diagnostic 128-update actors, and 794 at every command. Actor 794 has 3.31-degree running head RMS, but held shoulder biases, unequal swing/elbows/roll and gait-linked neck oscillation remain. CPU arm-history substitutions change effective lower outputs by 0.324 RMS at 3.5, showing learned feedback dependence without identifying mechanical coupling. Human/Unitree encoder likeness and optical gaze remain uncertified. The same supervisor resumed after 5,405.4 seconds. Block 776 completed 64 genuine updates at 8,192 environments with exact actor/critic/Adam/exploration load, zero KL/command mismatch and all three auxiliaries at 20 batches/update. Retained rates are 35.10k active, 29.23k worker-effective and 18.44k whole-chain samples/s, excluding documented suspensions through each endpoint. The last actual main block kept at least 1,495 MiB GPU and 6,210 MiB host memory free, with private growth 7 MiB and unchanged Torch allocation. No production source, reward, gain, cap, actor or optimizer reset was deployed. Original 35/1 gains, truthful 400-input/23-output inference, teachers, upper refinement and iteration 4 remain. Keep the same active five-hour check and computer on.

## Decision and current independent behavior

The focused failed-prior rehearsal candidate is rejected after two independent learning seeds and exact full-state 64/128 controls. Only the existing 24% prior bucket changes: the expected 0.15/1.5 fractions rise from 3% to 4.8%, while standing 32% and frontier 44% remain. At 128 updates, candidate 1.5 genuine flight is 1.021/0.969% versus control 1.222/1.240% for seed 157, and 0.611/0.559% versus 1.069/0.986% for seed 167; all are below 1.5%. Seed 157 candidate loses 0.15 travel and both 1 m/s support passes. Seed 167 recovers 0.15 and 3 m/s direction but also fails both 1 m/s support tests, where its control pushed test passes. Extra individually seeded initial states reproduce the failures. Neither seed qualifies 3.5 direction. Preserve the original learner; no diagnostic actor or sampler enters production. This rejects the tested allocation, not every retention method or the possibility of convergence.

The current complete stage is unqualified. These exact320-robot20-second nominal/pushed tests have32 trials per command, full-population contact/direction/COM checks and100% survival. Ordinary nominal resets are clones; they do not certify final independently seeded initial-state diversity. The additional diagnostic tests explicitly generate32 unique per-robot seeds per command. No4m,5m or actuator-limited completion is claimed.

Checkpoint: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\794_run_350_train_262\candidate.pt`. SHA256: `f9f143891ef48ff0c6e2ad66f21f4c290d0965fd1d39a10b319346766d0cb423`.

|Command m/s|Nominal signed|Pushed signed|Nominal / pushed failures|
|---:|---:|---:|---|
|0|-0.0000|-0.0000|pass / pass|
|0.15|0.1237|0.1256|forward_net_travel / forward_net_travel|
|0.35|0.3798|0.3802|pass / pass|
|0.65|0.6973|0.7007|pass / pass|
|1|1.0436|1.0389|pass / pass|
|1.5|1.4903|1.4884|flight, geometric_flight / flight, geometric_flight|
|2|2.0455|2.0357|pass / pass|
|2.5|2.5295|2.5224|pass / pass|
|3|2.9992|3.0001|pass / pass|
|3.5|3.4653|3.4716|world_lateral_speed / world_lateral_speed|

Exact7943.5 genuine flight is39.632/39.642%; retained1.5 flight is only0.615/0.396%. Force gaps, current-command tracking and survival alone do not qualify the stage. Compared with767, direction improves,3m direction recovers and1.5 gains some flight;0.15 remains failed. Improvement is nonmonotonic. Preserve605's separately passing3.5 case,267's all-prior pair and original349's separate3m certification.

## Bounded rehearsal experiment

The existing source already assigns exact prior commands. Expected standing/prior/frontier fractions are32/24/44%; each of eight prior anchors gets3% of all robots. Duplicating0.15 and1.5 only inside the existing prior list gives each4.8%, and each other prior2.4%. All random draws, standing/frontier membership and other frontiers are bitwise preserved. Rewards, teacher formulas, upper auxiliary, truthful history, actions, physics, gains and caps are unchanged. This is a tested allocation hypothesis, not a missing-command bug or arbitrary weight repair.

Both branches start the exact773 actor,critic,Adam and trained exploration, using4096 environments and the existing restart-LR ceiling for64 then128 updates at learning seeds157/167. Each saved actor has matching nominal/pushed tests;128 actors also receive4231 nominal-physics and4251 pushed tests with `SeedSequence([seed,environment_id])`,32 unique smallq/qd states per command and paired initial-state hashes across branches. Original classifiers stay unchanged. All8 learning/evaluation combinations and8 diverse tests completed. Diagnostic actors are kept outside production; original signatures do not certify the changed sampler.

|Learning seed|Branch|Updates|1.5 true flight N/P %|3.5 lateral N/P m/s|Earlier failures N/P|
|---:|---|---:|---:|---:|---|
|157|control|64|1.385/1.403|0.207/0.195|0.15:forward_net_travel; 1.5:geometric_flight; 3:world_lateral_speed / 0.15:forward_net_travel; 1.5:geometric_flight; 3:world_lateral_speed|
|157|control|128|1.222/1.240|0.282/0.251|1.5:geometric_flight; 3:world_lateral_speed / 1.5:geometric_flight; 3:world_lateral_speed|
|157|candidate|64|0.552/0.417|0.169/0.160|0.15:forward_net_travel; 1.5:flight,geometric_flight; 3:world_lateral_speed / 0.15:forward_net_travel; 1.5:flight,geometric_flight; 3:world_lateral_speed|
|157|candidate|128|1.021/0.969|0.290/0.269|0.15:forward_net_travel; 1:walking_support; 1.5:geometric_flight; 3:world_lateral_speed / 0.15:forward_net_travel; 1:walking_support; 1.5:flight,geometric_flight; 3:world_lateral_speed|
|167|control|64|0.174/0.170|0.317/0.298|0.15:forward_net_travel; 1.5:flight,geometric_flight / 0.15:forward_net_travel; 1.5:flight,geometric_flight|
|167|control|128|1.069/0.986|0.304/0.286|0.15:forward_net_travel; 1:walking_support; 1.5:geometric_flight; 3:world_lateral_speed / 0.15:forward_net_travel; 1.5:geometric_flight; 3:world_lateral_speed|
|167|candidate|64|0.000/0.042|0.370/0.347|0.15:forward_net_travel; 1:walking_support; 1.5:flight,geometric_flight / 0.15:forward_net_travel; 1.5:flight,geometric_flight|
|167|candidate|128|0.611/0.559|0.301/0.279|1:walking_support; 1.5:geometric_flight / 1:walking_support; 1.5:geometric_flight|

Seed157 candidate128 adds both1m support failures and fails0.15 where control passes. Seed167 recovers0.15 and3m but fails both1m support tests and has less1.5 flight. Diverse tests reproduce those failures. Head and arm changes are also mixed, despite an unchanged upper auxiliary. The execution validation passed; behavioral deployment criteria did not.

Validation:27 meaningful CPU sampler fixtures,53 protected authored-source equalities,all51 unchanged-actor physical arrays and every case bitwise equal,eight64-environment PPO smoke updates,four full-state reload updates,four passively observed4096-environment PPO updates and all matched main-trial loads. All pre-update KL and action-command mismatches are zero; existing auxiliaries remain20 batches/update. See [experiment validation](experiment_validation.json), [matched review](matched_focused_review.json) and [CPU fixtures](focused_prior_cpu_tests.json).

## Native physics, contact and motor attribution

The separate448-robot20-second probe uses two seeds,32 independent small initial joint states per cohort,current773 at0.15/1.5/3/3.5 and qualified349 at0.15/1.5/3. Free base,gravity,self-collision,truthful observations,original35/1 gains and native CAD limits remain. All survive without resets; command mismatch is zero and full reward terms reconstruct. Statistics below use the post5s window and are a physical comparison, not substituted qualification gates.

|Seed|Actor|Command|Signed m/s|Lateral m/s|True flight %|Full mean reward rate|
|---:|---|---:|---:|---:|---:|---:|
|4291|current|0.15|0.1066|0.0780|0.0000|14.0565|
|4291|current|1.5|1.4820|0.1304|0.0000|14.2679|
|4291|current|3|2.9778|0.0862|21.8500|14.0121|
|4291|current|3.5|3.4419|0.2018|36.4888|13.8742|
|4291|qualified_prior|0.15|0.1839|0.0884|0.0000|14.2292|
|4291|qualified_prior|1.5|1.4683|0.0779|1.8309|14.5446|
|4291|qualified_prior|3|3.0196|0.1089|24.4424|13.9894|
|4311|current|0.15|0.1063|0.0780|0.0000|14.0554|
|4311|current|1.5|1.4819|0.1303|0.0000|14.2678|
|4311|current|3|2.9776|0.0859|21.7585|14.0133|
|4311|current|3.5|3.4431|0.2000|36.4597|13.8764|
|4311|qualified_prior|0.15|0.1839|0.0884|0.0000|14.2293|
|4311|qualified_prior|1.5|1.4683|0.0776|1.8559|14.5455|
|4311|qualified_prior|3|3.0186|0.1089|24.3550|13.9894|

Teacher1.5 reproduces1.831/1.856% true flight and higher complete mean reward than current773's0%. Current1.5 has force gaps but no bilateral sole-clearance flight. The same limits admit this retained gait; this does not prove one particular training repair will recover it.

All4000 native5ms actuator calls save requested/applied torque,motor velocity and cap ratios. Independent masks reconstructed from saved arrays exactly match driving-clip flags. Every tested cohort has zero driving-speed clips. Current3.5 ankles overspeed in about1.2% of samples, but their requests are braking or zero; they are not velocity-limited driving. Ankle requests peak around21Nm below the modeled45Nm cap. Sparse teacher elbow torque clipping is not a locomotion ceiling. These are declared simulation motor limits, not hardware/thermal identification.

![Current and qualified prior physical response](motor_retention_response.png)

See [raw motor attribution](motor_substep_attribution.json), [contact/objective analysis](contact_objective_analysis.json), [qualified teacher receipt](motor_retention_response/qualified_teacher_receipt.json) and saved200Hz NPZ. Exact794's separate label-corrected10Hz audits have sparse ankle overspeed but cannot inherit773's5ms torque attribution. They do not establish a later-actor ceiling. Full nominal/pushed COM/contact/action arrays are finite,synchronized and command-consistent; no resets occur.

## Actual PPO derivatives and arm feedback

Four real passively captured minibatches reproduce incoming moving/standing derivatives. Standing subtraction error is below4.5e-10. Total PPO mean-network norms are28.4-59.6,prior3.13-3.46 and upper2.25-2.45; running/prior cosine ranges-0.008 to0.391,upper/PPO-0.411 to0.243. Per-command teacher contributions are smaller and have mixed alignment. These are pre-global-clipping/Adam mean-network contributions, not full optimizer steps or evidence of universal interference. Positive normalized advantage does not prove global return optimality. The diagnostic learner is discarded.

The first captured actor's CPU same-command/phase-bin donor armq,qd and previous-action histories change effective clipped lower outputs by about0.324RMS at3.5. These donors were truthful saved histories, but the recombined whole-body state need not be dynamically realizable. This reproduces learned input dependence, not identified mechanical coupling, a safe trajectory or justification for production observation substitution. See [gradient summary](passive_gradient_summary.json) and [arm sensitivity](arm_feedback_sensitivity.json).

## Upper-body motion over time

All fourteen767 plots,all four plots for each128-update diagnostic actor,and all fourteen794 nominal/pushed plots were visually inspected, with saved SHA256 receipts. CSV/JSON/SVG/PNG contain actual head yaw/pitch,shoulder pitch/roll,elbow angles,physical targets and velocities by every command. Full-population32/cmd sampling is10Hz;16 total selected representatives per gate are50Hz,reset/terminal rows segmented. Sparse population sampling cannot certify fast jitter.

Exact794 at3.5: head RMS3.312deg,p954.950deg. Median shoulder pitch spans9.81/7.27deg; slow means-14.94/1.74deg. Hip cadence2.67Hz; ipsilateral shoulder/hip phases167.5/-160.9deg. Representative target/joint RMS error8.90deg and selected raw upper clamp fraction0%. The5deg head band is diagnostic only.

Held shoulder biases,unequal amplitude/elbows/roll and gait-linked head oscillation persist. The794 nominal3m representative slowly reverses bias;767's shown2.5/3m robots instead held bias. Do not transfer that description between actors. Training seed and changed lower exposure can indirectly alter upper behavior. The0.5Hz slow/residual split uses2.02s actual time origin,excludes3s filter edges and does not certify periodic natural motion.

![794 actual angles and physical targets](final_upper/upper_joint_angles.png)

![794 actual joint velocities](final_upper/upper_joint_velocities.png)

![794 slow bias and faster residual](final_upper/upper_bias_decomposition.png)

The primary [human arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) uses wrist/trunk IMUs,not mapped shoulder encoders; comparison is feature-based and morphology-limited. The [official Unitree runtime](https://github.com/unitreerobotics/unitree_rl_mjlab) is not a verified speed-matched encoder trajectory for this CAD. No reference curves are fabricated and no human/Unitree likeness is certified. Native-zero head is torso-relative; historical Isaac/FK parity does not calibrate the original face/camera optical axis. Existing upper20-batch refinement continues; no new gain or imposed arm curve is deployed.

## Owned pause,resume and retained samples

Supervisor39000 is suspended at exact773 saved boundary20:04:48.637UTC; worker drain takes32.8s. Work ends within the6000s budget in5371.7s; the harness exits0 and resumes the same supervisor21:34:54.017UTC. Total suspension5405.4s is timestamped and excluded through each rate endpoint. No silent pause remains. First resumed776 loads exact773 actor,critic,Adam and exploration and completes64 real8192 updates; all later loads and all KL/command checks pass.

Since the previous saved learning evidence: 1,536 retained main updates,301,989,888 samples,8030.6s active collection+PPO,9814.5s worker-effective and14976.6s whole-chain excluding newly documented pauses. Interval rates are37.61k/30.77k/20.16k samples/s. Current-run totals:2,481,389,568 main samples,16,857 retained updates,plus27507 inherited; current partial updates are separately unqualified. Capacity,smoke and diagnostic training never count as retained main progress. The32 unsaved interruptedR39 updates remain excluded.

Run rates are35.10k active/29.23k worker-effective/18.44k whole-chain. Whole-chain includes startup/capacity/gates,excludes18662.7s documented suspensions and conservatively includes an old untimestamped brief bookkeeping pause. These denominators differ. Last actual8192 block minimumGPU1495MiB and host6210MiB,private growth7MiB,Torch allocation growth0MiB. One worker and55 frozen hashes verify.8192target,300MiB reserve,host fallback and180 continuous healthy TRAINING-second ramps remain. Fusion is closed; evaluation/startup free memory is not training capacity.

Preserved failed attempts: initialization quoting before any pause/CUDA,and an unrecognized CPU audit option after resume. Both corrected locally; logs retained,neither a production fault. No training source,reward,cap,gain,optimizer or actor reset is deployed. No unsaved user/Fusion work is altered.

## Next focused tests and endpoint

1. Preserve the current R40 full state and compare fresh matching 3.5 m/s direction, 0.15 m/s net travel, 1.5 m/s genuine flight and every prior gate. The focused prior-bucket reweighting is rejected at 64/128 updates across learning seeds 157/167 and paired individually seeded initial states; do not repeat it without new evidence. Under native limits, qualified actor 349 still gives 1.83–1.86% genuine 1.5 m/s flight and higher complete mean reward than exact 773. Resolve phase-specific teacher effective-target errors, post-clamp/joint/slew response and sole/contact timing on truthful settled and transitioning histories before one bounded dynamics-aware, command-local continuation. Require exact-current full-state 64/128 controls, two learning seeds and all unchanged gates. Positive normalized advantages, smaller teacher norms and mixed alignment do not prove a sampling defect or universal gradient opposition. Preserve 605 current-case evidence, 267 all-prior and separately certified 349.
2. Inspect every actual head/shoulder pitch and roll/elbow angle, target and velocity timeline, bilateral amplitude, arm-leg phase and noncancelling slow bias. In the first actual 773 minibatch, same-command/phase-bin donor arm-history substitutions change effective lower outputs by 0.324 RMS at 3.5 m/s. This CPU sensitivity does not identify mechanical coupling or a safe arm repair. Before another shoulder derivative, run bounded truthful free-base arm-target/base-response cohorts and reset-safe sequential state coverage to separate physical coupling from learned feedback dependence. Preserve the existing upper 20 batches, forward head, legs, both teachers and 35/1 gains. Prior mean, stiffness, exposure, course and teacher candidates remain rejected absent new evidence. No production observation substitution, imposed reference curve, hybrid or cap increase; mapped human/Unitree encoders and optical gaze remain unverified.
3. Verify supervisor 39000, one actual worker, 55 frozen hashes, local ground and exact actor/critic/Adam/exploration plus both teachers. Count 27,507 inherited updates plus fresh completed and 50 verified R40 recoveries; exclude 32 unsaved R39 and all diagnostic samples. Exclude the documented 5,405.4-second pause only through each timing endpoint. Keep the 8,192 target, 300 MiB reserve, host fallback and 180 continuous healthy TRAINING seconds; Fusion stays closed. Exact 773 native 5 ms requests/applications have zero driving-speed clips through 3.5 m/s; overspeed ankles are braking. Later 794 sparse 10 Hz motor samples cannot inherit that substep attribution. If a newer high-speed plateau persists, measure actor-specific 5 ms requested/applied torque, CAD speed, available torque and sustained driving saturation. Only fully qualified 5 m/s or qualified 4 m/s with causal native limits blocking 5 ends training/automation, with all gates and 32 independently seeded states per command.

Keep the same ACTIVE five-hour heartbeat,thread and notification preference. Endpoint remains all-gate qualified actual5m/s at≥4.95,or all-gate qualified4m/s at≥3.96 with controlled causal native limits blocking5,and32 independently seeded initial states per command. At that endpoint save,pause owned training/prevent restart,pause this automation,brief and leave the computer on. Fulfilled full-body videos remain complete.

<!-- adaptive_check_20261008_1946_closing_owned -->

Closing owned evidence 2026-10-08T23:02:37.264481+00:00: the same R40 supervisor 39000 has one actual worker and 55 verified frozen hashes. It retains 16,871 completed plus 50 recovered = 16,921 current-run main updates, plus 27,507 inherited; 13 observed partial updates remain separate. Exact actor 797, matching nominal 798/pushed 799, reaches 3.461/3.469 m/s signed at command 3.5, with 0.151/0.146 m/s lateral drift. Both direction tests still fail. Retained 0.15 net travel, 1.5 geometric flight and 2/3 m/s direction now fail both tests; all other saved cases pass and all robots survive. Retained 1.5 true flight is 0.0000/0.0729%. Improvement is nonmonotonic: preserve 794's separately recovered 2/3 m/s cases. Four newer 797 angle/target, velocity and slow-bias plots were visually inspected; all-command population statistics are saved. Running head RMS is 3.27 degrees, but held shoulder bias and unequal swing/elbow/roll remain. Full all-command nominal/pushed trace QA stays explicitly bound to 794; the 773 native 5 ms motor attribution stays separate. Block 797 completed 64 genuine updates at 8,192 environments, loaded exact 794 actor/critic/Adam/exploration, had zero KL/command mismatch and all three auxiliaries at 20 batches/update. Minimum free GPU/host memory was 1493/6010 MiB, private growth 9 MiB and Torch allocation growth 0 MiB. Retained rates are 35.11k active / 29.24k worker-effective / 18.45k whole-chain samples/s, with 18662.7s documented pauses excluded. No diagnostic actor, sampler, reward, gain, cap or optimizer reset entered production. The current operation is 800 at 8,192. Inventory: 40 physical runs, zero differing hashes/read warnings; eight historical missing manifests remain documented. Read final_owned_gate_review.json, final_owned_learning_verification.json, final_owned_cpu/live_handoff.json and closing_newer_upper, then fresh ACTIVE_RUN/state. The same five-hour check remains active toward qualified 5 m/s or qualified 4 m/s with causal native limits blocking 5; computer on.

Evidence: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261008_1946\closing_owned_addendum.json.
<!-- /adaptive_check_20261008_1946_closing_owned -->

Native automation update is confirmed: same heartbeat/thread, ACTIVE, five-hour interval, unchanged notification preference and durable endpoint. Current owned supervisor identity and one actual worker are live. See [closeout verification](final_closeout_verification.json) and [automation verification](automation_verification.json).
