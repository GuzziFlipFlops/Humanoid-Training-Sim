# Five-hour check — October 8, 09:46 UTC

The same R40 learner and supervisor 39000 are running at 2026-10-08T10:08:27.898716+00:00, with one actual worker and 55 verified hashes. It retains 12,839 completed plus 50 recovered = 12,889 current-run main updates, plus 27,507 inherited; 0 partial updates are separate. Reviewed actor 605, nominal 606/pushed 607, passes both 3.5 m/s running cases: signed speed 3.472/3.474, lateral 0.117/0.120 m/s, genuine flight 37.72/37.49%. The whole stage remains unqualified because 0.15 net travel, 1.5 flight and 2.5 slip fail both tests. Saved gates never qualify newer partial actors. Required actor-605 upper timelines were inspected; head RMS 4.67 degrees, held shoulder biases and asymmetry remain. No human/Unitree motion or optical-gaze certification. Passive truthful-state and actual PPO captures establish a same-actor state-distribution/arm-feedback lead; no new learning candidate was deployed. All 51 unchanged physical replay arrays and cases are bitwise equal. Four diagnostic PPO updates pass full-load, KL and command checks. The same supervisor resumed after 110.1 seconds for the failed import attempt and 197.7 seconds for successful capture; failed evidence is preserved. Resumed block 608 completes 64 genuine 6144-environment updates with exact actor/critic/Adam/exploration. Retained rates are 35.17k active, 29.43k worker-effective and 18.48k whole-chain samples/s. Existing upper and both prior auxiliaries remain at 20 batches/update; 35/1 gains and all production sources, rewards, native caps and learned state remain unchanged. A CPU motor-case attribution error was corrected; no production fault or actuator ceiling is established. The same five-hour automation stays active; computer on.

## Decision and exact qualification

Preserve the useful current-speed direction improvement. Full retention and upper motion remain incomplete. Four passive gradient batches show both signs of alignment; they do not justify a new deployed loss. Continue existing refinements and test truthful sequential individual-cycle/state-conditioned coverage before a narrow candidate. Preserve actor 605 SHA `cae8b03b28b6701e689e975f4adec2596a811a2cc22381e247bf92a6e35ecbe0` with matching 606/607 gates, actor 267's all-prior pair and original 349's separate 3 m/s certification. No 5 m/s, conditional 4 m/s or actuator-limit endpoint is established. The owned operation `609_run_350_gate_201_nominal` is recorded separately. Evaluation robot count is not PPO capacity.

The reviewed pair uses 320 robots, 20 seconds and 32 trials per command, with 100% survival. Nominal clones do not establish the independently seeded initial-state diversity required at the final endpoint.

| Command m/s | Nominal signed / lateral m/s | Pushed signed / lateral m/s | Failures |
|---|---|---|---|
| 0 | -0.000 / 0.001 | -0.000 / 0.020 | pass both |
| 0.15 | 0.107 / 0.069 | 0.110 / 0.075 | forward_net_travel |
| 0.35 | 0.346 / 0.071 | 0.348 / 0.076 | pass both |
| 0.65 | 0.690 / 0.077 | 0.689 / 0.083 | pass both |
| 1 | 1.009 / 0.077 | 1.008 / 0.088 | pass both |
| 1.5 | 1.493 / 0.057 | 1.493 / 0.064 | flight; geometric_flight |
| 2 | 2.050 / 0.111 | 2.043 / 0.125 | pass both |
| 2.5 | 2.554 / 0.108 | 2.547 / 0.111 | foot_slip |
| 3 | 2.992 / 0.077 | 2.993 / 0.093 | pass both |
| 3.5 | 3.472 / 0.117 | 3.474 / 0.120 | pass both |

Compared with the previous check's actor 524, 3.5 lateral speed improves from 0.184/0.163 to 0.117/0.120 m/s. The 3 m/s case recovers direction. The 0.15 net-travel and 1.5 flight failures persist, while 2.5 slip regresses after actor 602 briefly passed it. These nonmonotonic prior results prevent promotion despite both current-speed cases passing.

## Upper motion over time

Actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets and velocities were plotted and visually inspected for actors 602 and 605. The audits include bilateral amplitudes, ipsilateral hip phase, elbow coordination, cadence, smoothness, saturation and a descriptive 0.5 Hz slow/faster decomposition. Thirty-two robots per command at 10 Hz are distinguished from 16 total representatives at 50 Hz, one or two per command. Resets and terminal rows are segmented. Population sampling at 10 Hz cannot certify fast jitter.

At actor 605's 3.5 m/s command, head RMS is 4.666 degrees; shoulder pitch ranges are 9.59/7.20 degrees and arm-versus-ipsilateral-hip phases are 169.7/-174.0 degrees at 2.67 Hz. Phase alone does not certify natural motion. Earlier slow 2.5/3 m/s reversals become more nearly held offsets in 605; physical targets carry those offsets. Roll asymmetry, elbow coordination and gait-linked head oscillation persist. All angle/target, velocity, bias and target-bias plots in `final_upper` were inspected and QA receipts saved. The [individual cycle plot](individual_cycle_means.png) was also inspected.

The unchanged passive replay captures truthful 400-input, five-frame histories at 10 Hz for all 320 robots and 50 Hz for 16 representatives. It returns identical actions. All cases and 51 physical arrays match original nominal 606 bitwise. CPU actor replay differs by at most 2.86e-05 action units from saved GPU outputs.

There are 256 complete eight-bin first-episode cycles between 5 and 17 seconds. Individual cycle means avoid cancellation between robots and slow bias reversals. At 3.5 m/s the representative's effective policy cycle-mean RMS is 28.14/9.99 degrees and actual joint cycle-mean RMS is 21.81/11.11 degrees. This is one robot over 31 cycles, not 32 independent robots. The five-degree reporting band is diagnostic only. Three CPU counterexamples cover quiet motion, zero-mean opposed swing and ensemble cancellation. An undeployed detached-feature temporal derivative affecting only the two final shoulder-pitch output rows is quantified; no production temporal loss or 64/128-update candidate learning was tested this check.

## Same-actor training-state diagnosis

The first actual 4096-environment PPO minibatch mean-network tensors are bitwise equal to actor 605. Restricting both datasets to the exact command in all five frames and balancing eight phase bins still gives:

| Command m/s | Training effective shoulder pitch means, deg | Evaluation effective shoulder pitch means, deg |
|---|---|---|
| 1 | -27.64 / 6.38 | -29.89 / 25.88 |
| 2.5 | -16.61 / -28.70 | -34.40 / 23.12 |
| 3 | -14.79 / -32.10 | -26.75 / 11.64 |
| 3.5 | -16.04 / -28.52 | -28.15 / 9.90 |

CPU-only phase-matched history substitutions measure fixed-state policy sensitivity. At 3.5 m/s, borrowing actual training arm q/qd/previous-action histories changes shoulder outputs by 1.708 RMS action units and lower outputs by 0.439 RMS; head-history substitution changes shoulders by only 0.038 RMS. These artificial combinations are not physically consistent trajectories, proof that mechanical effects are absent, or deployable observations. The lower-output response rejects an immediate history-spoof or hybrid approach. The reproduced training/evaluation state difference is a research lead, not an observation bug or established causal repair. Production inference remains truthful.

Four actual PPO minibatches have shoulder final-row PPO norms 4.29–15.74 versus existing upper auxiliary norms 2.50–2.92. Alignment with the proposed evaluation-cycle derivative ranges from -0.915 to +0.924; no universal opposition is established. Per-command advantages, ratios, teacher coverage, effective targets and pre-auxiliary derivatives are saved. The current 256-row auxiliary has only 1–8 exact 2.5 m/s samples in these batches. This describes coverage; it does not prove that increasing weight or sample count fixes motion. CPU stopped-hook subtraction matches directly captured pre-standing PPO derivatives; no production hook fault is inferred.

The [primary human arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) uses wrist/lower-back IMUs and supports feature comparisons of amplitude, asymmetry and smoothness by speed. It supplies no morphology-matched shoulder encoder targets. The [official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab) was revisited; mapped, speed-matched G1/H1 encoder curves were not obtained. No reference curves are fabricated and human/Unitree likeness remains uncertified. Native head zero is torso-relative. Known Isaac/FK parity does not calibrate the original CAD face/camera optical axis or certify world gaze.

## Synchronized physics and corrected motor attribution

`closing_com_contact_audit.json` verifies actor 605's representative 50 Hz COM, joints, physical targets, actions, forces, contacts and commands: finite values, no resets and zero observed/action command difference. Full-population contact/support gates remain unchanged. Moving COM offsets and simplified collision envelopes are not full-body pressure certification.

The older 0741 CPU motor helper assumed contiguous cohorts, but evaluation saves interleaved `case_labels`. The corrected audit stably sorts q/qd by saved labels before the unchanged exact CAD Jacobian/reduction mapping. A synthetic label fixture and directional finite-difference checks pass. This is a diagnostic accounting correction; production physics, native caps and aggregate samples are unchanged. Preserve older artifacts and treat their per-command attribution as unreliable. No historical ceiling had been established from those samples.

Correctly attributed actor-605/3.5 right ankle motor 2 has p99 speed 1.301 times its 400 RPM cap, with 1.962% of 10 Hz samples at/above cap; left motor 2 p99 is 0.853. These pose-derived samples are not native 5 ms driving-clipping or available-torque evidence. A fresh actor-specific substep audit is required if a plateau appears; older different-actor zero-clip probes cannot certify this actor. No sustained relevant ceiling or conditional 4 m/s success is inferred.

## Failure preservation and exact continuation

The first passive wrapper imported Isaac math before AppLauncher and failed with missing `pxr`. Its cleanup resumed the original supervisor after 110.1 seconds, including 106.9 seconds for drain. Logs and receipts are preserved under `attempt01_initial_import_failure`. The saved script copy there was made after the diagnostic import correction; failed logs remain exact. Only diagnostic import ordering changed. The successful retry completed passive evaluation plus four actual PPO/full-load updates, with zero KL and command mismatch; the smoke actor is discarded. The same supervisor resumed after 197.7 seconds, including 28.8 seconds for drain. A separate CPU analysis syntax error was corrected without simulator impact. The successful retry harness exited zero, with wait/drain budgets separated from diagnostic execution.

First resumed block 608 finishes 64 genuine 6144-environment updates and loads actor 605's actor, critic, Adam and exploration exactly. Both teachers and the upper auxiliary remain at 20 batches/update. Minimum free GPU is 1923 MiB, available host memory 8135 MiB, private-memory growth 11 MiB and Torch allocation growth 0 MiB. All pre-update KL and command differences are zero. One worker and 55 frozen hashes verify.

## Actual samples, time and resource guards

At 2026-10-08T10:08:27.898716+00:00, the learner retains 12,839 completed plus 50 recovered = 12,889 current-run updates, plus 27,507 inherited. Thirty-two interrupted unsaved R39 updates are excluded; 0 observed partial updates are separate. Capacity, smoke and diagnostic actors never count as main learning.

Current retained main totals are 1,814,495,232 samples, 51,592.3 collection-plus-PPO seconds and 61,661.2 worker seconds. Since the previous owned observation at 2026-10-08T05:51:11.727972+00:00, this adds 1,728 retained updates and 254,803,968 samples in 7,771.3 active seconds, 9,427.6 worker seconds and 15,100.6 chain seconds excluding documented pauses. The interval rates are 32,788/27,027/16,874 samples/s respectively.

Cumulative retained rates are 35,170 active, 29,427 worker-effective and 18,478 whole-chain samples/s. Whole-chain excludes 9,449.6 documented suspension seconds through its endpoint. Denominators differ; startup, capacity and gates remain in chain time. An earlier untimestamped short bookkeeping pause remains conservatively included. Steady training uses 6144 environments with about 1.9 GiB free GPU; evaluation memory is not capacity. Keep target 8192, reserve 300 MiB, host guards/fallback and 180 continuous healthy training seconds before upward ramps. Fusion stays closed, queued work and completed videos are preserved, and the computer remains on.

## Next focused tests

1. Preserve actor 605, whose 3.5 m/s running case passes both tests, and compare newer exact actors with all stopped and earlier-speed gates. Actor 605 still fails 0.15 net travel, 1.5 flight and 2.5 slip. Audit their actual per-command advantages, teacher targets and landing/force events before a narrow retention continuation. Current-speed success alone does not promote the stage; preserve actor 267 and separately certified 349.
2. Continue mandatory upper angle, target, velocity, bilateral and arm-leg timelines. Same actor 605 gives phase-balanced 3.5 m/s shoulder outputs (-16.04, -28.52 degrees) on stable-command training states versus (-28.15, +9.90) on deterministic evaluation states. Measure truthful sequential per-robot cycle coverage, command transitions, exploration and arm-history/hidden-feature response. Stage an individual-cycle or state-conditioned two-final-row derivative only after reset-safe CPU fixtures, physical response, exact full-state 64/128 controls, all gates and independent seeds. The ensemble-mean candidate remains rejected. Preserve forward head, legs, teachers, 35/1 gains and all learned state; no observation substitution, hybrid or imposed reference curve.
3. Verify supervisor 39000, one worker, 55 hashes, local ground, exact full-state teachers and actual retained samples. Preserve 27,507 inherited updates plus completed and 50 verified R40 recoveries; exclude 32 unsaved R39 updates. Exclude this check's 110.1-second failed-import and 197.7-second successful-capture pauses through each timing endpoint. Keep target 8192, reserve 300 MiB, host guards and 180 continuous healthy training seconds; do not force a ramp from evaluation memory. Correctly mapped actor-605 motor samples show right ankle motor 2 at/above cap in 1.96% of 10 Hz samples. If a high-speed plateau appears, measure current-actor 5 ms requested/applied torque and sustained driving clips before any limit conclusion. Only qualified 5 m/s or qualified 4 m/s with causal limits blocking 5 ends the same five-hour heartbeat.

Keep the same active five-hour heartbeat, thread, notification preference and durable authority. Finish only after actual signed 5 m/s speed at least 4.95 with every matching nominal/pushed/stopped/prior gate and 32 independently seeded states per command; or signed 4 m/s at least 3.96 with every gate plus controlled native limits proving 5 is blocked. At that verified endpoint save exact evidence, safely pause training and automation, prevent restart, brief the user and leave the computer on.

## Closing saved-gate evidence — actor 608

Closing owned evidence 2026-10-08T10:19:41.335832+00:00: same R40 supervisor 39000, one actual worker and 55 hashes. It retains 12,903 completed + 50 recovered = 12,953 current-run updates, plus 27,507 inherited. Newer exact actor 608, matching nominal 609/pushed 610, reaches 3.480/3.488 m/s at command 3.5, but lateral 0.265/0.245 fails both direction tests. Preserve actor 605's separately passing current-speed pair; improvement is nonmonotonic. Actor 608 recovers 2.5 slip in both tests; 0.15 net travel, 1.5 flight and 3.0 lateral still fail. All survive. Actor-608 upper angle/target/velocity/bias plots were also inspected: head RMS 4.00 degrees, shoulder pitch ranges 9.77/6.87 degrees, arm/hip phases 161/155 degrees. Held biases, slow 2.5/3 m/s reversals and fast asymmetry persist; natural motion remains uncertified. Latest completed block 611 has 64 genuine 6144-environment updates, exact full-state load, zero KL/command mismatch and all three auxiliaries at 20 batches/update. Minimum free GPU 1923 MiB and host 8391 MiB, private growth 9 MiB, Torch allocation unchanged. Retained rates 35.15k active / 29.41k worker-effective / 18.47k whole-chain samples/s. The 605 passive state/gradient and motor diagnoses remain actor-specific. No production change or diagnostic actor deployed; same active five-hour goal, computer on.

Read `closeout_gate_review.json`, `closeout_learning_verification.json` and `closing_newer_upper` for the newer pair. The owned operation at this snapshot is `613_run_350_gate_202_push`; its newer actor has an incomplete matching pair and receives no qualification here. The main report's physical replay, state substitutions, individual cycles and motor audit stay explicitly tied to actor 605.

Final owned check at 2026-10-08T10:24:43.919332+00:00: the same supervisor 39000 has one actual worker, all 55 frozen hashes and current operation 614 learning at 6144 environments, with 48 observed partial updates. Completed state remains 12,903 plus 50 verified recoveries; partial updates are not qualification. Fusion is closed. The final inventory has 40 physical runs, zero differing hashes/read warnings and eight documented historical missing manifests. See `final_verification.json`.
