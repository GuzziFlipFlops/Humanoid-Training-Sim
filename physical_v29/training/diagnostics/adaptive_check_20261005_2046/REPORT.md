# Three-hour training check — October 5, 20:46 UTC

<!-- closing_summary_2046 -->
Closing verification at 2026-10-05T21:08:10.906596+00:00: **same learner running at 6,144; running still unqualified.** The newer exact checkpoint 142/nominal143/pushed144 still fails 0.35 retention, and the 1.5 command gives 1.415/1.419 m/s with 0.0000%/0.0278% true flight. The measured head RMS is 3.41/3.41 degrees. Current completed-operation updates are 2880, plus 58 recovered updates. [Closing evidence](final_liveness.json). Earlier measurements below retain their original timestamps and checkpoint binding.

Evidence consolidated at 2026-10-05T21:00:49.915595+00:00. R37 remains the same full-state learner and supervisor 32492. **Running is unqualified.** Current exact checkpoint `139_run_150_train_044` has matching nominal/pushed tests; stopped, 0.15, 0.65 and 1.0 m/s pass, while 0.35 overspeeds again. At the 1.5 command, signed speed is 1.434/1.436 m/s and genuine flight is 0.0000%/0.0069%, below the unchanged 1.5% minimum. All robots survive. Head RMS is 3.08/3.08 degrees; natural arm motion is not certified. No 5 m/s or actuator-ceiling endpoint is established.

The tested fixed timing advances and box-smoothed force candidate are **rejected**. They supply no justified deployment. Production reward v8, verifier v6, truthful 400-input/23-output history, teacher context, upper-body module, native joint/CAD/motor caps, geometry/mass/contact, PPO and iteration 4 execution are unchanged. The full actor, critic, Adam and exploration are preserved; no optimizer reset or diagnostic actor is deployed. All 55 frozen hashes verify. [Decision](decision.json), [owned evidence](final_owned_observation.json).

## Exact independent qualification

Checkpoint SHA `7e8fcd84918ea6de2ce0bfaaa047d9f39de1ec0acd47343b0acbac5ee78bd970`. Each test has 192 robots for 20 seconds, 32 trials per command. Results bind this saved actor; later learner updates are separately unqualified.

| Command m/s | Nominal/pushed signed speed | Nominal/pushed pass | Nominal/pushed true flight % | Failure |
|---:|---:|---|---:|---|
| 0.00 | 0.000/0.000 | True/True | 0.0000/0.0000 | None |
| 0.15 | 0.237/0.238 | True/True | 0.0000/0.0000 | None |
| 0.35 | 0.464/0.459 | False/False | 0.0000/0.0000 | speed, world_forward_speed |
| 0.65 | 0.738/0.740 | True/True | 0.0000/0.0000 | None |
| 1.00 | 1.021/1.021 | True/True | 0.0000/0.0000 | None |
| 1.50 | 1.434/1.436 | False/False | 0.0000/0.0069 | flight, geometric_flight |

Recent checkpoints 130 and 133 passed all earlier commands and had small 1.5-command flight fractions (133: 0.2361% nominal/0.3646% pushed). This is seed-sensitive and nonmonotonic: current 139 loses 0.35 speed retention and almost all flight. Preserve the independently passing actors, walking teacher and earlier 038 flight actor. Do not claim the newest actor retains all speeds or call short flight running. [Matching gates](closing_matching_gates.json), [gate trends](gate_trends.json), [last prior-passing actor](last_all_prior_passing_actor.json), [upper-body diagnostic history](upper_audit.json).

## Controlled physical timing test

The supervisor was suspended at saved checkpoint 136; its existing pushed evaluation was allowed to finish, then its worker drained before diagnostics. The exact actor is tested at seed 91, six fixed cohorts of 32 robots, each 20 seconds: zero offsets, previous best-flight offsets, the same offsets 20/40/60 ms earlier, and previous best-return offsets. This independently tests prior search ideas on the CURRENT actor. It does not pair identical initial states across cohorts or qualify a new policy.

The leg offsets use pre-action phase and fade in after 3 seconds. Amplitudes/widths are held fixed for the timing interventions; center bounds are explicitly expanded to [0,0.5), rather than silently treating new centers as the earlier search bounds. All offsets pass through the existing +/-3 action clamp, soft joint limits, slew limiter and CAD feasibility checks. Base/gravity/self collision are active; observations are truthful. No head override, root support, pose/phase write or hybrid is used. Every offset remains diagnostic only. Seven CPU-check categories pass; actor tensors load exactly, observed/action commands match and full reward sums reconstruct within 2e-6. [Timing selection](timing_selection.json), [CPU checks](timing_cpu_checks.json), [physical results](timing_probe/result.json), [telemetry](timing_probe/round_00.npz).

| Cohort | Signed speed m/s | True flight % after 5s | Complete 20s return difference | Head RMS degrees |
|---|---:|---:|---:|---:|
| unchanged_control | 1.428 | 0.0000 | +0.000 | 2.83 |
| previous_best_flight_current_actor | 1.366 | 0.4785 | -4.274 | 2.59 |
| advance_1_control_ticks | 1.348 | 0.0000 | -4.130 | 2.68 |
| advance_2_control_ticks | 1.349 | 0.0000 | -3.615 | 2.72 |
| advance_3_control_ticks | 1.359 | 0.0000 | -3.931 | 2.73 |
| previous_best_return_current_actor | 1.480 | 0.1332 | +1.958 | 3.04 |

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

Resumed `139_run_150_train_044` completes 64 genuine updates at 6,144. Actor/critic/Adam/exploration load equality is exact; maximum pre-update KL and observed/action-command difference are zero. Minimum free GPU 1585 MiB, available host 6527 MiB, private growth 8 MiB and Torch allocation growth 0 MiB.

The same supervisor resumes after this **337.6-second** pause. Both R37 diagnostic pauses total 679.7 seconds and are excluded from whole-chain time. Pushed evaluation 138's wrapper includes this pause, so it is not a physics-throughput measurement. Current selected training capacity is 6,144; target 8,192 and the 300 MiB reserve remain. The live ramp requires about 2,091 MiB free GPU for the additional tier, not simply more than 300 MiB. It also requires healthy host RAM and 180 continuous healthy TRAINING seconds; startup/evaluation memory never qualifies.

An earlier 069 VRAM reserve breach saved `memory_recovery.pt` after 58 logged updates; 070's full-load proof verifies exact actor/critic/Adam/exploration from that checkpoint. Those 58 preserved updates are omitted from the supervisor's completed-operation counter. This is a reporting distinction, not lost training. Current completed-operation count is 2816; this run actually retains 2874 main updates including that recovery, plus 7,361 declared inherited updates. The inherited ancestry's incomplete-operation count is not newly audited here. No frozen supervisor counter is edited merely to change the display.

Completed-operation rates remain 44.58k active collection-plus-PPO / 38.60k worker-effective / 25.50k whole-chain samples/s. Including the exact retained failed-operation samples gives **44.62k active / 38.60k worker-effective / 26.10k whole-chain**, with 489,553,920 actual retained main samples. Denominators are separate; no capacity/smoke/diagnostic frames are main progress. [Pause](diagnostic_pause.json), [resume](diagnostic_resume.json), [full-state/rates/recovery evidence](resumed_learning_verification.json).

One actual Isaac/CUDA worker and all 55 authored hashes verify. The previous actor-specific 200 Hz motor probe found no driving-speed clips through 2.0 m/s; brief clips after extrapolated 2.5 m/s falls do not prove the endpoint. This check adds no new motor-ceiling certification. Full-body videos are already complete; no new recording is pending. Unrelated unsaved Fusion work is preserved.

## Next focused checks

1. Prioritize actual per-command advantage-weighted exploration and gradient evidence: checkpoint139 now loses 0.35 speed retention in both tests (0.464/0.459 m/s), while 0.0/0.15/0.65/1.0 pass. Compare qualified teacher labels/coverage, lower effective targets, PPO versus teacher gradients, and tracking-kernel credit on slow and running samples. The CPU running-state audit finds 14.53% raw means outside +/-3, with predicted Gaussian clipping 57.8/63.3% at left/right ankle pitch and 39.0/73.2% at left/right ankle roll. Synthetic tail-score tests are not actual PPO evidence. Passively compare actual minibatch ordinary versus clipped-Gaussian CDF scores, real flight credit, entropy and importance ratios. Only stage a narrow training-only estimator after unbiasedness/numerical tests, unchanged inference/physics/full-state load, and matched current-state 64/128-update control with every nominal/pushed/prior/head/direction gate. CAPG is a primary-source research lead, not deployed or proven causal; the old teacher-clipping-equivalence candidate remains rejected.

2. Preserve current neutral-head and walking state and compare exact matching prior/current gates with the preserved 133 actor and newest actor. Recent true flight rose to 0.236/0.365% before this pause but remains below 1.5% and is seed-sensitive. The 192-robot current-actor timing probe rejects 20/40/60ms advances: zero flight and lower complete return. Existing offsets produce only 0.479% flight and -4.274 return; never deploy these pulses/labels or repeat fixed shifts without new evidence. Box phase smoothing changes full return by too little to repair the tradeoff and reintroduces edge redistribution credit; it is rejected before learning. Inspect full objective and feasible natural gait, not grounded local-return maximization, arbitrary weights, observation spoofs, head rollback or hybrids.

3. Verify 6144/8192 moving capacity, one actual worker, 55 frozen hashes, native caps and full-state recovery. The 069 VRAM failure preserved 58 genuine updates in memory_recovery.pt and 070 loaded actor/critic/Adam/exploration exactly; supervisor completed-operation count omits those 58. Report completed and recovered updates separately and all actual retained samples/time, excluding both R37 diagnostic pauses. Current healthy ramp requires about 2091 MiB free GPU for the 6144-to-8192 tier, not merely 300 MiB; keep the reserve and 180 continuous healthy training seconds. Earlier 200Hz no-driving-clip evidence through 2.0 is actor-specific; fallen 2.5 spikes are not a ceiling.

Keep the same adaptive three-hour heartbeat active. At verified actual 5 m/s or a causally established relevant velocity ceiling, save, pause training and this automation, brief the user and leave the computer on.


<!-- final_liveness_2046 -->
Closing observation 2026-10-05T21:08:10.906596+00:00, owned snapshot 2026-10-05T21:06:43.628199+00:00 and actual PPO progress through 2026-10-05T21:08:06.357791+00:00: same supervisor, one actual worker and 55 hashes verify. Current `145_run_150_train_046` has 22 observed main updates at 6,144, zero KL/command mismatch, free GPU 1587 MiB and host 6754 MiB. Completed-operation main 2880, plus 58 verified recovered updates. Latest exact matching actor `142_run_150_train_045` still fails 0.35 speed retention at 0.452/0.448; all other earlier commands pass. Running command 1.5 reaches 1.415/1.419 m/s, true flight 0.0000%/0.0278%, head RMS 3.41/3.41 degrees. Running remains unqualified. These gates bind that saved actor, not current partial updates. Same heartbeat ID/thread/three-hour cadence/notification preference/durable scope are verified active. History inventory: 37 physical runs, 1620 operations, zero differing hashes and no read warnings; eight old missing manifests remain documented. [Final evidence](final_liveness.json), [newer matching pair](closing_newer_matching_gates.json).
