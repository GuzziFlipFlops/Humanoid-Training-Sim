# October 5, 14:44 UTC adaptive check — startup recovery and rejected gradient candidate

Training is running in **R37_offline_ground_full_state**, continuing the exact R36 checkpoint 080 actor, critic, Adam and exploration. It preserves **7,361 earlier genuine main updates**, including 1,536 from R36. At 2026-10-05T15:52:49.806335+00:00, R37 has 64 completed new main updates. The newest complete independent pair belongs to `009_run_150_train_001`; stopped/prior cases all pass: **True**. Running case passes: **False**. Full running-stage qualification: **False**. No 5 m/s or actuator-ceiling endpoint is established.

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
| 0.00 | -0.000 | -0.000 | True/True | 0.0000/0.0000 |
| 0.15 | 0.190 | 0.191 | True/True | 0.0000/0.0000 |
| 0.35 | 0.412 | 0.408 | True/True | 0.0000/0.0000 |
| 0.65 | 0.724 | 0.724 | True/True | 0.0000/0.0000 |
| 1.00 | 1.000 | 1.002 | True/True | 0.0000/0.0000 |
| 1.50 | 1.420 | 1.423 | False/False | 0.0000/0.0000 |

Each pair is192 robots,20 seconds and32 independent trials per command, with the unchanged nominal/pushed/stopped/prior/direction/contact/gait gates. At1.5, combined head yaw/pitch RMS is 3.51/3.51 degrees. This retains the forward-head improvement from the original51.7-degree posture; natural arms are not certified. [Exact matching gates](closing_matching_gates.json). Independent walking success does not turn a grounded1.5 m/s gait into running.

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
| 0.65 | 0.725 | 100% | 0.0000 |
| 1.00 | 0.995 | 100% | 0.0000 |
| 1.50 | 1.378 | 100% | 0.0000 |
| 1.75 | 1.478 | 100% | 0.0000 |
| 2.00 | 1.671 | 100% | 0.1540 |
| 2.50 | 1.103 | 0% | 1.3024 |

There is **zero driving-speed clipping in all motors at0.65,1.0,1.5,1.75 and2.0** in this experiment. At1.5 the ankle p99 cap ratios are about0.70/0.69 for the faster ankle outputs; an isolated1.03 cap crossing is braking, not removed driving torque. At2.5 every robot eventually falls. Its isolated5–10ms driving clips include an ankle and elbow and do not establish a sustained locomotion ceiling. Post-fall cap spikes must not qualify an endpoint. These observations also do not rule out torque-speed droop, torque limits, posture, target timing or policy faults. The brief2.0 flight and pre-fall2.5 samples are not qualifications. [Physical speed response](physical_speed_response/result.json), [200Hz motor attribution](physical_speed_response/substep_motor_audit.json).

Coarse fall attribution finds 19 isolated driving-clip samples in alive control ticks and 19 in failed ticks at 2.5. The post-step alive mask cannot locate fall onset inside its terminal 20 ms tick. A peak cap crossing while still flagged alive is therefore also insufficient to identify a persistent plateau. [Fall attribution and its temporal limit](physical_speed_response/fall_attribution.json).

The primary timing lead remains [Periodic Reward Composition](https://arxiv.org/html/2011.01387v2), which uses phase-dependent force/velocity composition and smooth interval boundaries. A feasible complete-return trajectory or a matched learning improvement must precede any production timing/reward repair. No paper's numerical weights were copied this check.

## Genuine resumed learning and resources

`009_run_150_train_001`: 64 genuine main updates at8,192, exact actor/critic/Adam/exploration load, zero KL/command difference. Minimum free GPU496MiB, host4925MiB, private growth7MiB and Torch allocation growth0MiB.

Completed-main rates are 46.66k active collection-plus-PPO, 40.43k main-worker-effective and 13.19k whole-chain samples/s. Denominators differ. Whole chain includes R37 startup/capacity/gates; the preceding R36 asset failure and diagnostic downtime are not counted as R37 training. One actual worker; source hashes55. Current operation `012_run_150_train_002` at8192 environments may be an evaluation count, not learning capacity. Target8192,300MiB reserve,host guards/fallback and180 continuous healthy training seconds before upward ramps remain. [Resumed full-state learning](resumed_learning_verification.json), [fresh owned observation](final_owned_observation.json).

## Preserved failed diagnostics and decision

The initial ownership recorder rejected an absent supervisor; its CPU-only failed-state branch now records a non-running checkpoint without claiming ownership. No runtime inference changed. An initial CPU analysis head-index shape error is preserved and corrected only in diagnostics. The first128 control launch found no convenience qualified-teacher certificate beside a direct-worker checkpoint before spawning CUDA; the empty operation/logs are retained. The corrected diagnostic runner reads the unchanged embedded certificate, verifies the exact qualified teacher path/hash, and lets the original strict worker validate it. It certifies the teacher, never the diagnostic student. The local ground's missing relative MDL lookup and one nonfatal graphics-interface cache log are preserved; all completed physics/PPO/recorded checks remain explicit.

**Decision:** continue the exact useful learner with verified offline startup maintenance. Reject clipping-equivalence fine-tuning for deployment because it does not improve flight. Preserve checkpoint080, all prior passing walking actors, earlier flight actor038, teachers, failed candidates, logs/hashes and unrelated unsaved Fusion work. Full-body GUI recordings are already fulfilled; no repeat is pending.

Next priorities:

1. Running still plateaus with retained neutral head and walking. Perform a bounded constraint-aware search or trajectory optimization around truthful current-state leg targets, resolving pre-action/reward phase and actual sole-force/target delay. Optimize complete return, not flight alone; validate any feasible normal-limit take-off trajectory in separate seeds and all prior/head cases before a 64/128-update matched-control continuation. Smoothed periodic force/velocity composition is a research lead; do not change arbitrary weights or deploy pulses, observation rewrites or hybrids.

2. Compare fresh exact-checkpoint nominal/pushed stopped,0.15,0.35,0.65,1.0 and1.5 cases, head/arm motion and true flight. Preserve independently passing actors. The teacher-clipping-equivalence candidate failed to improve flight at64 and128 and is rejected; running-row transfer and44% exposure remain rejected. Diagnose the full running objective/phase/action timing, not repeated reward-identical waiting or raw-action loss alone.

3. Verify local-ground provenance, exact actor/critic/Adam/exploration recovery, one worker/55 frozen hashes, moving8192/6144 memory,300MiB reserve and180 healthy training seconds before ramps. Use actual5ms CAD motor speed and driving-clip attribution; cap spikes during extrapolated falls never establish the endpoint. Retain all higher-speed/lower-speed gates and leave the computer on.

Keep the same adaptive three-hour heartbeat active until independently verified actual5m/s or a causally demonstrated relevant velocity ceiling. At that endpoint save,pause training and this automation,brief the user and leave the computer on.


<!-- closing_liveness_20261005_1444 -->
Closing owned observation at 2026-10-05T15:55:47.345617+00:00: same learner,64 saved new main updates plus40 observed updates in block012 at8192,one actual worker/55 hashes,506MiB free GPU and5489MiB available host RAM. Every observed KL/command mismatch remains zero. Partial updates do not qualify the new actor. [Closing liveness](final_liveness.json).
