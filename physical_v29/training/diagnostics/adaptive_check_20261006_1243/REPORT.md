# Five-hour locomotion and upper-body check — 2026-10-06T13:30:03.963362+00:00

The same **R39** learner is running after a bounded diagnostic pause. Its separately preserved R37 actor349 passed the existing complete3m/s nominal/pushed/prior contract; the current3.5 stage remains unqualified. No production source, reward, weight, PD gain, physical/CAD cap, actor, critic, Adam or exploration reset was deployed. The tested lateral-command candidate is rejected. Existing upper-only posture/symmetry refinement continues20 batches per update, preserving learned head and gait. Neither natural arm motion nor5/conditional4 completion is certified.

## Exact saved gates

The closing pair evaluates **132_run_350_train_041**, SHA2fb7d4d062e013f013091fec7fdea838ae4d5ebd47452ca6e20bc7ff24bdcf7a, in separate320-robot20s processes,32 robots per command. All survive. The table binds only this checkpoint; newer135 learning has not been qualified by this pair. Matching files: [closing_gate_review.json](closing_gate_review.json).

|Command|Nominal signed / lateral m/s|Pushed signed / lateral m/s|Failures|
|---|---|---|---|
|0.00|-0.000 / 0.001|-0.000 / 0.014|passes both|
|0.15|0.109 / 0.084|0.112 / 0.089|forward_net_travel|
|0.35|0.344 / 0.107|0.345 / 0.112|passes both|
|0.65|0.668 / 0.085|0.669 / 0.093|passes both|
|1.00|1.011 / 0.093|1.008 / 0.099|passes both|
|1.50|1.461 / 0.054|1.458 / 0.063|geometric_flight|
|2.00|1.950 / 0.049|1.948 / 0.064|passes both|
|2.50|2.530 / 0.078|2.528 / 0.089|passes both|
|3.00|2.999 / 0.131|3.000 / 0.160|no_hopping, world_lateral_speed|
|3.50|3.497 / 0.667|3.507 / 0.628|lateral_speed, world_lateral_speed|


Direction and retention are nonmonotonic. Initial126 had3.5 signed3.529/3.536 and lateral0.625/0.594; closing132 differs. Old349 certification must not be attributed to the newer actor. No sustained relevant actuator ceiling or valid4/5 endpoint is established.

## Upper-body motion and physical response

Actual head yaw/pitch, shoulder pitch/roll and elbow angle, target and velocity timelines were saved and visually inspected for126, trusted349, control/candidate128 and closing132. Population sampling is10Hz with32 robots per command;16 representative robots per gate have50Hz traces, typically only1–2 representatives per plotted command. Resets/terminal rows are segmented. This does not certify population-wide fast jitter. [Closing angle plot](closing_upper/upper_joint_angles.png), [velocity plot](closing_upper/upper_joint_velocities.png), [all data/statistics](closing_upper/upper_joint_time_audit.json).

Initial126 at1m/s has shoulder ranges40.06/34.24deg, ipsilateral correlations-0.975/-0.935 and phases-173.5/+174.4deg. At3.5 its ranges shrink to16.40/10.09deg, with unequal120.6/173.9deg phases, shoulder biases and gait-linked head oscillation. These are features, not human-likeness gates. Native head zero is torso-relative, not verified world gaze. [Human arm-swing research](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) uses wrist/back sensors and supports feature-level speed, symmetry and smoothness comparison; it is not a mapped shoulder/elbow encoder reference. No verified speed-matched official Unitree encoder curves were obtained, so no exact reference curves were fabricated.

The current132 physical cohort has no upper raw action clipping or upper target-limit clipping. Shoulder target/joint RMS errors are about11.4/12.6deg at3.5. Descriptive target-to-joint gait-frequency phase is-150/+174deg at2.66Hz, versus-47/-50deg near1.27Hz. The fast right shoulder component gain is0.288. This is closed-loop motion, not causal system identification. Arm subtree masses are2.5623kg each; locked shoulder pitch inertia differs0.58%, so no gross mass asymmetry is reproduced. The uncoupled35/4 PD analogy gives1.59–1.60Hz, excluding free elbow/roll/base coupling, gravity and history; it cannot prove resonance or prescribe gains. [Physical response](upper_physical_response.json), [inertia audit](upper_inertia_audit.json). Do controlled normal-limit frequency/coupling tests before changing posture weights or PD gains.

## Truthful steering, native caps and actual gradients

The132 actor was loaded exactly in320 physical robots,32 per cohort,20s at1/3.5 with truthful zero/left/right yaw and lateral commands. Free base, gravity, self collision and native constraints remain; no pulses, root support, pose/input rewrites or hybrid. Commands match exactly and weighted reward sums reconstruct within9.54e-7. All survive with zero resets. Lateral commands barely change motion; yaw commands alter heading, with asymmetric high-speed response. [Steering plot](steering_response.png), [results](steering_analysis.json).

Actual CAD Jacobian/reduction motor states were copied every5ms for4000 substeps. Straight1 and3.5 cohorts have zero driving-speed clips. A yaw cohort has one5ms clip; isolated saturation is not an actuator ceiling. Torque-difference fractions combine several constraints and are not torque-ceiling attribution. [200Hz native audit](steering_response/substep_motor_audit.json).

Four real4096-env PPO minibatches reproduce means on CPU within4.06e-5 maximum (about2.7e-6 RMS). Per-command older-head teacher conditioning changes effective lower labels about0.11–0.23 action units. Conditioned/truthful whole-parameter teacher gradient cosine is0.962–0.984; this is not a proven primary failure cause. Current-frontier PPO/teacher cosine varies-0.023 to+0.029; upper auxiliary norm is1.86–2.01 with mixed PPO alignment. The earlier stopped hook is reconstructed/subtracted for the whole-PPO comparison; moving masks already exclude its rows. Global clipping/Adam and later dynamics can change effects. [Actual gradient audit](current_gradient_analysis.json). All diagnostic learned states are discarded.

## Bounded learning candidate — rejected

Official [Unitree G1 configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py) includes lateral-command exposure. We tested a declared local adaptation: uniform+/-0.1m/s only on current frontier x>3 at stage3.5; all prior/stopped commands and evaluation retain zero lateral request. This is not proof that copying the range works. Ten CPU mode/stage counterexamples, selected-row resets, symmetry/bounds and protected-source comparisons pass. All53 authored runtime files are byte-identical; an explicitly hashed diagnostic wrapper changes only training command distribution. Extra random draws intentionally change subsequent rollout sampling.

Control and candidate each retain exact132 actor/critic/Adam/exploration, train4096 environments64+64 updates, and run all nominal/pushed320-robot20s gates at64/128. All four full loads are equal; every pre-update KL/command mismatch is zero. Candidate64 has modest lateral improvement but a new3.0 pushed failure. At128, candidate3.5 lateral is0.773/0.740 versus control0.592/0.564, with additional1.0 lateral and2.5/3.0 gait failures. Better isolated posture metrics cannot offset lost gait. **Reject**; no diagnostic actor or sampler enters production. [All exact pairs](matched_exposure_review.json), [experiment manifest](matched_exposure/experiment_manifest.json), [CPU checks](exposure_cpu_tests.json).

The initial diagnostic harness failed a WindowsPath/string bookkeeping expression after preserving completed control64. Retry reused that completed checkpoint, with logs retained. The first CPU allclose fixture was tighter than observed CPU/CUDA reduction error; measured errors are saved and bounded explicitly for this diagnostic, not locomotion gates. An audit-only median spelling was fixed before executing its revised version. Missing read-only lookup paths changed no training evidence. No simulation failure was concealed or treated as success.

## Resume, memory and accounting

Same supervisor35688 resumes exact132 SHA2fb7d4d062e013f013091fec7fdea838ae4d5ebd47452ca6e20bc7ff24bdcf7a. The complete suspended interval is1823.6s, including58.0s of worker drain; saved-state mtime immediately after suspension bounds its start. Paused nominal133 wrapper time includes that interval. No unrelated work or unsaved Fusion document was touched.

Post-resume135 completes65 genuine6144-env updates from exact actor/critic/Adam/exploration, zero maximum KL/command mismatch, upper20 batches/update. Minimum free GPU1550MiB and host5616MiB; private growth7MiB, Torch allocation growth0MiB. Active collection+PPO throughput44.30k samples/s. One actual worker and55 hashes verify at the closing owned snapshot. [Resume](diagnostic_resume.json), [learning proof](resumed_learning_verification.json), [fresh owned snapshot](closing_cpu/live_handoff.json).

Before suspension,2639 completed-operation plus33 hash-proven recovered updates=2672 actual retained R39 updates, with16936 earlier actual updates preserved. Those main samples give46.19k active,39.98k worker-effective and26.97k whole-chain samples/s. Startup/evaluation clocks remain separate; current diagnostics/smokes never count as main learning. Adding resumed65 gives2737 retained R39 main updates at that saved boundary. Target8192/reserve300, host fallback and180 continuous healthy TRAINING-second upward ramps remain;6144 is actual current capacity. [Accounting](throughput_and_loads.json).

## Verification diversity and next work

Source audit shows nominal clones share the exact same reset state/domain; pushed processes independently vary per-robot disturbances. Nominal count32 is not32 independently randomized initial-state trials. Existing classifiers and historical nominal-condition passes are preserved, but final5/conditional4 certification also needs explicitly seeded nominal-physics initial-state diversity under those same classifiers. [Diversity audit](evaluation_trial_diversity_audit.json). No endpoint or premature shutdown is claimed.

1. Current3.5 direction and retained0.15 net travel/1.5 geometric flight/3.0 hopping are primary. Uniform +/-0.1 frontier lateral exposure is rejected after exact full-state4096env64/128 control tests:128 lateral0.773/0.740 versus control0.592/0.564 and additional prior failures. Do not repeat it without new evidence. Diagnose advantage-weighted fixed-frame/body-frame objective and heading/yaw response; teacher-context label changes are measured, not a proven cause. Test a narrow justified current-state continuation, not arbitrary weights or reward-identical waiting.
2. Mandatory upper timelines remain. Normal-limit50Hz current physical cohorts show shoulder target/actual phase differences about-150/+174deg at2.66Hz, versus-47/-50deg near1.27Hz; locked-subtree masses/inertias are nearly symmetric. These are descriptive, not causal ID. Run bounded truthful free-base arm/PD frequency-response and coupling tests before changing gain or posture weights. Preserve head, gait and exact learner; any justified upper refinement needs full-state64/128 control plus all gates. No spoof/hybrid/cap raise. Human/Unitree exact curve likeness remains uncertified.
3. Verify one worker/55 hashes/full-state recovery, actual retained updates incl33 current-run recovery,6144/8192 target/reserve300/host guards/180healthy training seconds. Straight1/3.5 physical cohorts have zero native5ms driving speed clips; one isolated yaw-cohort5ms clip is no ceiling. Before final5 or conditional4 endpoint, establish32 independently seeded nominal-physics initial-state trials per command; current nominal clones share one reset condition. Preserve all existing nominal/pushed classifiers and save explicit diversity evidence.

The same automation remains every FIVE hours, active until verified5 or verified4 with controlled native-limit evidence blocking5. At that endpoint save/pause training and automation/brief; computer stays on. Provisional inertias and simplified collision envelopes still limit physical hardware claims. Existing full-body videos remain fulfilled.

## Final CPU observation

At 2026-10-06T13:34:21.699775+00:00, the same supervisor35688 owns one worker, all55 frozen hashes verify, and operation141 is starting another6144-environment main block at3.5m/s. Completed-operation main updates are2769; adding33 verified recovered gives2802 retained R39 updates, plus16936 inherited. The independent132 gates reported above do not qualify these newer actors. [Final observation](final_cpu/live_handoff.json), [verification](final_verification.json).

CPU history inventory records39 physical runs and2085 operations, zero differing manifest hashes and no read warnings. Eight historical missing manifests remain documented. The verified same heartbeat is ACTIVE at a five-hour interval; no duplicate, shutdown or endpoint pause was performed.
