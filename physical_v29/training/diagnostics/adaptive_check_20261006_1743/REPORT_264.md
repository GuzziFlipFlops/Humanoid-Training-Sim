# Five-hour training and upper-body check — 2026-10-06T18:20:26.764486+00:00

The same **R39 full learner and supervisor35688** are running again after controlled diagnostics. Reject this fixed-frame kernel for immediate deployment; resume the same original full learner and existing upper refinement. No production source, reward, PD gain, native cap, actor/critic/Adam/exploration reset or diagnostic actor was deployed. Existing upper-only posture and symmetry refinement continues20 batches per update. The current3.5 stage remains unqualified; the separately preserved R37checkpoint349 passed the existing3m/s nominal/pushed/prior contract. No5m/s or conditional4m/s endpoint, motor ceiling or human-like upper motion is certified.

## Exact saved locomotion evidence

The latest reviewed pair belongs to **264_run_350_train_085**, SHA7cc7756447cb12d668653015ddf6954f7716b0e56803cd28e5157de761f675e0,320 robots per process,32 per command,20 seconds. It does not qualify a newer partial learner. [Exact pair](closing_gate_review.json).

|Command m/s|Nominal signed / lateral m/s|Pushed signed / lateral m/s|Failures|
|---|---|---|---|
|0.00|0.000 / 0.001|0.000 / 0.016|passes both|
|0.15|0.160 / 0.071|0.162 / 0.076|passes both|
|0.35|0.367 / 0.080|0.367 / 0.084|passes both|
|0.65|0.690 / 0.055|0.691 / 0.064|passes both|
|1.00|1.008 / 0.065|1.003 / 0.075|passes both|
|1.50|1.487 / 0.058|1.485 / 0.067|geometric_flight|
|2.00|1.980 / 0.059|1.980 / 0.078|passes both|
|2.50|2.526 / 0.058|2.534 / 0.071|passes both|
|3.00|3.037 / 0.098|3.036 / 0.145|world_lateral_speed|
|3.50|3.492 / 0.393|3.500 / 0.359|world_lateral_speed|


At the starting261 actor,3.5 signed speed was3.472/3.479m/s but fixed-frame lateral travel was excessive. Retained0.15 net travel and1.5 geometric flight failed. Every robot survived. Useful running progress does not solve direction or retention. Earlier certified349 must remain separately identified.

## Actual upper motion and physical response

Actual head yaw/pitch, shoulder pitch/roll and elbow angle/target/velocity timelines were saved and visually inspected for261, qualified349, both128-update diagnostic actors and the closing actor. Population32 per command is sampled10Hz;16 representatives per test have50Hz traces, generally1–2 representatives per command. Resets are segmented. [Closing angle timelines](closing_upper/upper_joint_angles.png), [velocity timelines](closing_upper/upper_joint_velocities.png), [data/statistics](closing_upper/upper_joint_time_audit.json). Fast pitch swing remains small and uneven, with shoulder bias, larger roll motion, elbow coordination differences and gait-linked head oscillation. The existing training-only soft bands and symmetry loss do not impose an arm-leg phase trajectory.

Closing nominal1m/s shoulders have p95-p05 pitch ranges26.80/31.99deg and ipsilateral phases-159.2/173.6deg. At3.5 ranges shrink to9.63/10.12deg, phases145.6/-122.6deg and native head RMS5.61deg. These descriptive phases and bounds are not human-motion certification.

The bounded free-base experiment uses exact264,320 robots,10 cohorts of32,20s,seed71. Native caps, gravity and self collision remain; feedback is truthful. All survive without resets, commands match and complete reward sums reconstruct within9.54e-7. Training stiffness is35Nm/rad and **damping1Nms/rad**. The earlier1243 uncoupled35/4 analogy used a default configuration rather than the environment's explicit damping1; it did not describe production. This report corrects that interpretation without rewriting its historical evidence.

At1m/s actual-minus-target shoulder phases are-51.8/-47.5deg and component gains1.16/0.60. At3.5 they are+164.3/-109.9deg and gains0.30/0.16; target/joint RMS errors14.2/16.2deg. Doubling/quadrupling damping further reduces fast pitch response and is **not selected**. Four small stopped-command sine cohorts at0.75,1.5,2.66,3.25Hz reproduce strongly frequency-dependent motion. These closed-loop free-base responses include base/elbow/roll/policy coupling; they are not open-loop identification or hardware validation. [Physical results](arm_response_analysis.json), [response plot](arm_response.png), [frequency traces](arm_frequency_response.png).

The3.5 control shoulder p99 efforts are15.1/19.4Nm, peaks15.5/20.0Nm, versus declared peak71.25Nm. All10 cohorts have zero native5ms driving-speed clips. Low sampled effort and zero clips give no proof of a relevant speed/torque/thermal ceiling. Declared continuous caps and provisional inertias are not bench measurements. [Load audit](arm_load_audit.json), [200Hz motor audit](arm_response/substep_motor_audit.json).

A new CPU URDF forward-kinematics audit combines actual root, waist and neck transforms. In the selected261 nominal3.5 representative, modeled head-forward heading error is16.11deg RMS/27.54deg p95;1m/s representatives give6.43deg RMS. Native-zero head is torso-relative and does not ensure world-facing forward. This calibrates the zero-joint modeled axis to pelvis+X, not the CAD face/camera optical axis, and has not been cross-checked against Isaac head-link orientation. It is a representative diagnostic, not a population gaze gate. [Axis data](world_head_axis_audit.json), [timeline](world_head_axis.png).

Primary [human arm-swing research](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) supports feature-level symmetry, amplitude and smoothness comparisons, with different sensors and morphology. Mapped speed-matched official Unitree shoulder/elbow encoder trajectories remain unavailable. No curves or natural-motion certification were fabricated.

## Fixed-frame direction objective — bounded comparison

The original [Unitree G1 velocity task](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py) scores translation in the current yaw frame. Our independent gates require reset-frame travel. On901 truthful pushed3.5 representative states, raw tracking quality is0.865 in the current yaw frame but0.409 in the fixed reset frame; fixed lateral is0.476m/s versus body lateral0.157. These are unweighted/raw-score counterfactuals and do not alone prove an advantage or PPO failure. Isaac Lab's separate [direct locomotion task](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/locomotion/locomotion_env.py) offers a fixed-target heading/progress research reference; it is a different task, not proof of this repair.

The diagnostic candidate rotates yaw-frame linear velocity into the reset frame only for commanded x>3. Existing exponential width0.5, weight4 and every stopped/prior<=3 fixture remain identical.33 CPU checks cover signs/wrap/bounds/derivatives, stationary/reverse/lateral counterexamples and9216 bitwise prior fixtures. All53 protected authored physical/runtime files are byte-identical; the original55-hash production inventory remains unchanged. The runtime override is explicitly hashed and **the original reward signature does not certify its execution**. Candidate checkpoints are experimental only and cannot enter production without a new frozen reward version and narrowly bound migration.

Exact264 actor/critic/Adam/exploration starts both4096-env branches. Each performs64+64 updates and every unchanged320-robot20s nominal/pushed case at64/128. All four exact full loads and every pre-update KL/command mismatch pass. Diagnostic PPO never counts as production learning. [Matched experiment](matched_direction_review.json), [manifest](matched_direction/experiment_manifest.json), [CPU checks](direction_cpu_tests.json), [extra checks](direction_extra_tests.json).

|Branch / updates|3.5 signed nominal / pushed|Lateral nominal / pushed|Prior failures nominal; pushed|
|---|---|---|---|
|control / 64|3.404 / 3.408|0.722 / 0.694|0.15: forward_net_travel; 1.5: geometric_flight ; 0.15: forward_net_travel|
|control / 128|3.438 / 3.441|0.590 / 0.571|1.5: geometric_flight ; 3: world_lateral_speed|
|candidate / 64|3.514 / 3.513|0.435 / 0.417|1.5: geometric_flight ; 0.15: forward_net_travel; 1.5: geometric_flight|
|candidate / 128|3.527 / 3.525|0.254 / 0.242|1.5: geometric_flight; 2.5: no_hopping ; 1.5: geometric_flight; 2.5: no_hopping|


Decision: At128 updates lateral improves about57%, but both3.5 direction tests still fail and a new2.5 no_hopping failure appears in both, while control passes2.5. Preserving gait outweighs isolated direction improvement. The frame hypothesis remains a research lead requiring focused retention/gradient evidence. The original actor resumes; all diagnostic actors remain evidence only. Previously rejected lateral exposure, timing offsets, box smoothing, running transfer and clipping-equivalence candidates remain rejected absent new evidence.

The first physical diagnostic attempt failed before simulation because the copied protected ground USD was excluded. Original groundUSD/manifest/textures were restored verbatim, all53 protected files verified and the experiment rerun successfully. Its failed logs and [repair evidence](ground_copy_repair.json) remain. A CPU upper-audit metadata lookup was corrected to attach the exact historical349 gate pair; no classifier/source/actor changed. Startup shader-cache warnings are preserved and are not treated as policy failures or success.

## Resume and accounting

The same supervisor resumes exact264 SHA7cc7756447cb12d668653015ddf6954f7716b0e56803cd28e5157de761f675e0 after1599.9s of suspension, including58.0s worker drain. Saved-state mtime immediately after suspension bounds its start. The paused265 evaluation wrapper includes this interval; exclude it from throughput. No unrelated unsaved work was touched.

The first resumed block **267_run_350_train_086** completes64 genuine updates at6144 with exact actor/critic/Adam/exploration equality, zero KL/command mismatch and20 upper batches/update. Minimum free GPU1550MiB, host6062MiB, private growth8MiB and Torch allocation growth0MiB. [Resume](diagnostic_resume.json), [genuine learning proof](resumed_learning_verification.json), [owned snapshot](closing_cpu/live_handoff.json).

Before diagnosis,5495 completed-operation+33 hash-proven recovered=5528 actual retained R39 updates;16936 earlier actual updates remain. Main rates are45.26k active collection+PPO,39.22k worker-effective and26.60k whole-chain samples/s, excluding the previous1823.6s pause. Current diagnostics begin after that timing endpoint and are excluded from learned sample counts. Current capacity6144, target8192/reserve300MiB, host fallback and180 continuous healthy TRAINING-second upward ramps remain. Startup/evaluation memory is not steady capacity. [Accounting](throughput_and_loads.json).

1. Direction correction is a promising but undeployed hypothesis: exact4096-env128 comparison lowers3.5 lateral0.590/0.571 to0.254/0.242m/s, yet world-lateral still fails and retained2.5 hop fraction2.299% exceeds the unchanged2% cap in both tests;1.5 flight also fails. First analyze per-command actual advantages/PPO versus prior gradients and2.5 landing events on these saved actors. Require a bounded retention-preserving candidate, independently seeded replication and full-state64/128 control/all gates before deployment; original signatures do not certify diagnostic reward overrides. Do not blindly repeat the same kernel, permanently infer the frame hypothesis false, arbitrarily change weights, or deploy diagnostic actors.
2. Every check inspect actual upper angle/target/velocity timelines. Actual shoulder35/1 free-base tests show high-frequency attenuation; more damping worsens pitch and is not selected. Before an upper repair, test bounded normal-limit stiffness/damping or feasible target-velocity/phase response with truthful feedback and motor-effort/base coupling, then narrow full-state matched control/retention tests. Preserve current head/gait/full learner. Cross-check the CPU modeled world head axis against actual Isaac head-link orientation and original CAD face axis before any world-forward posture label; native head zero alone is insufficient. No cap raise, spoof, hybrid or fabricated human/Unitree curves.
3. Verify same owned worker55 frozen hashes/local ground/exact full-state recovery, completed plus33 proven recovered R39 updates and16936 inherited, moving6144/8192 target/reserve300/host fallback/180continuoushealthyTRAINING-second ramp. Exclude this1599.9s diagnostic pause and earlier1823.6s pause from clocks. All10 current normal-limit physical cohorts have zero native5ms driving-speed clips, not proof of a high-speed ceiling. Before final5 or conditional4 endpoint establish32 independently seeded nominal-physics initial-state trials per command with all unchanged classifiers; nominal clones alone are insufficient. Preserve useful actors, unsaved Fusion and fulfilled videos.

Same active FIVE-hour automation, same thread/notifications; final actual5 or qualified4 with proved native limits blocking5. At that endpoint save,pause training/automation,brief,computer stays on. Nominal clones currently share reset conditions: final certification needs32 explicitly independent seeded nominal-physics initial-state trials per command under unchanged classifiers. Current historical gates remain preserved. Full-body videos are fulfilled; Fusion unsaved work remains protected.
