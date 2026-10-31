# Humanoid training check — October 2, 2026, 15:09 EDT

R22 is running at **8192 environments** in the **standing stage**. At 19:09:35 UTC, 230 full PPO updates were checkpointed and the resumed worker reported iteration 0 after the latest host-RAM recycle. The most recent progress sample was 5.6 seconds old. The supervisor and its owned simulation worker were verified against their command lines and the active run path. There is one simulation worker; the Windows Python wrapper is not a second simulation.

The independent 128-robot, 20-second nominal standing evaluation after 150 full updates showed **128/128 robots surviving**, compared with 0/128 in the initial smoke gate. It failed only the anchor check: mean final drift was **0.3212 m**, above the **0.25 m** limit. Mean speed was 0.05165 m/s, foot slip 0.03864 m/s, pelvis tilt 2.82 degrees and torso tilt 2.64 degrees. There were no invalid differential states or non-foot contacts in that evaluation. Nominal standing has not passed, so pushed standing has not been tested and walking/running have not started. The next nominal gate follows the second 150-update block.

The full saved 50 Hz telemetry covers 1000 control ticks for the first 16 robots. All joint, action, root and force samples are finite. There are no resets or physical failure flags in that trace. Both feet are loaded for 99.62% of the samples, with no flight. Mean root displacement is 0.3217 m; mean left/right foot displacement vectors are (0.3368, 0.0869) m and (0.1728, 0.0808) m. This is balanced standing with slow foot sliding, not walking or hidden reset survival.

All **201 CPU reconstruction frames** for robot 0 were reviewed in chronological contact sheets, covering 20 seconds at 10 Hz plus the final control tick. They show steady balance corrections with bent knees, no falls and no alternating steps. These are telemetry reconstructions, **not Isaac screenshots**. Forward kinematics from the frozen URDF agrees with the saved simulated foot positions within 0.00001182 m. The contact-sheet renderer was corrected to display all four rows of its 12-frame layout; the original telemetry and PNGs are preserved.

The latest 20 stochastic training updates have a 99.93% healthy fraction. Rare fall and differential-fault resets still occur during exploration; the zero-failure statement above applies to the independent deterministic evaluation. Every saved full-training progress row retains zero nominal-pose drift. All frozen source, physics, mass, actuator, asset and reward hashes in the chain manifest match, and the protected tested workspace files match their frozen copies.

The live sample had **1623 MiB free VRAM** and **14673 MiB available host RAM**, above the existing 300 MiB VRAM reserve. The target remains 8192, with fallback and a configured 180-second healthy-memory ramp. No upward ramp is needed at the maximum target. Host RAM recycling continues to preserve checkpoints, actor/critic, Adam and learning rate; the underlying allocation-growth cause remains unresolved. The latest recycle checkpointed 230 total full updates and fresh PPO progress was verified after startup at 8192. No additional Isaac worker, rendering camera or stereo camera was started for this review.

The survival improvement is independent behavioral evidence of learning. One remaining drift failure does not establish stalled learning, and the existing gate blocks promotion of that behavior. **Rewards and running training were left unchanged.** Checkpoints and logs remain in the source-frozen run. The chain is incomplete and the computer was **not shut down**.

Evidence:

- [Live process, source-hash and rollout audit](status_and_rollout_audit.json)
- [201-frame standing review](standing-gate-review.html)
- [Frame metadata and kinematic check](standing_gate_150_frames.json)
- [Independent nominal evaluation](../../runs/locomotion_20261002_r22_unitree_nominal/017_stand_gate_002_nominal/evaluation.json)
- [Active run pointer](../../ACTIVE_RUN.json)

This is a dated check. Read the active pointer and fresh run state for subsequent progress.
