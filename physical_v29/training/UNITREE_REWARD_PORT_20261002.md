# Unitree reward port and validation — October 2, 2026

**Historical baseline:** R20 was stopped and preserved after 68 full updates.
Its learned policy breached the waist motor-angle envelope after 0.08 seconds.
The active replacement is R22 with declared Unitree lifetime adaptations and
the nominal-reference alias corrected. Read [the current evidence](UNITREE_NOMINAL_REPAIR_20261002.md)
and [the episode repair](UNITREE_LIFETIME_REPAIR_20261002.md).
The source comparison and original diagnostic results below remain useful;
dated R20 liveness statements do not describe the active run.

The replacement is `locomotion_20261002_r20_unitree_g1`. It trains the frozen
physical-v29 robot with the published Unitree RL Lab G1 reward function bodies
and weights. Entity names and dimensions are explicitly adapted. This is a
fresh policy; no Unitree pretrained weights or old v29 actor are loaded.
Read [ACTIVE_RUN.json](ACTIVE_RUN.json), the indicated live state and current
worker progress for status. This report is a dated record, not a live display.

The implementation has passed reward arithmetic/adversarial tests, real Isaac
physics/PPO/checkpoint reload, and 4096/6144/8192 capacity checks. These tests
establish execution and particular reward responses. **They do not establish
learned standing, walking, running, or a Unitree-looking gait.** The four-update
smoke candidate fails the independent standing/walking gate. No reliable
research or finite test can guarantee PPO convergence on a different robot.

## What was copied and researched

Primary sources are pinned and stored with SHA256 checks:

- Unitree RL Lab commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`:
  [H1 task](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py),
  [G1 29-DOF task](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py),
  [custom rewards](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py),
  [PPO configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py).
  Exact installed Isaac Lab general/velocity reward sources are retained too.
  Unitree's Apache-2.0 license remains in [unitree_vendor](unitree_vendor/).
- Unitree RL Gym commit `276801e46c5d433564f24658bac64f254b7d2d4b`:
  [older H1 configuration](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/h1/h1_config.py)
  and H1/G1 environment code. This is a different reward/PPO version.
- Unitree RL MJLab commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`:
  [H2 configuration](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/config/h2/env_cfgs.py),
  [velocity task](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/velocity_env_cfg.py),
  and reward implementations. H2 has a public task here; it uses a different
  simulator/sensor arrangement from the chosen Isaac Lab port.

The active port executes AST-extracted original function bodies and reads
weights/parameters directly from the pinned H1/G1 configurations. It does not
replace them with vaguely similar formulas. General Isaac functions are
overridden in the same order as Unitree's `mdp` imports. Local quaternion/entity
adapters are exercised by analytical tests. Reward accumulation uses weighted
floating point values multiplied by the 0.020-second control interval, matching
Isaac's RewardManager convention.

## H1 versus G1

The table uses rates before multiplying by control time. Missing terms are
absent from that published configuration; they were not assigned zero by a
new tuning exercise.

| Term | G1 RL Lab weight | H1 RL Lab weight |
|---|---:|---:|
| Forward/lateral velocity tracking | 1 | 1 |
| Yaw velocity tracking | 0.5 | 0.5 |
| Alive | 0.15 | 0.15 |
| Vertical velocity cost | -2 | -2 |
| Roll/pitch angular velocity cost | -0.05 | -0.5 |
| Joint velocity cost | -0.001 | absent |
| Joint acceleration cost | -2.5e-7 | -2.5e-7 |
| Action change cost | -0.05 | -0.05 |
| Soft joint limit cost | -5 | -5 |
| Mechanical energy cost | -2e-5 | absent |
| Arm pose deviation | -0.1 | -1 |
| Waist pose deviation | -1 | -1 |
| Hip roll/yaw pose deviation | -1 | -1 |
| Flat orientation cost | -5 | -1 |
| Base height cost | -10 | -10 |
| Gait contact match | 0.5 | 0.5 |
| Contacting foot slip cost | -0.2 | -0.2 |
| Foot clearance reward | 1 | 20 |
| Undesired contacts | -1 | -1 |
| Excess foot force over 500 N | absent | -0.0002 |

Both tracking rewards use an exponential squared error with standard-deviation
parameter 0.5. G1's walking gait period is 0.8 seconds, versus H1's 0.6;
both use opposite foot phases and 55% stance duty. G1 disables gait reward for
command magnitude at or below 0.1 m/s. H1 does not have that stop gate.

The original clearance formula multiplies height error by foot horizontal
speed. At zero speed the exponential is 1 regardless of height. This is an
upstream property: stationary feet receive the maximum clearance component.
H1 assigns it twenty times G1's weight. In-place alternating contacts can also
receive gait credit. Copying either reward therefore cannot replace tests of
net travel, bilateral real steps, survival or visual appearance.

The older RL Gym H1 instead uses a *negative* swing-height squared-error cost
with weight -20 and contact-phase credit 0.18. Its small LSTM policy and action
change weight differ too. Those are not the positive-20 RL Lab clearance term.
We did not blend the two versions into a claimed original Unitree task.

## H2 findings

H2's MJLab task uses a posture reward whose joint tolerance changes between
standing, walking and running, a stand-still cost, body angular-momentum cost,
soft landing and self-collision costs. Its shared task has an explicit
termination cost, command-gated foot clearance/slip terms, and opposite-foot
gait phases. Clearance is a penalty in that task rather than H1 RL Lab's large
positive exponential. These are useful comparison points; the active run is
the G1 baseline, not a mixture renamed H2. The public tasks also do not prove
that every commercial Unitree demonstration uses those exact releases.

## Robot-specific adaptations

The robot has 23 simulated joints, rigid wrists/fingers, tilted hip pitch axes
and coupled rod differentials. G1's 29-joint pretrained tensors cannot be
loaded into that interface just because the joint names look similar.

- Six shoulder/elbow joints replace the arm entity selection. Three waist
  joints replace the waist selection; four hip roll/yaw joints replace the hip
  deviation selection. Left/right foot links and sensors map in fixed order.
  Non-foot bodies are the undesired-contact selection. Full mappings are saved
  in each worker's `unitree_reward_manifest.json`.
- Base target height is the measured nominal pelvis height, 0.9058366 m.
  Standing foot link origins are about 0.0040229 m above the ground. The G1
  clearance target becomes 0.1040229 m; this is link-origin height, not measured
  sole clearance. H1 diagnostic target uses the corresponding 0.15 m offset.
- Walking through 1.0 m/s retains the original 0.8 s / 0.55 gait parameters.
  For the requested 1.5–2.5 m/s running stages, the *same* reward body/weight
  receives period `1 / (0.9 + 0.5 * command_speed)` and duty 0.43. Opposite phases
  imply 14% desired flight. This is an explicit running extension outside this
  G1 task's published command range, not a copied official running result.
- Position targets are nominal standing joint positions plus 0.25 times the
  policy action. The old moving CAD-IK target and its imitation rewards are
  removed. The old standing teacher/MSE and discovery payout are absent.
- Actor uses five frames of G1-scaled proprioception; critic also has base
  linear velocity. Actor dimension is 400, critic 415. Two phase-clock fields
  per frame are a declared local addition; upstream's clock fields were
  commented out. Observation noise follows the G1 task's scales.
- PPO copies 24-step rollouts, 512/256/128 ELU networks, five epochs/four
  minibatches, initial standard deviation 1, adaptive learning rate 0.001,
  entropy 0.01, gamma 0.99, lambda 0.95 and target KL 0.01. Actor/critic
  observation normalization is off as upstream. Local exploration bounds,
  zero initial mean, physical safety limits and curriculum are explicit
  differences from the complete upstream environment.

CAD transforms, geometry, mass, collision proxies, physical joint stops,
controller/transmission limits and motor assumptions are unchanged. The
conservative simulation mass remains 34.46334 kg including the battery. New
CAD v32 covers/brackets are a separate release, not silently added to this run.
No training camera is instantiated.

## Tests and deliberately bad movements

[CPU evidence](diagnostics/unitree_port_20261002/counterfactual_rewards.json)
contains 5,589 passed assertions, including exhaustive phase/contact
combinations. This is an assertion count, not 5,589 separate physics tests.
Checks cover source hashes, closed-form tracking, yaw-frame invariance, time
scaling, mapping, limits, rough actions, acceleration/energy, height/tilt,
contacts/slip, zero-command behavior and walking/running phase patterns.
An additional CPU parity check compared 4,233 random unit quaternions, including
scalar and multidimensional batches, against the installed Isaac helpers.
Inverse rotation was identical and maximum yaw-quaternion error was 1.12e-16.
Evidence: `diagnostics/unitree_port_20261002/math_helper_parity.json`.

Controlled whole-cycle fixtures compare matched walking against stationary,
in-place alternation, skating, hopping, reverse/sideways travel, leaning,
falling, one-foot stepping and rapid foot flicks. Matched walking scored above
these fixtures for each profile. Selected G1 rates: matched walk 3.625,
stationary 2.813, in-place alternation 3.238, skating 2.845, hop 2.416,
fallen -5.615. These are synthetic telemetry inputs with asserted truth, not
dynamically achieved gaits. Positive bad-motion scores remain possible.

Seven additional scripted probes ran in the actual free-base simulation.
Normal torque/speed/differential limits, gravity, floor and self-collision
were enabled. A few cases inject a one-time velocity impulse. No base brace
or balance helper was used. Resets were deliberately held for four seconds
so falls remain visible; this is different from normal training termination.

| Actual scripted probe | First physical termination | Mean G1 rate | Mean H1 rate |
|---|---:|---:|---:|
| Nominal stand | none within 4 s | 2.357 | 21.691 |
| Alternating joint targets | 1.48 s | -8.154 | 11.249 |
| Alternating ankle flick | none within 4 s | 2.591 | 21.386 |
| Sliding impulse | 1.34 s | -9.109 | 11.682 |
| Hopping impulse | 3.44 s | 0.281 | 19.110 |
| Roll impulse | none within 4 s | 2.486 | 21.228 |
| Combined pitch and roll | 1.86 s | -6.340 | 14.033 |

H1's positive totals during these failed trajectories expose how its clearance
component can obscure failures in a score-only dashboard. The held-reset
probe does **not** prove a normal H1 optimizer would choose falling, and H1
was not the reward used by the previous failed v29 run. Nominal stand drifts
0.302 m forward; the ankle-flick case also translates, so its name is not proof
of staying in place. Neither case passes the independent stationary gate.

## Frame review and geometry audit

[Frame review](diagnostics/unitree_port_20261002/frame-review.html) provides
each captured image with synchronous G1/H1 terms, joint positions, targets,
contacts and root velocity. Seven MP4s and all 280 screenshots are retained
in `diagnostics/unitree_port_20261002/scripted_visuals_r2`.

All 280 probe frames and all 47 frames of the two official G1 GIF references
were inspected in consecutive contact sheets. Reference clips show alternating
leg flexion/planting and a stable torso. The scripted stepping falls backward;
sliding falls forward; hopping fails after landing; combined pitch/roll also
falls. Those motions do not match the reference gait. Short clips and gray
collision proxies do not support claims about full cosmetic CAD or commercial
motion quality. The physical reference has restrained arms; exaggerated arm
swing is not a required inference from it.

The review also includes 50 CPU-reconstructed frames from actual recorded
four-update PPO smoke rollouts. Both example trajectories remain mostly
double-contact and tilt backward rather than forming alternating steps. Their
first failure times are 2.36 s at rest and 2.44 s under 0.15 m/s command.
Reconstruction stops before automatic reset; it is **not an Isaac screenshot**.
FK reconstructed foot origins agree with simulator telemetry within 0.014 mm.
All 50 frames were inspected. The full 64-trial independent reload gate fails;
successful checkpoint loading is still recorded separately from behavior.

[Kinematic audit](diagnostics/unitree_port_20261002/kinematic_audit.json)
checks all 23 saved axes and foot derivatives. Hip pitch axes have the CAD's
opposite 25-degree tilt. Knee and ankle pitch axes are consistent with the
saved transforms. Roll rotates around the foot origin, so its zero origin
translation derivative is expected. Nominal COM projection lies inside the
combined sole AABB, but that broad static test does not establish dynamic
balance or the stricter independent support gate. No evidence justifies
moving the motor axes or enlarging torque/joint limits.

## Launch, capacity and promotion

Old `r19_standing_retention` was drained and preserved, including
`020_walk_015_train_002_part_05/candidate.pt`. It had no qualifying walking.
Its latest candidate combined failed survival, asymmetric/hopping landings,
ankle clipping and lost standing retention. Similarity to Unitree's PPO
architecture had not made it an actual copy of Unitree rewards.

R20 freezes reward/observation/action source and vendor bytes with checkpoint
provenance. Incompatible old history/action contracts are rejected. Real
physics preflight passed: 337.610 N support versus 338.085 N expected, no falls
in four seconds, no invalid transmissions, and simultaneous differential
samples within recorded linkage closure tolerance. Four actual PPO updates
and a reload completed without implementation failure.

| Actual PPO capacity check | Environment steps/s | Minimum sampled free VRAM |
|---|---:|---:|
| 4096 | 8336 | 3845 MiB |
| 6144 | 10227 | 2689 MiB |
| 8192 | 11674 | 1533 MiB |

The chain targets 8192 with a 300 MiB reserve, memory fallback and a 180-second
continuous healthy-live-PPO upward ramp. Evaluations/startup cannot supply ramp
evidence. Host RAM checkpoint recycling preserves actor/critic/Adam/LR; it
contains observed growth but the underlying host allocation leak is unresolved.
Only one Isaac worker owns CUDA at a time. Render probes finished before launch.

Stages are stand, walk 0.35/0.65/1.0, then run 1.5/2.0/2.5 m/s; 0.15 m/s
remains an independently checked retention command. Training blocks request
150 updates, which may be divided by safe RAM recycling. Each promotion needs
separate nominal and pushed 20-second gates for the current and prior commands.
Failed standing/walking remains unqualified while training continues. Reward
alone, a crash, time limit, smoke test or partial video cannot promote or
authorize the user's conditional completion shutdown. Natural gait still
needs visual review of a learned candidate.

Terminal monitor, without launching a second simulator:

```powershell
& 'E:\Aditya\robot_sim\physical_v29\training\monitor_training.ps1' -Watch
```

CPU-only frame reconstruction/review rebuild:

```powershell
& 'E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe' `
  'E:\Aditya\robot_sim\physical_v29\training\build_unitree_review.py' `
  --run 'E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261002_r20_unitree_g1'
```

Rendering uses a separate single-robot diagnostic process and is run only when
the training worker has safely stopped. The initial Vulkan renderer and a
subsequent integer-metric display failure were preserved, repaired, and the
Direct3D12 retry completed. They are not counted as successful first attempts.
Source, checkpoints, reports and journal changes remain local; no Git push,
external logger, stereo camera, or unsaved Fusion work is involved.

## Observed training handoff

At 12:29 EDT, actual standing PPO was advancing at 8192 environments. The
supervisor had saved 34 updates across two RAM-guard checkpoints; the third
worker had completed eight further updates. Its fresh sample had 1407 MiB
free VRAM and 7990 MiB available system RAM. Both owned processes were live;
the frozen reward/environment/worker bytes matched the tested workspace files.
These are runtime observations, not behavioral passes. The current candidate
is unqualified and the independent baseline standing/walking cases failed.
Read `diagnostics/unitree_port_20261002/handoff_status.json` for the dated
process/state/progress evidence and the active run for subsequent changes.
