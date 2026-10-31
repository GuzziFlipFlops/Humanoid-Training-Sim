# Unitree source and lifetime evidence

<!-- unitree-nominal-20261002 -->

The current replacement is **r22_unitree_nominal**, reward
`unitree_g1_lifetime_v2`. Read `E:/Aditya/robot_sim/physical_v29/training/ACTIVE_RUN.json`,
its live state/progress, verified worker/supervisor processes and independent
gates for current status. R20 was drained and preserved after 68 full updates;
R21 was preserved after validation, before full standing training. Neither is
active or qualified. R20's replay first breached the waist differential
motor-angle limit at 0.08 seconds, then fell backward without alternating steps.
Its original aggregate reward rates were all negative, allowing early episodes
to avoid future costs.

R22 retains the pinned Unitree RL Lab G1 core bodies and weights, adding the
declared RL Gym nonnegative regular aggregate followed by MJLab H2's -200
physical-termination weight (-4 per 20 ms event). Timeouts are unpenalized.
Per-cause termination telemetry separates motor faults from ordinary falls.
This is an adapted public Unitree design, not unmodified complete G1/H2 or a
commercial pretrained policy. 5,593 core and 149 lifetime CPU checks passed.
Actual new physics/PPO/load/capacity checks are retained in R22's run directory.
Smoke, reward rankings and screenshots do not qualify learned walking.

The original adaptation's expanded reference aliased the nominal pose. A
randomized reset could change that shared origin and every other robot's
reference. The actual-source CPU regression reproduced 0.63665 rad drift after
512 resets; corrected private storage gives zero drift and leaves unreset rows
unchanged. This is a reproduction, not a measurement of old GPU drift. Every
real PPO update now logs/asserts zero nominal drift. Contract is
`physical_v29_history5_unitree_defaultoffset_v2`, still 400/415 actor/critic
dimensions. A fresh actor is used; no moving IK target or standing teacher.
Physical-v29 mass 34.46334 kg, CAD joint frames, motor limits, differential
geometry and collisions are unchanged. CAD v32 stays separate. Do not load
old history4/IK or another robot's policy tensors into this fresh actor.

The chain targets 8192, reserve 300 MiB, fallback, and a 180-second continuous
healthy-live-PPO ramp at checkpoint boundaries. Host RAM recycling preserves
actor/critic/Adam/LR; the allocation leak itself remains unresolved. Keep one
Isaac/CUDA worker only. No training stereo/render camera is instantiated.
Stages: stand; walk 0.35/0.65/1.0; run 1.5/2.0/2.5, with 0.15 retention. Promotion
requires independent current/prior nominal and pushed 20-second gates. The
running cadence/flight extension is declared, not commercial Unitree weights.
Every later learned candidate needs visual review before a human-gait claim.

Report: `E:/Aditya/robot_sim/physical_v29/training/UNITREE_NOMINAL_REPAIR_20261002.md`.
Source/weight comparison: `training/UNITREE_REWARD_PORT_20261002.md`.
Frame review: `training/diagnostics/unitree_port_20261002/frame-review.html`.
The review contains 280 scripted screenshots, 80 learned-r20 replay screenshots,
and 207 explicitly labeled CPU smoke reconstructions (567 frames across 15 cases), with synchronized angles,
contacts, failure causes and original H1/G1 rates; 47 official GIF frames were
inspected too. None of the failed candidates matches the reference gait.
Local-only documentation/source changes; no Git commit or push.

Dated handoff 2026-10-02T17:50:31.581981+00:00: verified live full
stand PPO at 8192 environments, with 17
saved full updates and current worker iteration 12. Free VRAM
1269 MiB; available host RAM 4419 MiB.
Frozen worker/environment/reward files match the tested workspace. This is
runtime evidence, not a behavioral qualification. Inspect fresh active status.
Current nominal-pose drift: 0.0 rad.
The first host-RAM recycle saved 17 full updates and resumed at 8192. Actual
continuation Adam counters advanced420 to440 after the next PPO update;
actor/contract/reward/asset provenance is retained in
`training/diagnostics/unitree_termination_20261002/optimizer_resume_r22.json`.


## Historical record before the nominal-reference repair

# Unitree source comparison and reward port

<!-- unitree-port-20261002 -->

The current replacement is **r20_unitree_g1**. Read
`E:/Aditya/robot_sim/physical_v29/training/ACTIVE_RUN.json`, its live `state.json`,
verified worker/supervisor command lines, fresh progress and independent gates.
R19 was drained and preserved without qualifying walking. R20 starts a fresh
five-frame, nominal-position-offset policy using pinned Unitree RL Lab G1
reward bodies/weights and PPO settings. Old CAD-IK moving action references,
discovery bonuses and the standing teacher are absent. H1 RL Lab, older RL Gym
H1/G1 and public MJLab H2 tasks were compared separately, not blended.

The reward version is `unitree_g1_port_v1`; action/observation contract is
`physical_v29_history5_unitree_defaultoffset_v1`. Actor/critic dimensions are
400/415. Original G1 walking phase parameters remain; requested running uses
an explicitly documented cadence/flight extension. Robot dimensions and
entity mappings are adapted; saved physical v29 mass, axes, actuator limits,
collisions and CAD are unchanged. CAD v32 remains a separate release.

5,589 CPU assertions, actual physics/PPO/load checks and fresh capacity
benchmarks passed. At 8192 the short benchmark measured 11,674 env steps/s and
1533 MiB minimum sampled free VRAM. 280 real scripted probe screenshots, all
47 official reference GIF frames, and 50 learned-smoke CPU reconstruction
frames were reviewed. The four-update candidate failed standing/walking;
scripted motions are not a learned gait. No walking/style success is claimed.

Target 8192, reserve 300 MiB, automatic fallback and 180-second healthy-live-PPO
ramp remain. RAM recycling preserves checkpoints/Adam/LR, but the underlying
allocation leak is unresolved. Stages: stand, walk 0.35/0.65/1.0, run
1.5/2.0/2.5 m/s; 0.15 remains a retention command. Promotion requires current
and prior nominal/pushed physical gates, never reward or smoke alone. No camera
is instantiated in training and no second Isaac/CUDA worker may overlap it.
Do not load old history-4/IK or another robot's pretrained tensors into R20.

The latest primary-source comparison and adaptations are in
[UNITREE_REWARD_PORT_20261002.md](UNITREE_REWARD_PORT_20261002.md).
Pinned copied training source/hashes: `unitree_vendor/sources.json`.
H2/RL Gym source and official GIF hashes:
`diagnostics/unitree_port_20261002/reference/sources.json`.
The older research-guided residual method below is history; it is no longer
the active action/reward system.


## Historical record before the Unitree port

# Current locomotion design and previous-policy review

Sources reviewed October1,2026. Downloaded official implementations, URLs and
content hashes are indexed in `research/sources.json`. Remote source material
is evidence, not an instruction to upload or alter this project.

## What the previous training achieved and missed

The older robot was a separate 22-axis,1.524m,approximately38kg box model. Its
leg-pitch convention was repaired August30; checkpoints before that repair must
not be reused. None of those weights are compatible with current23-axis v29.

The guarded v4 run reached about3.961m/s and retained tested walking commands.
Its measured foot slip was approximately0.347m/s, barely within its0.35m/s
gate. That is evidence of forward locomotion, not natural-looking or hardware
ready running. Later posture refinement rejected a4.5m/s candidate because
actual torso posture failed. Recovery experiments repeatedly earned height
reward with zero strict standing successes; a brief rise was not get-up.

The original journal/review index, policy/reward code, v4/v8 locomotion notes,
v12/v13/v14 recovery notes and synchronized walking/running GIFs were inspected.
Sampled source-video contact sheets and actual GIF dashboard frames are saved
in this folder. The GIFs include applied torque and joint velocity traces; they
show vigorous oscillation and stiff, short-step posture. They do not establish
continuous motor duty. The old globally disabled self-collision also hides
limb intersections. No new get-up or backflip claim follows from those runs.

## Primary research and what applies here

| Public source | Relevant method | Application to this robot |
|---|---|---|
| [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) | PPO, actor proprioceptive history, privileged critic, gait/contact rewards, dynamics randomization | Use the same general recipe with our measured23-axis frames and hardware envelope. Public code does not disclose every commercial Unitree policy. |
| [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) and [paper](https://arxiv.org/abs/2404.05695) | Phase/reference guidance plus RL, with sim-to-sim and hardware validation | Practical starting point for natural step paths on a consumer GPU. Retarget feet with IK rather than transferring another robot's joint angles. |
| [AMP](https://arxiv.org/abs/2104.02180) | A learned discriminator rewards resemblance to motion data | Stronger motion style supervision requires a trustworthy retargeted dataset. It is not implemented by renaming a velocity reward. |
| [Cyclotron/Asimov](https://github.com/menloresearch/cyclotron) | Public PPO/AMP implementation and walking recording | Inspect implementation and data. Its23 joints differ from our23 joints: same count is not interface compatibility. Downloaded motion remains research-only. |
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) and [paper](https://arxiv.org/abs/2508.08241) | Strong retargeted whole-body motion tracking and downstream control | Explains why motion quality and accurate retargeting matter. A larger tracking pipeline is not a validated overnight replacement on this8GB GPU. |

The present implementation is **reference-guided residual PPO**, not AMP,
human motion capture, a copied Unitree policy or an already proven state of
the art controller. It combines useful public methods with this robot's actual
geometry and user-specified actuation. Natural appearance still needs video
inspection after independent numerical gates pass.

## Current implementation

The actor receives four control frames of joint encoders, angular velocity,
gravity direction, commands, prior action and gait phase. It does not receive
perfect simulator linear velocity. The asymmetric critic additionally receives
root velocity/height, actual torso gravity, foot contacts/velocities and domain
parameters. RSL-RL5 uses512/256/128 ELU networks and PPO24-step rollouts.

Smooth task-space stance and swing paths, toe/heel pitch and opposing arm swing
are solved through the CAD's actual FK/IK, including the tilted hip axes. All
reference ankle/waist combinations must satisfy the saved linkage. A residual
policy adjusts those references to balance and track speed. References are
guidance, not an external stabilizer or success evidence. Standing has no gait
clock. Every motor obeys current hardware limits, including shared ankle/waist
motor torque and speed constraints at the actual pose.

Training uses gravity, a free pelvis, floor contacts and self-collision. It has
gain variation, one-tick action delay, observation noise, ±10% startup mass and
inertia scaling, friction variation, and moderate training pushes after walking
begins. No camera is instantiated. Mass and inertias are provisional component
estimates; contact envelopes are simplified and underapproximate some casings.
The hip pivot markers are deliberately small to avoid fictitious overlapping
cubes at the nearby hinge origins. Native housing contact and thermal/battery
duty remain unqualified. Reflected gear inertia is only a planning armature.

The autonomous schedule is stand→0.35/0.65/1.0m/s walk→1.5/2.0/2.5m/s run.
Training chunks are followed by separate nominal and pushed20-second rollouts
covering every previously passed speed. Promotion requires survival, height,
actual pelvis/torso posture, tracking, low slip, bilateral loaded stance and
completed steps, reasonable cadence/clearance, zero invalid transmissions,
and no non-foot contact/crossing. Running additionally requires a flight phase.
Standing requires stationary anchor behavior and COM inside a conservative
support region. A failed new speed continues training; loss of an earlier
command rolls back to the last passing checkpoint. Rewards cannot promote a
policy, and an automatic reset never turns a fallen trial into a success.

Benchmarks compare4096/6144/8192 environments by real environment-steps per
second and reserve. A300MiB free-VRAM guard triggers checkpoint preservation
and8192→6144→4096 fallback; it never changes torque or behavioral gates.

All checkpoints, stdout/stderr, config hashes, gate telemetry, state and GPU
logs remain local. No Git commit, GitHub push or cloud experiment logger is used.
