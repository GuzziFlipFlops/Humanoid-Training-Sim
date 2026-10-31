# Corrected Unitree adaptation and active training — October 2, 2026

The fresh replacement is **r22_unitree_nominal**. Read
[ACTIVE_RUN.json](ACTIVE_RUN.json), its live state/progress, verified processes
and independent gates for status. The G1 core bodies/weights remain pinned;
the lifetime adaptations described in
[UNITREE_LIFETIME_REPAIR_20261002.md](UNITREE_LIFETIME_REPAIR_20261002.md) remain.
R20 and R21 are preserved diagnostic runs, not qualified policies.

## An implementation defect beyond reward incentives

The first Unitree adaptation assigned its nominal reference with
`self.stand.expand(num_envs, -1)`. That is a shared view, not per-robot storage.
The inherited reset writes `self.reference[env_ids] = randomized_pose`.
Consequently a randomized reset could overwrite the single standing-pose
tensor and change references for robots that had not reset. Both the action
origin and relative-position observations use that nominal tensor. The target
the actor was supposed to learn could therefore drift while training ran.

This was a bug in the adaptation. It undermines any inference about learning
from R20/R21. It is separate from the demonstrated negative-return termination
incentive, which is still repaired. The recorded r20 waist-limit failure is
real, but its existence alone cannot prove that an optimizer deliberately
chose it rather than being affected by a changing nominal pose.

The CPU regression executes the actual `_reference` method from frozen R21
and the inherited reset assignment. With the same seeded 0.012-rad reset
noise, 512 one-robot resets changed the original nominal pose by up to
**0.63665 rad (36.48°)** and changed unreset rows. This is an isolated code
reproduction, not a measurement of the precise drift in the old GPU run.
The fixed method produced **zero nominal drift**, independent reference rows,
and no changed unreset rows under the same sequence.

The repair allocates independent reference storage with `.clone()`. It does
not change the saved nominal angles, scale actions differently, or enlarge a
motor's valid region. Every real PPO update records and asserts
`nominal_pose_max_abs_drift_rad == 0`. The observation audit now correctly says
five history frames. The new checkpoint contract is
`physical_v29_history5_unitree_defaultoffset_v2`; old affected policies are
rejected. Actor/critic dimensions remain **400/415**.

The installed Isaac Lab `JointPositionAction` also clones its default-position
offset (`IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`,
line 195). The corrected private nominal origin follows that storage contract.

[Reset regression evidence](diagnostics/unitree_termination_20261002/nominal_reset_regression.json)
also verifies byte equality for the previous physical environment, independent
evaluator, motion data, mass, URDF, actuator/differential code and geometry, and
the G1 reward port. Only the declared nominal-reference implementation and
worker contract/diagnostic guard changed. No CAD joint axes or hardware limits
were altered. A fresh source-frozen actor is required; old actor, value and
optimizer tensors are not reused.

## Reward design and evidence retained

The [source comparison](UNITREE_REWARD_PORT_20261002.md) analyzes current
public H1/G1 RL Lab, older RL Gym H1/G1 and MJLab H2 tasks separately. The
active design retains G1's original core functions and weights, including
velocity tracking, contact-phase alternation, clearance, balance, slip,
joint/action smoothness and posture terms. Anatomy/dimensions are explicit.
H1's positive clearance coefficient is twenty times G1's, and its contact
phase reward is active at zero command. These properties were tested, not
assumed to prove that H1 itself cannot learn.

Two published components handle episodes:
[RL Gym](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py)
floors regular rewards before adding a fault term, while the
[MJLab H2 task](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/velocity_env_cfg.py)
uses a -200 physical-termination weight. The active aggregate is
`dt * (max(sum(original weighted G1 terms), 0) - 200 * physical_termination)`.
Timeouts are unpenalized. Per-cause termination rates distinguish waist/ankle
linkage faults from low height, tilt and impact. This is a G1 core with
**declared adaptations**, not an unchanged complete commercial Unitree policy.

The retained validation includes **5,593 core checks**, **149 lifetime checks**,
the 512-reset regression, and original helper/FK parity checks. Controlled
telemetry covers twelve intended/adversarial motion classes. Real free-base
Isaac probes test alternating targets, foot flicks, slide/hop/roll impulses,
and combined pitch/roll with normal hardware constraints. All **360 actual
screenshots**, **207 separately labeled smoke-pose reconstruction frames**,
and **47 official reference GIF frames** were inspected. The local
[frame review](diagnostics/unitree_port_20261002/frame-review.html) loads
**567 frames across fifteen cases**, with angles, targets, contacts and reward
terms. Future learned policies require their own visual review; the inspected
failed candidates do not resemble the reference gait.

R21's four-update deterministic candidate first fell at 4.18/4.40 seconds for
stand/0.15 m/s and did not walk. It is preserved as a test result; correcting
return incentives did not by itself qualify behavior. R21's nominal reference
bug was found before full standing training and its running gate was drained
cleanly. Its validation logs and checkpoints remain. R22 redoes real contact
physics, small PPO, reload and capacity checks with the corrected source.

R22's matching-source four-update smoke first fell at **3.28 seconds** while
standing and **3.38 seconds** at 0.15 m/s; the first causes were low height and
pelvis tilt. Both feet stayed planted before the forward fall, without
alternating steps. All four sampled PPO updates had zero nominal drift, and
the real weighted aggregation and physical-termination charges matched their
expected values. These are implementation checks, not learned locomotion or
evidence of improvement from a four-update candidate. See the
[live smoke audit](diagnostics/unitree_termination_20261002/live_smoke_audit_r22.json).

## Continuing and evaluating the chain

The corrected R22 capacity checks completed with the actual contact physics,
actuator mapping, reward port and PPO updates:

| Environments | Warm PPO environment steps/s | Minimum sampled free VRAM (MiB) |
|---|---:|---:|
| 4,096 | 8,348 | 3,843 |
| 6,144 | 10,141 | 2,689 |
| 8,192 | 11,391 | 1,531 |

Full standing PPO started at 8,192 after its first independent standing gate
failed. The first RAM recycle saved 17 full updates and resumed from
`008_stand_train_001/resource_recovery.pt` in the second worker. The actual
continuation checkpoint has matching contract/reward/physical hashes and all
17 Adam parameter counters advanced **420 → 440**, as expected for one PPO
update (five epochs × four minibatches). The actor also changed; optimization
continued rather than restarting. See the
[optimizer continuation evidence](diagnostics/unitree_termination_20261002/optimizer_resume_r22.json).

The RAM guard recycled the first worker when available RAM reached 1,789 MiB;
free VRAM was still 1,149 MiB. This was host-memory maintenance, not a VRAM
fallback. The first continuation also stayed at 8,192. At the **13:53 EDT**
handoff, 34 full updates were saved and the third worker was advancing at
8,192, with zero sampled nominal drift and 1,599 MiB sampled free VRAM.
Read the timestamped
[handoff snapshot](diagnostics/unitree_termination_20261002/handoff_status.json)
and live state for current counts. The unqualified initial 128-robot standing
gate had 0% twenty-second survival. No stage has been promoted at this dated
handoff; more reward or executed PPO updates alone are not progress evidence.

Target **8192 environments**, **300 MiB reserve**, automatic memory fallback,
and a **180-second healthy-live-PPO ramp** remain. Host RAM recycling retains
actor/critic/Adam/LR, but the allocation-growth cause is unresolved. One Isaac
CUDA worker runs at a time. The training has no stereo/render cameras; the
diagnostic recorder ran separately, and CPU frame reconstruction needs no
second simulation worker.

Stages are stand; walk 0.35/0.65/1.0; run 1.5/2.0/2.5 m/s, retaining 0.15 as an
independent walking test. Every promotion requires separate nominal/pushed
20-second current/prior gates. Reward, a visually appealing clip, a smoke
test or a time limit cannot promote a candidate. No policy is claimed to stand
or walk reliably before those gates pass. Exact imitation of Unitree's style
is not guaranteed by reward copying.

The robot remains physical-v29: **34.46334 kg** conservative mass estimate,
23 saved CAD axes, coupled rod transmissions, existing motor limits and
provisional contact/inertia model. CAD v32 is separate. Similar anatomical
joint names do not make those transmission dynamics identical to Unitree's.
The public task/PPO configuration is adapted, not its pretrained network
tensors. The explicitly declared running cadence/flight extension still
requires its own later behavior and retention tests.

Inspect live status without starting a second simulator:

```powershell
& 'E:\Aditya\robot_sim\physical_v29\training\monitor_training.ps1' -Watch
```

Source, reports, checkpoints and journal changes remain local; no Git commit
or push. Conditional shutdown applies only if the entire chain passes and
unsaved work is secured; these implementation checks do not satisfy that.
