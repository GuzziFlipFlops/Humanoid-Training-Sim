# Recovery v9: measured reverse curriculum, September 4

This candidate is isolated from the approved walking/running checkpoints and
from rejected recovery versions v1-v8. It is a simulation candidate, not a
hardware safety certification.

## Why the earlier policies stayed down

The failure was not just insufficient iterations. The old uniform action scale
could command the knee only to about 1.06 rad, although this robot's URDF allows
2.62 rad and successful humanoid recovery references use roughly 2.1-2.2 rad.
The policy therefore could not reach a useful tucked pose. The reward ordering
also made early attempts restart and pay failure costs more often than a policy
which remained on the floor.

v9 maps each normalized action asymmetrically across that joint's real URDF
range. Zero remains the validated neutral standing target. A direct retarget of
the official HumanUP reference was tested and rejected: it reached only 0.314 m
on this robot, so no result from that replay is labeled a success. The useful
HumanUP design lessons retained here are the two-stage discovery/refinement
structure, proprioceptive history, and curriculum instead of one impossible
floor-to-stand jump.

## Verified physical anchor

With the corrected action range, the neutral-target controller recovered all
16 robots from a 0.55 m crouch and passed the exact two-second dwell plus five
seconds of continued quiet standing. It had zero unsafe terminations and about
0.0023 m/s final planar drift. Evidence is in:

`E:/Aditya/robot_sim/humanoid_project/logs/guarded_getup_v9_crouch_hold_20260904_072857`

The learned actor starts with a zero output mean so it reproduces this measured
controller before PPO changes it. This is an initial controller, not copied
HumanUP weights.

## Reward and anti-exploit contract

- Signed potential-difference progress rewards net recovery progress. Repeated
  rise/fall cycles sum back to zero instead of farming height.
- The potential combines height, upright torso, and foot support. Uprightness
  while lying down cannot earn a standing reward.
- A discovery timeout has no event penalty. The immediate floor cost makes
  waiting unattractive without punishing moving policies more often.
- An unsafe reset costs 40 points: more than the entire 24-point progress
  budget plus the floor cost it could avoid. A violent high launch cannot win.
- Completion pays only after two seconds in the strict basin and five more
  seconds without walking, slipping, using hands/knees, drifting, or falling.
- Losing the stand revokes completion. A latched neutral-target balance
  controller takes over only after the learned recovery enters a conservative
  upright, two-foot-support basin.
- Actor observations contain four frames of deployable proprioception. The
  critic additionally sees geometry and reward history.

## Training schedule

The reverse curriculum tests fixed five-degree endpoints from 5 through 85
degrees, then a tucked supine state, audited supine poses, all four lying pose
families, and finally an optional low-energy refinement. Training at a tilt
samples every easier five-degree state as replay; evaluation stays fixed at the
named endpoint. Every candidate is independently evaluated and rollback is
automatic after repeated regression.

The base ceiling is 3,500 PPO iterations and the absolute ceiling is 4,200.
The full-size v9 preflight measured about 5.8 seconds per 8,192-environment
iteration, so training plus evaluations is expected to fit approximately 6-8 hours. Stages
that already pass are skipped. The supervisor requests 8,192 environments,
reserves 512 MiB VRAM, and retries at 6,144 then 4,096 only after measured
memory pressure.

Official references used:

- HumanUP: https://github.com/RunpeiDong/humanup
- HumanUP paper: https://arxiv.org/abs/2502.12152
- HoST: https://github.com/InternRobotics/HoST

No checkpoint is called successful solely from training reward. Promotion
requires measured height/orientation, two loaded feet, no other ground support,
low body/joint motion, bounded foot/body/yaw drift, and the full seven-second
hold. Supine and four-pose stages also require a later independent push test.
