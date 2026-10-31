# Walking reward repair — October 1, 2026

Historical v2 report. The later authorized r11 repair is documented in
[NET_TRAVEL_REPAIR_20261001.md](NET_TRAVEL_REPAIR_20261001.md); use ACTIVE_RUN.json
for current state. V3 fixes remaining rocking credit and repeated rollback.

The old r4 policy passed nominal and pushed standing, then failed walking.
Its inspected 0.35 m/s rollout stayed upright but averaged 0.0057 m/s with no
completed steps. Hip/knee/ankle oscillation was real, not forward locomotion.
Later blocks did not establish a reliable gait. Increasing training reward
was not useful evidence of improvement.

## Current continuation

R6 revalidated standing. R10 now resumes the same frozen worker, reward,
optimizer and physics sources with automatic three-minute VRAM ramps,
timing-adjusted blocks, a 300 MiB reserve and an 8,192 ceiling. Walking is still
unqualified. Read ACTIVE_RUN.json and the live gates for current progress;
the r6 restart/count/budget details below are historical. See
[capacity and monitoring](AUTOMATIC_VRAM_20261001.md).

## Cause and changes

The previous Gaussian velocity term paid a stationary robot about 54% of its
tracking maximum at a 0.35 m/s command. Posture, joint reference averaged over
23 axes and contact-pattern matching added reward while both feet stayed down.

`measured_steps_v2` replaces the commanded-walking terms:

- Tracking needs signed forward progress, with zero tracking credit at zero
  speed. Loaded-foot sliding attenuates it. Backward motion has negative progress.
- A genuine step needs at least 60 ms loaded stance, 80 ms uninterrupted air,
  12 mm clearance of the lowest sole corner, forward foot placement, at least
  3 mm pelvis advance and 60 ms loaded landing. Forward placement is measured
  along the heading captured at liftoff. Credit alternates sides, is consumed
  once and has a minimum interval. Simultaneous walking landings do not get two
  step bonuses. A rejected landing cannot become a step by sliding afterward.
- Actual unloaded swing/clearance and CAD-derived foot positions supply dense
  guidance before a complete step succeeds. This shaping is not success.
  Planted-foot contact-pattern matching no longer earns walking gait credit.
- The joint guide uses the 12 leg axes; healthy arm/neck axes cannot dilute a
  leg mistake. Its positive reward requires actual progress or unloaded swing.
- A small walking posture/survival floor remains to avoid making immediate
  falling the most profitable exit. Zero-command standing keeps its settled
  reward and has no gait reference; positive commands start at 0.12 m/s.
- Added a strictly verified 0.15 m/s walking stage before 0.35 m/s. Every later
  gate includes this command and previously passed commands. No camera,
  external balancing force, actuator increase or success reset is added.

Contact means positive vertical sole load rather than the magnitude of any
foot force. Clearance uses the eight corners of the unchanged provisional
sole box, including foot orientation. Gait duty interpolates the existing IK
clips, fixing the old running-duty mismatch. This still does not qualify real
housing collisions, thermal duty or natural human appearance.

The public [Unitree rewards](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py)
and [Humanoid-Gym environment](https://github.com/roboterax/humanoid-gym/blob/main/humanoid/envs/custom/humanoid_env.py)
were revisited for command-gated contact/clearance guidance. The ordered step
and signed-progress repair is specific to this robot's observed failure; it is
not a copied commercial policy or AMP implementation.

## Validation and evidence

28 CPU reward/step/reference/normalization/capacity tests passed. They cover actual alternating steps,
stationary bobbing, sliding, chatter, in-place and backward kicks, insufficient
clearance, same-foot farming, hopping, delayed credit, yaw, resets, termination
and control-dt integration. The CAD foot table matches the original FK.
All 14 independent behavioral gate tests and three Windows status-write tests
also passed. Existing numerical gate thresholds are not relaxed.

`checks/reward_repair_20261001/recorded_walker_audit.json` replays 11,520
both-loaded samples of r4's `039_walk_035_gate_004_nominal/telemetry.npz`.
Positive locomotion reward rate fell from 2.2411 to 0.07999, a **96.43% reduction**.
Gait and completed-step credit are exactly zero. The old record lacks per-foot
positions; ambiguous contact samples are excluded and no missing swing data is
fabricated. This is an incentive regression, not evidence that walking learned.

New PPO progress JSON contains actual moving speed, progress, completed-step
rate, swing quality, loaded-foot slip and falls. New gate telemetry adds root
poses, per-foot positions, contacts and lowest-sole clearance for future audits.
The independent evaluator does not use the training step tracker or reward to
promote a policy; it still requires speed, bilateral steps, survival, clearance,
slip and previous-command retention in separate nominal and pushed processes.

## Historical restart and preservation

New run: `runs/locomotion_20261001_r6_measured_steps`. Read `ACTIVE_RUN.json`
and its live state before claiming a current stage or environment count.

The r4 tree was stopped by checking its supervisor/launcher command line. All
completed candidates, frozen sources, gates and GPU logs remain. Its state is
`superseded_for_reward_repair`. Two previously completed preflight processes
were also checked and closed; they were still holding GPU allocations.

The new chain starts from the trusted standing actor at
`r4_8192/027_stand_train_013/candidate.pt`, not the failing walker. Its standing
normalizer had zero variance for the unused command and phase channels across
112 million observations, producing normalized phase inputs up to 100 when
walking was first requested. A preserved checkpoint copy changes only those
20 bounded input statistics to mean 0/std 1. Physical-input statistics, count
and actor weights remain unchanged. All 256 CPU standing-input trials give
exactly identical actor outputs; the independent live stand gates revalidate
the transformed copy before promotion. See `repair_actor_normalization.py`
and `checks/reward_repair_20261001/stand_normalization_repaired.json`.
The checkpoint copy is checked unchanged on load. Because the reward
changed, old critic estimates and Adam moments are discarded; training keeps
a modest 0.12 residual exploration floor on that first migration. Subsequent
v2 blocks resume their own critic and optimizer normally. Every save, including
periodic saves, records reward, asset, motion and actuator provenance.

R5 completed contact/transmission preflight, four real PPO updates, checkpoint
reload execution and new 4096/6144/8192 benchmarks. Smoke and reload did not
pass walking behavior; their scope is runtime/checkpoint execution. The 8192
benchmark crossed the reserve at 243 MiB free with Fusion open. R5 then stopped
on a capacity-selection assertion, which has been fixed. R6 reuses these
unchanged-source checks with explicit provenance and selects the eligible
6144 fallback; no rejected 8192 result is treated as eligible. It independently
revalidates the normalized standing checkpoint before learning the easier
walking stage. It requests 8192, retains the user's
300 MiB reserve and 8192 -> 6144 -> 4096 fallback, uses 25-update blocks and a
12-hour ceiling approximately preserving the prior run's remaining budget.
Memory fallback never changes hardware or pass thresholds.

No claim of learned walking follows from this repair or its synthetic tests.
Local source, snapshots, checkpoints and documentation only; no Git commit,
GitHub push or external experiment logging.
