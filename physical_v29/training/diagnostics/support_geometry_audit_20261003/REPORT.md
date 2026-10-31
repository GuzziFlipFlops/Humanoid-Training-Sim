# R23 status and sole-support audit — October 3, 2026 UTC

At 03:24 UTC, R23 qualified standing after 512 full PPO updates. Both independent
128-robot, 20-second nominal and pushed evaluations passed every existing check.
The supervisor automatically promoted the saved candidate to walking at 0.35 m/s.
The first independent walking evaluation is running; learned walking is not claimed.

| Standing measurement | Nominal | Pushed |
| --- | ---: | ---: |
| Survival | 100% | 100% |
| Pelvis tilt | 0.732 degrees | 0.829 degrees |
| Final drift | 0.0407 m | 0.0414 m |
| Mean planar speed | 0.0114 m/s | 0.0472 m/s |
| Foot slip | 0.00306 m/s | 0.00342 m/s |
| Both loaded feet | 100% | 100% |
| Support-outside samples | 0% | 0.872% |

Qualified checkpoint:
`runs/locomotion_20261002_r23_standing_curriculum/047_stand_train_008_part_04/candidate.pt`.
Evidence: `048_stand_gate_009_nominal` and `049_stand_gate_009_push` evaluation
JSON and synchronized telemetry. One actual CUDA worker and the owned supervisor
were verified; all 41 frozen-source hashes matched. PPO targets 8192 with the
300 MiB VRAM reserve, fallback and three-minute healthy-memory ramp. Independent
evaluations use 128 environments. Host-RAM growth and checkpoint recycling remain.

## New diagnostic idea: support bounds versus actual sole geometry

The 448-update gate survived quietly with both feet loaded but reported 95.05%
of global samples outside support. The existing helper uses a yaw-frame rectangle
centered on the foot-link origins, with half sizes 55/35 mm. The frozen sole boxes
are 238.44 by 101.51 mm and their centers are offset 47.47 mm forward from those
origins. This conservative region is different from the actual sole footprint.

The new CPU-only `training/audit_sole_support.py` reproduces the saved old support
flags and compares actual simulator COM with frozen-URDF FK-transformed sole
geometry. It does not launch Isaac, infer COM from an approximate mass model,
modify the active run or award qualification. The audit samples every tenth
post-warmup tick of the first 16 saved robots: 90 ticks and 1440 samples per gate.

| Diagnostic outside fraction | 448-update sampled gate | 512-update sampled gate |
| --- | ---: | ---: |
| Existing ankle-origin rectangle | 93.40% | 0% |
| Same rectangle sizes, actual sole centers | 0% | 0% |
| Full transformed loaded-sole convex hull | 0% | 0% |
| Sole hull inset 10 mm longitudinal / 5 mm lateral | 0% | 0% |
| Hull of corners within 2 mm of lowest loaded corner | 0% | 0% |

The reconstructed foot transforms agree with simulated foot positions within
0.000017 m. Mean sole vertical corner span was 0.066 mm at 448 updates and
0.010 mm at 512, so the inspected sole bottoms were nearly horizontal.

This supports a possible overly conservative or miscentered support-gate failure
in the earlier pose; it does not establish actual pressure distribution, CoP,
ZMP or hardware balance. Global 128-robot results and first-16 audit fractions
have different scope. The latest policy passes even the original conservative
gate, so no gate/reward edit or restart was made. Retain progress, and investigate
actual contact geometry if this discrepancy becomes a future blocker.

Raw evidence: `sole_support_audit.json`, `latest_512/sole_support_audit.json`,
and `training/diagnostics/stage_rewards_20261002/live_handoff.json`.

## Recurring follow-up restored

Automation `humanoid-training-research-and-repair-every-four-hours` is ACTIVE,
attached to the current thread and recurring every four hours. The thread's
read-only runtime metadata was verified as `gpt-6.1-sol`, reasoning `max`.
The heartbeat inherits those settings; it has no independently pinned model.
First expected check: October 3 at approximately 03:16 EDT / 07:16 UTC.

The prompt requires fresh status/process/resource/gate/telemetry evidence,
primary-source research when progress stalls, new diagnosis experiments,
tested justified changes, checkpoint preservation and a fresh frozen run when
training source changes. It prohibits arbitrary churn or promotion from reward,
and retains nominal/pushed/prior-skill gates. It may redesign and restart without
asking for confirmation within the user's authorization. Conditional shutdown
remains limited to independently qualified completion of the entire requested
chain, after final artifacts and unsaved-work checks.
