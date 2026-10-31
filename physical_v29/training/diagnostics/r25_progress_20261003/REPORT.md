# R25 walking progress and landing replay — October 3, 2026

R25 is making measurable independent progress, so this check preserved its
actor, optimizer, rewards and frozen source. Standing has qualified again in
this run. Walking and running have **not** qualified. The latest matching
nominal/pushed evaluations are at 320 walking updates; subsequent training is
continuing at 8192 environments. Read `ACTIVE_RUN.json` and the refreshed
[live handoff](../walk_regression_20261003/live_handoff.json) for current counts.

Final ownership refresh at 11:50 UTC / 07:50 EDT: **384** walking updates are
checkpointed. Operation `047_walk_035_gate_007_nominal` is running at 128
evaluation robots; next training count remains 8192. Supervisor 7400 owns the
single actual worker 31964. Available host RAM is 18,865 MiB and evaluation
free VRAM 6103 MiB; this evaluation reading is not an 8192 capacity benchmark.
All 42 frozen hashes still match. The 384-update gate result is pending; the
last completed matching pair and gait analysis below remain at 320 updates.

## What improved, and what still fails

Every gate uses 128 free-base robots for 20 seconds. Walking gates split fixed
commands across stopped, 0.15 and 0.35 m/s cases. The table below refers to the
42 robots commanded to walk at 0.35 m/s; step counts are means per robot after
the two-second warmup. All startup falls still count against survival.

| Walking updates | Nominal speed | Survival | Left/right qualifying steps | Qualifying cycles/s | Loaded-foot motion metric |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.002 m/s | 100% | 0 / 0 | 0 | 0.0035 m/s |
| 64 | 0.002 | 100% | 0 / 0 | 0 | 0.0025 |
| 128 | 0.002 | 100% | 0 / 0 | 0 | 0.0041 |
| 192 | 0.001 | 100% | 0 / 0 | 0 | 0.0038 |
| 256 | 0.405 | 100% | 0 / 0.05 | 0.0013 | 0.1396 |
| 320 | 0.315 | 100% | 1.79 / 2.43 | 0.1171 | 0.1068 |
| 320, pushed | 0.316 | 100% | 1.50 / 2.76 | 0.1184 | 0.1079 |

The move from mostly translating without qualifying steps at 256 updates to
several real qualifying steps at 320 is progress. Loaded-foot motion also fell
about 23.5%. It is still a shallow asymmetric shuffle. The nominal gate fails
minimum bilateral step counts, step symmetry, cadence and alternating landings.
Repeated same-foot landings are 39% nominal and 36% pushed, versus a 2% limit.
The nominal target cycle rate is 1.075 Hz; 0.117 Hz is far below it. Both legs
must make at least five qualifying steps. The 0.15 m/s case still mostly stands
and fails movement/step gates. No promotion is justified by forward speed.

Standing retention passes both latest gates: nominal drift 0.01586 m and pushed
drift 0.01589 m, with 100% survival in each stopped group. Fresh standing gates
009/010 previously passed in R25; qualification was not inherited from R23.
The supervisor's generic `retention_failures` includes the unlearned 0.15 m/s
case and should not be read as repeated loss of independently verified standing.

## New diagnostic: why the left steps are rejected

[audit_r25_gait_progress.py](../../audit_r25_gait_progress.py) replays the exact
frozen independent landing classifier on three saved 1000-tick, first-16-robot
traces. All 192,000 scalar landing/hop/repetition comparisons match exactly.
This does not replace the full gates; it checks their recorded event accounting.

The five sampled nominal walkers have 78 settled, air-qualified left-foot
candidates at 320 updates. Four qualify; **74 fail sole clearance alone**.
Forward foot placement and body advance pass for these candidates, and none is
rejected as a paired hop. Right-foot candidates qualify 17/18. The pushed sample
shows the same pattern: left 3/75, with 72 rejected for clearance; right 13/13.
At 256 updates this sample had no left candidates and only one accepted right
step. The classifier has not hidden repeated same-foot landings.

The sampled left feet reach only about 10–13 mm maximum clearance, compared
with about 32–34 mm on the right. Robot 8 has a 14-degree left-knee excursion
versus 21 degrees on the right. Its left ankle is near the lower URDF stop in
13% of post-warmup ticks and has about 0.24 rad target tracking RMS error.
These are concrete follow-up measurements if improvement stalls; they do not
establish an incorrect joint axis or justify changing CAD positions. Live
training ankle targets clip roughly a third of the time, while sampled motor
torque saturation remains zero. Differential/nonfinite failures are absent in
the latest nominal walking gate.

I inspected the [frame sheet](041_walk_035_gate_006_nominal_frame_sheet.png) and
[synchronized plot](041_walk_035_gate_006_nominal_robot_8_telemetry.png). The
frames show small knee/ankle pulses and unequal unloading. They are CPU
reconstructions of saved simulation states using the frozen URDF, **not Isaac
screenshots**. Foot FK agrees with measured positions within 0.0192 mm. These
frames do not show a Unitree-like natural gait or certify contact physics.
Robot 8's lowest sole corners have appreciable motion while loaded; the mean
minimum near-ground corner motion is 0.0769 m/s. An interior contact point is
not sampled, so this is a kinematic diagnostic rather than measured frictional
slip or contact pressure.

## Support geometry and public Unitree rewards

The new pushed sole audit confirms the standing support-frame issue remains
separate from walking. Among 540 sampled stopped first-episode poses, the
ankle-origin rectangle reports 2.04% outside; transformed sole footprints,
inset footprints and near-ground corner hulls each report 0%. The collision
sole is centered approximately 47.5 mm forward of the foot-link origin. This
diagnostic changed no gate and does not qualify a policy. Moving COM projection
outside one instantaneous support footprint is not itself a dynamic fall test.
See [grouped audit](progress_audit.json) and [raw sole audit](sole_support_push_320/sole_support_audit.json).

I reviewed current public G1/H1 source again. Both use contact-phase gait
feedback with weight 0.5 and a loaded foot-velocity cost of -0.2. G1 uses an
0.8-second phase period; H1 uses 0.6 seconds. Their clearance weights are 1
and 20 respectively, with different target foot-link heights. These are
robot-specific scales, not interchangeable sole-clearance weights.
[G1 configuration](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py),
[H1 configuration](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py).

The public phase reward matches contact timing; it does not verify meaningful
forward landings. Its clearance kernel can reward zero foot velocity even
without lift. R25's declared adaptation already uses actual unloaded sole
clearance in expected swing and maintains separate landing gates. Copying H1's
clearance weight 20 onto that different formula is not an evidence-backed fix.
[Public reward implementation](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py).

The verified original five-foot walker achieved about 0.994 m/s with repeated
bilateral steps and zero inspected falls. Its approximately 0.473-second stride
and 55,518 environment steps/s differ markedly from R25. It used 80 actor
inputs/22 actions and simplified drives; R25 uses 400/23, CAD constraints,
self-collision and torque/slew limits. Old weights cannot be loaded directly.
See the [preserved comparison](../old_walker_comparison_20261002/REPORT.md).

## Learning and resources

At the 11:39 UTC ownership snapshot there were 339 checkpointed main walking
updates and one actual Isaac worker. A subsequent CPU row read observed 358
optimizer updates, including the active part. Every sampled pre-update KL,
including reset rows, remains zero across these full updates. The observation
ownership repair remains effective. Latest adaptive LR is about 0.00017,
post-update KL 0.0122 and raw action clipping about 0.33%; these diagnostics do
not prove convergence. All 42 frozen source hashes matched.

Main training averages approximately 11,686 environment steps/s during PPO,
8,782 including completed worker startup/exit/recycling, and 5,600 including
independent gates and scheduling since standing promotion. These denominators
are different. Bounded capacity tests still favor 8192 over 6144 by 12.1% raw
and 7.8% effective throughput. The original walker sampled about 4.75 times
faster than current raw training; elapsed time alone cannot explain gait defects.

Host private memory grows roughly 710–720 MiB/update in long workers. The guard
saves checkpoints near 2048 MiB available RAM, releases the worker and resumes
actor/optimizer state. Operation 043 saved after 19 updates and operation 044
continued successfully. Minimum saved free VRAM is 1105 MiB, above the 300 MiB
reserve. The 180-second healthy upward ramp and fallback remain configured.
The prior CPU observation-lifetime test stayed stable; native memory growth is
still unresolved. A shader-cache initialization message did not stop the
completed worker. See [resource/PPO snapshot](training_health.json).

## Decision and next check

Continue the improving R25 run without changing reward or physical source.
Recheck actual bilateral clearance, air duration, placement, cadence and stopped
retention at the next matching nominal/pushed gates. If these plateau or
regress, use the new rejection decomposition and joint tracking data to test
one justified change rather than restarting on the clock. Running has not
started; its alternating steps, flight, cadence, clearance, travel and prior
skills remain mandatory before promotion. No live reward edit, new simulator,
CAD migration, Git commit/push or shutdown occurred in this check. The existing
four-hour heartbeat remains ACTIVE.

Reproduce the saved-event audit with:

```powershell
& .\env_isaaclab\Scripts\python.exe physical_v29\training\audit_r25_gait_progress.py `
  --run physical_v29\training\runs\locomotion_20261003_r25_owned_observations `
  --output physical_v29\training\diagnostics\r25_progress_20261003 `
  --evaluation 035_walk_035_gate_005_nominal `
  --evaluation 041_walk_035_gate_006_nominal `
  --evaluation 042_walk_035_gate_006_push `
  --support-audit physical_v29\training\diagnostics\r25_progress_20261003\sole_support_push_320\sole_support_audit.json
```
