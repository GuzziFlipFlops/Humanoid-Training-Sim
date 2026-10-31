# Net-travel repair — October 1, 2026

The user authorized checking stalled training, repairing reward exploits,
closing finished Fusion work and restarting with 8,192 environments as the target.
The new run is `runs/locomotion_20261001_r11_net_travel`; consult `ACTIVE_RUN.json`
and that run's live `state.json` for actual liveness, count and stage.

## Observed failure

R10 completed 14 nominal walking gates without qualifying walking. Its last
nominal candidate survived the 20-second trial but moved at only **0.0080 m/s**
against 0.15 m/s, with zero left/right steps. Standing failed the support check;
other recent candidates also fell. Fourteen training launches loaded the same
standing prior, plus one memory-recovery continuation. Thus 409 completed PPO
updates in that run were not 409 cumulative updates of one walking policy.

The v2 rectified velocity reward paid the forward half of rocking without
requiring net travel. Replaying 11,482 recorded planted-foot, pre-fall samples
gave positive tracking/progress rates of 0.0750/0.0709 despite no foot lifts.
V3 makes both zero on those samples. The separate reference term used a declared
zero-error upper-bound proxy. This is evidence of an incentive defect, not a
claim that all reward gains were deliberate exploitation.

Evidence: `checks/net_travel_20261001/audit.json`, including gate history,
checkpoint sources and recorded joint excursions. The old run, frozen source,
all completed checkpoints and telemetry remain intact. Its last worker was
allowed to finish and save before the owned processes were stopped.

## Changes

- `net_travel_v3` uses a fixed world heading and the furthest forward position
  reached during each moving bout. Positive tracking/progress require new
  territory and an unloaded foot or recent verified landing. Repeated travel
  over the same rocking excursion does not renew that credit. Backward motion
  still costs reward. Loaded-slip and physical-health restrictions remain.
- Dense swing guidance initially lasts 2.5 gait cycles and is renewed only by
  a valid step. Stationary leg lifting cannot collect it indefinitely. Positive
  command resampling cannot renew the frontier or discovery window.
- The candidate continues learning across failed retention tests. The trusted
  checkpoint remains separate and is replaced only after all nominal/pushed
  current- and earlier-command gates pass. No numerical gate was loosened.
  This removes the repeated 25–33-update rollback loop; it does not certify an
  unqualified candidate for use.
- One fifth of environments rehearse whole standing episodes, in addition to
  the existing mixed command sampling. Blocks have a 64-update minimum.
  The initial PPO learning rate falls from 0.0006 to 0.0002; the existing
  adaptive schedule remains. Critic/Adam reset on reward migration, while the
  standing actor and its repaired normalizer load exactly.
- New telemetry records fresh travel, unloaded-foot fraction and separate
  standing falls. Actor/critic dimensions, motor limits, contact asset, mass,
  reference file and independent evaluator are unchanged. CAD V32 stays separate.

## Validation and runtime

63 reward, step, reset, reference, gate, VRAM and status regressions passed.
One additional chain integration test passed: it exercises an intervening
retention failure, verifies candidate continuation and checks that the trusted
policy/promotion remain unchanged until all cases pass (64 tests total).

Fresh 64-environment PPO and checkpoint reload completed. The reload retained
standing in its nominal smoke trial; walking remained unqualified after only
four smoke updates. Startup benchmarks walking-command PPO at 4,096, 6,144 and
8,192 before selecting the highest safe requested count. Read `benchmarks.json`
and live state for measured results.

The completed fresh benchmarks measured:

| Environments | Environment steps/s | Minimum free MiB |
|---|---:|---:|
| 4,096 | 13,693 | 4,072 |
| 6,144 | 16,964 | 3,124 |
| 8,192 | 18,013 | 2,198 |

All counts preserved the reserve; the supervisor selected **8,192**. These are
short startup measurements, not an ETA or sustained walking qualification.

The preserved standing prior then passed fresh nominal and pushed gates in
`006_stand_gate_001_nominal` and `007_stand_gate_001_push`: survival 1.0 and
0.921875 respectively, with the unchanged 0.97/0.90 required thresholds.
The supervisor promoted standing and entered `walk_015`; walking still needs
its own bilateral-step, speed and stability gates.

The first full walking block, `010_walk_015_train_001`, was observed through
ten PPO updates at 8,192 environments with at least 1,992 MiB free VRAM,
no memory fallback and finite telemetry. See
`checks/net_travel_20261001/runtime_validation.json` for the exact final snapshot.
This startup evidence still shows no completed gait and is not a walking pass.

Fusion was closed through its normal exit after pausing its queued upload.
The native V32 archive and verified copy remain saved locally in
`C:/Users/rshan/Downloads/Humanoid_v32_CAD`.

The 300 MiB reserve, automatic memory fallback and 180 continuous seconds of
real training headroom before an upward step are retained. The chain has an
18-hour limit. A failed gate continues learning; expiry leaves the stage
unqualified. Runtime/regression success is not proof of learned humanlike gait.

```powershell
& 'E:\Aditya\robot_sim\physical_v29\training\monitor_training.ps1' -Watch
```
