# Automatic training capacity — October 1, 2026

The active chain is `runs/locomotion_20261001_r11_net_travel`. Always read
`ACTIVE_RUN.json` and the indicated live `state.json` before quoting progress.
R11 closed Fusion and freshly benchmarked 8,192 at a minimum 2,198 MiB free;
it selects 8,192 with 64-update minimum blocks. The 300 MiB reserve, fallback
and 180-second upward qualification below that ceiling remain. See
[net-travel repair](NET_TRAVEL_REPAIR_20261001.md). The r9/r10 history follows.

The old supervisor was replaced after its owned worker finished; its checkpoint,
optimizer and frozen policy/physics source were preserved. Source identity is
checked before resuming. No simultaneous second GPU simulator was started.

The r9 increase from 3,072 to 4,096 used 326.8 continuous seconds of actual
PPO allocation measurements: the final sample had 1,342 MiB free, against an
812 MiB requirement. This proves a capacity decision, not walking progress.
Standing has a passed prior; walking has not passed its independent gates.

R10 also applied the 4,096 → 6,144 increase after a completed checkpoint.
Its final qualification was 232.05 seconds, with 1,374 MiB free against the
1,324 MiB requirement. The following 6,144-environment block was observed
advancing real PPO updates. The exact decision is
`003_walk_015_train_001/environment_ramp.json` in the r10 run. This is an
observed capacity change, not a walking pass; read live state for later counts.

`vram_ramp.py` observes free memory every three seconds while real PPO updates
are fresh. Startup and 128-environment evaluation memory cannot qualify an
increase. It requires 180 continuous seconds, resets on low/missing memory or
a sampling gap, and adds one tier at a time:

`512 → 1,024 → 2,048 → 3,072 → 4,096 → 6,144 → 8,192`.

Required free memory is **300 MiB plus the estimated allocation for the added
environments**. The initial estimate is 0.5 MiB per added environment, above the
measured 0.357 MiB/env slope. More expensive measured slopes raise that estimate.
An increase is applied after a completed checkpoint, to the next training
worker. Independent gates still run first; their temporary free memory is not
evidence for a larger training allocation. The count cannot be changed inside
an already constructed Isaac scene.

The worker still saves a recovery checkpoint and falls back if free memory
drops below 300 MiB. A memory failure clears the ramp, adds a three-minute
cooldown and increases the estimated incremental cost by 25%, avoiding repeated
immediate retries at the same count. Available memory is an estimate; a newly
opened app or later allocation can still cause fallback. Hardware assumptions,
rewards, 23-axis dynamics, evaluation thresholds and retention checks are
unchanged. The new supervisor has an 18-hour ceiling and preserves unpassed
stage evidence when that ceiling is reached.

Fast 25-update blocks sometimes end before the three-minute observation window.
The r9 block reached 171.75 seconds despite sufficient headroom, so r10 adjusts
block length from measured update time: at least 25 updates, enough for roughly
210 seconds below the 8,192 ceiling. It still observes actual memory for 180
seconds; predicted timing alone never approves a ramp. At 8,192 it returns to
the configured 25-update minimum. The worker/checkpoint/reward sources are unchanged.

Nine CPU capacity-ramp/timing tests pass. A combined 31-test run also passed existing
capacity-selection, behavioral-gate and Windows status-write checks. Live
`state.json` records `vram_ramp`, `next_training_environment_count`, GPU samples
and policy behavior. Each actual upward decision writes `environment_ramp.json`
beside the completed training block. `gpu.csv` records the underlying samples.

Run this PowerShell command for live status:

```powershell
& 'E:\Aditya\robot_sim\physical_v29\training\monitor_training.ps1' -Watch
```

Fusion v30 changes are **not** silently incorporated into this chain. The
training asset remains frozen physical v29, with its provisional 34.46 kg mass
estimate and configured actuator/linkage assumptions. V30 mass, contact meshes
and actuator packaging need a separate export/requalification before a future
training migration. All records and documentation remain local; no Git push.
