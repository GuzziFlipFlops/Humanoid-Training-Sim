# Recovery v8: safe floor-escape repair, September 3

## Why v5 stayed down

v5 did not crash. Its first stage reached only 0.106 m mean maximum pelvis
height after 400 iterations. The main reward totals show why: new-best progress
paid about 1,792 points, while the episode-end failure term charged about
421,890 points. An attempt that moved and failed early restarted sooner and
could be charged repeatedly; lying still delayed that charge until timeout.
The 0.25-rad action scale and 0.18 exploration also left too little recovery
motion. Evidence is preserved in
`E:/Aditya/robot_sim/humanoid_project/logs/guarded_getup_v5_20260903_185757`.

## v8 changes

- Failure at a discovery timeout is no longer a negative event. Strict success
  is still determined only by independent evaluation.
- A continuous height-times-upright reward is zero near the floor and increases
  during a real rise. Upright rolling on the floor cannot farm it.
- Positive vertical velocity receives a small reward before the standing basin.
- Remaining below 0.22 m receives a small continuous cost that immediately
  decreases as the pelvis rises.
- Early discovery style, power, ankle and collision-margin costs are weakened;
  severe impact/penetration remains a hard termination.
- Recovery action scale is restored to 0.70 rad. Migration neutralizes the old
  standing action head and begins with 0.15 stochastic exploration. A smoke at
  0.30 exploration caused excessive unsafe resets; 0.15 reduced that to about
  3% in the four-iteration smoke while remaining over twice v5's joint-space
  exploration amplitude.
- A production-size v6 probe exposed a large-batch-only failure: unsafe resets
  jumped from 0.9% before the first PPO update to 66.9% after it. v7 caps raw
  actions at +/-1 (still +/-0.70 rad), reduces the discovery learning rate to
  1e-4, and bounds every optimizer-step change in the action head. The failed
  v6 run is preserved at `logs/guarded_getup_v6_20260903_215836`.
- The longer v7 smoke exposed a new escape: an unsafe early termination avoided
  the continuing floor cost. v8 charges 10 points only for an objectively
  unsafe recovery termination. Safe failed attempts and timeouts remain free.
  Dense rise credit now begins just above the measured 0.08 m reset height.
- Once a recovery enters the conservative upright/two-foot basin, a separately
  audited controller takes over. It previously passed 16/16 strict trials with
  zero unsafe events and 0.0023 m/s final drift. A rise followed by walking or
  falling still fails the 2-second dwell plus 5-second continued-hold audit.

## Schedule

| Stage | Assistance | Iteration ceiling | Gate |
|---|---:|---:|---|
| Supine discovery A | 20% body weight | 600 | mean max height >=0.50 m, <50% unsafe |
| Supine discovery B | 10% body weight | 400 | mean max height >=0.62 m, <50% unsafe |
| Supine recovery | 0% | 800 | >=90% nominal and >=85% pushed success |
| Supine/prone/sides | 0% | 600 | same per-pose gate |
| Refinement | 0% | 400 | same per-pose gate |

The 2,800 iterations are ceilings; the supervisor evaluates every 200 and can
advance early. It requests 8,192 environments with a 512 MiB reserve and falls
back only on measured memory pressure. No checkpoint is called successful
unless the unassisted evaluator observes correct height/orientation, two loaded
feet, no other support, low motion, bounded drift, and the full seven-second
hold. Training can still fail to converge; the gates prevent false promotion.

This is a simulation policy candidate, not hardware certification.
