# Training paused and performance measured — October 4, 2026

The user stopped the recurring four-hour requests and proposed pausing for an
optimization assessment. The heartbeat is PAUSED. R29's supervisor was stopped
after its identity and exact output directory were verified; the current worker
finished normally and saved all 64 updates. R29 is paused at **256 walking
updates**, with no training or profiling process remaining. The full actor,
critic, Adam moments and adaptive learning rate are preserved. All 43 frozen
source hashes still match. The offline status recorder now recognizes this
explicit paused state without incorrectly requiring a live supervisor.

Saved checkpoint:
`runs/locomotion_20261004_r29_host_capacity_checked/014_walk_035_train_004/candidate.pt`.
SHA256: `12b02dd709175b1e2ddf1f3cc6a6f4334f943bc21bd423879acf92d18b8031b1`.
See [pause_completion.json](pause_completion.json) and [evidence.json](evidence.json).

## What passed, and why this is still the 0.35 stage

Standing has matching independent nominal and pushed qualification. The R29
baseline's **0.35 m/s nominal case** passed: 0.322 m/s fixed-frame forward speed,
100% survival, 22/23 alternating landings, 4.63 degrees final heading error.
The entire stage did not pass. Its pushed 0.35 case failed heading, and retained
0.15 cases failed heading/net travel. Promotion requires all cases at the same
checkpoint in both tests. An earlier reference to a 0.35 pass must be understood
as that individual case, not complete stage qualification.

R29 therefore trains commands up to 0.35, retaining stopped and lower-speed
commands. It has not advanced to 0.65 or 1.0 m/s. The configured sequence is
stand, 0.35, 0.65, 1.0, 1.5, 2.0 and 2.5 m/s. The older R25 did reach 1.0 under
its old gates, but subsequent full-population direction analysis showed those
gates could accept turning. Its actor remains the seed, not proof of straight
1.0 m/s qualification.

The newest independently evaluated actor is at 192 updates; the newly preserved
256 actor has not been tested independently yet. The 192 actor survives every
case and takes 22/23 alternating steps. Its nominal 0.15 final heading improves
from 64.10 to 7.79 degrees, but speed overshoots to 0.262 m/s. Its nominal 0.35
final heading regresses to 68.41 degrees. This is real gait with incomplete
command/direction control, not evidence that the full reward system is solved.
The current actor should be preserved rather than discarded because one speed
improves while another regresses. Complete gate histories are in evidence.json.

## The training asset is already geometrically simple

The frozen training URDF contains **24 links, 23 revolute joints, 24 box
colliders and 24 box visuals**. It contains no CAD mesh colliders, camera,
finger articulation or simulated closed rod loops. Fixed-joint merging,
instanceable conversion and replicated physics are already configured. The
rod mechanisms are represented through their pose-dependent motor-space
transmission equations, rather than dozens of additional passive links.

Consequently, simplifying Fusion's shells or CNC parts would not address the
largest measured training cost. Any further geometry simplification belongs in
a separate simulation proxy, with retained joint frames, mass, inertia, sole
geometry and motor limits.

## Bounded same-physics profile

After draining training, one isolated Isaac worker loaded the saved actor and
unchanged frozen USD, at 4,096 environments. It used the same 5 ms physics,
four physics steps per control tick, 8/2 solver iterations, self collision,
contact sensors and CAD limits. Eight warmup ticks preceded 24 uninstrumented
ticks; eight additional ticks recorded nested CPU/CUDA-event section times.
This was a performance experiment, not a walking gate or a PPO benchmark.

Uninstrumented environment plus mean-actor inference: **7,416 samples/s**.
Instrumented control tick: **548 ms** on average.

| Section | Calls per control tick | CPU dispatch/wait time per control tick |
|---|---:|---:|
| Differential state/Jacobian | 18 | 304.7 ms |
| Hardware actuator compute, including differential calls | 4 | 250.4 ms |
| PhysX simulation step | 4 | 89.0 ms |
| Reward and diagnostic accumulation | 1 | 39.4 ms |
| Actor inference | 1 | 1.24 ms |

The differential row is nested inside other rows: **do not add all rows**.
It consumes approximately **55.6%** of sampled wall time; simulation stepping
consumes about 16.2%. CUDA-event intervals include launch gaps and are not
isolated GPU kernel busy time. This identifies a costly code path without
proving that every millisecond is Python overhead or that physics is free.
See [physics_profile/result.json](physics_profile/result.json), the complete
log and [profile_physics.py](profile_physics.py).

All four production blocks completed 64 updates at 6,144 environments without
resource recycling. Private RAM remained approximately 12,664–12,690 MiB;
CUDA allocation remained stable. Sampled pre-update KL and action/reward command
difference are zero. Across 37,748,736 main samples, active collection plus PPO
throughput averages **9,368 samples/s**. Collection occupies **94.1%** of active
time; optimizer work occupies 5.9%. Worker-effective rate is 8,502 samples/s.
Including maintenance startup/smokes and eight independent gates, rate to the
last saved result is 4,212 samples/s. These are distinct timing scopes.

The archived original five-foot walker delivered approximately 55,518 active
PPO samples/s at 6,144 environments, about 5.93 times this current active rate.
It used implicit PD, 80 actor inputs, 22 actions and disabled self collision;
it was a different physical/controller model. The current larger network is
not the largest measured bottleneck.

## Recommended optimization order

1. **Fuse or compile the exact CAD transmission calculations and batch the
   ankle/waist work.** Reduce repeated trigonometric/vector operations and GPU
   launches while preserving motor angles, analytic Jacobians, closure validity,
   singularity handling, combined pitch/roll torque limits and braking behavior.
   Cache only immutable constants or explicitly bounded results for a verified
   unchanged pose; invalidate on physics updates and resets. Do not reintroduce
   the old transient-tensor cache leak or reuse a stale Jacobian.
2. **Reduce overhead in diagnostics and duplicate data access.** Profile the
   reward/audit section separately; keep reward/contact computation and all
   independent gates. Preserve returned observation ownership and command timing.
   Optimize buffer lifetime and scalar summaries before shrinking policy history.
3. **Benchmark selective collision filtering or a minimal collision proxy as
   a separate physical change.** Keep soles, meaningful leg/body intersections
   and non-foot ground contact detectable. Globally disabling self collision or
   replacing coupled transmissions with independent ideal drives changes what
   the actor can physically do and requires fresh qualification.
4. **Resume compatible learned weights after tests**, rather than restarting
   from random weights. Benchmark complete stable capacity at 4,096/6,144/8,192
   and effective throughput, then rerun nominal, pushed and retention gates.
   Target8192/reserve300MiB/healthy180-second ramp remain the requested settings.

Exact-code optimization should precede changing the physical CAD. In this short
sample, completely eliminating the differential cost would yield a theoretical
ceiling of approximately **2.25 times** environment-step speed. An implementation
will retain some cost; matching the old 55k rate would require improvements
elsewhere as well. No speedup or convergence guarantee is justified yet.

The rewards produce alternating gait but are not proven across the requested
speeds/directions. More samples per hour help learning; they do not automatically
repair inaccurate commands, retention interference or reward loopholes. No
production reward, actuator, contact, mass, joint or CAD change was made in this
assessment, and training has not been restarted.

## Primary references and local comparison

Isaac Lab's [G1/H1 asset configuration](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/unitree.py)
uses implicit drives in the standard G1/H1 definitions and disabled self
collision. G1's minimal configuration specifically removes collision meshes.
Its standard solver settings are 8/4, so fewer solver iterations are not a
sufficient explanation for its advantage over this robot's 8/2 configuration.

NVIDIA recommends primitive colliders, avoiding unnecessary collision work,
asset instancing and profiling in its [performance handbook](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html).
Several of those measures are already present here.

The published [Isaac Lab v2.2.1 benchmarks](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/overview/reinforcement-learning/performance_benchmarks.html)
list G1 at 94k environment-only, 88k environment-plus-inference and 82k full PPO
samples/s, with 4,096 environments on RTX4090/Ubuntu22.04. This computer is
RTX4060/Windows. Those figures are useful references, not this hardware's target
measurement. The table has no directly comparable H1 row.

The [archived walker comparison](../old_walker_comparison_20261002/REPORT.md)
preserves the original policy, telemetry and throughput evidence. Neither it
nor this microbenchmark establishes hardware readiness or humanlike gait.

No CAD migration, Git commit/push, forced Fusion close or shutdown occurred.
Unrelated unsaved Fusion work is preserved. Recurring checks and the training
chain remain paused pending optimization work and a deliberate restart.
