# Five model-execution optimization iterations — October 4, 2026

Selected and applied **iteration 4**. The controlled 4,096-environment inference
benchmark rose from **11,505 to 37,741 samples/s
(3.28×)**. Actual same-count PPO rose from
**8,086 to 25,451
samples/s (3.15×)**.
At 6,144 environments it measured **31,858 PPO samples/s**.
This speeds data collection; it does not prove the current walker or guarantee
convergence. The current 0.35 m/s chain still requires matching independent gates.

## All iterations, including the rejected attempt

The user requested at least three and at most five iterations. Exactly five
source variants were created and tested. The unchanged baseline is a reference,
not a sixth optimization. Repeated tests and integration checks did not create
additional model variants.

| Iteration | Actual implementation | Controlled inference samples/s | Decision |
| --- | --- | --- | --- |
| Baseline | R29 frozen model and full 256-update checkpoint | 11,505 | Reference |
| 1 | Prepared immutable geometry and shared sine/cosine values; original arithmetic ordering retained. | 12,707 (1.10× baseline) | Valid, superseded |
| 2 | Bounded owned CUDA graphs for each exact differential state calculation; fresh pose copied every call. | 22,205 (1.93× baseline) | Valid, superseded |
| 3 | Full actuator capture, initial attempt | Failed capture | Rejected: CPU-to-GPU index conversion during capture. Logs/source retained. |
| 4 | Fixed capture-unsafe Python joint-index conversion with prepared CUDA index tensors; full actuation graph. | 37,741 (3.28× baseline) | Selected |
| 5 | Added pure graph calls for target projection and fresh post-physics differential validity; two declared environment-loop replacements. | 38,305 (3.33× baseline) | Valid, rejected: +1.5% is within short-run variation; additional environment changes |

Each directory has a frozen `source/physical_v29`, SHA manifest, launch arguments,
logs, results and trace arrays. The iteration-3 capture failure and iteration-4
guarded startup retry remain preserved. Failed CUDA capture and reserve-breach
workers hung in Kit close; only their verified owned processes were drained.
The one-worker guard blocked the next launch until they were drained. No
unrelated process or unsaved Fusion document was closed.

## Fidelity and why this is not an excessive simplification

The training asset already has 24 box colliders/rigid links and 23 revolute
joints, with no stereo vision, finger dynamics or passive closed-loop solver.
The speedup executes the same pose-dependent rod differential math more
efficiently. No physical CAD part or motor transform was moved or removed.

Iteration 4 changes only `cad_differential.py`, `hardware_actuators.py`, and the
new `owned_cuda_graph.py`. All other experiment runtime files are byte-identical
to the frozen baseline. Preserved: joint axes/limits, nominal offsets, collision
shapes/offsets, masses/inertias (34.46334 kg estimate), 95% assumed transmission
efficiency, motor torque and speed limits, simultaneous pitch/roll envelopes,
overspeed braking, gravity, self-collision, contact sensors, 5 ms physics steps,
four substeps, 8/2 solver iterations, actions, observations/history ownership,
resets/randomization, commands, rewards and independent gates.

Immutable geometry is prepared once per device/dtype. Each graph copies the
current inputs into static storage; no Jacobian is reused across a changed
pose. Results are cloned, so another replay cannot overwrite saved rollout
data. Caches have at most two graph entries per operation. CPU/gradient-bearing
inputs use eager tensor calculations.

Verification evidence:

- [Source audit](source_audit.json): every experiment manifest matches; every
  undeclared file is identical. R29's checkpoint was unchanged.
- [CPU math](iteration_01/math_cpu.json) and [CUDA graph math](iteration_02/math_cuda.json):
  bitwise equality for 13,513 wide-grid/random/zero poses for each of three
  differentials and both float32/float64, five coupled torque/speed/braking
  cases, changed shapes, noncontiguous inputs and owned prior outputs. Iteration
  4 has the exact same differential-file SHA as the CUDA graph test.
- [Full actuator parity](iteration_04/benchmark_4096_retry/actuator_parity.json):
  all eight optional position/velocity/feedforward modes, both dtypes, changed
  gains/poses/velocities and previous-output ownership pass bitwise comparison.
- [4,096-robot sampled replay](iteration_04/physical_parity_4096.json): all saved
  observations, joints, torques, root states, actions, targets, commands, rewards,
  termination flags, contacts, foot forces/positions and COM are bitwise equal.
- [Every-frame 72-robot replay](iteration_04/mixed_replay_parity.json): all 400
  frames and all 72 robots, 0/0.15/0.35/0.65/1/1.5 m/s commands, forced resets at
  tick 100 and lateral velocity pushes at tick 150, remain bitwise equal in
  every saved field. High commands in this diagnostic are not gait qualification.
- Actual eight-update PPO runs at 4,096 and 6,144 retain the full actor, critic,
  exploration, Adam, adaptive LR and training-only standing teacher. Every
  sampled pre-update KL is zero; checkpoint save and full reload succeed.
- [Continuation integration](continuation_integration.json): exact three-file
  allowlist and pinned reports; ten source-mutation counterexamples and a false
  locomotion-qualification claim are rejected. The real freeze path includes
  and hashes the new graph helper. Original checkpoints/frozen runs are intact.

## Throughput denominator and resources

Inference comparisons used seed 901, the same R29 actor, 4,096 environments,
24 warmup ticks and three 64-tick measured windows in fresh processes. The
observation returned by the previous step is consumed, matching PPO. The
earlier 24-tick profiler requested an extra observation each tick and added
instrumentation; its 7.4k figure is not this controlled baseline.

| Actual PPO test | Active samples/s | Effective samples/s | Scope |
| --- | --- | --- | --- |
| Baseline 4,096 | 8,086 | 7,903 | Eight updates, full continuation |
| Iteration 4, 4,096 | 25,451 | 24,143 | Eight updates, full continuation |
| Iteration 4, 6,144 | 31,858 | 30,492 | Eight updates, full continuation |
| Iteration 4, 8,192 | Not eligible | Not eligible | VRAM dropped to 214 MiB; reserve check stopped test |

Active PPO counts collection plus optimizer time. Effective PPO also counts
in-process logging/GC/guards, but excludes Isaac process initialization and
independent qualification. The whole-chain rate will be lower and must be
measured separately. These short tests do not certify long-term memory stability.
The archived old-model rate and public G1 RTX4090/Linux benchmarks are different
models/hardware. No comparable throughput guarantee is made.

## Selected continuation and preservation

Original R29 checkpoint SHA:
`12b02dd709175b1e2ddf1f3cc6a6f4334f943bc21bd423879acf92d18b8031b1`.
All benchmark-updated actors are diagnostic and discarded for main learning.
The exact original full checkpoint and qualified standing teacher seed the
fresh source-frozen continuation; the critic and Adam are not reset for an
execution-only change. The supervisor performs a fresh physical preflight,
PPO/load smoke, complete capacity tests and current nominal/pushed/stopped/
earlier-speed gates. Execution parity does not qualify walking or running.

Continuation at report time: `locomotion_20261004_r30_exact_graph_execution`, status `running`,
operation `train`. The authoritative fresh
state is `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r30_exact_graph_execution/state.json`. Target remains 8,192 with a 300 MiB VRAM reserve,
host guard, fallback and 180 continuously healthy seconds before an upward ramp.
Four-hour prompts remain disabled. No reward redesign, CAD geometry change,
commit/push, forced unsaved-document close or shutdown occurred.

The CUDA graph design follows [PyTorch's primary CUDA graph guidance](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/):
static storage, warmup and replay restrictions. See [experiment README](README.md),
[CPU/hardware helpers](actuator_parity.py), [reproducible benchmark](benchmark.py),
and [pinned continuation proof](continuation_proof.json).

## Fresh production capacity after restart

These use the actual frozen worker at stopped command, without the walking
standing-teacher gradient. They are distinct from the walking diagnostic above.

| Environments | Completed updates | Active samples/s | Minimum free VRAM MiB | Eligible |
| --- | --- | --- | --- | --- |
| 4,096 | 8 | 29,803 | 2725 | True |
| 6,144 | 24 | 36,363 | 1657 | True |
| 8,192 | 24 | 40,913 | 568 | True |

Current stage: `walk_035`; operation `train`. The selected next training count is 8192 environments. The 8,192 stopped-command test completed 24 updates above the reserves; walking must still obey the runtime guard and may fall back. No current walking qualification is inferred from capacity.
