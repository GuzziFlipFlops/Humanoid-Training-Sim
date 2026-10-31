# Guarded running candidate — 2026-09-02

This is a separate experiment, not a replacement for either archived policy. Changes live in Python code, not an Isaac GUI reward editor. No CAD, joints, masses, actuator limits, action clipping, or old task registrations are modified.

## Evidence and starting point

- Approved walker: `E:\Aditya\robot_sim\humanoid_project\logs\rsl_rl\humanoid5ft_natural_walk\2026-08-31_17-06-10\model_2098.pt` (native action scale 0.35).
- Existing runner: `E:\Aditya\robot_sim\humanoid_project\logs\rsl_rl\humanoid5ft_natural_run_from_working_walk\2026-09-01_19-54-34\model_3800.pt` (native action scale 0.45).
- Detailed earlier evidence: `E:\Aditya\robot_sim\humanoid_project\logs\gait_diagnostics_20260902\REPORT.md` and its nominal/randomized telemetry.
- New evaluator preflight: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v1_preflight\runner_baseline\evaluation.json` and `telemetry.npz`.

The earlier data found phase-aligned knee target asymmetry, not merely a lagging actuator. The fresh matched nominal test measured stance-duration asymmetry at 1 m/s of 0.338 for the approved walker and 0.561 for the runner. At 0.5 m/s the figures were 0.561 and 0.726. Zero would mean equal average stance durations. The reference walker is therefore not perfectly symmetric either.

The supervisor tests the runner against the archived walker at 0.5 and 1 m/s, under nominal conditions and matched lateral kicks. If the runner fails, it selects the approved walker and verifies its self-comparison. The selected native action scale is preserved throughout the candidate chain. Loading a policy with a different action scale without accounting for physical targets would change its behavior immediately.

The candidate uses the existing NaturalRun reward base and observation layout. If the walker is selected, switching from its walking objective to the existing running objective is an additional transfer-learning difference: the first block is not a pure ankle-only ablation of the walking training objective. Physical action interpretation is unchanged, and the walking gates still apply.

## Requested changes

| Change | Implementation | Initial weight |
|---|---|---:|
| Correct ankle indexing | Explicit names grouped as left pitch/roll, right pitch/roll; runtime assertion | Existing -2.0 |
| Weak stance-balance cost | Completed left/right stances; squared relative duration mismatch with 10% deadband | -0.20 |
| Unreachable target cost | Requested joint target outside existing hard bounds; no clipping | -0.25 |
| Actual torso lean | Torso-frame tilt beyond a speed-dependent allowance, not pelvis orientation | -0.40 |

The ankle term retains the existing torque/speed-envelope proxy and only fixes its indexing. This is NOT a newly validated nonlinear RSU mechanism or motor thermal model. Implicit actuator torques remain PD estimates rather than measured drive/impact torques.

Stance filtering is causal: 30 ms confirmation requested, quantized to 40 ms at the existing 50 Hz control rate. Isolated short contacts/dropouts do not complete steps. Initial partial stances and stances longer than 1.5 seconds are excluded; both sides need completed, recent steps. State resets independently for each environment, on command changes, and after pushes. The cost is gated off during turns, poor tracking, excessive lateral/yaw motion, and one second of command/push recovery. Missing steps are not interpreted as success by the evaluator.

There are no new exact knee-angle targets, fixed stride lengths, arm-swing targets, cadence targets, or instantaneous equal-force penalties. Existing NaturalRun terms are retained. Running flight is allowed; the final high-speed gate checks for some filtered flight but does not prescribe its duration.

Unreachable-target and torso costs activate above 1 m/s and ramp to full weight by 2 m/s. Torso allowance is about 24 degrees at 1.8 m/s and 29 degrees at 4 m/s, a tunable hypothesis rather than a proven biomechanical optimum. They are introduced separately, before speed advancement. The failed standalone hard-waist-clamp diagnostic is not deployed.

## Learning and promotion

40% of non-standing command resamples are 0.3–1.0 m/s; the rest are 1.0 m/s through the current ceiling. Standing resamples remain 5%. Speed ceilings are fixed inside each block: there is no time-based curriculum that silently advances regardless of performance.

PPO retains the original 80 observations / 22 actions and existing model architecture. Fine-tuning uses a fixed learning rate 1e-4, entropy coefficient 0.002, and PPO clip parameter 0.10. Original optimizer state is discarded on transfer/new stages; successful same-stage blocks retain their optimizer. A walking-regressed update is rejected and the next bounded attempt starts from the last accepted checkpoint with a lower learning rate (floor 2.5e-5).

| Stage | Speed ceiling | Maximum additional iterations |
|---|---:|---:|
| Correct indexing / transfer check | 1.8 m/s | 100 |
| Weak stance balance | 1.8 m/s | 600 |
| Unreachable targets | 1.8 m/s | 400 |
| Actual torso lean | 1.8 m/s | 400 |
| Speed step | 2.3 m/s | 400 |
| Speed step | 2.6 m/s | 400 |
| Speed step | 3.0 m/s | 400 |
| Speed step | 3.5 m/s | 400 |
| Final speed | 4.0 m/s | 600 |
| **Maximum** | | **3,700** |

Each stage evaluates after at most 200 iterations (100 for the first). Two test batches are required: seed 47 nominal, seed 83 with matched lateral kicks at 8 and 13 seconds. Each has 24 replicas per policy/speed, 20 seconds, first-episode data only, and a three-second warmup excluded from averages. Falls during warmup still count. The approved walker is reevaluated every time, not compared against an unrelated historical scalar.

Walking checks compare speed error, lateral drift, foot slip, torso tilt, completed-stance asymmetry, limb crossing, non-foot contact and fall fraction. Fixed tolerances are in `gates.py`; they were chosen before training, not adjusted to make a candidate pass. These are practical regression checks, not statistically certified equivalence. Missing/nonfinite data or insufficient completed steps fail closed. Both batches must pass. A checkpoint failing walking is never used as the input for the next speed stage.

The balance stage additionally requires an improvement in 1 m/s stance balance relative to the selected seed. Higher stages require speed tracking, low fall/contact/slip rates, and progressively target/torso gates. A 4 m/s command with approximately 3 m/s actual motion cannot pass. A stage exhausting its budget stops the chain (`stopped_gate`) instead of automatically promoting. The last accepted checkpoint and rejected checkpoints are preserved separately.

At the previously measured approximately 6.3 seconds/iteration, 3,700 iterations alone would be about 6.5 hours. Startup, benchmarking and repeated evaluations add overhead, and stages may finish early or stop at a failed gate. The chain logs current throughput; neither duration nor successful 4 m/s running is guaranteed.

## Startup, memory and records

Launcher: `E:\Aditya\robot_sim\humanoid_project\experiments\guarded_run_v1\chain.py`.

Active launched output: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v1_20260902_1726`.

The earlier `_1723` preflight was deliberately interrupted to fix Windows process-tree cleanup before unattended memory retries. Its completed smoke/reload/4,096-environment results remain archived. No main candidate stage from that launch was promoted.

Preflight: 64-environment smoke train, saved-checkpoint reload, then 4,096/6,144/8,192 short stability benchmarks. Four-iteration benchmark throughput includes startup/warmup and is not a precise steady-state estimate. Use the highest requested count that passes. Retain a 512 MiB GPU reserve; three low-memory samples trigger a retry of only that block at the next lower count. Arbitrary Python/config errors stop with logs rather than being mislabelled as VRAM failures.

Critical output files:

- `state.json`: active phase/child PID, stage, last accepted checkpoint, gate decisions, paths, GPU sample.
- `gpu.csv`: sampled GPU memory/utilization/power/temperature. Unsupported power is kept as N/A.
- `source_snapshot/` and per-worker `env.yaml`, `agent.yaml`, `command.json`: reproducibility.
- `original_hashes.json` / `originals_preserved.json`: archived policy and production asset/source integrity.
- Each train folder: stdout/stderr, TensorBoard data, intermediate models, `candidate.pt`, `result.json`.
- Each evaluation folder: `evaluation.json`, `telemetry.npz` (50 Hz joint angles/targets/speeds, estimated torque, motion/contact/limit data, first-fall masks), stdout/stderr.
- `*_gate.json`: machine-readable individual pass/fail reasons. A completed worker is not a passed behavioral gate.

Monitor without launching another simulator:

```powershell
Get-Content "E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v1_20260902_1726.launcher.log" -Tail 20 -Wait
```

Do not run another copy of this supervisor while this one is active. A new invocation requires a fresh output directory; this script is deliberately not a blind resume/overwrite tool. Future playback must use the `action_scale` stored in the selected candidate's result, not assume the standard NaturalRun scale.

## Limits and research context

The design follows the general evidence that symmetry-aware learning can improve locomotion, but this candidate does not implement that paper's augmentation or equivariant networks: [Su et al., Leveraging Symmetry in RL-based Legged Locomotion Control](https://arxiv.org/abs/2403.17320). Isaac's [symmetry configuration documentation](https://isaac-sim.github.io/IsaacLab/v2.2.1/_modules/isaaclab_rl/rsl_rl/symmetry_cfg.html) provides a future alternative; it is not silently enabled here.

Known limitations remain: simplified overlapping collision geometry with self-collision disabled, an approximate differential constraint, no validated electrical/thermal model, and nominal matched evaluation plus two push events rather than exhaustive domain/random-terrain testing. Passing these gates still requires visual review and does not certify real hardware safety or perfectly humanlike gait.

All experiment data are local under E:\Aditya\robot_sim, which is not a Git repository. No files are copied into the public humanoid repository or pushed online by this chain.
