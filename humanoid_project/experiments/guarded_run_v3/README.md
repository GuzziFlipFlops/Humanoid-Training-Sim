# Guarded v3 — calibrated completed-stance shaping

Created September 2, 2026. Local candidate only; no CAD, motor, joint, production task, or archived policy changes.

Live state: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v3_20260902_2244\state.json`.
Supervisor log: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v3_20260902_2244.launcher.log`.
Always read live state before claiming the chain is running or a policy succeeded.

## Evidence and diagnosis

v2 stopped intentionally after 600 balance iterations. Both nominal and lateral-push batches passed all walking-retention checks but failed the required 5% improvement over the original walker's stance asymmetry. Nominal original seed: 0.33916; v2 at 200: 0.34201; at 400: 0.32780; at 600: 0.34360. This is not a monotonic learning trend and does not justify assuming more iterations alone will solve it.

The last training log reports a completed-stance contribution of -0.0124 versus +3.8222 for velocity tracking. The old squared, deadbanded cost is weak at the observed imbalance. It also compares only the latest completed stance per foot rather than the per-leg mean used in evaluation. These are plausible learning-signal weaknesses, not experimentally isolated proof of the failure's cause.

Archived evidence: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v2_20260902_2103`; especially `01_balance_1_0600_gate.json`, both `*_eval_47` / `*_eval_83` reports, and training `stdout.log`.

## Narrow revision

- Continue the tested v2 600-iteration checkpoint, retaining native action scale 0.35. A fresh matched walking check is required before main training. The smoke checkpoint is never promoted.
- Keep the original walking command distribution, PPO/network/observation setup, ankle pairing fix, events, and other rewards.
- Only the balance cost changes: mean of up to the last two **completed** stances on each leg; same causal contact debounce, reset/push/command cooldown, stale-data rejection, and valid-motion mask.
- Preserve the 0.10 relative-asymmetry deadband. Above it, use a continuous Huber-like linear-tail penalty instead of a squared tail. Weight -0.60 instead of -0.20. No prescribed knee angles, stride length, cadence, or instantaneous equal loading.
- At relative excess `x = clamp((asymmetry - .10)/.90, 0, 1)`, unweighted cost is `5*x*x` below .10 and `x-.05` above. Maximum weighted magnitude is .57. At recorded nominal 1 m/s contacts it is approximately .131, versus .0147 previously (ungated replay after warmup). Replay is a signal calibration, not evidence of a better policy.
- Add training diagnostics: fraction of valid stance samples, ungated/gated mean cost, and weighted cost in every worker's `result.json`. A never-active balance term fails preflight.
- Check balance every 100 iterations, with an 800-iteration upper bound and immediate progression on success. Walking regressions trigger rollback and lower learning rate. Failure to meet the gate still stops the chain; it is not bypassed.

`gates.py` and `evaluation.py` are byte-for-byte identical to v2. In particular, the required 5% improvement remains relative to the **original approved walker**, not the worse resumed candidate. The evaluator continues to use independent mean completed-stance durations over each rollout, not the new training cost. Both nominal and push tests must pass. This is evidence under those tests, not a guarantee against all gait exploits.

## Chain and time

| Stage | Speed ceiling | Maximum additional iterations |
|---|---:|---:|
| Revised stance balance | 1.0 m/s | 800 |
| Walking-to-running reward transition | 1.8 m/s | 400 |
| Unreachable joint targets | 1.8 m/s | 400 |
| Actual torso lean | 1.8 m/s | 400 |
| Speed progression | 2.3, 2.6, 3.0, 3.5 m/s | 400 each |
| Final running target | 4.0 m/s | 600 |
| Total ceiling | | 4,200 |

At the observed 2.5–3.3 seconds/iteration, the full iteration ceiling alone is roughly 2.9–3.9 hours, plus simulator startups and repeated evaluations (roughly another 1–2 hours if every block is used). A stage may pass early or the chain may stop at a failed gate. No promise of 4 m/s feasibility or human-like gait is made.

Preflight: 64-environment 32-iteration smoke, reload/evaluation in a separate process, and fresh 4,096/6,144/8,192 checks. Request 8,192 with a 512 MiB reserve; memory fallback 6,144 then 4,096. Worker process-tree cleanup is restricted to processes owned by this supervisor.

## Files and tests

- `candidate.py`: training penalty, command mixture, later-stage reward blending.
- `contact_state.py`: old evaluator-compatible tracker plus new `SmoothedStances` training subclass.
- `test_candidate.py`: ten CPU tests, including mirrored/bounded costs, contact history, reset isolation, staleness, no partial first step, debounce, ankle pairing, missing-data rejection, and false 4 m/s rejection.
- `replay_signal.py`: read-only reward-magnitude calibration using archived telemetry.
- `worker.py`: runtime assertions, checkpoint loading/training, signal diagnostics.
- `chain.py`: resume lineage, preflight, rollback, gates, GPU reserve, source snapshots.
- `monitor.ps1`: concise stage/iteration/GPU display. The terminal must explicitly distinguish evaluating from training.
- Updated read-only monitor (`monitor.py`, launched by `monitor.ps1`): recent 20-iteration mean in iterations/sec and seconds/iteration; stage count; ETA to the next checkpoint, next 200-iteration milestone, full current-stage budget, and final evaluation. Estimates include observed evaluation durations and startup overhead and are conditional, not guaranteed stage-promotion times. Shows latest checkpoint quality-check pass/fail and evaluated no-fall rate separately; survival is not overall task success. No checkpoint success rate is inferred from training reward. Six monitor unit tests, synthetic training/terminal snapshots, and live evaluation/archived stopped-state snapshots validate parsing and ETA arithmetic. Use `-Once` on the PowerShell wrapper for a single snapshot.
- Output `state.json`, `gpu.csv`, per-block `env.yaml`, `agent.yaml`, `command.json`, stdout/stderr, `result.json`, `candidate.pt`, gate JSON, and evaluation `telemetry.npz` retain the audit trail.

Unit tests: ten passed before launch. Initial test fixture was extended to allow the second full right stance to actually finish; no production logic was changed to satisfy an incomplete fixture. Runtime preflight result must be checked separately.

## Research and limitations

[Leveraging Symmetry in RL-based Legged Locomotion Control](https://arxiv.org/abs/2403.17320) demonstrates symmetry-aware architectures/data augmentation as a route to improved gait. This revision does **not** implement that paper's augmentation or architecture; this completed-step reward is our narrower, testable hypothesis for preserving the existing checkpoint. An [Isaac Lab discussion about air-time reward tuning](https://github.com/isaac-sim/IsaacLab/discussions/1977) also illustrates that reward weights do not simply transfer across setups; it is an unanswered discussion, not validation of this weight.

The simplified collision model still has disabled self-collision; crossing tests are proxies and cannot certify arbitrary collision-free motion. Implicit drive torque is a PD estimate, not measured physical motor/impact torque. No hardware certification, gait guarantee, or threshold relaxation is implied.
