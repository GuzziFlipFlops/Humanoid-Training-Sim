# Guarded candidate v2 — walking-preserving transfer

Active output: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v2_20260902_2103`.

## What happened to v1

The first 100-iteration block completed in 305.172 seconds at 8,192 environments. It was not a crash. Both matched evaluation batches rejected its 1 m/s stance asymmetry. In the nominal batch, the approved walker scored 0.338 versus 0.395 for the candidate (lower is better), exceeding the existing +0.04 tolerance. No speed promotion occurred; the original walker remained the last accepted checkpoint.

Evidence: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v1_20260902_1726\00_index_1.8_0100_gate.json` and the neighboring `*_eval_47` / `*_eval_83` telemetry. That candidate tracked 1.8 m/s well but did not retain the reference's 1 m/s gait quality. It must not be called approved.

The transfer changed more than ankle indexing: after selecting the walker, v1 used the running reward base and commands up to 1.8 m/s immediately. This confounds the result; it does not prove the ankle correction caused the regression. v2 isolates the early changes properly.

## v2 changes

- Start from the archived approved `model_2098.pt`, native action scale 0.35. Neither original checkpoint is overwritten.
- Index stage: preserve the actual NaturalWalk reward configuration, 0.2–1.0 m/s range, 15% standing fraction, original events and physical model. Change only the ankle term's joint pairing (plus conservative PPO fine-tuning settings).
- Balance stage: retain that walking configuration and add just the requested weak completed-stance cost, weight -0.20.
- Separate transition stage: expand toward 1.8 m/s with the low-speed mixture. For the seven existing terms that differ between NaturalWalk and NaturalRun, smoothly blend their original weighted contributions using `alpha = clamp((command_speed - 1)/1, 0, 1)`. At <=1 m/s each contribution is exactly the walking contribution; at >=2 m/s it is the running contribution. The simulator asserts those endpoint identities before training.
- Introduce unreachable-target and actual-torso-lean costs in separate stages, as in v1, before further speed increases. No new fixed knee angles, stride length, cadence or instantaneous equal-foot-load targets.
- Learning rate reduced to 5e-5. Index checkpoints are evaluated every 50 iterations; later blocks every 200. Walking regression still causes rollback, not threshold relaxation.

Matching the low-speed reward does not guarantee policy retention, because one network still learns from multiple speeds. Repeated matched evaluations are therefore still mandatory.

## Maximum iteration budget

| Stage | Ceiling | Maximum iterations |
|---|---:|---:|
| Index correction | 1.0 m/s | 100 |
| Weak stance balance | 1.0 m/s | 600 |
| Smooth run-objective transition | 1.8 m/s | 400 |
| Unreachable targets | 1.8 m/s | 400 |
| Actual torso lean | 1.8 m/s | 400 |
| Speed steps | 2.3, 2.6, 3.0, 3.5 m/s | 400 each |
| Final speed | 4.0 m/s | 600 |
| **Total maximum** | | **4,100** |

The first v2 indexing block took 187 seconds for 50 iterations (3.74 seconds/iteration including first-iteration overhead; later individual iterations were about 3.3 seconds). At that average, 4,100 iterations alone would take about 4.3 hours, with startup/evaluations adding time and later reward stages potentially changing throughput. Budget is an upper bound, not a success guarantee. A passing stage can finish early; an exhausted failed stage stops advancement.

## Tests and safeguards

Seven CPU tests carried forward, including rejection of false 4 m/s tracking, missing-step rejection, correct ankle grouping, causal contact filtering, partial-step exclusion and reset isolation. The same `gates.py` thresholds are unchanged.

Preflight uses a longer 32-iteration / 64-environment smoke train so rollout time reaches scheduled pushes. It reloads that saved model in another process. The completed v1 4,096/6,144/8,192 scaling checks are reused only after hashes verify unchanged robot assets and original policies; v2 also performs a fresh 8,192-environment training smoke check with all new reward code. Original policies are the only real training seeds, never smoke models.

The background supervisor retains the 512 MiB reserve, exact per-worker process-tree cleanup on Windows, memory-only fallbacks, checkpoint/config snapshots, stdout/stderr, GPU CSV, telemetry and machine-readable gate decisions. It rejects walking-regressed candidates and refuses speed promotion without passing both test batches. See v1's README for the unchanged filter definitions, metric tolerances and simulator limitations.

## Files

- `candidate.py`: isolated reward/command changes and the walking-to-running blend.
- `contact_state.py`: causal, completed-stance state and independent ankle grouping math.
- `gates.py`: unchanged walking and stage-quality thresholds.
- `evaluation.py`: reference/candidate comparison with joint/contact/target telemetry.
- `worker.py`: bounded train/evaluation process and runtime reward assertions.
- `chain.py`: stage sequencing, rollback, GPU reserve and worker supervision.
- `test_candidate.py`: CPU regression tests.
- Output `state.json`: current phase, selected checkpoint, candidate action scale and gate state.
- Output `*_gate.json`: individual failure/pass reasons; reward return is not a success criterion.
- Output `source_snapshot/`, per-block YAML/command files: exact experiment definition.
- Evaluation `telemetry.npz`: first-episode, pre-reset 50 Hz joint angles, targets, velocities, approximate drive torque and gait measurements.

Monitor:

```powershell
Get-Content "E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v2_20260902_2103.launcher.log" -Tail 20 -Wait
```

The torque estimates, simplified collision model and ankle-envelope approximation retain the limitations documented in v1. No hardware capability is newly certified. Original CAD, actuator configuration and old task code are unchanged. Everything remains local; the private repository file index points here.
