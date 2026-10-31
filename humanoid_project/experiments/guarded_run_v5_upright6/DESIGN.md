# Guarded v5 — upright 4 m/s fine-tune, then measured progression to 6 m/s

This experiment starts from, and never overwrites, the completed guarded-v4 4 m/s checkpoint:

`logs/guarded_run_v4_20260903_0305/06_speed_4_0600_8192/candidate.pt`

Its exact path and SHA256 are written to each run's `input_checkpoint.json`. The approved walker, completed v4 checkpoint, shared `humanoid_cfg.py`, USD, and task sources are immutable inputs. All new checkpoints live under a new v5 log directory.

## Evidence for the change

The final v4 nominal rollout measured 3.961 m/s at a 4.0 m/s command with no falls, but the articulated torso averaged 33.0 degrees from vertical. The same policy measured only 0.82 degrees at 1 m/s. This isolates the problem to high-speed shaping rather than the CAD joint axes or the retained walking policy.

The old high-speed reward allowed 23–29 degrees before penalizing lean, and its gate did not fail until roughly 35 degrees at 4 m/s. V5 replaces that permissive cone with a smooth tolerance of `8 + command_speed` degrees (12 degrees at 4 m/s and 14 degrees at 6 m/s). It uses a smooth-L1 cost, not a hard pose target. Walking at and below 1 m/s keeps the existing reward contribution unchanged and is re-evaluated against the archived approved walker after every candidate.

## Why 4 m/s can look slower than expected

The telemetry says the simulated base really averaged 3.961 m/s. It also measured about 5.21 foot-strikes per second (2.61 same-leg strides/s), approximately 0.76 m per step and 1.52 m per same-leg stride. Camera tracking, the large floor grid, lack of motion blur, and the 1.524 m robot scale make that translation look slower than a close real-world running video. The speed value comes from root velocity in physics telemetry.

## Stages and gates

Maximum budgets are 600 iterations for upright 4.0 m/s, then 400 at 4.5, 500 at 5.0, 600 at 5.5, and 800 at 6.0 m/s. A stage can stop early only after both a nominal batch and a lateral-kick batch pass:

- measured target-speed error;
- no-fall, sufficient pre-fall duration, and completed-step evidence;
- torso mean tilt and lean occupancy;
- foot slip, non-foot contact, crossing, and stance-balance checks;
- flight phase for running;
- 0.5 and 1.0 m/s retention against the archived approved walker.

The command sampler keeps 40% walking samples, broad high-speed samples, and endpoint-biased samples. It does not prescribe knee angles, stride length, or equal instantaneous foot loading.

## Velocity-limit rule

V4 at 4 m/s did not prove that a limit increase was required. Most joints were far from their ceilings. The right knee and ankle pitch touched their limits intermittently, but the policy still met 4 m/s.

V5 starts with every approved velocity limit unchanged. A one-time 15% increase is allowed only if both nominal and kick evaluations are stable, both miss target speed by more than the gate tolerance, and the same bilateral locomotion actuator class spends at least 10% of valid samples at 95% or more of its configured limit. Falling, a single test, or merely approaching a limit cannot trigger the increase. Any increase is applied only inside this experiment's deep-copied articulation config and is recorded in `state.json`; `humanoid_cfg.py` remains untouched.

## Critical files

- `candidate.py`: mixed-speed sampling, upright reward, and isolated optional velocity-limit scaling.
- `evaluation.py`: first-episode telemetry, torso metrics, and per-joint velocity saturation.
- `gates.py`: walking-retention, task-quality, upright, and velocity-proof gates.
- `chain.py`: checkpoint provenance, smoke/8192 checks, block training, rollback, and conditional 15% rule.
- `worker.py`: one bounded train/evaluation process.
- `monitor.py`: read-only status and ETA display.
