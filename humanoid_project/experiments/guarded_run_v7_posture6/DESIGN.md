# Guarded v7 — isolated posture repair, then measured progression to 6 m/s

V6 is preserved byte-for-byte.  It completed cleanly but stopped at the 4.5
m/s gate because sustained articulated-torso lean exceeded the occupancy
limit in both independent batches.  The failure was not a crash, OOM, speed
deficit, fall, slip failure, contact/crossing exploit, balance failure,
unreachable target, or proven locomotion-actuator envelope limit.

The repair keeps the v5/v6 torso allowance (`8 + command_speed` degrees) and
all hard gates unchanged, but raises the actual-torso reward weight by 25%,
from -1.00 to -1.25.  This term is exactly inactive through 1 m/s, so the
approved walking reward remains unchanged; archived 0.5/1.0 m/s evaluation is
still required after every candidate.  Training resumes from v6's fully
passing 4.0 m/s traction checkpoint rather than either rejected 4.5 m/s
checkpoint.

The remaining ceilings are 4.5, 5.0, 5.5, and 6.0 m/s.  No actuator change is
made.  The existing experiment-local <=15% velocity rule remains fail-closed
and can activate only after two stable batches both miss speed and prove the
same bilateral locomotion class saturated for >=10% of valid samples.

## Preserved v6 design provenance

V6 resumes the separately preserved v5 upright checkpoint. V5 reduced measured
4 m/s torso tilt from 33.0 to about 10.1 degrees, kept 4.00 m/s speed, zero
falls and approved walking retention, but stopped because contact-foot slip was
0.374 m/s against the unchanged 0.35 m/s gate. All other final stage checks
passed.

The first v6 stage adds a high-speed-only stance-foot translation cost at weight
-0.60 on top of the unchanged built-in -0.25 slide term. It has a 0.10 m/s
deadband, uses measured >5 N foot contact, and activates only above 1 m/s while
the upright base is tracking its command. It cannot earn a lower cost by simply
standing still, contributes exactly zero to <=1 m/s walking, and does not alter
friction, geometry, torque or velocity limits. The traction stage has a maximum
300-iteration budget in 100-iteration measured blocks. It must retain the v5
torso gates, speed, flight, safety and archived-walker comparisons before the
4.5/5.0/5.5/6.0 m/s stages can begin.

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
