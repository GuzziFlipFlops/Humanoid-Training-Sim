# V6 audit and v7 repair report

Run audited: `logs/guarded_run_v6_traction6_20260905_0122`.

## Terminal diagnosis

- Supervisor state: `stopped_gate`; supervisor PID 25904 and child PID 0 are
  no longer active.  This was a deliberate hard-gate stop, not a process crash.
- Every worker produced a durable `result.json`; all `stderr.log` files are
  empty.  The final training log reached 200/200 iterations and emitted the
  explicit `[GUARDED] COMPLETE` marker.
- The 8,192-environment blocks remained stable at roughly 6,659 MiB sampled
  VRAM (6,668 MiB in the preceding block); no allocation or OOM evidence exists.
- Native action scale is 0.35.  The final rejected v6 checkpoint SHA256 is
  `22bb9dd585f19cb23e1c11e582004267357c1270e5c7b6e0551c7105c8b7cc05`.

Both final 4.5 m/s evaluations used 24 replicas for 20 seconds, with 17
post-startup seconds retained.  Nominal/push mean speed was 4.4923/4.4930 m/s,
falls 0/0, slip 0.2845/0.2864 m/s, non-foot contact 0/0, foot and knee crossing
0/0, valid-step replicas 24/24, stance asymmetry 0.0402/0.0387, target-out-of-
bounds 0.0095/0.0097, and flight fraction 0.3164/0.3170.  Mean torso tilt
11.573/11.532 degrees passed.  Lean occupancy was 0.3527/0.3360 and failed the
unchanged <=0.25 gate in both batches.

Synchronized `telemetry.npz` confirms the excess was sustained, not a kick
transient: nominal occupancy by four rollout windows was approximately
0.358/0.355/0.362/0.339, and push occupancy 0.348/0.360/0.275/0.356.  Median
replica occupancy was 0.350 nominal and 0.340 pushed.  V6 improved occupancy
monotonically from about 0.789 at the 4.5 m/s baseline to 0.606 after 200
iterations and 0.344 after 400, while speed and traction remained valid.

An actuator-speed change is forbidden by the evidence.  Although both ankle
pitch joints spent about 10.5-13.4% of valid samples near their velocity limit,
the target-speed gate already passed, and knee saturation was unilateral and
negligible.  The two-condition rule therefore fails closed.

## Isolated repair

Resume checkpoint:
`logs/guarded_run_v6_traction6_20260905_0122/00_traction_4_0200_8192/candidate.pt`
with SHA256
`a770187d4eaf274e5417b97061cc6b126a5a61b6044c473006927e9ea305ea4e`.
Its paired 4.0 m/s gate passed every walking and stage check.

V7 changes only the already high-speed-only actual-torso reward weight from
-1.00 to -1.25.  It preserves the tolerance cone, hard gates, evaluator,
action/observation interface, action scale 0.35, low-speed reward behavior,
traction term, command distribution, assets, and actuator limits.  Fresh unit
tests, a 64-environment signal smoke, checkpoint reload, and an 8,192-
environment 512 MiB-reserve benchmark are required before main continuation.
