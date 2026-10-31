# Recovery v14: supported-stand reverse curriculum, September 6, 2026

## Why v13 was rejected

V13 completed 600 iterations but produced 0/64 strict stands. Its final policy
briefly reached 0.960 m, moved at 0.763 m/s during the final five seconds, left
the 3.4 m local floor and fell to a nonsensical negative height. The failure was
not VRAM or a process crash. Refinement accidentally used `RiseOnlyEndings`, so
unsafe/run-off-floor states did not reset. Its cosine upright score also gave a
horizontal body 50% credit, its quiet reward did not require loaded feet, and
the inherited full-range action distribution had grown to std 2.30.

## V14 corrections

- Refinement uses the audited unsafe and terminal-fall terminations.
- A measured 14 rad/s physical target slew prevents the crouch-to-neutral
  one-tick impulse; this is now action contract `supported_slew14_v2`.
- Training starts from a clean zero-mean policy, not the failed v13 actor.
- Exploration starts at 0.05, is clamped to 0.04-0.08 on reload, and has no
  entropy bonus forcing it upward.
- The standing-height band is narrower and rapidly loses reward above 0.94 m.
- Uprightness uses exponential tilt scores; sideways/inverted poses earn
  effectively no credit.
- Quiet reward requires height, uprightness and both loaded feet.
- Motion, unsupported high states, workspace escape, floor stalling and unsafe
  resets receive explicit costs. These costs activate only where appropriate
  and do not prescribe a get-up trajectory.
- The exact seven-second independent unassisted stand gate remains unchanged.

The live run is:

`E:\Aditya\robot_sim\humanoid_project\logs\guarded_getup_v14_supported_stand_20260906_105955`

It passed 31 CPU contracts, the 64-environment train/reload smoke and the
8,192-environment allocation smoke (6,884 MiB sampled peak) before starting PPO.

V12 stopped cleanly after 200 iterations. Its first unassisted 45-degree gate
improved from 0.607838 m to 0.675758 m but missed the 0.75 m height threshold.
The run and checkpoint remain immutable at:

`E:\Aditya\robot_sim\humanoid_project\logs\guarded_getup_v12_reverse_20260906_033513`

## Failure diagnosis

Fresh 16-environment deterministic evaluations of v12's best checkpoint found:

| Start | Mean maximum pelvis height | Strict seven-second stands |
|---|---:|---:|
| tilt 5 degrees | 0.961776 m | 0/16 |
| tilt 15 degrees | 0.899944 m | 0/16 |
| tilt 25 degrees | 0.855818 m | 0/16 |
| tilt 35 degrees | 0.788774 m | 0/16 |
| tilt 45 degrees | 0.675732 m | 0/16 |

All cases ended near 0.172 m and remained on the floor. The policy had learned
to obtain transient height, not to stand. V12 made this possible by using
maximum height for early promotion, rewarding positive vertical velocity even
near standing height, and delaying standing-height, upright, two-foot, and
quiet-dwell rewards until the final stage. It also jumped directly to a
45-degree endpoint even though the reset generator supports five-degree steps.

Probe evidence:

`E:\Aditya\robot_sim\humanoid_project\logs\guarded_getup_v12_intermediate_probe_20260906`

## V13 correction

- Preserve v12's moving actor and observation normalizer.
- Discard its critic and optimizer because their height-only objective is no
  longer valid.
- Replace monotonically increasing refinement height reward with a positive
  0.85 m standing-height band. Jumping above the standing band earns less.
- Fade positive upward-velocity shaping to zero by 0.72 m.
- Reward uprightness, two loaded feet, and quiet standing from the first stage.
- Train with no negative energy, motion, collision, failure, or posture terms.
- Begin at five degrees and increase tilt only after an independent unassisted
  stable-standing gate passes.
- Replay every easier five-degree reset while training each endpoint.
- Evaluate every 100 PPO iterations and preserve the best checkpoint.

## Guarded training plan

The budgets are ceilings; a passing stage advances early.

| Stage | Ceiling | Training assistance | Unassisted stable success gate |
|---|---:|---:|---:|
| tilt 5 | 600 | 0% | 50% |
| tilt 15 | 400 | 5% | 45% |
| tilt 25 | 400 | 10% | 40% |
| tilt 35 | 500 | 15% | 35% |
| tilt 45 | 600 | 20% | 30% |
| tilt 60 | 700 | 25% | 25% |
| tilt 75 | 800 | 25% | 20% |
| tucked supine | 900 | 20% | 15% |
| randomized supine | 1,000 | 15% | 10% |
| randomized supine final | 1,400 | 0% | 50% |

Assistance is training-only, fades spatially to zero by 0.70 m, and is always
zero during promotion evaluation. Only the final independent unassisted
two-second dwell plus five-second continued stand can certify recovery.

## Launch

```powershell
Set-Location "E:\Aditya\robot_sim\humanoid_project"
& "E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe" `
  ".\experiments\guarded_getup_v14_supported_stand\chain.py" `
  --output ".\logs\guarded_getup_v14_supported_stand_YYYYMMDD_HHMMSS" `
  --num_envs 8192 --reserve_mb 512
```

Monitor:

```powershell
& "E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe" `
  ".\experiments\guarded_getup_v14_supported_stand\monitor.py"
```

Self-collision remains disabled for the overlapping simplified box geometry.
Passing the simulation gate is not hardware certification and still requires
visual and synchronized telemetry review.
