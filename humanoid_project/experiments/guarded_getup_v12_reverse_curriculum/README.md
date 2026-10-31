# Recovery v12: stronger reverse curriculum, September 6, 2026

V11 stopped cleanly after 600 PPO iterations. Its 20%-assisted floor-escape
stage passed unassisted at 0.396 m, but the following 10%-assisted stage ended
at 0.529459 m versus the 0.55 m unassisted height gate. It produced 0/64
strict seven-second stands. The v11 run and every checkpoint remain immutable.

V12 is an isolated continuation from v11's best measured checkpoint:

`E:\Aditya\robot_sim\humanoid_project\logs\guarded_getup_v11_height_first_20260905_232453\01_assist10_rise_0400_8192\candidate.pt`

## Major revision

Instead of repeating the failed direct 20% -> 10% -> 0% supine schedule, v12
actually uses the already-audited reverse-curriculum reset family. Training
replays easier endpoints while lowering the robot from 45 degrees to 65, 80,
tucked supine, and finally the randomized supine pose-bank family. Training-only
upward assistance begins at 35% body weight and tapers through 30%, 25%, 20%,
15%, and zero. The force itself also fades with pelvis height and is exactly
zero at and above 0.70 m.

Every promotion evaluation is unassisted. Height gates are discovery milestones
only; they cannot certify a stand. Final acceptance remains an independent
unassisted 25-second rollout from the requested supine family with at least 50%
of trials completing the two-second dwell plus five-second uninterrupted stable
stand. A leap, roll, collision exploit, brief height crossing, PPO return, GPU
activity, or saved checkpoint is not completion.

| Phase | Ceiling | Train start | Train assist | Unassisted gate |
|---|---:|---|---:|---|
| Reverse launch | 200 | through tilt 45 | 35% | mean max height >=0.75 m at tilt 45 |
| Reverse 65 | 300 | through tilt 65 | 30% | mean max height >=0.72 m at tilt 65 |
| Reverse 80 | 400 | through tilt 80 | 25% | mean max height >=0.68 m at tilt 80 |
| Tucked supine | 500 | through 90-degree tuck | 20% | mean max height >=0.62 m |
| Supine assisted rise | 600 | randomized supine | 15% | mean max height >=0.75 m |
| Supine unassisted rise | 800 | randomized supine | 0% | mean max height >=0.75 m |
| Stable stand refinement | 1,000 | randomized supine | 0% | >=50% strict seven-second success |

## Reward and safety contract

Discovery uses only pelvis height (+12) and positive vertical velocity (+1.5).
Refinement uses only height (+10), positive vertical velocity (+0.75), upright
(+6), two-foot support (+4), and quiet upright hold (+3). There are no negative
motion, energy, torque, action-rate, collision, joint-posture, joint-speed, or
episode-failure rewards. Timeout is the only training termination. Isaac's
finite numerical checks, URDF joint limits, actuator limits, and collision
solver settings are unchanged.

The actor controls all 22 joints, its normalizer remains trainable, and the
asymmetric normalized action maps the full policy range to every joint's audited
URDF range with a hard-stop margin. No joint is frozen and no controller takes
over after a rise.

## Preflight and launch

The supervisor runs all CPU contracts, a real four-iteration 64-environment
train smoke, a reload smoke, and a real four-iteration 8,192-environment
allocation smoke. It enforces a 512 MiB reserve and falls back to 6,144 then
4,096 only after three measured low-memory samples or a demonstrated allocation
failure. Preflight updates are discarded; main training resumes the exact v11
checkpoint above. Each completed train block records per-term finite/nonzero
reward diagnostics.

```powershell
Set-Location "E:\Aditya\robot_sim\humanoid_project"
& "E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe" `
  ".\experiments\guarded_getup_v12_reverse_curriculum\chain.py" `
  --output ".\logs\guarded_getup_v12_reverse_YYYYMMDD_HHMMSS" `
  --num_envs 8192 --reserve_mb 512
```

Monitor read-only state with:

```powershell
& "E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe" `
  ".\experiments\guarded_getup_v12_reverse_curriculum\monitor.py"
```

Self-collision remains disabled for the simplified overlapping box model, so a
final simulation pass is not hardware certification and still requires visual
and synchronized telemetry review before any real-robot use.
