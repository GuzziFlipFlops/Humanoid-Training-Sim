# Recovery v4: staged assisted discovery, September 3

## Why v3 stopped

The completed v3 run did not crash. It deliberately stopped after 600 additional
supine iterations because three measured blocks had no strict success or useful
height progress. The final nominal evaluation had 0/128 recoveries, no unsafe
terminations, and only about 0.237 m mean maximum pelvis height. The v3 safety
changes removed v2's violent collision exploit, but the policy converged to a
new local optimum: remain safely on the floor.

Evidence is preserved at
`E:/Aditya/robot_sim/humanoid_project/logs/guarded_getup_v3_20260903_115042`.
No v3 checkpoint is promoted as a successful get-up policy.

## Repair

v4 keeps every v3 collision, speed, support, two-foot hold, drift, and
post-success invalidation rule. It changes discovery rather than weakening the
final criteria:

1. First learn the terminal standing hold from upright resets. This replaces a
   stale hand-coded hold calibration that now fell in all 16 current trials.
2. Supine training then uses a global upward pelvis force equal to at most 20%
   of body weight, then 10%, then zero.
3. The force is full below 0.20 m pelvis height and linearly fades to zero by
   0.70 m. Strict standing cannot begin below 0.78 m, so assistance cannot
   complete or hold a successful stand.
4. The actor does not observe the assistance value. The privileged critic does.
5. Assisted 10-second evaluations are diagnostic height/safety gates only and
   are explicitly ineligible for promotion.
6. All actual promotion gates are unassisted 25-second trials, including a
   2-second qualifying dwell, 5-second continued hold, fixed foot/body anchors,
   no non-foot contact, no unsafe event, and a separate perturbation seed.
7. A saved-policy smoke checkpoint is reloaded with assistance disabled before
   full-size work. Main work branches from the preserved v3 200-iteration
   collision-aware candidate; smoke and benchmark gradient updates are discarded.

HoST suggests an upward pull around 60% of body weight, but that value was not
safe for this model. A 10-second 60% smoke evaluation had 128/128 unsafe trials
and 59.8 mm peak penetration. A preserved-actor 40% calibration had 48/64
unsafe trials and 38.0 mm peak penetration. The 20% calibration had 0/64 unsafe
trials, 9.65 mm peak penetration, and 0.391 m mean maximum pelvis height, so 20%
became the local upper bound. The staged concept follows HoST; starting with the
terminal hold follows the reverse-curriculum principle. This is an engineering
adaptation for this 38 kg robot, not a copied policy or a guarantee of convergence.

Primary references:

- https://github.com/InternRobotics/HoST
- https://github.com/RunpeiDong/humanup
- https://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf

## Guarded schedule

| Stage | Assistance | Maximum iterations | Gate |
|---|---:|---:|---|
| Learned terminal hold | 0% | 600 | >=90% nominal, >=85% perturb, zero unsafe |
| Supine discovery A | 20% body weight | 400 | assisted mean max height >= 0.50 m, <50% unsafe; retain unassisted hold |
| Supine discovery B | 10% body weight | 400 | assisted mean max height >= 0.62 m, <50% unsafe; retain unassisted hold |
| Supine recovery | 0% | 800 | >=90% nominal, >=85% perturb, zero unsafe |
| Supine/prone/sides | 0% | 800 | same gate for every pose group |
| Refinement | 0% | 400 | same gate for every pose group |

The 3,400 additional iterations are ceilings. Each block is evaluated after 200
iterations and may advance early. The chain stops on unsafe behavior, no
progress, or a missed gate; it never silently promotes the latest checkpoint.
It requests 8,192 environments with a 512 MiB reserve and only falls back to
6,144 or 4,096 after measured memory pressure. In the current launch,
`logs/guarded_getup_v4_20260903_163352`, the 8,192 probe fell below the reserve
and the first main block started at 6,144 environments. Read `state.json` for
live status; launch is not evidence of learned success.

This remains a simulation candidate. Torque telemetry is an implicit-drive
estimate and does not certify hardware safety.
