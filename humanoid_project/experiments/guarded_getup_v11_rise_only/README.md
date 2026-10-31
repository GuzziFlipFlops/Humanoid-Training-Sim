# Recovery v11: height-first supine-to-stand training, September 5

This experiment deliberately removes the negative behavioral constraints that
caused guarded recovery candidates to prefer remaining on the floor. It is
isolated from every approved walking and running checkpoint.

## What PPO is asked to do

The learning return contains exactly two nonzero terms:

| Term | Weight | Meaning |
|---|---:|---|
| `height` | +12.0 | More pelvis height is always better, from 0.06 m through 0.78 m. |
| `upward_velocity` | +1.5 | Positive pelvis Z velocity is rewarded; downward velocity is not. |

Discovery has no posture, balance, contact, torque, power, collision,
joint-speed, action-rate, standing-dwell, failure, or floor-stall terms.
Episodes terminate only on timeout. Self-collision is disabled for this
simplified box model. There is no hidden controller takeover after it rises.

The final refinement remains positive-only: height +10, uprightness +6,
two-foot support +4, quiet upright hold +3, and positive vertical velocity
+0.75. These terms never punish the exploratory movements required to leave
the floor. A strict seven-second standing hold is checked only by the external
evaluator; it does not suppress exploration during training.

The physics engine still enforces URDF joint limits, actuator torque/velocity
limits, and finite numerical state. Those are model validity constraints, not
reward-policy restrictions.

## Why this differs from the earlier HumanUP-inspired versions

The previous work borrowed HumanUP's broad idea of motion discovery followed
by refinement and used a reverse curriculum. It did not copy a trained policy
that was already successful on this exact robot. Directly replaying a HumanUP
reference on this morphology was measured and rejected because it reached only
about 0.314 m pelvis height.

Versions v5-v10 attempted to learn a finished, safe recovery in one guarded
objective. Several had a return-ordering error: an active attempt could end
early and pay penalties more often than lying still until timeout. v10 removed
that ordering error but still retained many constraints and froze most of the
actor, so it did not discover a full floor-to-stand motion.

v11 first asks the simpler question: can the full trainable actor discover a
motion that raises the pelvis? It uses locally calibrated training-only body
weight assistance, tapering from 20% to 10% to zero. Every promotion gate is
unassisted. Only after reaching 0.75 m does it add the positive-only standing
terms. The current scope is the supine initial-pose family; other lying
orientations are intentionally deferred until supine recovery works.

| Phase | Ceiling | Training assistance | Unassisted promotion gate |
|---|---:|---:|---|
| Floor escape | 400 | 20% body weight | Mean maximum pelvis height >=0.30 m |
| Rise | 400 | 10% body weight | Mean maximum pelvis height >=0.55 m |
| Full rise | 800 | 0% | Mean maximum pelvis height >=0.75 m |
| Stable stand | 1,000 | 0% | >=50% strict seven-second standing success |

## Important limitation

The discovery phases are intentionally exploitable. A checkpoint may leap,
roll, thrash, collide, or briefly pass through a height threshold. Those phases
are called rise discovery, not successful get-up. Only the final independent
standing-hold gate can complete the chain. This is still simulation evidence,
not hardware certification; self-collision is disabled in the simplified model.

Run the full chain only after the smoke test:

```powershell
Set-Location "E:\Aditya\robot_sim\humanoid_project"
& "E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe" ".\experiments\guarded_getup_v11_rise_only\chain.py" --output ".\logs\guarded_getup_v11_rise_only_YYYYMMDD_HHMM" --num_envs 8192 --reserve_mb 512
```

Monitor it with:

```powershell
& "E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe" ".\experiments\guarded_getup_v11_rise_only\monitor.py"
```

References that informed the earlier staged approach:

- HumanUP: https://github.com/RunpeiDong/humanup
- HumanUP paper: https://arxiv.org/abs/2502.12152
- HoST: https://github.com/InternRobotics/HoST
