# Unitree rewards: physical failure diagnosis and repaired run — October 2, 2026

**Validation follow-up:** R21 was preserved and stopped after a nominal-pose
tensor alias was found. The active replacement is R22 with that defect fixed
and an explicit immutable-pose guard. Read
[the current report](UNITREE_NOMINAL_REPAIR_20261002.md). The episode-reward
tests and primary-source comparison below remain valid; they alone did not
prove that the observation/action implementation was sound.

The replacement run is `locomotion_20261002_r21_unitree_lifetime`, reward version
`unitree_g1_lifetime_v2`. Read [ACTIVE_RUN.json](ACTIVE_RUN.json), the indicated
fresh state, actual supervisor/worker processes, and latest independent gates
for live status. This report records the design and evidence; it is not a live
progress display. Standing, walking and running remain unqualified until their
independent tests pass. No guarantee of PPO convergence is implied.

## Why the unmodified G1 aggregate was stopped

R20 used the pinned G1 RL Lab core reward bodies and weights, adapted to the
physical-v29 anatomy. It ran 68 full standing-training updates after its
four-update smoke. Its last preserved checkpoint is
`runs/locomotion_20261002_r20_unitree_g1/012_stand_train_001_part_04/resource_recovery.pt`.
The source, logs and checkpoints were drained cleanly and retained.

Across those 68 updates, the sum of original weighted reward rates was always
negative: -7.9590 to -1.5999 per second. Ending a negative-return episode can
avoid its future costs. The G1 task's alive reward disappearing on termination
does not itself impose a terminal fault cost. Frequent upright resets were
therefore investigated rather than accepted as improving survival.

A deterministic replay of the 68-update candidate in real Isaac physics first
failed at **0.08 s**, for both stand and a 0.35 m/s command. The reason was
`invalid_differential`, not initially low height or a normal fall. At the stand
failure, actual waist pitch/roll were **0.34895/-0.11863 rad**, despite a valid
**0.20/-0.20 rad** target. One motor required **-1.2158 rad**, outside its saved
**±1.0472 rad** range. Closure error was initially less than 0.00005 mm; this
was an angle-limit breach, not a numerical rod-length error. Valid position
targets do not guarantee valid dynamic trajectories or prevent overshoot.

The recorder held resets to expose the subsequent fall. All **80** screenshot
frames of the two replays were reviewed: no alternating steps developed, and
the robot fell backward. Values after the first fault are visualization, not
normal episode behavior or a successful rollout. The evidence supports a
termination incentive defect and actual transmission-limit failure; it does
not establish that the CAD axes were wrong or that every earlier failure had
this cause. The nominal contact-physics check and joint-axis audit passed.

Evidence: [transmission reconstruction](diagnostics/unitree_termination_20261002/transmission_failure.json),
[actual replay result](diagnostics/unitree_port_20261002/r20_policy_failure_visuals/result.json),
[frame-by-frame review](diagnostics/unitree_port_20261002/frame-review.html).

## Published components used for the repair

The G1 core remains identical. Two lifetime-handling components are explicitly
adapted from other public Unitree implementations:

1. [Unitree RL Gym's compute_reward](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py)
   floors the regular total at zero before adding a termination component. Its
   [configuration](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot_config.py)
   enables this option and identifies early termination as its purpose.
2. [Unitree RL MJLab's H2 velocity task](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity/velocity_env_cfg.py)
   assigns `is_terminated` a **-200** weight. The active port uses the equivalent
   non-timeout physical-termination indicator, covering invalid transmissions
   as well as falls. A 20 ms control tick applies a **-4** terminal cost.

The complete active aggregation is:

```text
reward = dt × (max(sum(original G1 weighted terms), 0) - 200 × physical_termination)
```

The fault term is applied **after** flooring; it cannot be hidden by flooring
the final aggregate. Timeouts are unpenalized. The original per-term rates,
floor adjustment, and per-cause termination rates are logged independently.

This is a **G1 core with declared Unitree lifetime adaptations**, not an
unchanged complete G1, H1, H2 or commercial Unitree policy. All copied primary
sources, revisions, licenses and SHA256 hashes are saved in
[unitree_vendor/sources.json](unitree_vendor/sources.json).
The [H1/G1 comparison](UNITREE_REWARD_PORT_20261002.md) records every original
weight, anatomy mapping, observations/actions, PPO parameters and limitations.
H1's positive foot-clearance weight is 20 versus G1's 1, and its gait term
remains active at a zero command. Those properties favor choosing G1 for the
requested stand/walk chain, but do not prove that H1 cannot learn locomotion.

## Tests and visual verification

- **5,593 CPU checks** cover pinned source hashes, analytic reward responses,
  yaw and time-step invariance, anatomy mapping, phase/contact combinations,
  running phase extension, and twelve deliberate motion classes. The many
  phase assertions are not 5,593 independent behavioral experiments.
- **149 lifetime checks** verify exact terminal timing/cost, unpenalized
  timeouts, original G1 core preservation, primary-source coefficient/order,
  adversarial cycle rankings after flooring, and discounted episode returns.
  Across fixed-prefix continuations at the observed negative rates and stress
  rates, early fault exits no longer improve the return. Recorded r20 faults
  re-scored through the repair have negative returns. This is a bounded return
  test, not proof of convergence or every possible exploit.
- **4,233 quaternion cases** previously verified the local helper adapters
  against installed Isaac Lab math. FK reconstruction of both old and repaired
  smoke rollouts agrees with sampled simulated foot locations within **0.017 mm**.
- **280 real scripted Isaac screenshots**, **80 real learned-r20 replay
  screenshots**, **138 separately labeled learned-smoke CPU reconstruction
  frames** (50 raw baseline and 88 repaired smoke), and **47 official Unitree
  reference GIF frames** were inspected.
  Scripted alternating targets, foot flicks, sliding, hopping, roll impulses
  and simultaneous pitch/roll expose reward responses. These are diagnostic
  motions, not a learned walking qualification.

Primary outputs: [core tests](diagnostics/unitree_termination_20261002/counterfactual_rewards.json),
[lifetime tests](diagnostics/unitree_termination_20261002/lifetime_tests.json),
[review browser QA](diagnostics/unitree_port_20261002/frame-review-qa.json).
The review has frame controls with synchronized joint targets, actual angles,
contacts, failure flags and the original H1/G1 reward terms. Its thirteen motion
selections load all **498** frames without browser errors.

[Actual repaired smoke audit](diagnostics/unitree_termination_20261002/live_smoke_audit.json)
confirms nonnegative regular aggregation and the correct terminal cost in real
Isaac/PPO logs. Its deterministic 4-update replay still fails: first standing
failure at **4.18 s** (height/tilt), and 0.15 m/s failure at **4.40 s** (height).
The inspected 88 reconstructed frames show two planted feet followed by forward
falling, without alternating steps. This is not a walking pass or evidence of
eventual convergence. It no longer exhibits the inspected r20 candidate's
0.08-second waist-limit failure, but four updates cannot establish learning
progress or eliminate every fault.

Positive stationary/in-place reward remains possible in the public core.
Independent gates must reject no travel, sliding, hopping, asymmetric steps,
invalid transmissions, loss of standing and falls. The official clips show
alternating knee flexion and foot plants with a stable torso; the inspected
failed candidates do **not** match that gait. Future learned candidates need
their own telemetry and visual review before a style claim.

## Launch and qualification contract

R21 starts a fresh actor from the measured nominal pose. Old-policy tensors
and qualifications are not transferred. The action/observation contract stays
`physical_v29_history5_unitree_defaultoffset_v1`, actor/critic dimensions
400/415. Physical-v29 mass (**34.46334 kg**), joint frames, motor torque/speed
limits, differential geometry, collision geometry and CAD are unchanged.

The new source-frozen run performs real 32-environment contact/transmission
preflight, four PPO smoke updates, checkpoint reload, and 4,096/6,144/8,192
capacity benchmarks. The freshly trained smoke is the first unqualified
candidate; the full 128-robot nominal standing gate remains mandatory.
Duplicate evaluation of that same initial candidate was removed.

The chain targets **8192 environments**, keeps the **300 MiB VRAM reserve**,
falls back on memory pressure, and ramps one tier after **180 continuous
healthy seconds of actual PPO** at checkpoint boundaries. Host RAM recycling
preserves actor/critic/Adam/LR. The allocation growth remains unresolved; this
guard contains it rather than claiming the leak is fixed. No training camera
or parallel CUDA recorder is running.

Stages remain stand; walk 0.35, 0.65 and 1.0; run 1.5, 2.0 and 2.5 m/s.
The 0.15 m/s command remains a retention test. Separate nominal and pushed
20-second physical evaluations for current and prior skills control promotion.
The running cadence/flight parameters are an explicit extension to G1's public
walking task, not a claim to Unitree commercial running weights. A crash,
budget expiry, high reward or failed gate is never completion.

Monitor the active chain without creating a second simulator:

```powershell
& 'E:\Aditya\robot_sim\physical_v29\training\monitor_training.ps1' -Watch
```

All source, reports, saved checkpoints and journal edits stay local. No Git
commit/push was made. The user's conditional computer shutdown applies only
after the entire chain passes; it is not authorized by these diagnostic tests.
