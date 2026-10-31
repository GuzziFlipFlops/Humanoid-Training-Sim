# Adaptive training check — October 4, 11:37 UTC

Final live observation at 12:16:30 UTC: R31 is active after 1,024 completed main
updates and is running the next pushed independent evaluation with 192 trials.
The PPO training count remains 8,192. All 46 frozen files still verify and one
actual simulator worker is present. [Final live snapshot](final_live_handoff.json)
preserves this newer observation separately from the matched gate evidence below.
The same three-hour automation was updated and its identity, cadence, thread,
notification preference and active status verified in
[automation verification](automation_verification.json).

## October 4, 11:37 UTC adaptive check: walking qualified; take-off diagnosis

Observed 2026-10-04T12:09:54.454301+00:00. R31 remains running at run_150 with
8,192 environments and 960
completed main updates, plus its separately recorded inherited history.
Exact saved checkpoints have now passed matching nominal/pushed/stopped/prior-speed
gates at 0.65 and 1.0 m/s. The 1.0 checkpoint is
`035_walk_100_train_006/candidate.pt`; nominal actual signed speed was
0.942 m/s, 100% survival and 22/23 alternating
landings. These passes qualify that saved actor; the current running candidate
still fails some slow-speed retention and has no flight.

Latest matching gates: `056_run_150_gate_007_nominal` and
`057_run_150_gate_007_push`, checkpoint `055_run_150_train_006/candidate.pt`.
The 1.5 case moves 1.414 m/s
with 100% survival, 29/30 alternating landings, about 1.64 Hz cadence and zero
flight after 384 running updates. The running stage remains unqualified. Slow tracking is nonmonotonic;
0.15 briefly recovered in gate005 but failed again in gate006. No 5 m/s or
hardware-ceiling claim is supported.

CPU-only new diagnostics reconstruct the exact 400-input actor history from
saved physical traces (maximum action discrepancy 4.8e-06).
Commands/actions match and the actor responds to command-channel interventions.
Saved 50Hz contact forces and actual sole clearances independently confirm that
one foot remains loaded in every intended running flight window. Sampled
pose-dependent CAD motor speeds and declared reductions show no sustained leg
velocity ceiling; 10Hz sampling cannot exclude brief substep clipping.
The actual runtime action clamp is +/-3; the audited running trace has 31.6%
of raw outputs outside it, concentrated in some hip/ankle channels. This is a
diagnostic lead, not grounds for enlarging physical limits.

11 new controlled reward counterexamples pass. An ideal run
beats fast walking, sliding, backward/in-place stepping and same-foot hopping
at 1.5 through 5 m/s in the fixtures. The ideal-run/fast-walk gap is only about
0.09-0.11 reward-rate units under held posture/costs. Reducing a loaded foot's
force from 338 to 18 N does not change the binary gait/flight objective while
the loaded-contact classification remains the same. A continuous phase-conditioned
force signal is an undeployed hypothesis requiring a controlled physical test.
Research and diagnostics do not prove convergence or hardware feasibility.

Preserve R31 for now: it newly qualified both walking stages since the preceding
check; standing, heading, step alternation and resources remain healthy, while
early running retention is still changing. Do not erase that actor or alter
live frozen sources solely because the timer fired. The next check must be
critical about persistent zero flight: if newer independent gates show no
flight/retention progress, perform a bounded take-off/phase intervention at an
owned checkpoint boundary, compare physical force/clearance and objective effects,
then implement a justified focused repair with fresh frozen sources and smoke
tests. Do not spend another check merely changing arbitrary weights.

Completed-main throughput: 40,206 active
PPO samples/s, 32,426 including
worker startup/exit, and 18,263 over
the whole R31 chain including startup, capacity and independent gates. Last
64-update block: 40,276/32,528.
All completed main blocks used 8192 without recovery; minimum free VRAM
512 MiB, host available 9279 MiB. Last block private memory grew
4 MiB; pre-update KL and command
mismatch stay zero. All 46 frozen hashes and one actual CUDA worker verify.
Keep target8192, reserve300, fallback/180s healthy ramp and selected iteration4.

Report and reproducible evidence:
`E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_1137/REPORT.md`.
The same adaptive three-hour heartbeat remains active until qualified actual
5 m/s or a reproduced relevant velocity ceiling; then pause and brief. Leave
the computer on and preserve unsaved user work.

## Latest candidate cases

| Command | Nominal signed speed | Pushed signed speed | Survival nominal/pushed | Failures |
| --- | --- | --- | --- | --- |
| 0 | -0.000 | -0.000 | 100%/100% | Pass |
| 0.15 | 0.262 | 0.263 | 100%/100% | speed, world_forward_speed |
| 0.35 | 0.456 | 0.452 | 100%/100% | speed, world_forward_speed |
| 0.65 | 0.694 | 0.696 | 100%/100% | Pass |
| 1 | 0.998 | 0.998 | 100%/100% | Pass |
| 1.5 | 1.414 | 1.416 | 100%/100% | flight |


## Exact checkpoint qualification

- stand (0 m/s): `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r28_heading_alignment_cache_fixed\002_ppo_smoke\transferred_actor.pt`.
- walk_035 (0.35 m/s): `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r30_exact_graph_execution\019_walk_035_train_004\candidate.pt`.
- walk_065 (0.65 m/s): `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r31_exact_command_rehearsal\015_walk_065_train_003\candidate.pt`.
  Matching gates: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r31_exact_command_rehearsal\016_walk_065_gate_004_nominal` and `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r31_exact_command_rehearsal\017_walk_065_gate_004_push`; SHA256 `15f4050b2396e349b7d622891d57fd063f171351bfc7e7642bc1d5fa9282f67c`.
- walk_100 (1 m/s): `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r31_exact_command_rehearsal\035_walk_100_train_006\candidate.pt`.
  Matching gates: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r31_exact_command_rehearsal\036_walk_100_gate_007_nominal` and `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r31_exact_command_rehearsal\037_walk_100_gate_007_push`; SHA256 `2fbe6823792d0a4a31909fda058f069333e1fcb6584d1286a3e6492bf9bdeec4`.


## Diagnostic files

- [Evidence snapshot and complete gate history](evidence.json), recorded at the time above.
- [CPU running/actor/CAD audit](running_audit.json) and [reproducible code](audit_running.py).
- [Running reward counterfactuals](running_counterfactuals.json) and [code](counterfactual_running.py).
- [Running contact timing](running_contact_timing.png): saved physical forces, sole clearance and knee angles, not a rendered simulation.
- [Independent speed trend](speed_tracking_trend.png).
- [Snapshot recorder](record_evidence.py). It does not start a simulator or edit a run.

## Primary-source review and decision limits

The public [G1 Isaac Lab configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py)
and [H1 configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py)
use alternating contact phases with duty 0.55. With half-cycle leg offsets this
implies overlapping stance rather than flight. That mathematical inference is
why this project's 0.43 running duty and flight term are declared local adaptations;
the public walking configuration alone does not validate 5 m/s running here.

[Siekmann et al., Periodic Reward Composition](https://arxiv.org/html/2011.01387v2)
build gait objectives around phase-dependent physical forces and foot velocities,
with smooth phase boundaries. This supports testing a continuous unloading signal
if the measured take-off plateau persists. It does not establish that the proposed
local exponential signal, its scale, this robot's transmissions or this actor will
work. The force-sweep fixture holds posture, velocity and other costs fixed; it
is neither a dynamically possible trajectory nor a substitute for a physical test.

This check changes diagnostics and documentation only. Runtime reward, PPO,
action mapping, geometry, collisions, mass/inertia, motor limits and gates are
unchanged. No new frozen run, policy reset, qualification relaxation or shutdown.
The current actor can continue learning; its preserved qualified walking actor
is available if a later tested repair needs a known good starting point.
