# Adaptive three-hour check — October 4, 08:36 UTC trigger

Recorded 2026-10-04T09:52:17.515776+00:00. Read ACTIVE_RUN.json and fresh state for later progress.

## Result and continuation

R30 independently qualified 0.35 m/s under matching nominal/pushed and retained
command gates, then improved straight alternating walking at 0.65 m/s. Its
0.65 cases pass repeatedly; promotion remains blocked by retained 0.15 m/s
overspeed. This is a specific tracking/retention failure, not evidence of another
circling, hopping or stationary-step qualification loophole. Reward alone still
does not certify the policy.

R30 was drained only after another matching nominal slow-speed failure, at its
saved full **1,472-main-update** checkpoint, with the pushed evaluation allowed
to finish. All owned workers exited; no updating training worker was interrupted.
The source-frozen **R31 exact-command-rehearsal** continuation is now
**running**, stage **walk_065**, operation
`015_walk_065_train_003`.
Current environment count: 8192; next training count:
8192. New main updates:
128. Preserve an improving candidate.

Initialization remains the exact original R30 checkpoint:
`E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r30_exact_graph_execution\078_walk_065_train_019\candidate.pt`

SHA256 `8a62733db47184082e9a1dcf5459893c9285f10018feb5af3a0b3f86f202e31d`. CPU full-load equality confirms all
9 actor, 8 critic and 51 Adam tensors. Exploration and Adam moments are retained.
The unchanged worker restart rule bounds the saved 0.00015 learning rate to the
existing 0.0001 ceiling; it is not an exact-rate preservation claim. Small smoke
actors and benchmark actors are discarded before main learning.

## Independent behavioral evidence before repair

The last R30 matching 128-robot, 20-second nominal/pushed evaluations are
`079_walk_065_gate_020_nominal` and `080_walk_065_gate_020_push`. Every command has32 trials.
All latest moving cases survive and record22/23 alternating landings, no same-foot
or paired-hop landings, low foot slip and close heading alignment.

| Command | Nominal measured forward | Pushed measured forward | Nominal failures | Pushed failures |
| --- | --- | --- | --- | --- |
| 0.00 m/s | -0.000 m/s | -0.000 m/s | pass | pass |
| 0.15 m/s | 0.277 m/s | 0.275 m/s | speed, world_forward_speed | speed, world_forward_speed |
| 0.35 m/s | 0.445 m/s | 0.442 m/s | pass | pass |
| 0.65 m/s | 0.732 m/s | 0.731 m/s | pass | pass |


The 0.15 command repeatedly measured approximately0.27–0.30 m/s over late
checkpoints, even while the 0.65 case improved. The earlier 0.35 qualified actor
did pass0.15, so this is not established as a mechanical minimum walking speed.
Current1.0 m/s and running stages have not qualified. Earlier R25 passes under the
old direction-permissive verifier remain historical, not new straight-walk proof.

## Focused repair and scope

Old training already sampled earlier speeds, but only about1.8% of all training
commands fell within0.02 m/s of0.15 at the0.65 stage. Exact-command rehearsal now
gives0.15 about12.1% of the population, a measured6.71-fold exposure increase.
The total mixture remains32% stopped,24% prior moving and44% frontier moving.
Prior rows select exact independently evaluated speeds; frontier rows include its
exact command and retain the original0.8–1.0 range. RNG draw count and the
permanently stopped one-fifth cohort remain. This is a declared local training
adaptation, not a claimed upstream Unitree sampler or proven convergence cure.

Actor/action/observation/history/reset semantics, all reward kernels/weights,
geometry, mass/inertia, contacts, solver settings, CAD Jacobians/reductions and
hardware bounds are unchanged. Selected iteration4 execution optimizations stay.
Evaluation delegates to the original sampler. Removing only the pinned
training override reconstructs the original environment bytes and signature;
restoring the narrowly listed metadata/signature changes reconstructs the full
old worker, including PPO and checkpoint loading. All other authored runtime
files must match. Current frozen hashes verified: 46.

The same saved checkpoint is re-evaluated before main learning. R30's matching
older-speed qualification is preserved for its exact trusted checkpoint; every
current and lower-speed command still must pass before further promotion.
No gate threshold was relaxed.

Latest completed post-rehearsal nominal independent result: {"folder": "013_walk_065_gate_003_nominal", "passed": false, "cases": [{"command_m_s": 0.0, "signed_forward_m_s": -7.57767751230208e-05, "survival": 1.0, "failures": []}, {"command_m_s": 0.15, "signed_forward_m_s": 0.2684594571132567, "survival": 1.0, "failures": ["speed", "world_forward_speed"]}, {"command_m_s": 0.35, "signed_forward_m_s": 0.4058920394358688, "survival": 1.0, "failures": []}, {"command_m_s": 0.65, "signed_forward_m_s": 0.6576858364128867, "survival": 1.0, "failures": []}]}
Latest completed post-rehearsal pushed independent result: {"folder": "014_walk_065_gate_003_push", "passed": false, "cases": [{"command_m_s": 0.0, "signed_forward_m_s": -7.174534504640036e-05, "survival": 1.0, "failures": []}, {"command_m_s": 0.15, "signed_forward_m_s": 0.26623869117078836, "survival": 1.0, "failures": ["speed", "world_forward_speed"]}, {"command_m_s": 0.35, "signed_forward_m_s": 0.40142139313253444, "survival": 1.0, "failures": []}, {"command_m_s": 0.65, "signed_forward_m_s": 0.6611599811670168, "survival": 1.0, "failures": []}]}
The early slow-speed improvement is not monotonic: nominal0.15 moved from0.277
before rehearsal to0.258 at64 updates and0.268 at128. Current0.65 tracking improved
from0.732 to0.685 then0.658, retaining survival and alternating steps. Preserve
the useful progress, but do not describe the slow-control issue as solved.
Matching pushed/retention qualification remains mandatory before promotion.

## Tested extension toward5 m/s

The frozen curriculum now continues through1.0 walking and1.5/2.0/2.5/3.0/3.5/
4.0/4.5/5.0 m/s running. Existing prior speed cases/classifiers are unchanged.
At5, an additional actual signed reset-frame speed check requires at least4.95
m/s, declaring a1% measurement tolerance; commanding5 is never success.
Population grows to keep at least32 trials per command, reaching416 robots per
nominal/pushed final test. Old cadence, clearance, flight, direction, net travel,
anti-fall/slide and retention requirements remain.

The endpoint is independently qualified actual5 m/s or a reproduced relevant
actuator velocity ceiling. A ceiling still requires motor speeds through the
actual differential Jacobians, sustained phase-specific saturation, torque
headroom and a controlled speed-plateau diagnosis. None is established here.
At either endpoint save, pause training and this heartbeat, and brief the user;
leave the computer on. No shutdown occurred.

## Validation and fidelity

- 51 CPU continuation/resource/heading/ownership/balance tests pass. Scheduler
  fixtures were made coherent with their own frozen source; no guard was weakened.
- 24 evaluation sampler/RNG comparisons are bitwise identical. All182 saved
  original case classifications match, and10 protected-source/false-qualification
  counterexamples are rejected. The current proof pins these tests and5 allowed
  runtime files; there is no broad signature bypass.
- 32 phase/lift/lifetime/standing-retention-gradient/PPO-audit assertions pass.
  Eight running command speeds through5 favor matched cycles over in-place,
  backward, sliding and synchronized-hop fixtures. The goal classifier rejects
  a4.8 m/s counterexample that the generic old5-command gate would accept.
- 5,593 upstream G1/H1 counterfactual checks pass. Those upstream tests also
  reproduce limitations of raw rewards; fixtures do not qualify real gait.
- Fresh32-robot physical/contact/combined-transmission preflight passes, with
  zero invalid poses and337.61 N support against338.09 N estimated weight.
- Actual CUDA64-robot four-update PPO and four-second checkpoint reload complete.
  Every smoke pre-update KL is zero; full-load equality is separately saved.
  The short reload did not pass every gait gate and is never a qualification.
- Same-checkpoint independent physical replay: {"passed": true, "independent_cases_identical": true, "files": {"telemetry.npz": {"all_arrays_equal": true, "arrays": 13}, "gait_diagnostics_50hz.npz": {"all_arrays_equal": true, "arrays": 26}, "startup_diagnostics.npz": {"all_arrays_equal": true, "arrays": 12}}}
  Details: [evaluation parity](evaluation_parity.json).

Files: [CPU proof](cpu_test.json), [full checkpoint load](full_checkpoint_load.json),
[running tests](running_extension_tests.json), [migration proof](continuation_proof.json),
[drain](r30_drain.json), [raw evidence](evidence.json). Small fixture/build attempts
initially caught mixed-newline restoration, fixture tick dtype and stale capacity
fixtures. Two harmless drain monitors used the wrong nominal-folder index and
were stopped before touching training; the final owned-process drain succeeded.
Those failures were preparation/test problems, not erased production failures.

## Throughput and memory

Latest six R30 main6144 blocks average
35,683 active PPO samples/s
(samples divided by collection+learning time),
28,709 worker-effective
samples/s including worker starts/exits. Full observed R30 main samples/time is
15,257 samples/s including startup,
capacity and independent gates. These denominators are distinct.

Those blocks retain at least1692 MiB GPU free and10326 MiB host available.
Worker private memory grows only6–9 MiB per64 updates; Torch reserve stays
1630–1632 MiB. No new unbounded cache leak is demonstrated. One earlier8192
main block hit the300 MiB reserve and saved the full recovery state; subsequent
6144 workers were stable. Its35 interrupted updates are preserved and included
in measured sample totals even though the supervisor's main count omits them.

R31 capacity results are saved in benchmarks.json:
`[{"environments": 4096, "environment_steps_per_second": 30461.986491097854, "minimum_free_mb": 2811, "folder": "E:\\Aditya\\robot_sim\\physical_v29\\training\\runs\\locomotion_20261004_r31_exact_command_rehearsal\\004_benchmark_4096", "updates": 8, "wall_seconds": 69.01147150993347, "effective_environment_steps_per_second": 11395.670644217482, "host_ram_recycled": false, "minimum_host_available_mb": 11461, "expected_updates": 8, "eligible": true}, {"environments": 6144, "environment_steps_per_second": 36767.35481463884, "minimum_free_mb": 1743, "folder": "E:\\Aditya\\robot_sim\\physical_v29\\training\\runs\\locomotion_20261004_r31_exact_command_rehearsal\\005_benchmark_6144", "updates": 24, "wall_seconds": 154.71773552894592, "effective_environment_steps_per_second": 22873.55090805284, "host_ram_recycled": false, "minimum_host_available_mb": 10266, "expected_updates": 24, "eligible": true}, {"environments": 8192, "environment_steps_per_second": 41593.722595390485, "minimum_free_mb": 659, "folder": "E:\\Aditya\\robot_sim\\physical_v29\\training\\runs\\locomotion_20261004_r31_exact_command_rehearsal\\006_benchmark_8192", "updates": 24, "wall_seconds": 185.41734743118286, "effective_environment_steps_per_second": 25448.49263228346, "host_ram_recycled": false, "minimum_host_available_mb": 9439, "expected_updates": 24, "eligible": true}]`

Latest completed new main block: `{"folder": "E:\\Aditya\\robot_sim\\physical_v29\\training\\runs\\locomotion_20261004_r31_exact_command_rehearsal\\012_walk_065_train_002", "iterations": 64, "environments": 8192, "checkpoint": "E:\\Aditya\\robot_sim\\physical_v29\\training\\runs\\locomotion_20261004_r31_exact_command_rehearsal\\012_walk_065_train_002\\candidate.pt", "active_PPO_samples_s": 40595.15960936023, "worker_effective_samples_s": 32435.04969852304, "minimum_free_vram_mb": 531, "minimum_host_available_mb": 9497, "first_private_mb": 15044, "last_private_mb": 15050, "pre_update_KL_max": 0.0, "action_command_difference_max": 0.0}`.

Current worker completed updates at this observation: 16.
Current collection+learning throughput: 40820.3137509981 samples/s.
Latest progress memory: {"used_mb": 7428, "free_mb": 529, "utilization": 84, "ram_available_mb": 9561, "ram_total_mb": 32580, "process_rss_mb": 7975, "process_private_mb": 15069, "torch_allocated_mb": 866, "torch_reserved_mb": 2158}
Latest command coverage: {"version": "exact_prior_frontier_rehearsal_v1", "stopped_fraction": 0.318359375, "exact_command_fractions": {"0.15": 0.123046875, "0.35": 0.1173095703125, "0.65": 0.2490234375}, "minimum_m_s": 0.0, "maximum_m_s": 0.6499999761581421, "scope": "Current full-population training commands; not a behavioral gate."}
These are training diagnostics, not an independent post-repair gait pass.

The target stays8192 with both host/GPU reserves, automatic fallback and180
continuously healthy training seconds before upward ramps. Stopped-command
capacity alone does not guarantee moving-command memory headroom. Unrelated
unsaved Fusion work is preserved. No second actual CUDA worker was started.

## Primary-source cross-check and next focus

[Unitree H1 configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py)
uses a widening velocity-command curriculum and nominal-offset joint actions;
[Isaac Lab command implementation](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py)
exposes sampled velocity commands and heading feedback. These support auditing
the command distribution; the exact retained-command rehearsal is our
evidence-based adaptation. Matching rewards alone does not make different robot
actuation, physics and implementation equivalent.

Next1–3 tests: compare post-rehearsal0.15/0.35/current-frontier independent speed
and survival with this baseline; preserve improvement and finish matching gates.
Monitor moving-command8192 memory/fallback and actual sample/time, retaining the
optimization. Before running promotion inspect cadence/flight/clearance and
actual motor velocity telemetry; if tracking stalls, measure per-command reward
gradient/response rather than repeatedly changing weights.
