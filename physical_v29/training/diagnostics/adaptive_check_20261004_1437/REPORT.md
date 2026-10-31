# Adaptive training check — October 4, 14:37 UTC

## October 4, 14:37 UTC adaptive check: running take-off repair

R31 was safely drained at a saved boundary after repeated 1.5 m/s running blocks
failed to develop genuine flight. Its final checkpoint107, 2,052 completed main
updates and all logs/qualifications remain preserved. The selected checkpoint101
passes every stopped and prior walking command in matching nominal/pushed gates102/103;
the latest107 regresses0.35/0.65. R32 retains101's exact mean and trained exploration,
with fresh critic/Adam for the changed running objective and fresh qualification.
Historical exact walking passes through1.0 m/s remain available; neither running
nor the5 m/s endpoint is qualified.

Repeated synchronized sole/force telemetry finds zero geometric flight despite
about1.43–1.45 m/s signed travel and100% survival. Small force-threshold-only
flight readings at101 are grounded contact gaps, not progress toward running.
A controlled96-robot normal-limit action/phase probe achieved brief small real
separation with a0.3rad knee-retraction pulse:0.285% geometric flight, maximum
minimum sole clearance1.318mm, all96 survivors, no resets. No training helper or
manual pulse was deployed. Its old total reward was lower, so the probe establishes
partial take-off feasibility, not that the whole objective favors that pulse.

Running v7 adds one bounded continuous phase-conditioned foot-force term,
weight0.5 and quarter-body-weight exponential scale. The flight bonus now requires
both actual soles above1mm and both normal forces below1N. Evaluation v6 adds
the same geometric evidence requirement with the existing1.5%–45% flight bounds;
the old flight gate and all prior gates remain. This is a local research-informed
adaptation, not an exact Unitree recipe or a convergence guarantee.

Forty-nine CPU checks pass, including600 bitwise standing/walking fixtures,
partial-unloading feedback, adversarial running cases through5m/s, grounded
false-flight rejection and source-bound mean/exploration transfer. A real64-env
four-update running PPO smoke and full checkpoint reload pass, all survivors,
pre-update KL0 and observation/action command difference0. Forty-one protected
authored runtime files are byte-identical; exactly3 files changed. Geometry,
mass/inertia, contacts, CAD differential constraints/motor limits, inference,
commands, PPO configuration and iteration4 execution remain unchanged.

Fresh observation 2026-10-04T15:46:21.172096+00:00: R32 status=running,
stage=run_150, current worker environment count=6144,
main updates=0. All46 frozen hashes verify;
1 actual worker, GPU used/free/utilization=7143, 814, 60,
host available8450MiB. Count during preflight/capacity is not
the selected next PPO count (6144). Fresh promotions are
[('stand', 0.0), ('walk_035', 0.35), ('walk_065', 0.65), ('walk_100', 1.0)], tied to the saved evaluated actor.
Target8192, reserve300MiB, host guards,
fallback and180 healthy seconds before an upward ramp remain enforced.

Read `E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_1437/REPORT.md`
and its replacement_status.json for fresh stage/qualification. Continue checking
true flight and retained speed tracking; do not count force unloading, smoke
updates or a5m/s command as qualification. The same three-hour heartbeat remains
active. No relevant sustained actuator velocity ceiling is demonstrated. At the
verified endpoint pause and brief; leave the computer on and preserve user work.


## Preserved actors and independent trends

| Command | Selected 101 nominal/pushed signed speed | Selected prior cases | Last 107 nominal signed speed | Last failures |
| --- | --- | --- | --- | --- |
| 0 | -0.000/-0.000 | Pass | -0.000 | Pass |
| 0.15 | 0.179/0.179 | Pass | 0.247 | Pass |
| 0.35 | 0.401/0.395 | Pass | 0.481 | speed, world_forward_speed |
| 0.65 | 0.702/0.706 | Pass | 0.763 | speed, world_forward_speed |
| 1 | 0.982/0.983 | Pass | 1.011 | Pass |
| 1.5 | 1.431/1.433 | Fail | 1.455 | flight |


Selected initial actor: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r31_exact_command_rehearsal\101_run_150_train_021_part_02\candidate.pt`.
SHA256 `7abb125cb7151c92a71b8ad49851a95cb4a811ffc46eba79cc374996c39f4ce3`. Last preserved actor: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r31_exact_command_rehearsal\107_run_150_train_023\candidate.pt`,
SHA256 `a7f499f9732669102ca91fb3516badb29fc5b8ac5e10a57db2e611e13d91fc28`.
R31 actor/critic/Adam/exploration remain intact on disk. The R32 transfer keeps
the selected mean/exploration exactly, discards value/moments because returns
changed, and discards smoke/benchmark-trained actors before qualification.
The actual v6 transfer is recorded in warm_start.json; the stock supervisor's
historical `v4 trained exploration` label is stale descriptive text, not its
source-bound transfer protocol. No qualification was reused.

R31 initial fresh evidence: 40,337 active
PPO samples/s, 32,600 worker-effective,
19,429 full-chain. These denominators differ.
An8192 block reached the300MiB VRAM guard and saved recovery; its remaining
updates finished at6144. Last saved6144 main block101 gave46,796 active/
40,057 worker-effective samples/s, minimum1,455MiB free VRAM and9,337MiB host
available, about6MiB private memory growth. Memory fallback preserved progress.
Do not describe all newer blocks as8192 or capacity tests as main throughput.

## Controlled physical experiment

| Physical intervention | Actual forward speed | Both soles >1 mm and forces <1 N | Maximum minimum sole clearance | Old reward rate |
| --- | --- | --- | --- | --- |
| baseline | 1.459 m/s | 0.000% | 0.082 mm | 13.934 |
| retract_010 | 1.454 m/s | 0.000% | 0.334 mm | 13.880 |
| retract_020 | 1.462 m/s | 0.000% | 0.928 mm | 13.841 |
| retract_030 | 1.460 m/s | 0.285% | 1.318 mm | 13.736 |
| phase_minus008 | 1.463 m/s | 0.000% | 0.063 mm | 13.783 |
| phase_plus008 | 1.469 m/s | 0.000% | 0.072 mm | 13.946 |
| slow_015 | 0.249 m/s | 0.000% | 0.020 mm | 14.488 |
| slow_035 | 0.485 m/s | 0.000% | 0.025 mm | 14.251 |


Eight groups of12 robots,10seconds at50Hz, same107 actor, normal gravity,
unfixed base, unchanged actuator limits. Interventions begin at3seconds.
Policy phase was shifted +/-0.08cycles, or a brief anatomically balanced
hip/knee/ankle retraction was applied only in intended flight windows. All96
robots survived; no resets. The0.3rad case has no both-sole clearance above5mm.
The pulse is a diagnostic intervention, absent from deployed action mapping.

The first diagnostic attempt failed on an inference-tensor ownership restriction;
its script/failure are preserved in probe_attempt_01. The probe was repaired by
cloning the action buffer before mutation and rerun successfully. Its own failed
worker was verified and stopped after preserving the exception; no unrelated
application or worker was stopped. This was a diagnostic failure, not a policy pass.

![Saved physical force/sole timing](physical_takeoff_timing.png)

Plot selects the maximum-clearance trial in each group; population statistics
are in result.json. It is telemetry, not rendered visual gait verification.

## Objective, physical and provenance limits

Old binary gait/flight feedback remains flat when a supported swing foot is
unloaded338→18N without crossing the contact classification. The added force
signal has a tested nonzero gradient under held posture/costs, positive bounded
range and no standing/walking reward change. The new aggregate objective's
physical effect and convergence still require independent learned rollouts.
The scripted retraction lowers the old average total reward even though it
slightly improves unloading. Do not claim the new0.5 weight reverses every cost
tradeoff, produces a full running stride or establishes hardware5m/s feasibility.

Saved actor-history reconstruction reproduces2240 actions within5.8e-6;
commands match. About35.1% of audited raw running outputs exceed the runtime
action clamp +/-3. Effective action/target clipping should continue to be audited.
Exact pose-dependent CAD Jacobian/reduction motor speeds show no sustained leg
velocity ceiling.10Hz sampled motor audit cannot exclude brief physics-substep
clipping, and isolated near-cap samples do not establish the endpoint.

The primary [Periodic Reward Composition paper](https://arxiv.org/html/2011.01387v2)
uses phase-conditioned force/velocity objectives with smoothed boundaries. It
supports testing force feedback; our positive exponential, quarter-weight scale
and0.5 rate are declared local adaptations. The public
[Unitree H1 velocity task](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py)
has walking phase duty0.55. Half-cycle alternation with that duty implies
overlapping support, not a running flight guarantee. Existing local running
duty0.43, motor limits and independent fixed-frame criteria remain.

## Tests, execution and next decision

Completed saved capacity attempts:

- `006_benchmark_4096`: 8 updates, completed=True, recycled=False, seconds=33.45300000003772.
- `007_benchmark_6144`: 24 updates, completed=True, recycled=False, seconds=81.79700000002049.

| Capacity count | Completed updates | Active PPO samples/s | Worker-effective samples/s | Minimum free GPU / host MiB |
| --- | --- | --- | --- | --- |
| 4096 | 8 | 28,900 | 10,210 | 2164 / 8621 |
| 6144 | 24 | 46,144 | 32,025 | 1081 / 7790 |

Selected next training count: 6144. The 8192 attempt reached the VRAM reserve and was rejected, not accepted as capacity. Fresh promotions at this observation: [('stand', 0.0), ('walk_035', 0.35), ('walk_065', 0.65), ('walk_100', 1.0)].


- [R31 initial evidence, gate inventory and memory/throughput](evidence.json).
- [Exact history, command, force/sole and CAD motor audit](running_audit.json).
- [Owned checkpoint drain](drain.json) and [selected retained actor](selected_seed.json).
- [Controlled96-robot results](physical_response/result.json), telemetry.npz and [probe code](physical_takeoff_probe.py).
- [49 CPU counterfactual/provenance checks](cpu_tests.json) and [test code](test_candidate.py).
- [Actual running PPO smoke](running_ppo_smoke/result.json), warm_start.json, progress.jsonl.
- [Full checkpoint reload diagnostic](running_checkpoint_reload/result.json):64envs/4sec, too small/short to qualify.
- [Three-file deployment and41-file protected equality proof](deployment_proof.json).
- [Replacement live state,46 hashes and owned worker](replacement_status.json).

The CPU-only handoff recorder originally rejected the owned supervisor because
Windows command arguments used forward slashes. Its output ownership check now
resolves the actual --output argument, accepting equivalent Windows path spellings
while rejecting another run, misleading name prefix or a checkpoint-only path.
Five ownership counterexamples pass. This diagnostic-only fix is outside the
frozen runtime inventory; it changes no simulator, policy or process behavior.
The prior recorder and failed diagnostic attempt are documented and preserved.

No stereo, external root forces, phase intervention, fixed base, enlarged motor
limits or relaxed gate were deployed. R31 frozen source and every checkpoint are
preserved. New R32 frozen source is independently checked. Fresh standing,
nominal/pushed walking and prior-speed retention must pass before running
promotion. Target5m/s remains actual signed>=4.95m/s with>=32 trials/command,
nominal/pushed and retained skills. Neither success nor a hardware ceiling is
proven. If the added feedback also plateaus, inspect effective target gradients,
force signs/distribution and costs per intended two-foot flight window with
controlled physical tests before another narrow repair; avoid arbitrary weights.

<!-- walking_gui_capture_20261004 -->
## User-requested GUI recordings and exact training resume

The user requested a temporary training pause, actual GUI capture at several
walking speeds for ten seconds of real-time motion per clip, then resume.
The exact owned R32 supervisor PID 29884 was suspended at
a saved evaluation boundary; its child drained before any GUI simulator started.
The same supervisor, in-memory curriculum and saved actor resumed after capture
and verification. Pause duration: 775.6 wall seconds. No training restart,
policy change or critic/Adam reset occurred for these recordings.

Four actual Isaac Sim 5.1 GUI clips were captured from a tracking front/side
view at commanded0.15,0.35,0.65 and1.0m/s. Every clip is1280x720 H.264 MP4,
500 distinct control samples at50FPS, exactly10 simulated seconds played in10
real seconds. Each took about56 wall seconds to record. All decoded frames
were checked and representative start/middle/end frames visually inspected;
the whole robot and feet remain visible. All clips have zero falls. They depict
the frozen training geometry and deterministic actor, not a new CAD export or
independent qualification. Actual average signed speeds are0.181,0.404,0.704
and0.989m/s; target labels are explicit. Full synchronized telemetry is retained.

The first recording attempt produced a valid0.15 clip but failed while resetting
cached Isaac inference buffers before the next speed. The failed script, logs,
startup and exception are preserved in videos/walking_gui_20261004/attempt_01.
Reset now runs in the same inference context as rollout; the valid first clip
was preserved and remaining three completed. This changes only the recording
script, with no frozen training or physical changes.

Fresh qualification before the pause passed matching nominal/pushed standing,
0.35,0.65 and1.0m/s stages, with stopped and all earlier-speed retention. These
passes belong to the exact002_ppo_smoke/transferred_actor.pt checkpoint. The
1.5 baseline fails genuine-flight gates; no running,5m/s or velocity ceiling
qualification is claimed. The current learning actor needs new matching gates.

Resumed evidence: 2026-10-04T15:46:21.172096+00:00; stagerun_150,
6144 environments, one actual worker,46 hashes verified.
The current block has58 observed PPO updates (through local iteration
57), with every pre-update KL/action-command mismatch zero;
latest free VRAM814MiB and host available
8450MiB. Completed main blocks remain separately
counted at0 until this block finishes.

The suspended supervisor's recorded wrapper time for evaluation018 includes
the user video pause (about12.9 minutes), although its physical
rollout completed normally. Whole-chain throughput/budget clocks also include
this pause. Subtract the documented pause when comparing active training
throughput; do not diagnose a simulator stall from that wrapper duration.
The original timing and pause audit are retained rather than rewritten.

Video files and timing/provenance: `E:/Aditya/robot_sim/physical_v29/training/videos/walking_gui_20261004/README.md`.
Pause/resume: video_pause.json and video_resume.json. Decode/hash verification:
video_verification.json. Training remains active; the same three-hour adaptive
check continues toward the existing pause-and-brief endpoint. Computer left on.
<!-- end_walking_gui_capture_20261004 -->



## Final saved boundary after recording

Final saved boundary after video resume: first main operation019 completed 68 worker updates, all pre-update KL and action-command mismatches zero, minimum free GPU 812 MiB, host 8330 MiB, private growth 6 MiB. Matching nominal020/pushed021 on candidate019 show 1.5-command signed speed 1.589/1.591 m/s and 100% survival. Genuine geometric-flight fraction is only 0.0799%/0.1806%, below the unchanged required 1.5%, so running remains unqualified. Retained 0.35 now fails bilateral stance in both and world lateral speed under push; other retained cases pass. These are first-block observations, not solved running or a reason to erase the actor immediately. Compare subsequent full-population contact timing, heading and retention against this baseline; preserve the qualified baseline actor used for the videos. Block022 is now learning at6144. Four completed videos and exact supervisor pause/resume are preserved; three-hour automation remains active.

The automation backend misencoded Unicode dash punctuation. Equivalent ASCII dashes were used in the existing automation name/prompt, with the same ID, thread, three-hour cadence and notification preference verified. No duplicate automation was created.
