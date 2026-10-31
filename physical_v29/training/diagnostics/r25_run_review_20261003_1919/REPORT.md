# October 3, 19:19 UTC check — direction fault repaired, learned actor retained

R25 learned real alternating locomotion and passed its existing nominal/pushed
standing and walking tests through **1.0 m/s**. Running at 1.5 m/s developed
bilateral steps and short flight, but still failed speed/lateral or retention
checks. A full-population audit then reproduced a qualification loophole:
body-relative forward speed could pass while the robot gradually turned.
The replacement is **R27**, `locomotion_20261003_r27_heading_commands_checked`.
Read [`../../ACTIVE_RUN.json`](../../ACTIVE_RUN.json), its fresh state, and
[`../walk_regression_20261003/live_handoff.json`](../walk_regression_20261003/live_handoff.json)
for execution after this dated report. This is not final locomotion qualification.

## Observed progress and the missing direction check

Matching gates use 128 robots for 20 seconds, nominal/pushed seeds 901/1301,
fixed speed commands, all startup falls counted and two seconds excluded from
averages. Pushed trials add horizontal velocity at 5 and 12 seconds.

| R25 checkpoint/gates | Frontier | Body-relative speed, nominal | Left/right steps | Outcome under old checks |
|---|---:|---:|---:|---|
| 106 / 107–108 | 0.35 m/s | 0.382 m/s | 22 / 23 | All current and retained commands passed |
| 114 / 115–116 | 0.65 m/s | 0.650 m/s | 22 / 23 | All passed |
| 128 / 129–130 | 1.0 m/s | 0.990 m/s | 22 / 23 | All passed |
| 140 / 141–142 | 1.5 m/s | 1.678 m/s | 29 / 30 | Running case passed; low-speed retention failed |
| 147 / 148–149 | 1.5 m/s | 1.721 m/s | 29 / 30 | Speed/lateral and pushed low-speed alternation failed |
| 155 / 156–157 | 1.5 m/s | 1.713 m/s | 29 / 30 | All old walking/standing cases retained; running speed/lateral failed |

Survival was 100% in these cited pairs. Running flight increased from about
4.6% at baseline to 7.7%; cadence was about 1.64 cycles/s and accepted clearance
about 47 mm. These are substantial changes from R23's zero-step collapse.
They do not establish straight travel or humanlike appearance.

[`full_population_direction.json`](full_population_direction.json) audits **all
128 robots**, using saved 10 Hz root poses and first-episode live masks from ten
completed evaluations. Evaluation reset headings were identity in the frozen
source; the audit measures travel against that original heading, not the final
body orientation. At the old 0.65 m/s nominal pass, mean final absolute heading
was 122.4 degrees, signed forward speed in the reset frame only 0.254 m/s,
absolute lateral speed 0.463 m/s, and forward/path ratio 0.392. At the old
1.0 m/s nominal pass, final heading averaged 89.1 degrees, reset-frame forward
speed 0.596 m/s and lateral speed 0.677 m/s. The latest candidate improved
1.0 m/s nominal direction considerably but still turned at retained 0.65 m/s
and under some pushed conditions. This is a task-verification fault alongside
real gait learning, not evidence that every motion was a shuffle or fall.

The frozen landing classifier replays **768,000** saved landing/hop/repeat
scalars exactly from twelve evaluations. Inspected synchronized joint/contact
plots and successive [walking frames](148_run_150_gate_003_nominal_robot_4_frame_sheet.png)
and [running frames](148_run_150_gate_003_nominal_robot_5_frame_sheet.png) show
alternating knee/hip flexion, real unloading and short running flight. They are
CPU forward-kinematic reconstructions of measured states, **not Isaac screenshots**
or Unitree motion/style certification. Contact-patch pressure and hardware
thermal duty remain unqualified.

## Narrow repair and source provenance

The gait reward kernels and coefficients are unchanged. Added desired-yaw-rate
feedback using the [documented Isaac Lab heading command](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py):
gain 0.5 times wrapped heading error, with a 0.2 rad/s bound taken from the
[published G1 angular-command range](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py).
The G1 configuration itself disables heading mode; enabling it here is a declared
local task adaptation. The target is this robot's reset heading for the whole
episode. Stopped commands stay zero. The actor receives the desired yaw rate in
its existing three-component velocity-command input; it still supplies every
motor action, with no steering action helper or camera.

Command resampling and heading updates now occur when constructing the next
observation, after the preceding reward. The command seen with an action is
held through that action and reward. Saved `observed_command`/`action_command`
traces and `action_command_difference` telemetry audit this timing explicitly.
The earlier private history-snapshot repair is retained.

Independent gates add reset-frame signed forward speed, absolute lateral speed,
heading and forward net/path measurements. Local heading bounds are 20 degrees
for walking and 25 for running, with less than 5% outside; forward/path ratio
must exceed 0.85. Fixed-frame speed tolerance and lateral limit use the existing
0.10 m/s-or-12% and 0.14 m/s criteria. These new heading/path margins are local
engineering choices, not claims of a published optimum. Every previous
survival, step, alternation, cadence, support, contact, slip, tilt, clearance,
flight and retention threshold remains unchanged.

R25 was drained at **1471 main PPO updates**. The final worker finished normally
and saved `161_run_150_train_004_part_04/resource_recovery.pt`; all earlier
actors and logs remain. R27 transfers candidate155, SHA256
`892a9514ce58adaf684589c4f01ea085abaa604202a711d7deb6329e65c3f039`, whose latest
matching pair retained every old walking/standing case. Its mean and trained
exploration are preserved exactly before smoke updates. The critic/Adam are
fresh; initial learning rate is 0.0001 with existing adaptive PPO. Qualification
starts from the preserved `transferred_actor.pt`, not the smoke-updated actor.
No standing/walking qualification is inherited across the command/gate repair.

Reward version `cad_heading_command_retention_v5`, command
`fixed_reset_heading_yaw_rate_v1`, gate `physical_landing_v5_fixed_direction`.
The 400/415 observation inputs, 23 actions, no empirical normalization, mass,
URDF contacts, CAD differential/Jacobian and motor constraints are unchanged.
Physical-v32 CAD remains separate. No joint-axis fault was reproduced by this
check; the fresh physics test includes simultaneous differential combinations.

## Resources, tests and current execution

Fusion had an unsaved **Strap cam** document and about 9.7 GB private system
memory. The native bridge did not answer the read-only document query. Native
computer-control tooling was unavailable in this session, so Fusion was left
open without discarding unrelated user work. R25 fell from8192 through6144,
4096 and3072 to2048 while preserving memory checkpoints. Other apps also use
memory. Native host-memory growth remains unresolved; checkpoint recycling is
still necessary and is not convergence or completion.

Reproduced another maintenance defect: host-RAM failures also increased the
estimated VRAM cost per environment from0.5 to2.30 MiB, obstructing later ramps.
Host and GPU failure accounting are now separate. A ramp requires180 continuous
seconds of fresh live PPO evidence with VRAM for the next increment plus300 MiB
and healthy host headroom. Evaluation/startup gaps cannot qualify. Capacity
testing can fall below4096 when apps prevent larger counts from fitting.
Target8192 remains; actual count follows measured capacity and guards.

57 focused counterfactual/provenance/API tests passed in
[`counterfactual_tests_final.log`](counterfactual_tests_final.log). They reject
circling, backward/sliding/stationary travel, heading drift and reset teleports,
exercise command timing/wrap/sign/stopped behavior and host-aware ramps, and
verify unchanged kernels, gate checks, geometry/hardware and pinned G1/H1 bodies.
The actual-observation/installed-PPO CPU regression still shows **zero** owned
snapshot corruption and pre-update KL; the archived R23 counterexample is
reproduced at KL10.0575.

The first new frozen startup, R26, passed physics but caught an early RSL-RL
checkpoint save's uninitialized logger writer. Its failure evidence is retained.
A CPU test reproduces that installed API error and verifies the exact actor
roundtrip after repair. R27's fresh physics, four-update PPO, phase/standing-
retention smoke and both checkpoint reloads completed with pre-update KL0.
The recorder verifies43 frozen hashes and one actual Isaac/CUDA worker.
These are execution checks, not full skill gates. Capacity and fresh nominal,
pushed and earlier-skill retention gates follow automatically.

The saved-artifact [heading smoke audit](heading_smoke_audit.json) confirms
exact equality of all nine actor/exploration tensors between candidate155 and
the pre-smoke baseline. Both 16-robot, 50 Hz reload traces have zero observed/
action command difference. The moving smoke supplies nonzero heading feedback
up to0.1048 rad/s, with a constant reset-heading target; every smoke PPO update
has zero pre-update KL and action/reward command difference.

At20:29 UTC, capacity testing has completed.4096 collected4839.6 raw and3208.5
effective steps/s over eight updates.6144 collected6050.0 raw and3380.9 effective
steps/s over six updates before host-memory recycling.8192 crossed the300 MiB
VRAM reserve and failed its guarded capacity test. The chain selected6144,
with target8192 and the host/GPU-aware180s upward ramp retained. Full standing
verification is running at128 robots using the exact preserved baseline.
An evaluation reading of free memory is not capacity evidence.

The original five-foot walker still serves as a behavioral comparison, not a
compatible actor: different22-action/80-input interface, simplified collisions
and controller. Its older6144 benchmark collected about55,518 steps/s; R25's
earlier8192 capacity test collected11,762 raw/9,060 worker-effective steps/s
(12.1%/7.8% above6144 in those bounded tests). Current app pressure makes those
historical numbers an unsuitable prediction of today's throughput. Read fresh
capacity and end-to-end evidence rather than multiplying environment counts.

The four-hour heartbeat remains ACTIVE with refreshed R27/direction context.
No final chain completion, shutdown, Git commit/push or new CAD export occurred.

Final verified handoff at **20:40 UTC / 16:40 EDT**:
R27 is running the full pushed standing gate010. The fresh nominal gate009
passed all checks across128 robots:100% survival,0.547-degree pelvis tilt,
0.249-degree torso tilt,0.0166 m final drift,0.00361 m/s foot slip and zero
support-outside/nonfoot/invalid-differential samples. Pushed qualification is
pending; no new walking/running stage is qualified. Main-stage PPO updates are
still0 because the preserved baseline is receiving mandatory fresh gates.
The chain will train at6144 after verification, targeting8192 when memory allows.
Supervisor36060 owns one actual worker36292;43 frozen hashes match. Host RAM
available7054 MiB and free VRAM4614 MiB are evaluation readings. See preserved
[final_handoff.json](final_handoff.json), then refresh the live recorder for
subsequent execution. The earlier actor and all failed/candidate evidence remain.
