# R25 walking evidence and remaining gait failures — October 3, 2026

This is the 15:19 UTC recurring check. **Preserve R25 and its optimizer.** It has
learned measurable low-speed walking: the 768- and 832-update checkpoints pass
every existing 0.15 m/s case in matching nominal and pushed tests. Standing is
qualified and retained. The full 0.35 m/s stage remains unqualified, with uneven
right-foot stepping and repeated left-foot landings. Running has not started.
New heading measurements also show why a gait-gate pass alone would not certify
straight or humanlike walking. No reward, controller, asset, frozen source or
gate was edited during this check.

Read `../../ACTIVE_RUN.json`, the referenced live state, and the CPU-refreshed
[`../walk_regression_20261003/live_handoff.json`](../walk_regression_20261003/live_handoff.json)
for execution after this dated report. Run: `locomotion_20261003_r25_owned_observations`;
reward: `cad_phase_feedback_retention_v4`; gate: `physical_landing_v4_horizontal_support`.

## Independent behavior, rather than reward totals

Each matching pair uses 128 robots for 20 seconds: 43 stopped, 43 commanded
0.15 m/s and 42 commanded 0.35 m/s. Nominal/pushed seeds are 901/1301. Pushes
add horizontal velocity at 5 and 12 seconds. Startup falls count; averages and
step counts exclude the first two seconds.

| Walking updates | 0.35 nominal forward speed | Left/right qualifying steps | Repeated same-foot fraction | Loaded foot-link motion |
|---:|---:|---:|---:|---:|
| 320 | 0.315 m/s | 1.79 / 2.43 | 39.0% | 0.107 m/s |
| 640 | 0.372 m/s | 21.83 / 20.50 | 1.01% | 0.0348 m/s |
| 704 | 0.364 m/s | 23.00 / 6.14 | 57.84% | 0.0442 m/s |
| 768 | 0.361 m/s | 23.00 / 20.48 | 5.81% | 0.0347 m/s |
| 832 | 0.335 m/s | 21.07 / 14.21 | 25.78% | 0.0291 m/s |

The 640-update nominal gate passed all existing cases. Its matching pushed
0.35 case failed alternation, so it did not qualify the stage. Its exact actor
and hash remain preserved in [`preserved_nominal_checkpoint.json`](preserved_nominal_checkpoint.json).
The 704 right-foot regression recovered substantially at 768, then worsened
again at 832. High-speed gait quality is fluctuating rather than steadily
improving; the consistent new low-speed skill is a reason to retain progress.

Latest completed pair: `089_walk_035_gate_014_nominal` /
`090_walk_035_gate_014_push`, at 832 walking updates. All commands survive 100%.
Stopped cases pass all retention checks. At 0.15 m/s, both cases pass all
existing checks: forward speeds 0.138/0.141 m/s, steps 23/22 and 22.84/21.72,
repeats 0/0.89%, accepted clearance about 27.8 mm. At 0.35 m/s, both fail only
step symmetry and alternation: pushed steps 21/14.98, repeats 21.91%, asymmetry
0.335 versus <0.18. Nominal asymmetry is 0.389. The repeated-foot limit remains
<2%; thresholds have not been relaxed. Other speed, cadence, clearance,
non-foot-contact, differential, crossing, tilt and walking-support checks pass.

This is walking evidence within the tested conditions, not a fully qualified
walker, running policy, hardware policy or Unitree-style motion demonstration.

## New contact, frame and heading diagnostics

Extended the **CPU-only** [`../../audit_r25_gait_progress.py`](../../audit_r25_gait_progress.py)
with selectable robot/evaluation frames, contiguous unload/recontact bouts,
global displacement, path distance and quaternion-derived heading. It replays
the frozen classifier against **512,000** saved landing/hop/repeat scalars from
eight completed gates at 640/704/768/832 updates. Every scalar matches.
[`progress_audit.json`](progress_audit.json) indexes trends and the eight
`*_replay.json` files retain individual candidates, bouts, joints and heading.

The earlier left-clearance bottleneck has mostly resolved. In the five sampled
768-update nominal 0.35 walkers, all 115 left and 102 right settled candidates
qualify. Right-foot interruptions occur before settled-candidate formation:
330 unloaded bouts average 94 ms, with overlapping short-air, preceding-load,
settling-load and clearance failures. At 832, 70/70 right settled candidates
qualify, but only 70 form across five robots. This is not a saved step-count
mismatch. Load flags mean vertical force above 5% of robot weight; these
50 Hz traces cannot measure the actual contact patch or prove sub-tick contact.

Inspected synchronized plots and successive side/front FK frames at 640, 704,
768 and 832. They show alternating leg flexion, low foot lift, crouched posture
and interrupted swings. Frozen-URDF foot FK agrees with measured positions to
about 0.017 mm in the newly visualized samples. The images are reconstructions
of measured joint/root states, **not Isaac screenshots or natural-gait certification**.
Sole-corner and foot-link motion include rolling/rotation; neither measures
contact-patch friction or pressure. Latest visual evidence:
[`0.15 frames`](089_walk_035_gate_014_nominal_robot_1_frame_sheet.png),
[`0.35 frames`](089_walk_035_gate_014_nominal_robot_8_frame_sheet.png),
[`synchronized plot`](089_walk_035_gate_014_nominal_robot_8_telemetry.png).

**Additional limitation: heading.** Evaluation commands have zero lateral and
angular components, verified in frozen reset/command code. Five sampled
nominal 0.35 walkers turn about 88 degrees after warmup at 768 updates and
43 degrees at 832; pushed 832 turns about 36 degrees. They still advance about
5.85 m net at 832, so this is not motion in place. Existing gates measure
forward speed in the current body frame and do not enforce heading tracking.
Do not describe a later existing-gate pass as straight walking without a
direction audit. If this persists as gait counts improve, counterfactually
test a physically appropriate direction guard and heading feedback in a new
frozen run, retaining the compatible actor rather than resetting learned legs.

No new observation-ownership, nonfinite or differential fault was reproduced.
Hip/knee axes and excursions remain plausible; sampled ankles approach their
pitch limits and show tracking lag. Stochastic training also clips many ankle
roll targets and intermittently exceeds yaw-joint continuous torque ratings.
These remain diagnostic priorities if progress stalls; instantaneous torque
exceedance is not a thermal model or proof of an actuator failure. Current
stopped gates pass, so the earlier transformed-sole support diagnosis does not
explain the present gait interruption. That preserved audit is still diagnostic.

## Research and decision

Rechecked Unitree's current primary
[G1 configuration](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py),
[H1 configuration](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py)
and [reward functions](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py).
Their gait feedback compares contact time with phase; it does not itself
require settled, forward, alternating landings. Their clearance kernels use
different link-height and velocity conventions. H1's clearance weight cannot
be transferred directly to this robot's sole-clearance kernel. These sources
support keeping separate behavioral verification, not treating phase reward
as gait success. They do not guarantee this robot's convergence.

No broad weight retune or policy restart is justified by the four-hour clock.
Preserve R25 while checking whether right swing/settling continuity and heading
improve over several checkpoints. If no new robust best emerges, the next
controlled repair should target those measured behaviors and test true swings,
brief toe recontacts, repeated-foot motion and turning counterexamples. Avoid
weakening the verifier or erasing the now-retained low-speed skill.

Running remains unstarted. Reviewed existing v4 full-cycle counterfactuals and
running parameters: positive forward-only flight feedback, faster alternating
phase with 0.43 duty, greater sole clearance and independent cadence/flight/
travel/retention requirements. Those saved formula tests passed; they are not
physical running qualification. Review actual running rollouts before accepting
promotion or claiming natural motion.

## Execution and throughput

15:43 UTC verified snapshot: 805 full walking updates checkpointed, 810 update
rows including the active worker; supervisor 7400, one actual worker 3940,
42/42 frozen hashes intact. Training uses **8192** environments. Free VRAM
1336 MiB and available host RAM 9524 MiB at that snapshot. The 300 MiB reserve,
fallback and 180-second healthy upward ramp remain. The later 832 gate pair
completed and the supervisor started block 14 at 8192.

[`training_health.json`](training_health.json) measures about 11,639 raw,
8,809 main-worker-effective and 6,003 end-to-end environment steps/s since
standing promotion, with scopes and timestamps separated. The saved capacity
comparison still favors 8192 over 6144 by 12.1% raw / 7.8% effective; that is a
bounded test, not a universal optimum. The original successful five-foot walker
sampled about 55,518 steps/s: current raw throughput is roughly 4.77 times
slower. Its 80-input/22-action actor, simpler drives/collisions and task differ
from this 400-input/23-action CAD-constrained actor, so loading its weights
directly or promising its training time would be invalid. See the preserved
[`old-walker comparison`](../old_walker_comparison_20261002/REPORT.md).

Across 810 full-training rows, pre-update KL is zero, including sampled reset
rows; latest post-update KL is 0.0139 and action clipping about 2.52%.
Host memory still grows, requiring 38 completed checkpoint recycles by that
snapshot. Actor and optimizer reloads continue successfully. Minimum recorded
main-training free VRAM is 1059 MiB. Native host-memory growth remains unresolved.
No second Isaac worker, cameras or new physical/PPO restart smoke was needed:
only the offline diagnostic changed and its actual replay completed.

The recurring four-hour heartbeat remains ACTIVE. Only standing is a promoted
stage; the final stand–walk–run chain is incomplete. Leave training running and
do not shut down. Journal and current handoff pointers were updated locally;
checkpoints/logs are preserved, with no CAD migration or Git commit/push.

Final monitored refresh: **16:06 UTC / 12:06 EDT**, 896 walking updates
checkpointed. Supervisor7400 and one actual worker11924 were verified,
42 hashes intact,20371MiB available RAM. Free6046MiB VRAM is an evaluation
reading, not8192 capacity. See [`final_status.json`](final_status.json).
Subsequent state read shows pushed gate096 active at128 evaluation robots;
next training8192. Nominal896 gate095 passes stopped and0.15 cases but fails
0.35 alternation only. Its matching pushed result is not available at this
handoff; the latest completed matching pair discussed above remains832.
