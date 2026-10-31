# Standing reward repair and actor-only warm start

The previous R22 training was running but never qualified standing. Its latest
750-update nominal evaluation survived, yet failed tilt, stationary speed,
anchor drift and bilateral support. The earlier 150-update actor had much less
tilt and motion. R22 was drained at a completed worker checkpoint and preserved
with 790 full updates; no previous checkpoint or diagnostic record was deleted.

The replacement is `runs/locomotion_20261002_r23_standing_curriculum`. Read
`ACTIVE_RUN.json`, the pointed run's fresh state and progress, its verified
process command lines and independent gates for actual status. Startup tests
are not locomotion qualification.

## What is retained

The actor comes from R22's 150-update standing checkpoint:

`runs/locomotion_20261002_r22_unitree_nominal/016_stand_train_001_part_09/candidate.pt`.

Its 400-input/23-action default-offset contract remains unchanged. The CAD
frames, mechanical differential, motor limits, explicit controller, target
scales, history observations, reset/randomization functions and collision asset
are unchanged. Actual source/artifact comparisons test that claim. The old
five-foot actor is not dimensionally compatible and is not loaded.

The new reward changes value targets. Actor tensors are loaded and checked for
exact equality before deliberately replacing exploration standard deviation
with 0.35. Critic parameters and Adam moments start fresh. Later ordinary
checkpoint resumes preserve actor, critic, Adam and adaptive learning rate.
No previous qualification is reused. The immutable nominal pose remains checked
at each PPO update.

## Reward changes

`stage_rewards.py` defines `cad_standing_g1_curriculum_v3`:

- Direct positive standing feedback: upright 3.5, height 2.5, torso upright
  0.75, bilateral loaded feet 1.5, stationary motion 1.0, anchor 1.0 and
  horizontal COM support 0.5. Stopped-only terms disappear under moving commands.
- Original pinned G1 penalty function bodies remain, with action-change weight
  reduced from -0.05 to -0.01, matching the successful old walker's coefficient.
  Positive orientation/height kernels replace their weak quadratic costs.
- The total regular penalty is smoothly bounded with a two-unit budget. A
  two-unit survival rate offsets it. There is **no aggregate reward floor**.
  Each penalty's contribution and the raw/bounded cost are recorded separately.
  This preserves posture feedback under action noise while removing the incentive
  to exit just to avoid negative regular future reward.
- Physical terminal ticks receive -200 rate, or -4 at the 20 ms control interval.
  Posture/survival rewards cannot offset that terminal event. Timeouts are
  unpenalized.
- Moving velocity tracking has weight 4.0 and is gated by actual uprightness and
  height. G1 phase/contact gait feedback retains weight 0.5 and its walking duty,
  with declared running cadence/duty adaptation. No moving IK reference is used.
- Swing-clearance feedback uses actual sole clearance and unloading. Loaded or
  stationary feet earn zero; in-place stepping receives no gait/clearance/flight
  activity credit. Running flight credit is running-only and requires actual
  movement. Independent speed, slip, alternating landing and retention gates
  remain mandatory: rewards alone do not certify a gait.

The initial exploration standard deviation is 0.35. Learned standard deviation
is bounded to 0.04–0.5 for standing and 0.04–0.7 for moving stages. The observation
normalizer remains unchanged for a compatible actor transfer; introducing a new
normalization contract is a separate experiment.

## Balance verifier repair

`support_geometry.py` compares the gravity projection of COM and feet in a
yaw-only frame. Full pelvis roll/pitch previously mixed height into the support
plane. The independent verifier now uses this helper and has version
`physical_landing_v4_horizontal_support`. Every `gate_case` threshold and
condition is identical to R22. Actual simulator COM is now recorded alongside
the bounded first-16-robot rollout traces, so future audits need not substitute
URDF-reconstructed COM. This support rectangle is conservative and does not
constitute a dynamic ZMP guarantee.

## Validation and resource policy

The 73 targeted CPU tests passed: posture/slide/anchor/support rankings, reward
feedback under noise, penalty monotonicity, discounted early-exit counterexamples,
terminal event units, full-cycle walking versus in-place steps/skating/hops,
running versus in-place hopping, root tilt/yaw/translation support cases, unchanged
physical/action/observation sources and independent gate logic, and actor
provenance rejection. All 5,593 pinned G1/H1 arithmetic/source checks also passed
into a new diagnostic directory; original test evidence was preserved.

Tests are telemetry fixtures, not simulated or learned gait success. The new
supervisor runs a 32-environment physical preflight, four real PPO updates at 64
environments, a fresh four-second checkpoint-load smoke, then actual 4096/6144/
8192 capacity tests. Smoke evaluations cannot promote any stage. Nominal/pushed
promotion and retention evaluations still run 128 robots for 20 seconds.

Capacity benchmarks record both update throughput and effective throughput
including process startup/exit, plus available host RAM and checkpoint recycling.
4096 receives eight benchmark updates; 6144 and 8192 receive up to 24, allowing
real host-RAM guard behavior to be observed. These are bounded measurements,
not a guarantee of long-run memory stability. Longer full-training behavior
must also be checked.

The chain targets 8192, preserves the 300 MiB VRAM reserve, falls back on resource
failure and ramps one tier after 180 continuous healthy training seconds at a
checkpoint boundary. Host-RAM recycling is retained; the underlying allocation
growth is not declared fixed. Stage blocks are 64 updates for more frequent
behavioral feedback. Stand → walk 0.35/0.65/1.0 → run 1.5/2.0/2.5 remains gated
independently, including previous commands. No stereo/render camera is created
in training, and only one Isaac/CUDA worker runs at a time.

## Evidence and operation

- `diagnostics/stage_rewards_20261002/tests.json`
- `diagnostics/stage_rewards_20261002/unitree_core/counterfactual_rewards.json`
- Old R22 `stage_reward_repair_drain.json`
- New run `chain_config.json`, frozen `source`, per-worker stdout/stderr,
  `warm_start.json`, `result.json`, `progress.jsonl`, benchmarks and evaluations
- Prior diagnosis: `diagnostics/old_walker_comparison_20261002/REPORT.md`

The replacement was launched by `launch_stage_rewards.ps1` after source-hash,
checkpoint-hash and stopped-process checks. The deleted four-hour automation
remains deleted. Changes are local; no Git commit/push or shutdown.

## Fresh live validation and capacity results

The replacement supervisor PID 20208 launched at 22:55:36 UTC. The physical
preflight passed with no zero-action falls or invalid differential samples and
337.61 N mean support against 338.09 N expected weight. Combined ankle/waist
pitch-roll closure was tested. Four real PPO smoke updates completed, and the
new checkpoint reloaded for a four-second rollout with 100% survival. These
checks establish execution and compatibility, not learned standing or walking.

The actual simulator COM saved in the smoke trace independently reproduced all
1,600 support flags for its first 16 robots, with zero mismatches. All 41 frames
of the CPU-reconstructed smoke were reviewed. They show upright balancing and
small corrections without a fall; all commands are zero. Reconstruction agrees
with the simulated feet within 0.000017 m. These are reconstructed frames, not
Isaac screenshots or proof of a learned walking gait.

| Environments | PPO samples/s after first update | Samples/s including startup/exit | Updates | Minimum free VRAM | Minimum available system RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 8,473 | 5,657 | 8 | 3,827 MiB | 14,804 MiB |
| 6144 | 10,427 | 8,344 | 24 | 2,343 MiB | 4,318 MiB |
| 8192 | 11,747 | 9,133 | 20 | 1,217 MiB | 1,918 MiB |

8192 hit the existing 2,048 MiB system-RAM guard and saved its recovery checkpoint;
6144 completed its 24-update benchmark without recycling. Even including process
startup and exit, 8192 delivered 9.46% more samples/s than 6144 in these bounded
tests (12.66% higher during updates). Reducing to 6144 would deliver 8.65% fewer
samples/s in this comparison. The 4096 test has fewer updates, so its startup
overhead is less amortized. This is not a matched long-run benchmark or proof
that 8192 is optimal over an entire learning run. System-RAM allocation growth
remains unresolved and checkpoint recovery is still required.

6144 contains 147,456 samples per PPO update; 8192 contains 196,608. During
updates, the measured rates correspond to roughly 4.24 vs 3.58 PPO updates/minute.
More updates/minute therefore does not mean more samples/minute. Learning
efficiency must be judged by fresh independent behavioral gates at comparable
elapsed time or sample budgets; these capacity tests do not establish it.

The target remains 8192 with the existing 300 MiB VRAM reserve, fallback and
three-minute healthy-memory ramp. Fresh 128-robot, 20-second standing qualification
began after benchmarks. At the 23:16:21 UTC snapshot it was still evaluating;
no qualification or learning convergence was claimed. All 41 frozen-source
fingerprints matched the recorded manifest. Latest live evidence is saved in
`diagnostics/stage_rewards_20261002/live_handoff.json`.

At 23:20:40 UTC the first **full nominal standing gate passed**, with 128
robots for 20 simulated seconds: survival 100%, mean pelvis tilt 3.125 degrees,
torso tilt 4.852 degrees, planar speed 0.0643 m/s, final drift 0.1674 m,
bilateral loaded contact 97.85%, support-outside 2.15%, zero invalid differential
samples and zero nonfoot contacts. The separate pushed batch follows. This
nominal result alone does not qualify standing or walking.

The full pushed standing batch also had 100% survival, 0.1790 m final drift,
0.0754 m/s planar speed, 3.219-degree pelvis tilt and zero invalid samples or
nonfoot contacts. It failed only `com_over_support`: 3.0321% versus the unchanged
strict <3.0000% limit. Bilateral loaded contact was 96.9731%. No threshold was
relaxed and no promotion occurred. The chain automatically started
`009_stand_train_001` at 8192 environments with 64 planned PPO updates.

At the 23:28:01 UTC handoff, the full training block had completed three fresh
updates (zero-based log iteration 2). Actual worker PID 32880 belongs to wrapper
25768 and supervisor 20208; only one actual simulation worker was verified.
Free VRAM was 1,713 MiB and available system RAM 12,837 MiB. The latest bounded
training audit had zero falls, invalid-differential terminations and nonfinite
terminations, with immutable nominal-pose drift exactly zero. These exploratory
samples do not substitute for a deterministic gate. All 41 frozen-source hashes
matched. Training is left active; nominal standing passed, pushed standing
failed narrowly, and walking/running have not started or qualified in R23.
