# Physical-v29 training — R29 paused for performance review

Start with [the consolidated iteration history](ITERATION_HISTORY.md) for all
R1–R29 changes, failed starts, earlier five-foot policies and CAD revisions.
[History conventions and evidence inventory](history/README.md) define the
record required for every subsequent meaningful iteration; append the outcome
and refresh the CPU-only inventory before handoff. Frozen runs stay untouched.

## October 4 user-requested pause

The four-hour heartbeat is PAUSED. R29 is `paused_for_performance_review` after
its current worker completed normally:256 walking updates, full actor/critic/
Adam checkpoint014 preserved,43 frozen hashes unchanged, no active training
worker. [Current assessment](diagnostics/r29_performance_20261004/REPORT.md)
and its evidence.json preserve the checkpoint/hash, gate history and bounded
4096-environment performance profile. Refresh the CPU-only live recorder; it
now handles this explicit paused state. Do not restart from an old heartbeat.

The current0.35 stage never qualified as a whole. Its baseline0.35 nominal case
passed, but pushed heading and retained0.15 direction failed. Latest independent
192-update actor survives and alternates steps, with improved0.15 heading but
overspeed and regressed0.35 heading.256 actor remains unevaluated. No current
promotion to1.0 or running. EarlierR25's1.0 old-gate pass had a direction loophole.

The asset already contains24 box visuals/colliders and23 joints. Differential
state/Jacobian work runs18 times per control tick and costs55.6% of the sampled
548ms tick; physics stepping16.2%, actor inference1.24ms. Four6144 blocks average
9368 active PPO samples/s,94.1% collection time; memory is stable. Optimize exact
transmission math and launch/data-access overhead before simplifying physical
CAD or weakening collision/actuator modeling. No production optimization or
reward change was applied in this assessment; preserve the learned checkpoint
for tested continuation. Both training and recurring checks remain paused.

## Historical October 3, 23:19 UTC check; October 4 restart

Current frozen run is `runs/locomotion_20261004_r29_host_capacity_checked`;
read `ACTIVE_RUN.json` and CPU-refreshed live handoff for current state.
[Latest report](diagnostics/r27_review_20261003_2319/REPORT.md) documents R27's
independent direction regression, controlled physical yaw-command response and
the reproduced differential cache leak. R27 remains preserved at 192 updates.

Transient tensors are no longer cached as dictionary keys. 6,144 CPU pose and
limit outputs and a 72-robot, 350-tick physical replay remain bitwise identical;
the repaired replay stops accumulating CUDA allocations and Python views.
Reward v6 adds a positive bounded fixed-heading rate, weight 1/width 0.35 rad;
all existing gait kernels/weights, physical limits and hard gates are retained.
R28 began with the better-direction R25 candidate155 actor/exploration preserved
exactly, fresh critic/Adam and fresh qualification. 50 unit and 44 phase/retention checks
pass, as do real physics/PPO/checkpoint-load smokes. Read later results for
capacity and independent gait qualification. Target 8192/reserve 300 MiB,
180-second healthy host/GPU ramp, one actual worker and no stereo remain.

R28 subsequently passed full nominal/pushed standing. R29 preserves its full
actor/critic/Adam checkpoint and standing teacher under41 byte-identical authored
runtime files. The supervisor now requires complete healthy-host capacity tests
and lowers a count tier after saved host-RAM recovery, including the last update
of a block.24 CPU scheduling/memory/provenance checks and fresh full-load
PPO/checkpoint-reload smokes pass. The stable6144 benchmark delivered9584 raw/
7599 worker-effective steps/s;8192 recycled after5 updates with2010 MiB available
host RAM and delivered5001 effective. R29 selects6144 and checks independent
walking gates. Walking/running remain unqualified; no success or shutdown from
these smokes. `diagnostics/r27_review_20261003_2319/record_r29_maintenance_handoff.py`
refreshes exact current continuation evidence without a simulator.

## Historical October 3, 19:19 UTC four-hour check

Current pointer is `ACTIVE_RUN.json`; the replacement is
`runs/locomotion_20261003_r27_heading_commands_checked`. Read its fresh state
and the CPU-only `diagnostics/walk_regression_20261003/live_handoff.json` before
quoting the current worker, environment count or qualification.

R25 learned alternating gait and passed its old nominal/pushed standing and
walking gates through 1.0 m/s. Running at 1.5 m/s did not qualify. An audit of all
128 robots exposed turning hidden by body-relative speed: the old 0.65 m/s pass
averaged 122 degrees final heading, only 0.254 m/s reset-frame forward travel
and 0.463 m/s lateral travel. Preserve that learned actor rather than restart
from random weights. R25 is drained at 1471 main PPO updates; all logs,
checkpoints and frozen source remain. Candidate155 seeds R27 with exactly
preserved actor/exploration, fresh critic/Adam and no inherited qualification.

R27 supplies reset-heading error through the existing desired-yaw-rate command,
holds the observed command through its action/reward, and adds independent
reset-frame direction/net-travel checks. All old gate thresholds and gait reward
kernels remain unchanged. This is a declared local Isaac Lab command adaptation;
published Unitree G1 disables heading mode. Physical CAD/assets/actuators remain
v29, with 400/415 observation inputs and 23 actions. R25 observation ownership
and zero pre-update KL are retained. Host-RAM failures no longer inflate VRAM
cost estimates; upward ramps also require healthy host headroom for 180 seconds.

[Repair and evidence report](diagnostics/r25_run_review_20261003_1919/REPORT.md)
contains full-population direction, 768,000 exact landing-event replays,
synchronized plots/FK frames and 57 passing counterfactual/provenance/API tests.
R27 passed physical, PPO, phase-retention and checkpoint-load smoke execution;
capacity measurements and fresh full gates follow. R26's early logger failure
is preserved. Fusion was left open because its unrelated Strap cam document
has unsaved work; current app pressure may require lower counts. Target8192,
300 MiB reserve, automatic fallback and healthy ramp remain. Four-hour research
and repair checks are ACTIVE with updated R27 context. Chain incomplete;
no shutdown, CAD migration, Git commit or push occurred.

## Historical R25 status

## October 3, 15:19 UTC four-hour check

Preserve R25. The 768- and 832-update checkpoints pass every existing 0.15 m/s
case in matching nominal and pushed tests, retaining qualified standing. The
0.35 m/s stage is still unqualified: the latest pair089/090 survives100% but
fails step symmetry and alternation. Its nominal left/right steps are21.07/14.21,
with25.78% repeated-foot landings. The 640 checkpoint passed the full nominal
gate but failed matching pushed alternation; its actor/hash remain preserved.
High-command gait quality fluctuates; low-speed progress is now consistent.

Read [`diagnostics/r25_progress_20261003_1519/REPORT.md`](diagnostics/r25_progress_20261003_1519/REPORT.md)
and CPU-refreshed `diagnostics/walk_regression_20261003/live_handoff.json`.
`audit_r25_gait_progress.py` exactly replays512,000 saved event scalars from
eight gates, decomposes unload/recontact interruptions, and saves joint/contact
plots and inspected FK frames. These are measured-state reconstructions, not
Isaac screenshots. Right-step scarcity occurs before settled classification;
the earlier left-clearance bottleneck has mostly resolved.

New world-travel/heading audit: sampled0.35 nominal walkers turn about88 degrees
at768 and43 degrees at832 despite zero-turn commands. Existing gates do not
enforce heading. A later gait pass alone cannot certify straight or humanlike
motion; follow the report's direction/continuity diagnosis before claiming it.
Running remains unstarted/unqualified; retain all independent/retention gates.

The supervisor started block14 at8192 after the832 gates. One actual CUDA worker,
42 frozen hashes and zero sampled pre-update KL verified;300MiB reserve/180s
healthy ramp remain. Raw/main-worker-effective/end-to-end throughput is about
11639/8809/6003 steps/s at the15:43 snapshot. Host-memory growth still requires
successful checkpoint recycling. Frozen training/reward/asset/gate bytes are
preserved; only offline diagnostics changed. Heartbeat ACTIVE, chain incomplete.
The earlier11:18 check is archived in `diagnostics/r25_progress_20261003`.

<!-- owned-observation-repair-20261003 -->

October 3, 2026 07:57 UTC: active run is
`runs/locomotion_20261003_r25_owned_observations`. R23 qualified standing at512
updates but then failed448 walking updates with zero qualifying steps and
eventually lost standing. Its qualified actor and all960 full updates/gates
are preserved. R24 is preserved intermediate validation, not a trained stage.

R25 fixes reproduced PPO observation aliasing across resets (policy/critic
snapshot corruption1.0 and pre-update reset-row KL10.0575 become0), restores
phase/lift feedback before translation, and selects latest history-frame
stopped commands for training-only standing retention. Observation values,
physical assets, controller and gate thresholds are unchanged.44 targeted,
9 retention and5593 pinned core checks passed. Real physics, both PPO smokes
and checkpoint reloads completed; CUDA pre-update KL is0. No new qualification
is reused. Phase reload correctly fails walking while surviving/standing.

Read `ACTIVE_RUN.json`, fresh state and
[`diagnostics/walk_regression_20261003/REPORT.md`](diagnostics/walk_regression_20261003/REPORT.md).
CPU-only `record_owned_observation_handoff.py` refreshes live provenance,
42 frozen hashes, one actual CUDA worker, PPO audits, gates and throughput.
Target8192/reserve300MiB/fallback/180s healthy ramp remain. Initial optimizer
overshoot and recurring host-RAM growth remain monitored; do not infer gait
from reward or elapsed time. Four-hour research/repair checks remain ACTIVE.
No CAD migration, commit/push or shutdown occurred. Historical entries follow.

08:11UTC handoff: capacity completed and8192 selected for training. Raw/effective
throughput11762/9060 steps/s is12.11%/7.80% above6144 in bounded tests. RAM
recovery saved the checkpoint after19 updates; minimum free VRAM1105MiB.
The chain is running full nominal standing gate009 at128 evaluation robots;
no fresh full qualification yet. See the report/handoff for subsequent progress.

## Historical R23 standing curriculum

<!-- stage-reward-repair-20261002 -->

Latest 2026-10-03 03:24UTC: R23 qualified standing at512 PPO updates, passing
both128-robot20-second nominal/pushed gates. Chain advanced to `walk_035`,
currently evaluating; walking unqualified. Four-hour recurring research/repair
checks are ACTIVE in this thread, whose model is verified GPT-6.1 Sol/max.
See `diagnostics/support_geometry_audit_20261003/REPORT.md` for live evidence,
new actual-sole/COM diagnostic, automation identity and scope. Recheck live state.

Current run: **r23_standing_curriculum**, reward
`cad_standing_g1_curriculum_v3`. Inspect `ACTIVE_RUN.json`, fresh state/progress,
verified owned processes and independent gates. R22 was drained and preserved
at 790 full updates; it never qualified standing. The original five-foot walker
did walk in saved physical telemetry; see the comparison report below.

R23 retains the matching R22 150-update standing actor, resets exploration to
0.35 and starts a fresh critic/Adam. The 400-input/23-action contract and all
physical CAD, controller, observation, reset and motor assets are unchanged.
Strong positive standing feedback replaces the almost entirely floored reward.
Pinned G1 cost/phase bodies remain with declared local shaping/scales. Costs
are smoothly bounded with a survival offset; no aggregate reward floor.
Physical terminal events still cost -4 at 20 ms. Stopped/loaded feet get no
swing-clearance credit; moving rewards require healthy actual movement.

Support verification now uses horizontal yaw-only COM projection and saves real
simulator COM. Every existing gate threshold/condition is unchanged, under
`physical_landing_v4_horizontal_support`. 73 targeted CPU checks and 5,593
pinned reward checks passed. Real 32-environment preflight, four-update PPO
and a four-second checkpoint-load smoke completed; smoke is not qualification.

The supervisor measures 4096/6144/8192 update and effective wall-clock throughput,
including host-RAM recycling. Target8192, reserve300MiB, fallback and180-second
healthy ramp remain. Stage blocks are64 updates; promotion still requires
independent128-robot20-second nominal/pushed/current/prior gates. No cameras or
parallel Isaac workers. The four-hour automation was restored October3UTC;
the old one-time/deleted automation remains historical.

Repair: `STAGE_REWARD_REPAIR_20261002.md`.
Tests: `diagnostics/stage_rewards_20261002/tests.json`.
Original/current comparison: `diagnostics/old_walker_comparison_20261002/REPORT.md`.
Inspect live evidence before claiming a learned stand/walk/run. Local changes
only; no Git commit/push.

23:28 UTC handoff: full nominal standing passed; pushed standing failed only
support-outside (3.032% versus <3%). The chain is continuing standing at8192,
with three fresh updates in its first64-update block. Free VRAM1713MiB and
available RAM12837MiB. Measured effective capacity throughput:6144=8344 and
8192=9133 samples/s including startup/exit; bounded tests, not a long-run optimum.
Fresh evidence: `diagnostics/stage_rewards_20261002/live_handoff.json`.

## Historical R22 corrected Unitree adaptation

<!-- unitree-nominal-20261002 -->

The current replacement is **r22_unitree_nominal**, reward
`unitree_g1_lifetime_v2`. Read `E:/Aditya/robot_sim/physical_v29/training/ACTIVE_RUN.json`,
its live state/progress, verified worker/supervisor processes and independent
gates for current status. R20 was drained and preserved after 68 full updates;
R21 was preserved after validation, before full standing training. Neither is
active or qualified. R20's replay first breached the waist differential
motor-angle limit at 0.08 seconds, then fell backward without alternating steps.
Its original aggregate reward rates were all negative, allowing early episodes
to avoid future costs.

R22 retains the pinned Unitree RL Lab G1 core bodies and weights, adding the
declared RL Gym nonnegative regular aggregate followed by MJLab H2's -200
physical-termination weight (-4 per 20 ms event). Timeouts are unpenalized.
Per-cause termination telemetry separates motor faults from ordinary falls.
This is an adapted public Unitree design, not unmodified complete G1/H2 or a
commercial pretrained policy. 5,593 core and 149 lifetime CPU checks passed.
Actual new physics/PPO/load/capacity checks are retained in R22's run directory.
Smoke, reward rankings and screenshots do not qualify learned walking.

The original adaptation's expanded reference aliased the nominal pose. A
randomized reset could change that shared origin and every other robot's
reference. The actual-source CPU regression reproduced 0.63665 rad drift after
512 resets; corrected private storage gives zero drift and leaves unreset rows
unchanged. This is a reproduction, not a measurement of old GPU drift. Every
real PPO update now logs/asserts zero nominal drift. Contract is
`physical_v29_history5_unitree_defaultoffset_v2`, still 400/415 actor/critic
dimensions. A fresh actor is used; no moving IK target or standing teacher.
Physical-v29 mass 34.46334 kg, CAD joint frames, motor limits, differential
geometry and collisions are unchanged. CAD v32 stays separate. Do not load
old history4/IK or another robot's policy tensors into this fresh actor.

The chain targets 8192, reserve 300 MiB, fallback, and a 180-second continuous
healthy-live-PPO ramp at checkpoint boundaries. Host RAM recycling preserves
actor/critic/Adam/LR; the allocation leak itself remains unresolved. Keep one
Isaac/CUDA worker only. No training stereo/render camera is instantiated.
Stages: stand; walk 0.35/0.65/1.0; run 1.5/2.0/2.5, with 0.15 retention. Promotion
requires independent current/prior nominal and pushed 20-second gates. The
running cadence/flight extension is declared, not commercial Unitree weights.
Every later learned candidate needs visual review before a human-gait claim.

Report: `E:/Aditya/robot_sim/physical_v29/training/UNITREE_NOMINAL_REPAIR_20261002.md`.
Source/weight comparison: `training/UNITREE_REWARD_PORT_20261002.md`.
Frame review: `training/diagnostics/unitree_port_20261002/frame-review.html`.
The review contains 280 scripted screenshots, 80 learned-r20 replay screenshots,
and 207 explicitly labeled CPU smoke reconstructions (567 frames across 15 cases), with synchronized angles,
contacts, failure causes and original H1/G1 rates; 47 official GIF frames were
inspected too. None of the failed candidates matches the reference gait.
Local-only documentation/source changes; no Git commit or push.

Dated handoff 2026-10-02T17:50:31.581981+00:00: verified live full
stand PPO at 8192 environments, with 17
saved full updates and current worker iteration 12. Free VRAM
1269 MiB; available host RAM 4419 MiB.
Frozen worker/environment/reward files match the tested workspace. This is
runtime evidence, not a behavioral qualification. Inspect fresh active status.
Current nominal-pose drift: 0.0 rad.
The first host-RAM recycle saved 17 full updates and resumed at 8192. Actual
continuation Adam counters advanced420 to440 after the next PPO update;
actor/contract/reward/asset provenance is retained in
`training/diagnostics/unitree_termination_20261002/optimizer_resume_r22.json`.

[Repair report](UNITREE_NOMINAL_REPAIR_20261002.md).
Use `monitor_training.ps1 -Watch` without starting a second simulator.


## Historical record before the nominal-reference repair

# Physical-v29 Unitree reward training

<!-- unitree-port-20261002 -->

The current replacement is **r20_unitree_g1**. Read
`E:/Aditya/robot_sim/physical_v29/training/ACTIVE_RUN.json`, its live `state.json`,
verified worker/supervisor command lines, fresh progress and independent gates.
R19 was drained and preserved without qualifying walking. R20 starts a fresh
five-frame, nominal-position-offset policy using pinned Unitree RL Lab G1
reward bodies/weights and PPO settings. Old CAD-IK moving action references,
discovery bonuses and the standing teacher are absent. H1 RL Lab, older RL Gym
H1/G1 and public MJLab H2 tasks were compared separately, not blended.

The reward version is `unitree_g1_port_v1`; action/observation contract is
`physical_v29_history5_unitree_defaultoffset_v1`. Actor/critic dimensions are
400/415. Original G1 walking phase parameters remain; requested running uses
an explicitly documented cadence/flight extension. Robot dimensions and
entity mappings are adapted; saved physical v29 mass, axes, actuator limits,
collisions and CAD are unchanged. CAD v32 remains a separate release.

5,589 CPU assertions, actual physics/PPO/load checks and fresh capacity
benchmarks passed. At 8192 the short benchmark measured 11,674 env steps/s and
1533 MiB minimum sampled free VRAM. 280 real scripted probe screenshots, all
47 official reference GIF frames, and 50 learned-smoke CPU reconstruction
frames were reviewed. The four-update candidate failed standing/walking;
scripted motions are not a learned gait. No walking/style success is claimed.

Target 8192, reserve 300 MiB, automatic fallback and 180-second healthy-live-PPO
ramp remain. RAM recycling preserves checkpoints/Adam/LR, but the underlying
allocation leak is unresolved. Stages: stand, walk 0.35/0.65/1.0, run
1.5/2.0/2.5 m/s; 0.15 remains a retention command. Promotion requires current
and prior nominal/pushed physical gates, never reward or smoke alone. No camera
is instantiated in training and no second Isaac/CUDA worker may overlap it.
Do not load old history-4/IK or another robot's pretrained tensors into R20.

Details, original source pins, reward comparisons, tests and scope:
[Unitree port report](UNITREE_REWARD_PORT_20261002.md).
[Frame/reward review](diagnostics/unitree_port_20261002/frame-review.html).
Use `monitor_training.ps1 -Watch` to inspect the running chain. The older
launch/play commands below belong to historical contracts; do not use them
for this active five-frame policy.


## Historical record before the Unitree port

# Physical-v29 locomotion training

Read [ACTIVE_RUN.json](ACTIVE_RUN.json), then the indicated live `state.json`,
process command lines and latest independent gate before reporting status.
The 08:45 EDT follow-up repair is **r19_standing_retention**, preserving the
latest learned actor/critic/Adam. R15's latest gate survived but showed rapid asymmetric
landings, hopping and failed standing retention. Discovery credit renewed
after each accepted step; v6 caps it per physical episode. The v3 independent
counter fixes overlapping hop counts and adds landing-order/walking-support
checks while preserving all old thresholds. Walking has not passed. Read the
[latest audit](SECOND_FOUR_HOUR_AUDIT_20261002.md) and its fresh evidence snapshot.
R17 removed the loophole but its first post-training gate lost standing and
remained stationary under a walking command. R19 adds a training-only MSE
against the verified standing actor for stopped commands. Inference uses no
teacher or action blend; all qualifications are freshly checked. 128 CPU tests
and a real four-update smoke passed. Fresh 4096/6144/8192 benchmarks and both
standing-prior gates passed; full walking PPO was verified active at 8192
through nine updates at 10:29 EDT. The retained candidate's baseline failed
both standing and walking, so improved gait is not yet established.
R16/R18 preserve preflight evidence;
R15/R17 and the [04:45 audit](
FOUR_HOUR_AUDIT_20261002.md) remain history.
The supervisor continues after this chat ends.

## Current chain

The frozen 23-axis asset uses a free base, gravity, floor contacts, self-collision
and the saved CAD joint frames. Wrists/fingers are rigid; no cameras are
instantiated. CAD v32 is a separate native release. Its new brackets/covers and
mass/contact changes are not silently inserted into this running policy.

Reference-guided residual PPO uses four-frame proprioceptive history, an
asymmetric critic and CAD IK gait guidance. This is not AMP or a pretrained
Unitree policy. [Research](RESEARCH.md) indexes primary code/papers and the
review of older 22-axis policies and videos.

Stages: stand → walk at 0.15/0.35/0.65/1.0 m/s → run at 1.5/2.0/2.5 m/s.
Every promotion requires separate nominal and pushed 20-second gates for the
new command and all earlier commands. Failure keeps training; loss of an
earlier skill blocks promotion while the candidate continues learning. The
trusted checkpoint remains preserved. Rewards cannot promote.
Gates check actual speed/posture/support, bilateral steps, cadence, clearance,
slip, running flight, invalid transmissions, crossing and non-foot contact.
Numerical passes still need visual review for natural movement.

## October 2, 08:45 gait integrity repair

Current reward is `gait_integrity_v6`; gate version is `physical_landing_v3`.
First-step discovery cannot renew after accepted landings. Reward step credit
rejects paired and excessively rapid impacts and freezes travel at touchdown.
Walking flight costs reward and cannot receive positive travel/swing credit;
running retains legitimate alternating flight. Standing costs now address
measured COM/support and stopped drift, anchored where the stop begins.
Independent gates count raw physical steps, including same-foot repeats,
without borrowing reward cadence filters. All old thresholds are preserved;
new checks enforce order and walking support. Every evaluation saves bounded
50 Hz action/joint/contact diagnostics for 16 robots. The later R19 learner
adds stopped-command standing-prior regularization in PPO only; v6 rewards,
hardware, observation/action mapping and v3 gates are unchanged. 128 CPU tests passed.
Fresh Isaac smoke, reload, capacity and behavioral results must be read from
the new run. No configuration or test alone qualifies walking.

## Historical October 2, 04:45 balance, reference and running repair

The four-hour audit identified profitable stationary moving posture after the
finite swing window expires. R15's `motion_deficit_discovery_v5` retained
balance gradients, penalizes missing valid net progress and gives bounded
incremental credit for actual unloading/lift/placement. Repeated kicks/rocking
and command resampling could not replenish that credit, but accepted steps
did renew it; the later audit above repairs that loophole. V5's gates were
unchanged. Its reward migration kept the 612-update learned actor, reset
critic/Adam for the changed reward, and independently rechecks prior skills.
102 CPU tests passed. Actual GPU smoke, reload, capacity and gate reports are
saved in r15; check them before claiming learning or qualification.
Its trusted standing requalification passed both nominal and pushed checks.
The 612-update candidate baseline still failed bilateral steps/cadence and
standing COM retention (about 0.10 m/s, zero left and two right steps). Those
baseline gates precede v5 learning. The repaired worker started at 8192 and
completed multiple real updates with reserve headroom; walking is unqualified.

R11's last completed nominal gate after 128 updates had zero 20-second survival
and zero steps. Its walking forward mean was 0.145 m/s, but lateral speed was
0.726 m/s: forward speed alone described a fall, not a walking policy. A third
64-update block was drained and preserved before the repair. R12 was setup
validation only. R13 requalified standing (100% nominal survival, 92.1875% pushed)
and completed 64 walking updates. Its next gates still had zero 20-second
survival and zero steps. At 0.15 m/s, forward mean was 0.139 m/s while lateral
speed was 0.383 m/s. Correct forward speed alone is still not walking.
The saved candidate and trusted prior are both preserved.

R13 also exposed allocation growth: process commit reached 67 GiB on a 32 GiB
host, with paging and slowing updates. R14 adds periodic garbage collection
and checkpoints after complete PPO updates when available RAM falls below
2 GiB or worker commit exceeds 1.15 times physical RAM. The owned worker then
restarts at the same environment count from its actor, critic and Adam state,
continuing the remaining updates before gate evaluation. Startup memory pressure
can still reduce the environment count. This contains exhaustion; the underlying
allocation source is not yet proven. Adaptive PPO learning rate now survives
worker restart rather than resetting to its configured ceiling.

Resource migration checks identical physics, reference, reward, verifier and
PPO configuration. It continues R13's 64-update candidate rather than restarting
from standing. Fresh smoke/reload and capacity checks precede independent
current-command and retention evaluations. The [updated measured joint audit](
checks/balance_audit_r13_20261002/measured_joint_ranges.csv) compares actual
pre-fall angles/speeds with native hard limits and the controller target band.
No sampled pre-fall hard-limit (0.005 rad tolerance) or motor-speed violation was
found. This does not establish sufficient balance authority or realistic contacts.

The [isolated allocation probe](checks/memory_probe_20261002/result.json) ran 32
synthetic CPU PPO updates under the same network/configuration, with process
commit changing by only 0.11 MiB after warmup. It does not include Isaac or GPU
physics and cannot rule out GPU/framework or driver allocation problems.
The smoke [optimizer continuity audit](checks/memory_probe_20261002/checkpoint_continuation.json)
confirmed Adam steps advanced from 2048 to 2176 while retaining the saved 1e-5
rate; optimizer state was not reset. R14's fresh 8192 benchmark measured 19095
environment steps/s and 2492 MiB minimum free VRAM. These four-update numbers
do not describe long-block effective throughput including recycling.

The first real R14 worker saved `resource_recovery.pt` after 20 complete updates
when available RAM reached 1831 MiB (worker commit 30706 MiB; free VRAM 1828 MiB).
It completed successfully with `recycled=true`, and the supervisor launched
`009_walk_015_train_001_part_02` at the same 8192 environments for the remaining
44 updates. No qualification occurred between these partial blocks. Periodic
garbage collection did not stop the allocation slope; containment through saved
worker recycling is verified, but the underlying growth remains unresolved.

`balance_events_running_v4` fixes a fall-event unit error: the old -30 rate
integrated to only -0.6 per fall at 50 Hz. Each fall now costs eight independently
of control dt. Full posture incentives remain active during locomotion; finite
load-transfer guidance helps unload a foot without reopening indefinite rocking
credit. Positive travel still requires fresh territory and foot activity.

Running commands allow phase-appropriate flight and a running slip scale, but
simultaneous or near-simultaneous hops cannot pay two alternating steps. The
independent `physical_landing_v2` verifier measures air/load dwell, sole clearance,
foot/body travel and paired impact timing directly from physics. All earlier
nominal/pushed skill gates still apply. Corrections in the running branch become
active only at running commands; the chain cannot skip the walking gates.

The CPU [joint audit](checks/balance_audit_20261002/diagnosis.json) and
[axis/limit table](checks/balance_audit_20261002/joint_axes_and_limits.csv) compared
all 23 saved native joint origins, axes and limits with the actual training URDF.
No mismatches were found; each leg's six-dimensional spatial Jacobian has rank
six at the standing pose. Hip pitch axes are tilted 25 degrees, so canonical
pitch/roll assumptions would be wrong. Original references exceeded ankle
controller soft limits in 16% of 0.35 m/s phases and 69% of 2.5 m/s phases;
the highest speeds also exceeded hip target limits. Joint and FK foot guidance
now use the same bounded controller targets. The original archive and hardware
limits are preserved. This projection is a feasibility guard, not proof of a
dynamically balanced or human-like gait. The reference still lacks an explicit
lateral COM plan, and mass/contact approximations remain provisional.

85 CPU reward, landing, reference, gate, continuation, host-memory, capacity and
status tests passed. R14's fresh real four-update PPO smoke and checkpoint reload
have separate saved results. Inspect actual behavior results rather than treating
these tests as convergence evidence.

`progress.json` now includes named termination rates, balance eligibility,
load transfer and flight, plus all 23 measured/target joint means, target errors,
clipping, slew limits and torque saturation. It also samples normalized input
outliers, exploration std, learning rate, system RAM and CUDA allocated/reserved
memory. Every evaluation adds `startup_diagnostics.npz` for the first two seconds
of 16 robots, including actions, targets, root states, loads and failure causes.
These arrays are bounded; cameras are not required. On a done tick the pose can
already be reset, while failure flags describe the pre-reset state.

```powershell
& 'E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe' `
  'E:\Aditya\robot_sim\physical_v29\training\diagnose_locomotion.py' `
  --run 'E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261001_r11_net_travel' `
  --output 'E:\Aditya\robot_sim\physical_v29\training\checks\balance_audit_20261002'
```

The native one-time thread automation `humanoid-training-four-hour-check` is
scheduled for approximately 04:45 Eastern October 2. It checks the authoritative
active pointer, repairs failures, and may shut down only after the entire chain
passes matching independent gates and artifacts/unsaved work are secured. A
crash, expired limit, static posture or high reward does not authorize shutdown.

[Net-travel repair](NET_TRAVEL_REPAIR_20261001.md) addresses remaining rocking
credit and repeated short-block rollbacks in r10. Positive locomotion credit
is zero on 11,482 replayed planted-foot samples. Full standing rehearsal and
64-update minimum blocks support retention and cumulative learning. All 64
CPU regression/integration tests passed. These checks do not establish walking.
The earlier [v2 repair](REWARD_REPAIR_20261001.md) remains historical evidence.

## Automatic environments and monitoring

The reserve is **300 MiB**, checked each PPO update. The supervisor samples GPU
memory every three seconds. It increases one environment tier after **180
continuous seconds** of enough free memory for the added allocation plus the
reserve, up to **8,192**. Only fresh real training counts; startup/evaluation
memory cannot qualify. Increases apply to the next scene at a saved checkpoint
boundary. Memory failures preserve a checkpoint where possible and fall back.

Blocks use a 64-update minimum and measured timing sufficient to collect the
three-minute evidence window below the ceiling. The local chain has an
18-hour ceiling. An unpassed stage remains unpassed if that ceiling ends.
Detailed decisions, tests and limits: [automatic VRAM](AUTOMATIC_VRAM_20261001.md).

```powershell
& 'E:\Aditya\robot_sim\physical_v29\training\monitor_training.ps1' -Watch
```

Each completed increase writes `environment_ramp.json` beside its training
checkpoint. `state.json` reports progress and `vram_ramp`; `gpu.csv` records raw
samples. Read `progress.jsonl` for partial iterations, since the completed-block
total excludes unfinished work. Launcher PIDs may differ from actual Python
PIDs. Stop only verified owned process trees. `result.json` confirms worker
completion even if Kit cleanup hangs; a `.pt` file alone does not.

## Files and qualification

| File or directory | Purpose |
|---|---|
| [Mass estimate](MASS_ESTIMATE.md), `mass_estimate.json`, `mass_by_link.csv` | Part basis, link COM and approximate inertia |
| `assets/robot_training.urdf` | Lightweight contact/mass asset |
| `assets/motion_reference.npz`, `motion_validation.json` | CAD IK guidance and feasibility checks |
| `checks/preflight_07` | Contact/transmission scope, 337.45 N support versus 338.09 N expected |
| `checks/reward_repair_20261001` | Recorded reward audit and standing normalizer migration |
| Run `source/physical_v29` | Immutable worker/physics/evaluation/hardware source |
| Run `supervisor.py`, `vram_ramp.py`, `chain_config.json` | Exact capacity controller and provenance |
| Worker directories | stdout/stderr, config, progress, checkpoints and completion/failure |
| Gate directories | Raw telemetry and named independent pass/fail checks |

Runtime, load and capacity tests prove execution, not behavior. Nine ramp/timing
tests and the combined 31 capacity/gate/status tests passed. Reward regression
and live standing qualification retain their separate evidence.

The v29 mass basis is **34.46 kg** including battery and a conservative 640 g/hand
allowance, or **33.98 kg** with the published 400 g complete-hand value. Neither
is weighed. Contact boxes simplify housings; armature/PD gains are test choices.
Loaded motor curves, cooling, thermal duty, battery sag and controller current
limits remain unqualified. User actuator assumptions include 95% added
transmission/linkage efficiency and shared pose-dependent ankle/waist limits.
Locomotion does not certify get-up, backflips or physical load capability.

Play checkpoints only with their matching frozen source/asset. Workspace edits
do not change a running worker. No old 22-axis checkpoint is resumed. Previous
run directories, original checkpoints and dated documentation are preserved.
All changes/logs remain local; no Git commit/push or external experiment logger.
