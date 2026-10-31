# October 3, 23:19 UTC training check: steering regression and CAD cache repair

R27 learned real alternating steps and retained standing, but failed independent
straight-line locomotion. Its heading worsened over 64, 128 and 192 walking
updates. R28 started October 4 at 00:15 UTC with a narrowly tested CAD cache fix
and a direct heading-alignment reward. The earlier, better-direction actor is
retained; R28 started with fresh critic/Adam and qualification. R29 continues
the full R28 checkpoint under an audited host-RAM scheduling repair. Read `ACTIVE_RUN.json`
and the CPU recorder for current execution rather than this dated summary.

**Latest execution, October 4 at 01:02 UTC:** R28 passed matching full nominal
and pushed standing evaluations. R29's fresh full-checkpoint PPO and reload
smokes passed; it is executing independent walking gates. The next training
worker is selected at 6,144 environments, with the target still 8,192. Walking
and running remain unqualified. All 41 authored worker/physics/reward runtime
files match R28 byte for byte; the 43 frozen configuration hashes verify.

**01:13 UTC full nominal result:** R29's retained actor passes every stopped
and 0.35 m/s check. At 0.35 it achieves 0.322 m/s fixed-frame forward speed,
100% survival, 22/23 alternating landings and 4.63 degrees final heading error.
The retained 0.15 case survives and takes 23/23 alternating steps but fails
fixed heading and forward path efficiency: 64.10 degrees final error, 0.127
m/s forward and 0.121 m/s lateral speed, efficiency 0.657. The full stage
therefore remains unqualified. Pushed checks are running. These gates evaluate
the retained baseline before R29 main PPO updates, and provide the heading
repair's starting comparison rather than evidence of new learning.

## Independently measured behavior

| R27 candidate | Moving command | Body-forward speed | Fixed-frame forward | Final heading error | Independent result |
|---|---:|---:|---:|---:|---|
| Baseline 011 nominal | 0.35 m/s | 0.326 m/s | 0.322 m/s | 4.63 degrees | All checks pass at this command; retained 0.15 fails |
| 128 updates, 038 nominal | 0.15 m/s | 0.324 m/s | -0.064 m/s | 169.62 degrees | Direction fails |
| 192 updates, 050 nominal | 0.15 m/s | 0.226 m/s | -0.096 m/s | 124.85 degrees | Direction fails |
| 192 updates, 050 nominal | 0.35 m/s | 0.411 m/s | 0.186 m/s | 104.68 degrees | Direction fails |

Latest moving cases have 100% survival and 23/23 qualifying alternating
landings; standing passes. These are meaningful skills, but they do not qualify
the whole walking stage. Earlier pushed checks also fail direction. See
[evidence.json](evidence.json), the original matching gate folders and
`050_walk_035_gate_004_nominal/result.json`. No qualification thresholds changed.

The 192-update actor and checkpoint SHA256
`6927edfe2ac0b941cd16d515f6e003d5466f003a4cb761c35c3e471525f180de`
are preserved at R27 `049_walk_035_train_003_part_10/candidate.pt`.
R27 was drained at an evaluation boundary after the final nominal result;
its supervisor/worker were verified by command line before stopping.

## Controlled steering diagnosis

[actor_command_audit.json](actor_command_audit.json) reconstructs 9,010 saved
policy inputs per sampled actor, matching measured actions within 0.00003.
Changing all five yaw-command frames affects actions, but only weakly. The
recorded instantaneous yaw kernel gives only about 0.002 to 0.037 reward-rate
gain when removing the measured mean yaw bias while retaining gait oscillations.
That is a state counterfactual, not an action-to-yaw gradient claim.

A separate 72-environment physical experiment applied constant yaw commands
-0.20, 0 and +0.20 rad/s at 0.15 and 0.35 m/s, retaining the unmodified actor,
contacts, mass and actuator model. The 192-update actor's 0.15 m/s yaw rates
were +0.198, +0.204 and +0.211 rad/s respectively. It turned right even with a
left-turn command. Baseline rates were about +0.053 to +0.058 rad/s. At 0.35,
the latest rates were +0.076 to +0.102 rad/s. Every trial survived seven seconds.
See [physical_response/result.json](physical_response/result.json) and raw NPZs.
These short interventions diagnose response; they are not qualification gates.

Observed/action/reward commands agree; pre-update KL is zero. Earlier joint
frame/limit and support audits remain relevant. This check does not establish
a new CAD-axis fault. Steering bias and weak objective discrimination are
separate from the cache fault below.

## Reproduced memory defect and numerical repair

`CadDifferential.state()` cached `tuple(v)` even when `v` was a transient
tensor. Those keys use tensor-object identity and retain scalar/row views and
old storage. Moving waist geometry makes the leak scale with environment count.
The live 4,096-environment worker grew roughly 359 MiB of private RAM and
13.5 MiB of CUDA allocation per PPO update, requiring frequent checkpoint
recycling. These observations are preserved in `evidence.json`.

The repair bypasses caching for tensor inputs and caches only immutable JSON
geometry constants. After 300 changing poses, ankle caches remain at 11 entries
and waist at 12; original counts were 611 and 1,212. Across 1,024 poses for
each of three pairs in float32 and float64, all angles, Jacobians, validity,
closure errors, mapped torques, motor requests/applied efforts and speeds are
bitwise identical. Jacobians also pass finite-difference closure checks, and
combined torque/speed limits remain enforced. See
[differential_cache_equivalence.json](differential_cache_equivalence.json).

The complete same-seed, same-actor 72-environment, 350-tick physical replay is
also bitwise identical for root state, joints, contacts, clearance and survival:
[cache_fixed_physical_equivalence.json](cache_fixed_physical_equivalence.json).
The repaired rollout's CUDA allocation stays at 11.235 MiB instead of increasing
each tick; private memory settles near 4,569 MiB and Python object count remains
around 516,000. See `physical_response_cache_fixed.stdout.log`.

The fixed replay's baseline batch completed and was preserved before stopping
its later slow full-GC boundary. The second actor batch/profile was not completed
in that diagnostic. The first unpatched diagnostic likewise preserved its
completed baseline before a slow inventory/GC boundary; the resumed experiment
completed the 192-update intervention. Do not label these partial diagnostics
as complete gates. Native/full-GC timing remains a separate unproven concern;
production evidence is required before a further maintenance change.

The synthetic installed-PPO memory isolation test, with and without audit and
standing retention, remained stable over 32 updates after warm-up. It excludes
Isaac and CUDA and does not diagnose every possible native leak. Its artifact is
[ppo_memory_isolation.json](ppo_memory_isolation.json).

## Reward repair and retained actor

Reward version `cad_heading_alignment_retention_v6` adds one positive bounded
rate: `exp(-(wrapped_reset_heading_error / 0.35 rad)^2)`, weight 1, multiplied
by the existing healthy and moving masks. It is zero for stopped commands and
physical-fault ticks. The heading target remains fixed throughout each episode.
All previous kernels, gait/clearance/flight coefficients, observation/action/
reset functions, actuator limits and independent thresholds are unchanged.
The pinned Unitree source bodies remain unchanged. This heading term is a
declared local adaptation, not an upstream Unitree reward.

The counterfactual reward tests hold body speed and gait fixed and compare
straight, 20-degree, sideways and reversed orientations. Straight motion wins,
with over 0.6 reward-rate difference at 20 degrees. In-place stepping and
sliding still lose to a matched walking cycle; terminal, stopped, wrapping and
global-heading invariance tests pass. This proves incentive rankings, not PPO
convergence. [cpu_tests.json](cpu_tests.json) records 50 passing targeted unit
tests; [v6_phase_tests/tests.json](v6_phase_tests/tests.json) records 44 passing
phase, first-step, running, retention-gradient and provenance checks. The
history-ownership reproduction also passes with zero corruption/pre-update KL.
Old archival provenance assertions were updated to compare the current R27
baseline, retaining R23 as the explicit alias/first-step failure reproduction.

R28 actor initialization uses R25 candidate155, SHA256
`892a9514ce58adaf684589c4f01ea085abaa604202a711d7deb6329e65c3f039`.
Its nominal 0.35 direction was substantially better than the regressed R27
candidate. Actor weights and trained exploration are retained; critic and Adam
start fresh at initial learning rate 0.0001. All previous actors/checkpoints
remain preserved. R28 must pass new physical/PPO/load and capacity tests and
full standing, nominal/pushed walking and prior-speed gates before promotion.

## R28 execution evidence

Fresh physical/contact/transmission preflight, both four-update PPO smokes and
their checkpoint reloads completed normally. Preflight included combined
pitch/roll samples for both ankles and waist; no falls or invalid transmissions,
support 337.61 N versus 338.09 N expected. Worker transfer assertions preserve
all actor/exploration tensors. Pre-update KL and observed/action/reward command
differences remain zero. The phase/standing-retention loss executes in physics;
its four-second reload does not supply enough landings to qualify walking.
Read [r28_execution_handoff.json](r28_execution_handoff.json) for exact artifacts.

The completed eight-update 4,096 capacity test kept CUDA allocation at 436 MiB
and private RAM settled at 10,470 MiB after 10,467/10,468/10,470 startup readings.
Collection-plus-learning throughput was 6,897 steps/s, versus R27's 4,840:
a 42.5% increase. Including worker startup/exit, measured throughput was 4,230
versus 3,209 steps/s: a 31.8% increase. Minimum free VRAM was 2,402 MiB and
available host RAM 4,301 MiB; no host recycling. This is a bounded capacity test
under current app load. Higher-count results are below; these rates exclude
independent gate time and do not predict convergence.

## Host-RAM capacity repair and R29 continuation

| R28 environments | Completed updates | Collection + learning steps/s | Worker startup/exit included | Minimum host RAM available | Outcome |
|---:|---:|---:|---:|---:|---|
| 4,096 | 8/8 | 6,897 | 4,230 | 4,301 MiB | Full capacity test, stable memory |
| 6,144 | 24/24 | 9,584 | 7,599 | 3,360 MiB | Full capacity test, stable memory |
| 8,192 | 5/24 | 12,212 | 5,001 | 2,010 MiB | Saved checkpoint and stopped at host-RAM guard |

The 8,192 trial preserved the 300 MiB VRAM reserve, with a minimum 462 MiB,
but host RAM crossed the existing 2,048 MiB guard. Private memory was stable
near 14,814 MiB; external app load and the count's base allocation exhausted
headroom without the former per-update cache growth. The supervisor incorrectly
counted the shortened recovery as eligible capacity and would keep restarting
at the same count. Under these bounded tests, 6,144 delivered about 52% higher
worker-effective throughput than the recycled 8,192 trial.

The repaired supervisor requires the full benchmark update budget, host RAM
above its existing guard, no resource recycle, and the unchanged VRAM reserve.
It reclassifies the saved identical-runtime capacity evidence and selects 6,144.
A later completed-update host recovery retains the checkpoint/optimizer,
lowers one count tier and resets the ramp hold/cooldown without inflating
per-environment VRAM cost. The final-update boundary is covered too. Actual
healthy training still needs 180 continuous seconds of both host and GPU
headroom before increasing one tier. No gate or actor objective changes were
made in this maintenance step.

[resource_capacity_tests.json](resource_capacity_tests.json) records 24 passing
CPU checks, including the actual 8,192 capacity replay, whole-supervisor remaining
update continuation, final-update recycling, strict runtime provenance, and
maintenance smoke without actor reset. These isolated scheduler fixtures do
not qualify any physical policy.

R28's supervisor was stopped at an evaluation boundary and its pushed standing
worker finished normally before replacement. Both full gates use the same
checkpoint, 128 robots, 20-second trials and the required nominal/pushed seeds.
Every standing case/check passes. The matching completed reports are recorded
in R28 `standing_gate_completion_after_drain.json`; no policy or threshold was
edited to obtain this qualification.

R29 `locomotion_20261004_r29_host_capacity_checked` continues the full checkpoint
SHA256 `53862d9bbfa34124601db1c103d685582ff2f7ce503b21267d7166bc66979fd4`.
Actor, exploration, critic, Adam and adaptive learning rate are loaded together.
All 41 authored runtime files must match, including rewards, observations,
actuators and verification; generated USD caches are regenerated from the same
protected URDF/converter sources in the new run. Matching standing qualification
and its training-only retention teacher are preserved. Current/earlier-speed
nominal and pushed walking gates are rerun before further promotion.

Fresh four-update, 64-environment PPO and checkpoint reload smokes completed.
Worker load assertions preserved actor tensors; pre-update KL and action/reward
command differences are zero. Private RAM readings are 6,193/6,206/6,206/6,206
MiB and CUDA allocation stays at 36 MiB. Smoke-updated actors are discarded;
the qualified original checkpoint continues. See
[r29_execution_handoff.json](r29_execution_handoff.json) and
[r28_to_r29_runtime_provenance.json](r28_to_r29_runtime_provenance.json).

## Throughput and operating constraints

R27 capacity evidence: 4,096 environments collected 4,840 raw steps/s and
3,209 worker-effective steps/s; 6,144 collected 6,050 raw and 3,381 effective,
with host-memory recycling. Its 8,192 attempt crossed the 300 MiB VRAM reserve.
The original five-foot model's older 6,144 benchmark collected about 55,518
steps/s, but it has different 22-action/80-input dynamics/controller/contact
settings. Official G1 numbers on a different GPU/OS are not a same-machine test.
The cache defect is concrete overhead; any claimed speedup needs fresh R28
capacity and sustained training measurements. No 10x speedup is claimed here.

R29 targets 8,192 with the existing 300 MiB reserve, automatic fallback and
180 continuous healthy-training seconds before one upward tier at a checkpoint
boundary. Host RAM must also be healthy. It uses one actual Isaac/CUDA worker,
no stereo vision, and the frozen physical-v29 assets. Fusion's unrelated unsaved
Strap cam work is preserved. No CAD migration, Git commit/push or shutdown.
The four-hour heartbeat remains ACTIVE with this handoff. The full chain is
incomplete; shutdown remains conditional on matching final independent gates.

## Primary sources rechecked

The official [Unitree H1 configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py)
uses yaw-frame linear tracking and an instantaneous angular kernel with weight
0.5 and width 0.5. That does not explicitly reward an episode's fixed heading.
The [Isaac Lab heading command](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py)
supports proportional wrapped heading feedback, already used in R27. The
[Isaac Lab direct locomotion task](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/locomotion/locomotion_env.py)
also includes direct heading alignment alongside progress. Its existence informs
this local choice; it does not prove the chosen width/weight or this robot's
convergence. The public [G1 benchmark](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/overview/reinforcement-learning/performance_benchmarks.html)
uses different hardware/OS and timing scope. Our vendor sources remain pinned
at commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`.

Refresh the CPU-only live recorder and `record_r29_maintenance_handoff.py`,
then inspect fresh active state/progress/results.
If fixed-heading quality improves, preserve the actor and let learning continue.
If it stalls, measure steering authority and temporal reward/command response
with controlled physical interventions before another redesign. Running still
requires independent alternation, clearance, cadence, flight and travel checks.
