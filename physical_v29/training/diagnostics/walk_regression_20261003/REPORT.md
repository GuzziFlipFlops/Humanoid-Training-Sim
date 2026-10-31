# Walking regression and PPO observation repair — October 3, 2026

Latest19:19UTC check supersedes the dated R25 startup/status below. Read the
[direction repair report](../r25_run_review_20261003_1919/REPORT.md) and fresh
[live_handoff.json](live_handoff.json). R25 passed its old walking gates through
1.0 m/s, but a full-population audit reproduced turning accepted by body-relative
speed. R25 is preserved at1471 main updates. Current R27 retains the candidate155
actor/exploration with fresh critic/Adam, adds observed heading commands and
fixed-frame direction checks, and keeps all previous gait checks and reward
kernels. All qualification is fresh; running and final completion remain
unqualified.57 focused CPU tests and real physical/PPO/phase/load smokes passed
execution.43 frozen hashes and one actual worker are verified by the recorder.
Target8192/reserve300MiB/host-aware180s ramp/fallback and four-hour checks remain.
The latest report contains research, resource limitations and R26 failure evidence.

## Historical R25 observation repair and progress

Latest15:19UTC four-hour follow-up: [R25 walking report](../r25_progress_20261003_1519/REPORT.md).
768/832 walking-update checkpoints pass all existing0.15 m/s nominal/pushed
cases and retain standing.0.35 remains unqualified due to step balance/order;
832 pair089/090 survives100%. Preserve R25's now-retained low-speed skill.
512,000 exact event replays and new unload/heading audits expose right-swing
interruptions and turning drift. Existing gates omit heading; inspect direction
before certifying straight/natural motion. The11:18 report remains historical.
Read fresh `live_handoff.json` for current training/count.
This report below records the earlier repair and startup evidence.

The previous R23 run learned stationary standing but did not learn walking.
After 448 walking updates, every saved walking evaluation still had zero
qualifying left/right steps. Its last nominal walking gate had 0% survival at
all tested commands. The compatible, independently qualified 512-update
standing actor is preserved and seeds a new frozen run, **R25**:
`locomotion_20261003_r25_owned_observations`.

Read [live_handoff.json](live_handoff.json), `../../ACTIVE_RUN.json` and the
referenced fresh state for current execution. R25 must pass new full independent
gates; no standing or walking qualification is carried across this repair.
The recurring four-hour research/repair check remains active. The final chain
is incomplete, so conditional shutdown is not authorized yet.

Final monitored handoff at **08:14 UTC / 04:14 EDT**: R25 is healthy and running
`009_stand_gate_001_nominal` with128 evaluation robots. Capacity tests selected
**8192 for the next training worker**. No fresh full stage is qualified yet;
the main-stage update count is0 because startup/capacity updates are diagnostic.
All42 frozen hashes match, one actual CUDA worker is owned by supervisor7400,
and available RAM is18581MiB after checkpoint recycling (free VRAM6101MiB in
evaluation). The four-hour
automation remains ACTIVE; its dated context now references R25. Read-only
thread metadata confirms `gpt-6.1-sol` and `max`; heartbeat inherits this thread.

## Why a restart was justified

R23 passed independent 128-robot, 20-second standing gates at 512 updates:
nominal/pushed survival 100%/100%, pelvis tilt 0.732/0.829 degrees, final drift
0.0407/0.0414 m and foot slip 0.00306/0.00342 m/s. Both gates passed all checks.
The actor in `047_stand_train_008_part_04/candidate.pt` is retained.

Walking then received seven complete nominal/pushed evaluation pairs and an
eighth nominal gate over 448 updates. No pair established a single qualifying
step. At `086_walk_035_gate_007_nominal` all commands had zero survival; the
0.35 m/s command moved backward at -1.532 m/s. At the latest
`092_walk_035_gate_008_nominal`, apparent forward speed 0.325 m/s at that command
still came with zero survival and zero qualifying steps. Speed tracking alone
would conceal this failure. See [audit.json](audit.json).

Sampled synchronized joint/contact/COM/action traces show actual falls, not just
a support-verifier rejection. For gate 086, the first saved stop robot failed
at 1.76 s and a walking-command robot at 2.24 s, both from low height. Their raw
action 99th percentiles were 1.325/1.363, below the action clipping threshold.
The [CPU FK frame sheet](086_walk_035_gate_007_nominal_robot2.png) reconstructs
the saved fall; it is not a simulator screenshot or a learned walking video.
No differential fault or nonfinite state explained those failures.

The separate [sole-support audit](../support_geometry_audit_20261003/REPORT.md)
remains valid: a conservative ankle-origin rectangle falsely rejected 93.4%
of sampled 448-update standing states, whereas transformed physical sole
footprints and inset/corner hulls rejected 0%. The sole center is 47.47 mm
forward of the foot-link origin. The qualified 512-update standing trace passes
all formulations. This discrepancy does not explain later physical falls or
certify contact pressure. No gate or sole geometry was changed in this pass.

## Reproduced PPO state corruption

`_get_observations()` returned reshaped views of mutable history buffers. PPO
kept that observation object when choosing an action. During the subsequent
environment step, resetting terminal or timeout rows zeroed those same buffers
before PPO copied the transition into rollout storage. The stored observation
could therefore differ from the state that produced its action and log
probability. Both policy and critic observations were affected.

The controlled CPU regression extracts the actual observation/reset statements
and uses the qualified actor plus the installed RSL-RL PPO storage sequence.
For reset rows, frozen R23 code changes both snapshots by a maximum 1.0 and
creates pre-update distribution KL 10.0575. The corrected code copies the
returned policy and critic snapshots; corruption and pre-update KL are both
zero. Nonreset rows remain unchanged. Evidence:
[observation_snapshot_tests.json](snapshot_tests/observation_snapshot_tests.json).

This is a reproduced integration defect, not evidence that RSL-RL is broken.
Its [PPO implementation](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py)
retains the action-time observations until post-step storage.
[PPO's objective](https://arxiv.org/abs/1707.06347) compares old/new action
probabilities conditioned on the same recorded state. Mutable reset aliases
violate that requirement.

R25 logs a bounded pre/post-update audit and fails before optimization if
pre-update KL exceeds 1e-5. Samples include terminal and nonterminal rows,
return/value variance, likelihood clipping, action changes, command groups and
learning rate. A twin CPU PPO test confirms the audit leaves losses and model
tensors identical. It introduces no extra optimizer update, simulator or RNG
draw. Actual CUDA smoke updates now record zero pre-update KL, including reset
rows. Initial post-update KL still overshoots and adaptive learning rate drops;
this separate optimizer-conditioning issue remains under observation. The
reset fix is not claimed to explain every KL spike or guarantee convergence.

## Walking and running feedback repair

The R23 gait and clearance terms were multiplied by actual body-speed magnitude.
At rest, phase-correct unloading and lifting consequently earned no useful
first-step feedback. A counterfactual with a 0.35 m/s command gives the same
11.7005 units/s to planted, correctly unloaded and correctly lifted feet.
The revised rates are 12.2005, 12.7276 and 12.9447; wrong-phase lifting is 11.7005.
These are synthetic comparisons, not learned gait results.

The public [Unitree contact-phase reward](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py)
does not require prior actual translation to recognize its contact pattern.
R25 restores that feedback under a moving command and healthy posture. Sole
clearance requires actual unloading and the expected swing phase. Loaded or
stopped feet receive no swing credit. Running flight additionally requires
signed forward movement; walking flight receives a bounded cost. This follows
the separation of standing/walking/running objectives supported by
[gait-conditioned multi-phase humanoid training](https://arxiv.org/html/2505.20619v2),
without claiming to copy that full method or a commercial Unitree policy.

Stopped-command training rows also receive a 0.25-weight mean-action retention
loss against the qualified standing teacher, at most 256 samples per minibatch.
The command selector was corrected from the old fourth-frame index 246 to the
latest history-five index 326. Exact-gradient tests prove moving rows are
excluded. There is no teacher action blending or teacher at inference. New
standing qualification is mandatory before selecting the ongoing teacher.

The 44 new counterfactual/provenance/gradient/audit checks pass in
[phase_tests/tests.json](phase_tests/tests.json). They cover matched phase/lift,
standing, skating, in-place stepping, hopping, backwards travel, running flight,
terminal costs and protected physical/gate bytes. Nine standing-retention
checks and 5,593 pinned G1/H1 source/arithmetic checks also pass. Full learned
walking still requires independent survival, bilateral alternating landings,
cadence, clearance, slip, net travel and previous-skill retention gates, followed
by visual gait review. Formula rankings alone cannot establish that outcome.

## Preserved model and execution

The simulation stays physical-v29: 23 axes, 400/415 actor/critic history inputs,
34.46334 kg provisional mass, original CAD joint frames and differential
geometry, self collisions, actuator limits and target-slew constraints. R25
changes ownership of observation storage, not observation values, dimensions
or scaling. No CAD revision, controller gains, mass, motor speed/torque,
collision asset or gate threshold was changed.

R23's 960 full updates (512 stand, 448 walk), all gates, checkpoints and its
41 frozen source hashes remain preserved. Intermediate R24 contains only
validation/capacity evidence; it was drained after the PPO audit revealed the
aliasing issue. Its diagnostic updates are not counted as main-stage progress.
R25 has 42 frozen source files and the recorded seed checkpoint checksum. Actor
tensors match exactly before the deliberate exploration reset to 0.35; the
changed objective starts with fresh critic/Adam state. Matching subsequent
resumes retain actor, critic, optimizer and adaptive learning rate.

Startup checks are sequential: 32-environment physics/transmission preflight;
64-environment standing and phase/retention PPO smokes plus checkpoint reloads;
4096/6144/8192 capacity tests; then full 128-robot, 20-second nominal/pushed gates.
Preflight reproduced weight support 337.61 N versus expected 338.09 N, no falls
or invalid differential samples, and combined pitch/roll closure errors below
0.000054 mm. These checks establish execution, not learned qualification.

One actual Isaac/CUDA worker runs under the owned supervisor. Target 8192,
300 MiB reserve, fallback and a 180-second healthy-PPO-memory upward ramp remain.
The ramp never treats startup/evaluation free memory as training capacity.
Host-RAM growth requires checkpoint-preserving recycling; its root cause is
still open. Compare effective throughput including startup and recycling,
not just environment counts. Read the latest capacity results in the handoff.

Completed capacity results:

| Environments | Updates | Raw steps/s | Effective steps/s | Minimum free VRAM | RAM recycle |
|---|---:|---:|---:|---:|---|
|4096|8|8392|5631|3801MiB|No|
|6144|24|10491|8405|2259MiB|No|
|8192|19|11762|9060|1105MiB|Yes, checkpoint saved|

8192 is12.11% faster in raw throughput and7.80% faster including startup/exit
and the observed recycle. These are bounded tests with unequal update counts,
not a long-run optimum or convergence guarantee. The8192 worker stopped at the
host-RAM guard (1883MiB available), preserved `resource_recovery.pt`, exited
normally, and released memory. No VRAM reserve breach or forced count fallback
occurred. The supervisor selected the user's8192 target. Pre-update KL remains
zero through all sampled updates/reset rows. Post-update KL after initialization
is near its0.01 target and adaptive LR recovers above the floor; the initial
overshoot is recorded separately and not hidden by this improvement.

A new [CPU observation-lifetime experiment](cpu_lifetime.json) runs32 updates
with the installed PPO and actual history methods, once with and once without
the update audit. Both have zero accumulated weakly tracked snapshot references
and zero private-memory growth over their final16 updates. It uses eight fake
environments and actual network dimensions, not physics or CUDA. Meanwhile
the real4096/6144 workers grow about13.57/20.24MiB of allocated torch memory and
358/539MiB of private host memory per update. This narrows the investigation
but cannot attribute or exclude native simulator/CUDA ownership. The companion
`audit_cpu_observation_lifetime.py` is CPU-only and leaves saved policies alone.

The [original five-foot walker comparison](../old_walker_comparison_20261002/REPORT.md)
verified repeated steps around 0.994 m/s with zero falls in its saved nominal
and randomized trials, although gait was asymmetric. It has different physics
and an 80-input/22-action contract, so its actor cannot load into this robot.
Its approximately 55.5k versus this model's earlier 11.8k environment steps/s
also makes elapsed-time comparisons misleading. R23's repeated independent
walking failures, however, are behavioral evidence beyond slower throughput.

## Follow-up and reproduction

Use CPU-only `record_owned_observation_handoff.py` to refresh provenance and
resource evidence without starting another simulator. `audit_walk_regression.py`
reconstructs preserved R23 behavior; `test_observation_snapshot.py` reproduces
the reset defect; `test_phase_feedback.py` runs the counterfactual suite.
`launch_phase_feedback.ps1` checks tested source hashes, preserved seed and
owned-process absence before launching a fresh directory. Do not rerun it over
the active R25 run. No live frozen source was edited and no Git commit/push
was performed.

The next check should compare fresh nominal/pushed/retention gates and bounded
PPO diagnostics before changing anything. Preserve measurable improvement.
Investigate persistent optimizer overshoot, reward ranking or contact/axis
faults with controlled evidence if gates plateau. Do not claim humanlike gait,
running, recovery, backflip or hardware qualification from these smokes.
For memory diagnosis, compare owned fixed-action physics-only rollouts with
PPO at matching counts/steps, and inspect CUDA/TensorDict versus native sensor
allocations in the same single-worker setup. Preserve the active actor first;
CPU lifetime stability does not exonerate those native paths.
