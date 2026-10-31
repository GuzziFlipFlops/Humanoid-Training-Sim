# October 4, 17:37 UTC adaptive check: running regression and qualified walking retention

R33 is the source-frozen replacement for R32. The selected full actor had learned
genuine running at1.5m/s but lost slower walking; later R32 updates lost flight.
The current repair addresses that demonstrated retention problem. It does not
claim the stand-walk-run chain or5m/s goal is qualified.

Latest observation: 2026-10-04T18:27:46.192452+00:00. Status **running**, stage
**run_150**, operation `017_run_150_train_003`, observed count
**8192**, completed new main updates
**128**. All **46** frozen
hashes verify. Actual workers at this observation: **1**.
Fresh trusted baseline nominal/pushed requalification complete: **True**.
Read `replacement_status.json` and fresh ACTIVE_RUN/state rather than assuming
this dated observation remains current. Smokes and capacity are not learned progress.

Latest main execution evidence: `014_run_150_train_002` has
64 observed updates. Exact actor, exploration, critic and Adam
load checks pass. Every recorded pre-update KL and action-command mismatch is
zero. The qualified prior hook ran20 minibatches per update on the certified
0.1-1.0m/s command window; it is absent from inference. Minimum free VRAM was
368MiB, host available 6263MiB,
private-memory growth 7MiB and Torch allocation growth
0MiB. Active collection-plus-PPO throughput was
50660 samples/s. See `new_main_evidence.json` for
the exact scope, completed result and matching independent behavioral gates.
These execution measurements alone do not qualify running.
The completed worker block measured 43614 samples/s including its startup and shutdown; it used 8192 environments. This is a different denominator from active collection-plus-PPO time.

After 128 new updates, matching nominal 015_run_150_gate_003_nominal and pushed 016_run_150_gate_003_push remain unqualified. Running signed speed is 1.536/1.537m/s; true flight is 0.0000%/0.1250%, against the unchanged1.5% minimum. Earlier commands passing both: 0.0, 0.15, 1.0. Earlier commands still failing: 0.35, 0.65. Judge the repair against both retained tracking and geometric flight, not either alone. Compare subsequent complete pairs before attributing a tradeoff to the new gradient, and preserve both the selected flight actor and qualified walking teacher.

| Test | Command | Signed speed | Geometric flight | Passed / failures |
| --- | ---: | ---: | ---: | --- |
| 009_run_150_gate_001_nominal | 0.00 | -0.000m/s | 0.0000% | Passed |
| 009_run_150_gate_001_nominal | 0.15 | 0.349m/s | 0.0000% | speed, bilateral_stance, world_forward_speed |
| 009_run_150_gate_001_nominal | 0.35 | 0.514m/s | 0.0000% | speed, world_forward_speed |
| 009_run_150_gate_001_nominal | 0.65 | 0.798m/s | 0.0000% | speed, world_forward_speed |
| 009_run_150_gate_001_nominal | 1.00 | 1.073m/s | 0.0000% | world_lateral_speed |
| 009_run_150_gate_001_nominal | 1.50 | 1.539m/s | 2.0833% | Passed |
| 010_run_150_gate_001_push | 0.00 | -0.000m/s | 0.0000% | Passed |
| 010_run_150_gate_001_push | 0.15 | 0.344m/s | 0.0000% | speed, bilateral_stance, world_forward_speed |
| 010_run_150_gate_001_push | 0.35 | 0.508m/s | 0.0000% | speed, world_forward_speed |
| 010_run_150_gate_001_push | 0.65 | 0.796m/s | 0.0104% | speed, world_forward_speed |
| 010_run_150_gate_001_push | 1.00 | 1.073m/s | 0.0035% | world_lateral_speed |
| 010_run_150_gate_001_push | 1.50 | 1.539m/s | 1.9236% | Passed |
| 012_run_150_gate_002_nominal | 0.00 | -0.000m/s | 0.0000% | Passed |
| 012_run_150_gate_002_nominal | 0.15 | 0.228m/s | 0.0000% | Passed |
| 012_run_150_gate_002_nominal | 0.35 | 0.473m/s | 0.0000% | speed, world_forward_speed |
| 012_run_150_gate_002_nominal | 0.65 | 0.774m/s | 0.0000% | speed, world_forward_speed |
| 012_run_150_gate_002_nominal | 1.00 | 1.047m/s | 0.0000% | Passed |
| 012_run_150_gate_002_nominal | 1.50 | 1.533m/s | 0.1215% | flight, geometric_flight |
| 013_run_150_gate_002_push | 0.00 | -0.000m/s | 0.0000% | Passed |
| 013_run_150_gate_002_push | 0.15 | 0.228m/s | 0.0000% | Passed |
| 013_run_150_gate_002_push | 0.35 | 0.465m/s | 0.0000% | speed, world_forward_speed |
| 013_run_150_gate_002_push | 0.65 | 0.775m/s | 0.0000% | speed, world_forward_speed |
| 013_run_150_gate_002_push | 1.00 | 1.045m/s | 0.0000% | Passed |
| 013_run_150_gate_002_push | 1.50 | 1.535m/s | 0.3194% | flight, geometric_flight |
| 015_run_150_gate_003_nominal | 0.00 | 0.000m/s | 0.0000% | Passed |
| 015_run_150_gate_003_nominal | 0.15 | 0.206m/s | 0.0000% | Passed |
| 015_run_150_gate_003_nominal | 0.35 | 0.457m/s | 0.0000% | speed, world_forward_speed |
| 015_run_150_gate_003_nominal | 0.65 | 0.759m/s | 0.0000% | speed, world_forward_speed |
| 015_run_150_gate_003_nominal | 1.00 | 1.012m/s | 0.0000% | Passed |
| 015_run_150_gate_003_nominal | 1.50 | 1.536m/s | 0.0000% | geometric_flight |
| 016_run_150_gate_003_push | 0.00 | 0.000m/s | 0.0000% | Passed |
| 016_run_150_gate_003_push | 0.15 | 0.210m/s | 0.0000% | Passed |
| 016_run_150_gate_003_push | 0.35 | 0.448m/s | 0.0000% | speed |
| 016_run_150_gate_003_push | 0.65 | 0.759m/s | 0.0000% | speed, world_forward_speed |
| 016_run_150_gate_003_push | 1.00 | 1.021m/s | 0.0000% | Passed |
| 016_run_150_gate_003_push | 1.50 | 1.537m/s | 0.1250% | geometric_flight |



## Independent evidence and diagnosis

R32 is preserved at1352 completed main updates and its final full checkpoint
`076_run_150_train_020/candidate.pt`, SHA256
`da69a94e3d178e3ccdd7359a4398ecb170c7bba216d6ff976cc1a84f031558d6`. All source files,
logs, evaluations and intermediate policies remain. R32 had46.15k active PPO,
39.35k worker-effective and20.95k whole-chain completed-main samples/s at the
17:41 observation. These denominators differ; whole-chain time includes the
previous documented775.6-second user GUI pause. Subtract that pause for an active
comparison, and keep the original clock evidence intact. Latest completed6144
block had804MiB minimum free VRAM,8321MiB host available and5MiB private growth.
No pre-update KL or action-command mismatch was observed.

| Actor / test | Actual 1.5-command signed speed | Genuine flight | Retention |
| --- | ---: | ---: | --- |
| 029_run_150_gate_005_nominal | 1.539 m/s | 2.0833% | Failed 0.15, 0.35, 0.65, 1.0 |
| 030_run_150_gate_005_push | 1.539 m/s | 1.9236% | Failed 0.15, 0.35, 0.65, 1.0 |
| 077_run_150_gate_021_nominal | 1.517 m/s | 0.0000% | Failed 0.15, 0.35 |

Every running case above used32 independent robots. The early269-update
checkpoint028 passed **all running-case checks in both nominal and pushed tests**,
including speed, direction, alternating steps, survival and geometric flight.
Its prior-speed cases failed, so the whole running stage never qualified. Flight
peaked early and returned to zero in nine later nominal tests; slow retention
also failed repeatedly. This is regression, not a crash or evidence that the
robot cannot take off. The qualified standing/walking baseline remains preserved.

A controlled96-robot10-second normal-gravity comparison reproduced the earlier
actor's genuine flight at2.2792% versus0% for the final actor, with all robots
surviving and no resets. Their signed-world forward speeds were1.546 and1.525m/s.
The earlier actor's reward rate was14.247 versus14.099 for the grounded final
actor. The qualified teacher tracked the0.15 command at0.180 versus0.272 for the
final actor and0.353 for the early flight actor. A bounded diagnostic knee pulse
on the final actor produced only0.285% flight and reduced reward; phase offsets
did not restore flight. No pulse, phase helper or root support was deployed.
See `physical_response/result.json` and its full reward-term/target/force telemetry.

The saved50Hz final trace had a loaded foot in every intended flight window.
Action reconstruction matched2240 saved samples within6.68e-6; command snapshots
matched.39.5% of sampled raw running actions exceeded the existing+/-3 clamp.
Exact CAD Jacobians/reductions show no sustained leg velocity ceiling at these
saved10Hz states. Rare right-ankle samples reached98.5% of their cap; this does
not attribute failure to clipping or exclude substep saturation.

## Narrow repair and source lineage

The old bounded training prior protected stopped-command standing only. Earlier
walking commands had ordinary reward/rehearsal but **no corresponding retention
gradient**. R33 adds a separate training-only mean-action gradient toward the
latest independently qualified prior locomotion actor, limited by the **latest**
observation command. Running-stage samples above the teacher's certified speed
receive no such gradient. The original stopped-command gradient is bitwise
unchanged. The teacher is frozen, bounded to256 samples per minibatch, uses the
existing0.25 regularization coefficient, and is absent from inference. This is
a local adaptation, and closed-loop improvement remains to be measured.

Teacher use fails closed unless exact checkpoint bytes, current observation,
asset/motion/actuator contract, reward signature, classifier version and matching
full20-second nominal/pushed gates bind the entire earlier command set with
at least32 trials each. New future running stages select their most recent
qualified prior speed, without imposing a walking teacher on running samples.

Research supports using trained teachers to transfer and consolidate behaviors;
it does not guarantee this robot's result. [Policy Distillation](https://arxiv.org/abs/1511.06295)
and [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835)
support that general method. The0.25 bounded Gaussian-mean MSE hook here is a
declared implementation, not their exact published loss. The earlier
[Periodic Reward Composition](https://arxiv.org/html/2011.01387v2) force/phase
review and [public G1 task](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py)
remain background; no reward kernel or weight was changed this check.

Selected full checkpoint: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r32_geometric_flight_feedback\028_run_150_train_004\candidate.pt`;
SHA256 `4a2891bd0bb751fb6409f2a5bb54ec8a23266980d25dee5aab087e889055ac1a`;269 selected main updates.
Its actor, trained exploration, critic and Adam tensors load exactly. The existing
restart learning-rate ceiling bounds the saved2.25e-4 rate to1e-4; adaptive PPO
continues normally. Smoke/benchmark actors are discarded. The later1352-update
actor is preserved independently, rather than miscounted as the selected lineage.

Exactly three authored runtime files change: `standing_retention.py`,
`worker_unitree.py` and the supervisor.42 other frozen authored files, reward v7,
geometric verifier v6,400-input23-output inference, command distribution/history,
PPO algorithm/hyperparameters, geometry/contact/mass/CAD motor limits, iteration4
execution and all earlier/final gates remain identical. The new sources are
frozen in R33; R32 frozen bytes were never edited.

## Verification, including failed diagnostic attempts

22 targeted CPU checks pass: exact gradients, only certified prior rows, latest
frame selection, unchanged standing gradient, additive disjoint hooks, frozen
teacher, no inference hooks/behavior changes, qualification binding and negative
provenance tests. Nine further migration checks reject wrong selected/final
hashes, signatures, classifier/schema, active prior processes and protected source
changes. `cpu_tests.json` and `migration_validation_tests.json` contain the results.

The complete192-robot20-second nominal replay of checkpoint028 reproduces **all
case results and every saved physical/50Hz telemetry array bit for bit**, including
actions, targets, roots, joints, contacts, forces, COM and failures. See
`physical_parity.json`. Small real64-env running PPO and full-checkpoint reload
complete; all four pre-update KL and action-command mismatch values are zero.
`retention_fullstate_smoke_02/full_load_proof.json` confirms exact actor/critic/Adam
and exploration load. These smokes are execution evidence, not qualification.

The first CPU test incorrectly imported simulator-only evaluation utilities
without AppLauncher; it was corrected to AST-read the version constant. A later
new load-audit assertion compared Adam's CPU scalar step to a CUDA-loaded scalar;
the scalar values were unchanged. The failed diagnostic is preserved under
`retention_fullstate_smoke`, with its traceback and exact owned-worker cleanup.
It had left Kit open after failure; that failed process was closed, and the
retry passes after device-neutral exact comparison. There was a brief overlap
of the failed Kit context and retry startup; it never represented two training
workers. All subsequent simulator operations are sequential. No failed candidate
or short-reload behavioral result was accepted as qualification.

The old supervisor was suspended at a saved boundary and its evaluation worker
drained before controlled physics. Its exact identity was then retired after
validation. R33 was launched with full continuation, fresh physics/PPO/load,
capacity tests and matching trusted-baseline gates. The current candidate must
still pass **all** running and stopped/prior-speed gates to promote.

## Next decision and endpoint

1. After 128 new updates, matching nominal 015_run_150_gate_003_nominal and pushed 016_run_150_gate_003_push remain unqualified. Running signed speed is 1.536/1.537m/s; true flight is 0.0000%/0.1250%, against the unchanged1.5% minimum. Earlier commands passing both: 0.0, 0.15, 1.0. Earlier commands still failing: 0.35, 0.65. Judge the repair against both retained tracking and geometric flight, not either alone. Compare subsequent complete pairs before attributing a tradeoff to the new gradient, and preserve both the selected flight actor and qualified walking teacher. Next compare newer full matched pairs against this saved baseline. If retained tracking or genuine flight remains stalled, investigate whether the qualified walking-prior gradient conflicts with running PPO/flight objectives, including effective clipped action/target gradient contributions and controlled physical tradeoffs. A behavioral tradeoff alone does not establish gradient interference. Do not repeat reward-identical waiting indefinitely or erase useful actors.
2. Inspect locomotion_retention MSE/loss and per-command clipped output/target response. The prior acts only on latest-frame commands above0.1 and at/below the certified earlier speed; no teacher exists in inference. If flight or retention plateaus, use controlled gradient/physics tradeoffs before another narrow repair, not arbitrary reward weights.
3. Check moving6144/8192 capacity, free VRAM/host RAM and effective samples/time including the bounded teacher cost. Verify full actor/critic/Adam recovery and180-second healthy upward ramps. Retain actual CAD motor-speed telemetry; sampled spikes do not prove a velocity ceiling.

Keep one actual simulator worker, no stereo,8192 target,300MiB GPU reserve,
host guards, fallback and180 healthy seconds before upward ramps. The unchanged
curriculum proceeds through5m/s. Only measured signed>=4.95m/s plus matching
nominal/pushed and retained skill gates, or a causal demonstrated relevant
actuator velocity ceiling, satisfies the endpoint. No such endpoint is proven.
The same adaptive three-hour heartbeat remains active; at endpoint save, pause
owned training and the automation, and brief. **Leave the computer on.**
