# Adaptive training check — October 05 02:16 EDT

**R35 continues on its original supervisor, preserving the forward head and full learned state. The proposed running-command exposure change was tested and rejected. Running and all-prior retention remain incomplete.** No training runtime, reward, observation, action, CAD, motor cap or optimizer setting was deployed this check.

## Fresh production evidence

The check began with2560 completed new R35 updates;2688 were preserved at the pause boundary, plus1473 prior R34 updates. Exact saved checkpoint135 SHA256 `788837ad7b431073575a2e3aa3ad0460fe1f97b3d480336a8e8c6e01576d06fb` remains intact. The latest matched full192-robot,20-second nominal/pushed pair available at report creation is shown below. Each command has32 independent trials. Original qualified standing/walking actors are still preserved; their certificates do not qualify today's learner.

| Command m/s | Nominal signed m/s | Pushed signed m/s | Complete pair |
|---|---|---|---|
| 0.00 | 0.000 | 0.000 | Pass |
| 0.15 | 0.189 | 0.187 | Pass |
| 0.35 | 0.412 | 0.403 | Pass |
| 0.65 | 0.749 | 0.749 | speed |
| 1.00 | 1.067 | 1.068 | bilateral_stance |
| 1.50 | 1.417 | 1.421 | flight, geometric_flight |

All recent cases survive, but flight and some earlier commands remain unqualified. At the2560-update matched pair130/131, head combined yaw/pitch RMS at1.5 was3.798/3.801 degrees, versus51.71 before upper refinement;90.0%/90.7% of sampled frames put both head axes within5degrees. Other moving commands range roughly4.3–5.6degrees RMS. The5degree band is diagnostic, not a qualification gate. Arm compactness improved previously, but complete natural arm swing is not certified. Current upper progress must not be discarded simply to retrieve an older flight actor. See [upper audit](upper_audit.json), [production gate history](gate_trends_final.json) and [owned evidence](owned_observation_at_report.json).

## Closed-loop physical separation of head feedback and head motion

Training paused at an owned saved boundary, supervisor45868 was suspended, and the eval worker drained. One diagnostic simulator at a time used unchanged gravity, base freedom, self collision, contacts and CAD/motor constraints.128 robots,32 per group, ran20seconds. Four groups compared currentR35, the preservedR34 flight actor038, old body with native-zero head targets and actual feedback, and the same native-zero head control with ONLY head q/qd/previous-action history supplied from the simultaneous original-head cohort. After5seconds both neutral-head groups have exactlyzero head actions and targets. No root support or pose writes were used.

| Policy/control diagnostic | Head RMS after5s | Actual flight | Absolute lateral m/s | Final heading error |
|---|---|---|---|---|
| latest_run | 3.41° | 0.000% | 0.051 | 3.25° |
| preserved_flight_actor | 51.78° | 1.868% | 0.110 | 1.19° |
| old_body_neutral_head_actual_feedback | 2.10° | 2.451% | 1.409 | 70.25° |
| old_body_neutral_head_reference_head_feedback | 1.81° | 1.793% | 0.109 | 0.89° |

Every robot survives. Under the same native-zero physical head control, changing head-feedback history changes final heading from70.25degrees to0.89degrees and lateral speed from1.409 to0.109m/s, while preserving about1.79% genuine flight in the latter. This is causal evidence that the old body controller depends strongly on its original head-feedback distribution. Actual head dynamics also differ slightly after body trajectories diverge; this does not prove mechanical effects absent. The experiment is a diagnostic, not a qualified hybrid controller or permission to falsify production observations. The original actor's51.8degree head pose is not restored in production. All diagnostic controllers/inputs are discarded.

[Result](head_causal_response/result.json), [post-transient and motor analysis](head_causal_analysis.json), [reviewed plot](head_causal_comparison.png), [full saved trace](head_causal_response/telemetry.npz).

The exact50Hz CAD Jacobian/reduction audit validates motor-speed units by directional finite differences. Current right ankle motor2 has72.8% p99 of cap and0.281% samples at/above cap; original flight actor has84.0% p99 and0.309% samples at/above cap. Other relevant leg motors are farther below their caps. Isolated spikes, including transient dynamics above nominal cap, do not prove a sustained locomotion ceiling; substep clipping/torque attribution remains untested. All caps are unchanged.

## Training-command hypothesis, controlled comparison and rejection

The old1.5 stage samples32% stopped,24% earlier commands and44% frontier. Only about24% of all robots receive exact1.5 running rewards: the remaining20% frontier samples lie between1.2 and1.5 and use walking duty0.55, while1.5 uses running duty0.43. Production block135 confirms23.980% running-command exposure. This is a reproduced command/reward boundary, not proof it causes the plateau.

A separately staged candidate moves that20% frontier into the current running gait, retaining all stopped/prior draws and unchanged evaluation/inference.55 CPU checks pass; every <=1.0 and >=1.875 sampler output is bitwise equal, as are all prior/stopped cohorts. A same-actor192-robot20-second replay under the candidate is bitwise identical in all saved startup/physical/gait arrays and all evaluation cases. No geometry, reward weight/kernel, action semantics or gate changed.

Matched control and candidate each load checkpoint135's EXACT actor, critic, Adam and exploration, use4096 environments,64 PPO updates, seed71, the same qualified prior teacher and upper loss, and the same existing restart LR ceiling. Adaptive PPO LR remains adaptive. Full192-robot20-second nominal901/pushed1301 evaluations include32 trials per command. Every update has zero pre-update KL and observed/action-command difference. A further actual64-env4-update candidate reload proves actor/critic/Adam/exploration equality. These tests check safe execution, not convergence.

| Matched experiment | Running exposure | Nominal/pushed genuine flight | Nominal0.35 | Running head RMS nominal/pushed |
|---|---|---|---|---|
| control | 24.53% | 0.656% / 0.757% | Pass | 3.35° / 3.36° |
| candidate | 43.91% | 0.000% / 0.021% | Fail | 4.00° / 4.01° |

**Reject the candidate.** It reduces flight relative to control, loses a nominal0.35 retention pass, and slightly worsens head posture. Neither arm qualifies the complete running stage.64updates on one matched seed do not rule out every future exposure strategy, but they do not justify deploying this one now. Both diagnostic learned actors are discarded; the exact production checkpoint and source stay preserved. Candidate code/logs/checkpoints remain nearby for reproducibility; they are not a new active training revision.

[CPU tests](cpu_command_audit.json), [physical parity](physical_parity.json), [full matched comparison](matched_comparison.json), [decision](decision.json), [candidate reload proof](candidate_full_reload/full_load_proof.json).

## Research, interpretation and next narrow experiment

[Siekmann et al., Periodic Reward Composition](https://arxiv.org/html/2011.01387v2) describes complementary force/velocity behavior in swing and stance, with gait parameters supplied to the policy. It motivated auditing this robot's gait boundary, not copying numerical coefficients or asserting that extra running exposure must help. The controlled comparison rejects that simple remedy.

[Ross, Gordon and Bagnell, DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses imitation under the observation distributions induced by a learner's own actions. It supports investigating the measured change in head-feedback distribution and collecting teacher/student counterfactuals on current states. This is a research lead; no DAgger algorithm or teacher conditioning was deployed and the paper does not guarantee this robot's convergence.

Next prioritize a bounded training-only posture-aware locomotion transfer test on current neutral-head states, with matched control and full exact-state preservation. First reproduce the head-feedback effect at retained0.35/0.65/1.0 commands and verify any teacher conditioning under normal physics before trusting its targets. Then measure teacher/student lower-output and effective target errors, objective costs, per-command gradient contributions, and matched independent direction/flight/retention/head behavior. Production observations must remain truthful. Do not roll back to038, deploy action hybrids or reference-input rewrites, increase motor caps, or repeatedly tune arbitrary weights. Do not repeat the rejected exposure change without new evidence. Preserve the qualified walking teacher and038 as separate evidence.

## Resume, memory, actual throughput and provenance

The SAME owned supervisor resumed at2026-10-05T06:06:52.143395+00:00 after939.1seconds. Checkpoint135 and source bytes are unchanged; no diagnostic actor was deployed. The eval136 wrapper duration includes this pause and must not be treated as simulation throughput. At the owned report snapshot2026-10-05T06:10:31.765398+00:00, statusrunning, stagerun_150, 2688 completed new main updates, operation `138_run_150_train_043`. One actual CUDA worker,47 matching authored source hashes. GPU sample `2026/10/05 02:10:31.755, NVIDIA GeForce RTX 4060, 8188 MiB, 6255 MiB, 1702 MiB, 64 %`; host available7631MiB.

Completed-main throughput is43811 active collection+PPO samples/s,37734 main-worker-effective samples/s, and24077 whole-chain completed-main samples/s excluding both R35 diagnostic pauses(1411.3s). Denominators differ; diagnostic A/B samples are excluded. Latest completed production block135 gives43099 active samples/s, minimumGPU1701MiB andhost7232MiB; pre-update KL0. This is not the old~9k bottleneck. Running failure still requires behavior/learning diagnosis.

Target8192,300MiB reserve, host guards, fallback and180continuously healthy training-second upward ramp remain. Iteration4 execution optimizations, environmentv8/verifierv6,23outputs/400inputs, full checkpoint recovery and all independent current/prior gates remain untouched. The physical model still uses simplified collision envelopes; earlier full-CAD frame QA and fulfilled recordings do not certify every detailed CAD collision. Unrelated unsaved Fusion work is preserved. No GUI repeat was requested or performed this check.

[Pause](diagnostic_pause.json), [resume](diagnostic_resume.json), [fresh source/process evidence](owned_observation_at_report.json), [throughput/evidence](evidence.json). A stale diagnostic-recorder path lookup failed once and was corrected to the canonical training/record_owned_observation_handoff.py; no simulator or training fault resulted. Nonfatal Kit shader-cache messages and all experiment logs are preserved. Neither actual5m/s nor a relevant persistent motor ceiling is proven. Keep the same three-hour heartbeat active; save/pause/brief at its verified endpoint, leave the computer on.



<!-- final_20261005_0540_observation -->
## Final resumed learning and original-walker comparison

Fresh owned snapshot2026-10-05T06:17:45.734587+00:00: sameR35, 2752completednewmain, `141_run_150_train_044`, oneactualworker,47hashes. Resumed block138 completes64actualupdates with exact actor/critic/Adam/exploration load from135, zero maximum pre-updateKL/command mismatch, minGPU1692MiB/minhost7626MiB/private growth7MiB. Active collection+PPO43095/worker-effective37035samples/s. Matching139/140 outcomes:

| Command | Nominal signed speed | Pushed signed speed | Nominal/pushed genuine flight | Pair qualification |
|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000% / 0.000% | Pass |
| 0.15 | 0.189 | 0.187 | 0.000% / 0.000% | Pass |
| 0.35 | 0.412 | 0.403 | 0.000% / 0.000% | Pass |
| 0.65 | 0.749 | 0.749 | 0.000% / 0.000% | speed |
| 1.00 | 1.067 | 1.068 | 0.000% / 0.000% | bilateral_stance |
| 1.50 | 1.417 | 1.421 | 0.000% / 0.059% | flight, geometric_flight |

Running head RMS2.544/2.557degrees. These fresh gates still do not qualify the entire stage. [Exact resumed-learning proof](resumed_learning_verification.json), [final owned snapshot](final_owned_observation.json).

The original five-foot `model_2098.pt` SHA is recomputed and matches the successful historical comparison. Preserved eight-robot24s nominal/randomized1.0-command traces achieved0.9938/0.9946m/s with0falls; they are historical evidence, not a newly rerun stricter32-trial gate. Its collection history recorded55,518steps/s; current completed-main active rate43811is78.9%ofthat rate. The former approximately10x-throughput concern is not representative of today's optimized learner. This is not a matched hardware/task benchmark: original80inputs22actions/38kg/implicitPD/no self collision versus current400inputs23actions/34.46kg/CAD torque-speed-slew constraints/self collision. Old walking success is not evidence that current running or bilateral/direction gates pass. [Reverified original evidence](original_walker_comparison_refresh.json), [full historical comparison](../old_walker_comparison_20261002/REPORT.md).

The existing automation is verifiedACTIVE, sameID/thread/three-hour cadence/notification preference/durable scope. History inventory:35physicalruns,1322operations,8historicalmissingmanifestartifacts,0differinghashes/0readwarnings. No5m/s or relevantvelocityceilingendpoint. Leavecomputeron.


Final live refresh 2026-10-05T06:24:44.313882+00:00: same R35 running, 2816 completed new main updates, 6144 training environments, one actual worker and all 47 frozen hashes verified. The latest matching142/143 remain unqualified; see final_pair_summary.json and final_live_status.json. CPU history refreshed to1325 operations, zero differing hashes and zero read warnings.
