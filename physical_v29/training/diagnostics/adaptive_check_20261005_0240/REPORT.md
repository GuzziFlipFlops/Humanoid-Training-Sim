# Adaptive training check — October 04 23:08 EDT

**R35 is preserved and resumed on its same supervisor. The head improvement is substantial; running and slow-speed retention remain unqualified.** No training parameter, reward, inference, physics, CAD or actuator limit was changed this check. This is an evidence and diagnostic iteration, not a new training revision.

## Independent behavior and upper-body improvement

At the saved 960-new-update checkpoint `054_run_150_train_015/candidate.pt` (SHA256 `44395c6782cacb98224ca1dab644e1fec002f9c41f7b1f16e5425877f692c2f3`), matching full 192-robot,20-second nominal055/pushed056 tests include32 trials per command. Running head yaw/pitch combined RMS fell from 51.71 degrees on the exact transferred R34 actor to 3.372/3.370 degrees. Both head axes are within the diagnostic5-degree band for 88.99%/88.94% of frames. At832 updates the corresponding RMS was3.29 degrees; this progress is real but not monotonic or a new qualification gate. Other commands still show larger errors, especially0.65.

Arm offsets are more compact and elbow means more balanced. At832 updates, running elbow means were approximately-33.6/-32.9 degrees versus-43.8/-54.6 originally; shoulder-roll p95 fell from 29.1/35.9 to 15.8/10.9 degrees. Some arm velocity measures improve while right elbow velocity increases. The frame review still shows limited arm swing and sometimes simultaneous forward elbow flexion. Do not call all arm motion natural or solved.

| Command m/s | Nominal signed speed | Pushed signed speed | Matching qualification |
|---|---|---|---|
| 0.00 | -0.000 | -0.000 | Both pass |
| 0.15 | 0.252 | 0.253 | Incomplete: speed, world_forward_speed |
| 0.35 | 0.445 | 0.443 | Incomplete: speed |
| 0.65 | 0.806 | 0.807 | Incomplete: speed, world_forward_speed |
| 1.00 | 1.112 | 1.114 | Incomplete: speed |
| 1.50 | 1.429 | 1.432 | Incomplete: flight, geometric_flight |

All trials survive, but the 1.5 case has 0% nominal and0.0208% pushed genuine flight, below the unchanged1.5% minimum. No running-stage promotion is justified. Historical standing/walking qualifications belong to exact prior actors, not today's candidate. At the report observation 1024 main updates had completed and the next nominal evaluation was active; those newer gates must be read before a status claim. See [evidence](evidence.json), [full upper audit](upper_audit_final.json) and [trends](trends.json).

## Normal-limit controlled physical experiments

Training was paused only at a verified owned saved boundary, its evaluation drained, and one diagnostic simulator ran at a time.128 robots with 16 per intervention, ten simulated seconds, compared the current actor, plus/minus0.05-cycle policy timing, early/late0.30 rad retraction,0.15 rad extension followed by retraction, the preserved R34flight actor038, and a0.35 control. No root support, pose writes or increased limits were used. Interventions were diagnostic only.

Current actor:0 flight,1.439 m/s signed travel,14.328 reward rate. Preserved038:1.976% genuine flight,1.483 m/s,14.357 reward rate. Timing/pulse groups survive but produce no genuine flight and generally lower total reward. The current actor has a loaded foot in99.75% of intended flight samples, averaging265 N on the more loaded foot. Plus0.05 phase improves total rate slightly but still has 0 flight. This is a learned-gait failure; these tests do not establish an actuator ceiling or justify deploying pulses. [Physical results](physical_response/result.json), [cost tradeoffs](physical_cost_tradeoff.json), [full timing/motor audit](physical_timing_and_motors.json).

A second64-robot experiment isolates upper-body coupling. Combining the older flight actor's body outputs with current head/arm outputs preserves3.170% flight, but overspeeds to 1.717 m/s with 0.655 m/s absolute lateral speed and25-degree final heading. Replacing head outputs alone preserves2.279% flight but yields1.834 m/s,1.002 m/s lateral travel and33-degree heading. Head RMS stays3.12/4.06 degrees. Both combinations have 100% survival but fail direction/tracking; neither is deployed or qualified. This demonstrates coexistence of head-forward and some flight under the caps, not a viable combined policy. [Decomposition evidence](hybrid_response/result.json).

CPU fixed-state history intervention on the older running actor reproduces its lower actions within5.8e-6. Substituting only head q/qd/previous-action histories changes lower effective clipped actions by0.626 RMS action units; the right ankle alone changes up to 1.566 RMS units in the head-only decomposition. Head q alone changes lower effective outputs0.333-0.353 RMS. This is evidence of learned input dependence. It does not isolate mechanical head/COM effects or prove that ignoring head feedback is valid. Next use a closed-loop normal-limit control experiment that holds the actual neutral head motion fixed while separating the older actor's head-history input, before considering a training-only robustness/transfer repair. Never spoof production observations. [CPU dependency audit](head_observation_dependency.json).

Exact pose-dependent CAD Jacobians and declared reductions were applied to every saved50 Hz q/qd sample, including directional finite-difference checks. Current actor's highest leg p99 is right ankle motor2 at 74.3% of cap, with 0.1% sampled at/above cap. The preserved flight actor reaches82.9% p99 with 0.3375% sampled at/above cap. These isolated samples are not sustained saturation or proof of a locomotion ceiling; physics substep clipping remains unexcluded. Motor caps remain unchanged.

## Actual PPO, teacher and posture gradients

An exact full-state64-env, four-update diagnostic ran with the original standing/moving teachers and upper module. Full-load proof confirms actor,critic,Adam and exploration equality. Six actual PPO minibatches were passively captured; no gradient hook was changed in behavior. Running PPO parameter-gradient norms17.1-31.2, qualified walking-teacher0.435-0.660, upper auxiliary0.447-0.714. Running/teacher cosine ranges-0.037 to+0.198, positive in four of six; this does not support a claim of consistently opposing teacher gradients. Running/upper cosine is mixed(-0.488 to+0.303). Finite samples before global clipping/Adam cannot settle all future interference. All four pre-update KL and command differences are zero. Diagnostic learned actors are discarded. [Gradient analysis](gradient_analysis.json), [full-load proof](gradient_ppo/full_load_proof.json).

## Current full-CAD frame inspection

Two fresh four-second GUI diagnostics at commanded0.65 and1.5 m/s use the current saved actor, all 140 original compatiblev29 meshes and materials, correctly attached to articulated links. Training joints/inertias/collision proxies/caps are identical to the previously validated visual-only asset. Both clips contain200 distinct physical/rendered frames at 50 FPS, play four simulated seconds in four real seconds, and have zero falls. All400 frames decode successfully; preview plus ten representative timeline frames and closeups were visually inspected. Head,chest,arms,hands,waist/thigh/shin parts and feet remain present. Actual clip speeds0.810/1.443 m/s illustrate the retained-speed overshoot and grounded running. These short diagnostics do not replace or repeat the already fulfilled five ten-second recordings, nor certify full-CAD collision fidelity.

[Frame sheet](full_cad_frames/frame_contact_sheet.png), [current1.5 video](full_cad_frames/locomotion_150.mp4), [current0.65 video](full_cad_frames/locomotion_065.mp4), [decode verification](decode_verification.json), [visual review](frame_visual_review.json).

## Resource use, provenance and resume

Same supervisor PID45868, same checkpoint SHA, preserved in-memory curriculum and no training restart. Pause472.2 seconds; production resumed at 2026-10-05T02:58:49.237428+00:00. This pause is excluded from the reported whole-chain throughput. The paused nominal gate's wrapper time includes it and must not be interpreted as simulator performance. [Pause](diagnostic_pause.json), [resume](diagnostic_resume.json), [owned observation](owned_observation_at_report.json).

Completed main data: 44922 active collection+PPO samples/s, 38715 worker-effective samples/s including startup/exit, 23976 whole-chain samples/s including benchmarks/gates but excluding this pause. Denominators differ. Production uses6144 after its existing moving8192 reserve fallback. Latest6144 completed blocks have about 1.6-1.7 GiB minimum free GPU; latest main updates have zero KL/command mismatch and bounded private-memory growth. Target8192,300 MiB reserve, host guards/fallback and180healthy-second upward ramp remain.47 frozen hashes verify, one actual owned CUDA worker. The optimizer LR is adaptive; the dated1e-5 handoff was an initial saved rate, not a promise that production remained fixed there. Current PPO had adapted as high as1.5e-4, with the existing restart ceiling1e-4. No LR change this check.

Iteration4 optimizations remain unchanged. Their controlled same-count PPO speedup was3.15x (8086 to 25451 samples/s); the prior slowed~9k model had expensive repeated differential math, already corrected. The old five-foot model's locomotion success is separate and is not proof the larger23-axis physical model should converge identically. Current lack of flight is not explained by the former throughput bottleneck alone.

## Decision and next focused checks

Preserve the head/arm improvement, exact current learned state, qualified walking teacher and earlier038 flight actor. Evidence supports a focused upper-input/body-coupling experiment, not rolling back posture, removing feedback, deploying action pulses/hybrid control, relaxing gates or changing arbitrary weights. No runtime fault or safe tested repair is established by these bounded tests, so the same R35 learner continues.

1. Compare newer full matched gates/head/arm telemetry, especially slow speed tracking; preserve measurable improvement and all exact actors.
2. Prioritize closed-loop separation of head-history sensitivity from physical upper-body coupling; also audit the abrupt1.5 m/s walk/run duty/cadence boundary and training command exposure. If the plateau persists, test a narrow training-only continuation against a matched control using current exact full actor/critic/Adam/exploration, including neutral-head direction, flight and retained commands before deployment. No reward-identical waiting as the entire next check.
3. Monitor6144/8192 reserve recovery, full-state preservation,180healthy training ramps and actual motor velocities; never call isolated speed spikes a ceiling.

All diagnostics, logs, unsuccessful interventions and nonfatal Kit shader-cache messages are preserved. Lookup/path normalization misses are recorded in [diagnostic read notes](diagnostic_read_notes.json). All simulator/PPO/GUI probes exited0. Same adaptive three-hour heartbeat remains active. Neither5 m/s nor a relevant actuator ceiling is proven; leave the computer on.

## Final resumed observation

Final owned observation 2026-10-05T03:09:55.081910+00:00: same R35 learner, 1024 completed new main updates, block060 learning at 6144 with 51 observed updates, one actual worker,47 hashes verified, GPU free1688 MiB and host7601 MiB. The first resumed64-update block057 loads exact054 actor/critic/Adam/exploration, has zero max pre-updateKL/action-command difference and stable memory(minGPU1580 MiB,host6929 MiB,private+5 MiB). Matching1024-update nominal058/pushed059 still have 0 true flight and fail0.15/0.35/0.65 speed retention; stopped and1.0 pass. These newer gates do not qualify running. No diagnostic actor/control/source change deployed. Read fresh state before assuming this observation persists.

[Exact resumed learning proof](resume_learning_verification.json), [final owned observation](final_owned_observation.json), [same automation verification](automation_verification.json). History inventory refreshed:35 physical runs,1241 operations,0 differing hashes/read warnings;8 historical missing manifest artifacts remain.
