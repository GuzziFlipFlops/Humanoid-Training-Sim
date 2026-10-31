# Humanoid CAD and training iteration history

Consolidated October 4, 2026 from preserved local runs, frozen sources, checkpoints,
diagnostic reports and CAD archives. This is the starting point for the complete
history, including failed starts and rejected designs. Earlier reports remain
intact; their dated statements that a run is “active” are historical.

**Current recorded state:** R33 qualified prior walking retention continues the
full earlier R32 actor that passed1.5m/s running cases but lost walking. Later
R32 is preserved at1352 updates. The current training-only repair leaves inference,
rewards and all physical/independent gates unchanged. Read the October4 17:37
report and fresh state. Full running-stage and5m/s qualification remain incomplete.
The same adaptive three-hour check saves/pauses/briefs at the verified endpoint;
computer stays on.

| Read next | Contents |
| --- | --- |
| [Physical run evidence](history/PHYSICAL_RUNS.md) | Every R1–R33 run, input lineage, versions, source comparisons, checkpoint hashes, promotions, failures and all saved worker operations. |
| [Older five-foot history](history/OLDER_MODEL_HISTORY.md) | Original corrected-axis walker, guarded running v1–v8, recovery v1–v14 and their measured outcomes. |
| [CAD and export history](history/CAD_AND_EXPORT_HISTORY.md) | Earlier lower-leg/full-robot archives, v29 export and wrist/joint tests, rejected v30 waist/adapters, retained v31/v32 work. |
| [Evidence index](history/EVIDENCE_INDEX.md) | Original reports, diagnostics, controlled tests and CAD validation files. |
| [Older run inventory](history/OLDER_RUNS.md) | Original RSL-RL sessions, guarded launches, smokes, controlled probes, failed and incomplete launches. |
| [Machine inventory](history/RUN_INVENTORY.json) | Exact recorded configuration/state, operation results, source hashes and file paths. |
| [Logging convention](history/README.md) | How every subsequent meaningful iteration must be recorded and how to refresh the inventory. |
| [Documentation verification](history/VERIFICATION.json) | October 4 coverage/link checks, unchanged R29 checkpoint/source hashes and confirmed paused state. |
| [Five execution optimization attempts](diagnostics/optimization_20261004/REPORT.md) | Every measured candidate, capture failure, fidelity replay, actual PPO/resource comparison and R30 continuation proof. |

| [Adaptive check and R31](diagnostics/adaptive_check_20261004_0836/REPORT.md) | Independent R30 progress/retention, tested exact-command rehearsal, full-state continuation and5 m/s extension. |

## How to interpret the records

“Main updates” below are the root state's completed-training count, excluding
startup smokes and capacity benchmarks. Interrupted periodic checkpoints can
contain additional updates not reflected in that count. Resource continuations
can inherit earlier updates, while some early workers repeatedly rolled back;
adding run totals does **not** measure one actor's accumulated learning.

A completed PPO/checkpoint-load smoke proves execution, not locomotion. A policy
is qualified only by matching independent nominal, pushed and earlier-command
retention gates under its recorded verifier. An old pass is not retroactively a
pass under a repaired verifier. Geometry or reward-counterexample tests cannot
certify natural gait, contact pressure, strength or real hardware behavior.

The old five-foot robot is a separate 22-axis, approximately 38 kg asset. The
physical-v29 trainer uses 23 axes and a provisional 34.46334 kg mass estimate.
Its 400-input default-offset policy is not the older 80-input walker. CAD v32
has not been exported into this frozen training model.

<!-- adaptive_check_20261004_1137 -->
## October 4, 11:37 UTC adaptive check: walking qualified; take-off diagnosis

Observed 2026-10-04T12:09:54.454301+00:00. R31 remains running at run_150 with
8,192 environments and 960
completed main updates, plus its separately recorded inherited history.
Exact saved checkpoints have now passed matching nominal/pushed/stopped/prior-speed
gates at 0.65 and 1.0 m/s. The 1.0 checkpoint is
`035_walk_100_train_006/candidate.pt`; nominal actual signed speed was
0.942 m/s, 100% survival and 22/23 alternating
landings. These passes qualify that saved actor; the current running candidate
still fails some slow-speed retention and has no flight.

Latest matching gates: `056_run_150_gate_007_nominal` and
`057_run_150_gate_007_push`, checkpoint `055_run_150_train_006/candidate.pt`.
The 1.5 case moves 1.414 m/s
with 100% survival, 29/30 alternating landings, about 1.64 Hz cadence and zero
flight after 384 running updates. The running stage remains unqualified. Slow tracking is nonmonotonic;
0.15 briefly recovered in gate005 but failed again in gate006. No 5 m/s or
hardware-ceiling claim is supported.

CPU-only new diagnostics reconstruct the exact 400-input actor history from
saved physical traces (maximum action discrepancy 4.8e-06).
Commands/actions match and the actor responds to command-channel interventions.
Saved 50Hz contact forces and actual sole clearances independently confirm that
one foot remains loaded in every intended running flight window. Sampled
pose-dependent CAD motor speeds and declared reductions show no sustained leg
velocity ceiling; 10Hz sampling cannot exclude brief substep clipping.
The actual runtime action clamp is +/-3; the audited running trace has 31.6%
of raw outputs outside it, concentrated in some hip/ankle channels. This is a
diagnostic lead, not grounds for enlarging physical limits.

11 new controlled reward counterexamples pass. An ideal run
beats fast walking, sliding, backward/in-place stepping and same-foot hopping
at 1.5 through 5 m/s in the fixtures. The ideal-run/fast-walk gap is only about
0.09-0.11 reward-rate units under held posture/costs. Reducing a loaded foot's
force from 338 to 18 N does not change the binary gait/flight objective while
the loaded-contact classification remains the same. A continuous phase-conditioned
force signal is an undeployed hypothesis requiring a controlled physical test.
Research and diagnostics do not prove convergence or hardware feasibility.

Preserve R31 for now: it newly qualified both walking stages since the preceding
check; standing, heading, step alternation and resources remain healthy, while
early running retention is still changing. Do not erase that actor or alter
live frozen sources solely because the timer fired. The next check must be
critical about persistent zero flight: if newer independent gates show no
flight/retention progress, perform a bounded take-off/phase intervention at an
owned checkpoint boundary, compare physical force/clearance and objective effects,
then implement a justified focused repair with fresh frozen sources and smoke
tests. Do not spend another check merely changing arbitrary weights.

Completed-main throughput: 40,206 active
PPO samples/s, 32,426 including
worker startup/exit, and 18,263 over
the whole R31 chain including startup, capacity and independent gates. Last
64-update block: 40,276/32,528.
All completed main blocks used 8192 without recovery; minimum free VRAM
512 MiB, host available 9279 MiB. Last block private memory grew
4 MiB; pre-update KL and command
mismatch stay zero. All 46 frozen hashes and one actual CUDA worker verify.
Keep target8192, reserve300, fallback/180s healthy ramp and selected iteration4.

Report and reproducible evidence:
`E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_1137/REPORT.md`.
The same adaptive three-hour heartbeat remains active until qualified actual
5 m/s or a reproduced relevant velocity ceiling; then pause and brief. Leave
the computer on and preserve unsaved user work.
<!-- end_adaptive_check_20261004_1137 -->

<!-- adaptive_check_20261004_1437 -->
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
<!-- end_adaptive_check_20261004_1437 -->

## Physical-v29 runs, October 1–4

### October 4: adaptive three-hour checks and updated 5 m/s endpoint

The user reauthorized the recurring check after the measured execution
optimizations. The existing heartbeat was updated in place to ACTIVE every
three hours; its historical ID and thread were preserved, with no duplicate.
Each check must update its own dated handoff and next focused tests while
retaining the durable constraints in
[the operating plan](automation/THREE_HOUR_CHECK.md) and
[next-check record](automation/NEXT_CHECK.json). It preserves an improving actor,
completes nominal/pushed/stopped/prior-speed gates and proceeds toward running.
Stalled progress requires a reproduced diagnosis and tested repair, not arbitrary
weight changes or time-based policy resets.

The new endpoint is independently verified **actual 5 m/s**, or a reproduced
persistent locomotion-relevant actuator velocity ceiling. Then save the full
checkpoint and evidence, pause owned training and this heartbeat, and brief the
user. This pause-and-brief endpoint replaces the historical shutdown instruction.
The current frozen R30 curriculum still ends at 2.5 m/s; the checks are instructed
to implement/test a fresh frozen extension toward 5 m/s at an appropriate saved
boundary, preserving all previous gates and unchanged motor limits. That
extension is planned, not deployed by this schedule-only change.

[Saved schedule verification](diagnostics/automation_3h_20261004/VERIFICATION.json)
confirms ACTIVE, three-hour cadence, preserved thread/notification preference and
one matching active humanoid heartbeat. Setup observed R30 at walk_035 after64
main updates; the second block began at8192 and automatic fallback subsequently
selected6144. No reward, actor, optimizer, physical source or running frozen
configuration was changed. Current walking remains unqualified. Original and
updated automation snapshots are preserved in the same diagnostic folder.

### October 4: five measured execution optimization attempts and R30

The user requested three to five optimization iterations before restarting.
Exactly five source variants were created beside the trainer. The unchanged
baseline and repeated verification runs are not extra optimization iterations.
[Full report and evidence](diagnostics/optimization_20261004/REPORT.md).

| Attempt | Change and actual measured inference rate at 4,096 environments | Outcome |
| --- | --- | --- |
| 1 | Prepared immutable rod geometry and shared trigonometry: 12,707 samples/s versus 11,505 baseline. | Bitwise CPU/CUDA/physical parity; valid, superseded. |
| 2 | Bounded owned CUDA graphs for fresh differential states: 22,205 samples/s. | Exact parity and output ownership; valid, superseded. |
| 3 | Captured full PD and coupled hardware actuation. | CUDA rejected Python-list-to-device joint-index conversion. Failed source/logs preserved. |
| 4 | Prepared CUDA joint indices outside capture and captured full actuation: 37,741 samples/s. | Selected; original environment/reward/gate bytes unchanged. |
| 5 | Captured target feasibility and post-physics validity loops: 38,305 samples/s. | Valid parity, but 1.5% gain is within short-run variation; extra environment changes rejected. |

All 400 frames of a 72-robot, six-command replay with resets and pushes are
bitwise identical to the frozen baseline in every saved joint/action/contact/
COM/reward/observation field. Coupled limits, braking, invalid poses, optional
control modes, changed gains and tensor ownership were checked separately.
No CAD layout, mass, contact, time step, self-collision, joint limits, rewards,
PPO objective or qualification threshold changed.

Actual same-count PPO improved from 8,086 to 25,451 samples/s (3.15×); 6,144
measured 31,858. These count collection plus learning, excluding process startup
and independent gates. Eight-update checks retained full actor/critic/Adam/
adaptive LR/standing teacher and had zero sampled pre-update KL. 8,192 was
rejected when free VRAM reached 214 MiB. All benchmark-updated actors are
diagnostic and discarded; main learning resumes the exact R29 full checkpoint.

[R30](runs/locomotion_20261004_r30_exact_graph_execution/state.json) uses a new
frozen source. Its explicit migration proof permits only the three pinned
execution files; eleven mutation/false-qualification counterexamples were
rejected, and the actual freeze path was tested. Fresh physical preflight and
full PPO/load smoke passed. Complete capacity tests and fresh nominal/pushed/
stopped/earlier-speed gates precede main continuation/promotion. R29, its frozen
source and checkpoints remain preserved; its former paused state/pointer are
backed up in the diagnostic. The four-hour automation stays paused.

Each run name below links to its untouched saved state. The detailed inventory
links its frozen implementation and every saved operation, including failures.

| Revision / recorded main updates | Reason and implemented change | Recorded outcome and evidence |
| --- | --- | --- |
| [R01](runs/locomotion_20261001/state.json) · 0 | Initial free-base stand → walk → run chain, estimated mass/contact asset, explicit hardware and CAD differential actuation, four-frame IK-residual policy. | Preflight failed with `Logger.writer` missing. No learned or qualified stage. [Initial design/research](RESEARCH.md). |
| [R02](runs/locomotion_20261001_r2/state.json) · 150 | Repaired initial logger startup and ran the same initial curriculum with memory fallback. | Nominal standing failed. VRAM reserve failures and then a Windows sharing violation on `progress.json` stopped logging. Interrupted block had additional periodic saves; root count is not their full age. [Restart evidence](RESUME_20261001.md). |
| [R03](runs/locomotion_20261001_r3/state.json) · 0 | Added bounded retry for Windows status-file replacement; restored provenance on a copy of the interrupted periodic checkpoint without changing actor/critic tensors. Restart requested 4,096 environments. | Superseded before a completed main block when the user requested 8,192. Three status-writer tests passed; this is not a policy pass. [Restart](RESUME_20261001.md). |
| [R04](runs/locomotion_20261001_r4_8192/state.json) · 550 | Requested 8,192 environments and shortened workers to 25-update blocks to contain memory growth. | Standing qualified under the then-current gates; walking did not. Inspected 0.35 m/s rollout moved only 0.0057 m/s with zero completed steps. Stationary posture/contact terms and broad velocity tracking paid rocking. [Count change](SWITCH_8192_20261001.md), [reward audit](REWARD_REPAIR_20261001.md). |
| [R05](runs/locomotion_20261001_r5_measured_steps/state.json) · 0 | `measured_steps_v2`: signed progress, actual unloaded swing, bilateral physical landing requirements, loaded-slip attenuation; added a 0.15 m/s entry stage. Repaired standing normalizer's previously unused command/phase statistics on a checkpoint copy. | Physical preflight, PPO/load smokes and capacity work executed, but 8,192 crossed the 300 MiB reserve and capacity selection failed. No main learned walker. [Repair/tests](REWARD_REPAIR_20261001.md). |
| [R06](runs/locomotion_20261001_r6_measured_steps/state.json) · 25 | Same v2 worker/rewards; rejected unsafe capacity results and selected eligible 6,144 fallback. Preserved exact standing actor outputs, fresh critic/Adam after reward change. | Fresh standing gates passed; walking did not. Resource failures ended this run. [Repair/provenance](REWARD_REPAIR_20261001.md). |
| [R07](runs/locomotion_20261001_r7_fusion_memory_recovery/state.json) · 0 | Attempted resource recovery from R6 while Fusion used memory. | Failed immediately because an escaped validation-directory path could not locate `preflight_07/result.json`. No worker/main learning; root traceback retained. |
| [R08](runs/locomotion_20261001_r8_fusion_memory_recovery/state.json) · 50 | Corrected recovery path and resumed R6's compatible learning state. | Standing promotion retained; walking remained unqualified. Superseded for automatic resource management. [R8 launcher](launch_r8.py). |
| [R09](runs/locomotion_20261001_r9_auto_vram/state.json) · 50 | Automatic upward environment ramp after 180 continuous healthy training seconds, preserving the 300 MiB reserve and 8,192 target. | Learning resumed from R8; no walking qualification. Superseded for duration-aware blocks. [Memory policy](AUTOMATIC_VRAM_20261001.md). |
| [R10](runs/locomotion_20261001_r10_auto_vram_duration/state.json) · 409 | Duration-aware blocks allowed the three-minute headroom test to accumulate; resource resumptions retained optimizer state. | Fourteen nominal walking gates failed. Last 0.15 case: 0.0080 m/s, zero bilateral steps. Repeated failed-retention rollback meant many blocks restarted the standing prior rather than accumulating one walker. Forward-half rocking could still earn credit. [Measured diagnosis](NET_TRAVEL_REPAIR_20261001.md). |
| [R11](runs/locomotion_20261001_r11_net_travel/state.json) · 192 | `net_travel_v3`: fixed-heading travel frontier, finite swing discovery, standing rehearsal, lower initial learning rate and ≥64-update blocks. Continued the candidate across failed gates while keeping trusted qualification separate. | Standing prior requalified; walker failed. At 128 updates a nominal rollout had zero survival/steps, 0.145 m/s forward but 0.726 m/s lateral motion. Third block preserved before repair. [Repair](NET_TRAVEL_REPAIR_20261001.md), [later diagnosis](README.md). |
| [R12](runs/locomotion_20261002_r12_balance_events/state.json) · 0 | Prepared `balance_events_running_v4`, corrected fall-event units and reference feasibility; physical landing verification replaced simpler step counting. | Setup/preflight-only branch. No completed main learning or qualification. Saved validation remains; final verified replacement is R13. [Balance chronology](README.md). |
| [R13](runs/locomotion_20261002_r13_balance_verified/state.json) · 64 | Eight-unit dt-independent fall event, full posture guidance, finite unloading help, bounded controller reference, `physical_landing_v2` and declared running-flight extension. | Standing prior passed. Walking still had zero survival and steps; 0.139 forward / 0.383 lateral m/s. Axis audit found all 23 frames/limits consistent; reference soft-limit violations were repaired without enlarging hardware travel. Host commit grew to 67 GiB. [Joint/resource evidence](README.md). |
| [R14](runs/locomotion_20261002_r14_memory_continuation/state.json) · 548 | Resource-only continuation from R13: complete-update checkpoint recycling, host-RAM guard and retained adaptive LR/critic/Adam. Periodic GC contained exhaustion but did not identify the leak. | Walker survived but remained nearly stationary: nominal 0.03505 m/s, zero left and 0.14 right steps/robot. Standing support failed. Actor continued across resource workers. [Four-hour audit](FOUR_HOUR_AUDIT_20261002.md). |
| [R15](runs/locomotion_20261002_r15_motion_discovery/state.json) · 691 | `motion_deficit_discovery_v5`: finite lift/placement discovery banks and a movement-deficit cost to offset stationary moving-command income. Compatible actor retained; critic/Adam refreshed for changed return. | Survival improved, but rapid asymmetric contact cycles paid recurring discovery and the old verifier double-counted hop pairs. Last nominal 0.15 case recorded 84.89 left / 14.97 right landings and 9.1% flight; this was rejected gait. [First audit](FOUR_HOUR_AUDIT_20261002.md), [counterexamples](SECOND_FOUR_HOUR_AUDIT_20261002.md). |
| [R16](runs/locomotion_20261002_r16_gait_integrity/state.json) · 0 | Prepared `gait_integrity_v6`: one discovery bank per foot per episode, touchdown-frozen travel, paired-impact delay, walking flight rejection and raw independent landing order. | Four-update smoke completed; review found stopped-anchor semantics needed correction before main training. Intermediate validation preserved; no qualified chain. [Audit](SECOND_FOUR_HOUR_AUDIT_20261002.md). |
| [R17](runs/locomotion_20261002_r17_gait_integrity/state.json) · 115 | Final v6 repair with stop anchor at the stopping location; `physical_landing_v3` independently checked alternating order and walking flight. Added synchronized 50 Hz actions/joints/contact telemetry. | Trusted standing passed, retained walker did not. Post-block nominal standing survival 0%; walking 0.00527 m/s with zero steps. Host recycling preserved Adam/LR, but standing rehearsal alone did not preserve the actor. [119-test repair and gates](SECOND_FOUR_HOUR_AUDIT_20261002.md). |
| [R18](runs/locomotion_20261002_r18_standing_retention/state.json) · 0 | Added training-only stopped-command mean-action regularization toward the independently verified standing teacher. | Preflight failed on missing optional legacy `motion_sha256` metadata; no training block. Compatibility handling was corrected in R19. [Failure/teacher contract](SECOND_FOUR_HOUR_AUDIT_20261002.md). |
| [R19](runs/locomotion_20261002_r19_standing_retention/state.json) · 128 | Repaired teacher metadata checks; retained R17 actor/critic/Adam/LR because reward was unchanged. Teacher gradients only, weight 0.25; no teacher controller in inference or promotion. | Fresh prior standing passed; candidate walking/retention failed. This did not establish a working IK-residual walker. [Retention tests and actual PPO](SECOND_FOUR_HOUR_AUDIT_20261002.md). |
| [R20](runs/locomotion_20261002_r20_unitree_g1/state.json) · 68 | Pinned public G1 reward bodies/weights after H1/G1/H2 comparison; new five-frame nominal-offset action interface, no moving IK reference. New actor required for incompatible contract. | Aggregate regular reward was negative; deterministic replay first breached a waist motor-angle limit at 0.08 s and fell. All reviewed frames showed no learned gait. Later nominal-reference aliasing also invalidated implementation assumptions. [Port/source/visuals](UNITREE_REWARD_PORT_20261002.md), [fault diagnosis](UNITREE_LIFETIME_REPAIR_20261002.md). |
| [R21](runs/locomotion_20261002_r21_unitree_lifetime/state.json) · 0 | `unitree_g1_lifetime_v2`: Unitree RL Gym regular-return floor, then H2-style physical-termination term after flooring; original G1 core retained. | Validation-only branch. Smoked actor still fell; reset review found shared nominal-pose storage. Reward arithmetic tests did not validate the whole adaptation. [Lifetime audit](UNITREE_LIFETIME_REPAIR_20261002.md). |
| [R22](runs/locomotion_20261002_r22_unitree_nominal/state.json) · 790 | Cloned per-environment reference storage and protected immutable action origin. Reproduced old reset drift up to 0.63665 rad; corrected contract v2 rejects affected old actors. Fresh actor/critic. | Zero nominal drift was verified, but standing never qualified. Later 750-update gate regressed versus the 150-update actor. Reward floor could erase useful posture gradients. [Reset bug/tests/visuals](UNITREE_NOMINAL_REPAIR_20261002.md), [stage diagnosis](STAGE_REWARD_REPAIR_20261002.md). |
| [R23](runs/locomotion_20261002_r23_standing_curriculum/state.json) · 960 | Actor-only warm start from R22's better 150-update checkpoint; stage-specific positive posture feedback, bounded penalties without aggregate flooring, conservative exploration and horizontal COM support frame. | Standing qualified at 512 updates; the next 448 walking updates regressed without qualified steps. Sole-offset audit later showed an ankle-origin support rectangle could falsely reject an otherwise supported trace. [Stage repair](STAGE_REWARD_REPAIR_20261002.md), [actual sole audit](diagnostics/support_geometry_audit_20261003/REPORT.md), [regression](diagnostics/walk_regression_20261003/REPORT.md). |
| [R24](runs/locomotion_20261003_r24_phase_feedback_retention/state.json) · 0 | Prepared phase/lift feedback before translation and stopped-command retention from R23's qualified standing actor. | Physical/PPO diagnostic branch reproduced reset-related policy/critic history aliasing. Preserved preflight evidence; no main training qualification. [Regression/ownership investigation](diagnostics/walk_regression_20261003/REPORT.md). |
| [R25](runs/locomotion_20261003_r25_owned_observations/state.json) · 1,471 | Owned policy/critic observation snapshots fixed reset aliasing; CPU and real CUDA smokes showed zero pre-update KL. Phase feedback and latest-frame stopped retention enabled learning from the qualified standing actor. | Real alternating locomotion developed. Old nominal/pushed/retention gates passed stand and 0.35/0.65/1.0 m/s; run 1.5 began but did not qualify. Full-population audit reproduced turning/circling accepted under a zero-turn command. **Old passes are not current straight-walking qualification.** [Progress](diagnostics/r25_progress_20261003_1519/REPORT.md), [direction defect](diagnostics/r25_run_review_20261003_1919/REPORT.md). |
| [R26](runs/locomotion_20261003_r26_heading_commands/state.json) · 0 | Retained R25 candidate155 actor/exploration; added reset-heading feedback in the existing yaw command and independent fixed-frame heading/forward/lateral/net-travel guards. | Startup caught the early RSL logger-save `writer` fault. Failed preflight preserved; repaired/retested replacement is R27. [Direction repair](diagnostics/r25_run_review_20261003_1919/REPORT.md). |
| [R27](runs/locomotion_20261003_r27_heading_commands_checked/state.json) · 192 | Logger compatibility fixed; same heading-command repair, consistent command through action/reward, fresh critic/Adam and fresh qualification. Original gait weights retained. | Standing/alternation retained, but walking direction regressed. A 72-robot left/zero/right intervention measured weak steering, including right turn under left command. Commands and pre-update KL were consistent. Independent CAD transient-tensor cache leak reproduced. [Controlled diagnosis](diagnostics/r27_review_20261003_2319/REPORT.md). |
| [R28](runs/locomotion_20261004_r28_heading_alignment_cache_fixed/state.json) · 0 | Fixed transient-tensor cache without changing differential outputs; 6,144 CPU comparisons and 72-robot/350-tick replay were bitwise identical. Added declared local bounded reset-heading reward (weight 1, width 0.35 rad), retaining R25 candidate155 actor/exploration. | Matching full standing gates passed. 4,096/6,144 capacity completed with stable memory; 8,192 recycled after only 5/24 updates because host RAM crossed reserve. Supervisor wrongly treated the partial capacity result as eligible. Main walking did not begin here. [Cache/reward/capacity evidence](diagnostics/r27_review_20261003_2319/REPORT.md). |
| [R29](runs/locomotion_20261004_r29_host_capacity_checked/state.json) · 256 | Resource-only repair: require full capacity budget with no recycling and healthy host/GPU reserves; host recoveries lower a tier without inflating VRAM estimates, reset the 180 s ramp hold and preserve full checkpoint. All 41 authored worker runtime files retained from R28. | Preserved matching standing gates. Four 64-update walking blocks at 6,144 had stable memory. Latest full 192-update gates fail speed/heading despite real steps. User paused after checkpoint256 completed; no current 1 m/s/run promotion. Performance assessment only; no production optimization deployed. [Pause/profile](diagnostics/r29_performance_20261004/REPORT.md). |

## What the iterations established

The early failures were not one problem solved by a larger reward weight. Saved
counterexamples found stationary reward, forward-half rocking, repeated discovery
payout and landing-counter errors. Separate implementation reproductions found
nominal-pose and observation-history aliasing. Separate resource evidence found
Windows logging failures, transient-tensor cache growth and incorrect acceptance
of a host-recycled capacity benchmark. The linked tests state which defect they
isolate; they do not prove every historical failure had that cause.

R25 established that this frozen robot can develop sustained alternating motion
through the older 1 m/s gates. The direction audit then established that those
gates could accept turning when straight travel was requested. R27–R29 therefore
retain a useful learned actor while requiring stronger direction verification;
they do not justify calling the current entire chain complete.

The exact sole-support audit is also retained. R23's sampled 448-update trace was
93.4% outside the old ankle-origin rectangle but 0% outside the transformed sole
footprints/inset footprints/near-ground hull. This exposed a support-frame
diagnostic mismatch, not a contact-pressure certificate or permission to ignore
independent qualification. R23 also had a separate genuine walking regression.

## Latest performance assessment, not an implemented revision

The paused same-physics 4,096-robot short profile found 18 differential
state/Jacobian calls per control tick, consuming about 55.6% of its sampled tick;
physics stepping was about 16.2%. Nested actuator/CAD times must not be added.
The asset already uses 24 box colliders/visuals and 23 revolute joints, with no
camera, finger joints or passive closed-chain constraint solver in training.

Four 6,144 production blocks averaged approximately 9,368 active-PPO samples/s,
8,502 worker-effective samples/s and 4,212 samples/s over the whole observed run
including starts, maintenance and eight gates. These are different denominators.
The archived older walker measured approximately 55,518 active-PPO samples/s
with a different 22-axis implicit-PD model. A short microbenchmark is not a
guaranteed speedup or a comparable public Unitree hardware benchmark.

Exact batching/fusion of CAD calculations and measured data/audit overhead
reduction are proposed next work. Changing collision filters, differential
physics or solver quality would be a separate physical-model iteration requiring
fresh checks. None of those proposals is recorded as implemented.

## Documentation and handoff iteration

October 4: this retrospective consolidates previously scattered notes, records
all 29 physical run revisions and 757 saved worker-operation folders, catalogues
97 older guarded/diagnostic launches and 61 original RSL-RL sessions, and indexes
31 CAD/evidence archive groups. These counts describe this inventory snapshot;
the refresh script computes subsequent counts. Some old source READMEs inherit
earlier titles and some launches have stale states or no final result. Those
gaps are explicit rather than filled with invented outcomes.

The inventory also identifies eight early missing supervisor copies (R1–R8
record the SHA but lack the corresponding source copy in their run tree).
Surviving declared files have no differing manifest hash; current R29 verifies
all 43. This preservation gap is documented, not repaired by substituting a
later supervisor or altering an old manifest.

New narrative entries must record trigger, diagnosis, actual changes, unchanged
physical/interface constraints, actor/optimizer lineage, tests, independent
behavioral outcomes, failure/stop reason and next decision. Append them here
before handing off a new revision. Refresh generated evidence with
`history/record_inventory.py`; never edit frozen runs to make history look clean.

<!-- adaptive_check_20261004_0836 -->
## October 4, 08:36 UTC adaptive check: R31 retained-command rehearsal

Current continuation: `locomotion_20261004_r31_exact_command_rehearsal`,
running, walk_065. Read fresh ACTIVE_RUN.json/state.
R30 independently qualified0.35, improved straight alternating0.65, but late
0.15 retention repeatedly oversped at0.27–0.30. It is preserved at1,472 main
updates, full checkpoint078, SHA8a62733db47184082e9a1dcf5459893c9285f10018feb5af3a0b3f86f202e31d.
R31 retains that full actor/critic/Adam/exploration and standing teacher. The
existing restart learning-rate cap remains1e-4. Training-only exact prior/frontier
command rehearsal gives0.15 about6.71 times more exposure at the0.65 stage;
all reward weights, inference/history/action/reset behavior, physical assets,
CAD Jacobians and motor limits remain. Iteration4 execution stays selected.

Current0.65,1.0 and running qualification is incomplete; prior0.35 remains tied
to its original exact trusted checkpoint. No threshold is relaxed. The frozen
speed list now extends through3/3.5/4/4.5/5 m/s with32 trials per command and an
additional measured-forward-speed requirement at5. The entire128-robot,
20-second same-actor nominal baseline and all saved synchronized arrays match
R30 bit for bit. CPU command/provenance/running counterfactuals and real CUDA
physics/PPO/full-checkpoint-load checks passed; these are not learned qualification.

Latest R30 main6144 throughput was about35.7k active PPO,28.7k worker-effective
and15.3k full-chain samples/s, with stable memory. R31 stopped-command capacity
completed at8192:41.6k active PPO,659 MiB minimum GPU free and9439 MiB minimum
host available. Moving training still must obey reserve300, host guards,
fallback and180 continuously healthy seconds for an upward ramp. Preserve unsaved
Fusion work. One actual CUDA worker; no stereo.

Report/proof/tests: `E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_0836/REPORT.md`.
Complete local chronology: `physical_v29/training/ITERATION_HISTORY.md`.
Next focused tests: slow/current-speed retention, moving memory/throughput,
then running gait and motor velocity diagnostics. Keep improving actors.
The same adaptive heartbeat remains active every3 hours, GPT-6.1 Sol/max;
actual5 m/s qualification or a reproduced relevant velocity ceiling ends it
with evidence saved, training/checks paused and a briefing. Computer stays on.

R31's latest completed main block has 64 updates at8192 environments, 40,595 active PPO and32,435 worker-effective samples/s. Minimum free VRAM531MiB; host available9497MiB. Pre-update KL and action/command mismatch remain zero.

Latest post-rehearsal nominal test (013_walk_065_gate_003_nominal): retained0.15 now0.268m/s versus0.277 baseline; current0.65 now0.658m/s versus0.732. Survival remains100%. Slow tracking still fails: nominal speed fell from0.277 to0.258 at64 updates, then rose back to0.268 at128. Current-speed tracking continues improving. Keep the actor and require matching pushed/retention gates; the slow-control issue is not solved.

Preparation failures are retained in the dated report: exact-newline restoration,
test fixture dtype/source alignment, and harmless drain-monitor folder lookup.
No production source was edited in place and no checkpoint was discarded.
Historical R30 partial35-update recovery remains in its original file and is
counted in measured samples, separate from its main-update state count.
<!-- end_adaptive_check_20261004_0836 -->

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

<!-- adaptive_check_20261004_1737 -->
## October4 17:37 UTC: reproduced flight regression; R33 qualified prior retention

R32 is preserved at1352 updates. Its269-update checkpoint028 passed the1.5m/s
running case in matching nominal/pushed tests (1.539m/s;2.0833%/1.9236% real
geometric flight), but earlier walking failed, so no whole-stage qualification.
Later flight regressed to zero. A96-robot physical comparison reproduced this;
all survived. The standing prior had no gradient on moving prior commands.
R33 continues the exact028 actor/exploration/critic/Adam, with bounded training-only
qualified prior-speed teacher gradients; inference, reward v7, geometric gate v6,
all physics/motor constraints and iteration4 are unchanged. Teacher never controls
inference.22 CPU plus9 migration checks, real PPO/load and full192-robot20-second
bitwise replay pass. A failed diagnostic Adam-device comparison is preserved and
fixed only in the new load audit. Read [the dated report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_1737/REPORT.md),
proofs and fresh ACTIVE_RUN/state. Observed2026-10-04T18:27:46.192452+00:00: running run_150,
count8192, new main128;
46 frozen hashes verify. Fresh baseline teacher requalification complete:True.
After 128 new updates, matching nominal 015_run_150_gate_003_nominal and pushed 016_run_150_gate_003_push remain unqualified. Running signed speed is 1.536/1.537m/s; true flight is 0.0000%/0.1250%, against the unchanged1.5% minimum. Earlier commands passing both: 0.0, 0.15, 1.0. Earlier commands still failing: 0.35, 0.65. Judge the repair against both retained tracking and geometric flight, not either alone. Compare subsequent complete pairs before attributing a tradeoff to the new gradient, and preserve both the selected flight actor and qualified walking teacher.
No5m/s or relevant velocity-ceiling endpoint is proven. Same three-hour checks
remain active; endpoint saves/pauses/briefs with the computer on.
<!-- /adaptive_check_20261004_1737 -->

<!-- body_visual_review_20261004 -->
## October 4: user-requested body/frame review

Decoded all 2,000 walking-video frames and inspected five frames per speed.
The recorder used 24 provisional box visuals and no CAD meshes; all 140 source
visual meshes remain present. This predated the October 4 execution optimization.
Prepared a separate playback URDF restoring those visuals with exact unchanged
training joints, collision and inertial elements; conversion/rendering is not
claimed. The full collision/body-fidelity limitation remains. Recorded head pose
and deep knee flexion need separate gait review. An initial CPU-only array-label
error is preserved and corrected using actual Isaac joint order. No training,
CAD source, original videos, rewards, motor limits or policies were changed.
Read [video_body_review_20261004/REPORT.md](E:/Aditya/robot_sim/physical_v29/training/diagnostics/video_body_review_20261004/REPORT.md) and audit.json.


## October 4, 2026, 20:39 UTC check - R34 bilateral swing unloading

R33 is preserved at 1401 new main updates. All stopped/prior walking commands through 1.0 recovered, but 1.5 m/s genuine flight plateaued at zero. CPU and actual-physics tests reproduced a v7 load-redistribution incentive during intended two-foot flight; v8 exponentiates the summed swing loads with the existing weight and scale. Single-foot swing and 600 standing/walking fixtures are bitwise unchanged. One real PPO parameter-gradient sample did not show running/prior opposition; walking retention remains. R34 retains exact checkpoint021 mean/exploration, fresh critic/Adam for the changed objective and mandatory fresh qualification. Two runtime files changed, 42 protected authored files identical; real PPO/full reload and all 192-robot/20-second replay arrays/cases match. Normal-limit probes reproduce flight; no helper pulse is deployed. Target8192/reserve300/180s healthy ramps and all gates remain. See E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_2039/REPORT.md and replacement_status.json. No running-stage/5m/s/velocity-ceiling completion inferred. The separate video-body audit documents box-only visuals and simplified collision-fidelity limits.


Verified R34 follow-up at2026-10-04T21:39 UTC: fresh full nominal/pushed/stopped/prior gates requalify standing and walking through1.0m/s on the exact transferred baseline. Running PPO resumed at8192; first43 observed updates have exact actor/critic/Adam/exploration load, zero maximum pre-update KL/command mismatch,20 prior-retention batches/update,37.73k active samples/s,minGPU370MiB,host5655MiB,private growth11MiB,Torch allocation growth1MiB. These are partial main updates, not whole-chain throughput or running qualification. The same three-hour heartbeat is verified active with updated diagnosis and next focus; read adaptive_check_20261004_2039/REPORT.md, replacement_status.json and new_main_evidence.json.


## October 4, 2026, 22:45 UTC user status review

R34 remains run_150 after449 new main updates, selected6144 after saved8192
VRAM fallback. Exact checkpoint038 now passes the running case in matching
nominal039/pushed040: signed1.476/1.477m/s, true flight1.8854%/1.7569%,
100% survival and all running checks. Complete stage remains unqualified:
retained0.15/0.35 overspeed at0.262/0.455 nominal and0.263/0.451 pushed.
Stopped,0.65,1.0 pass. Parameters unchanged because this is fresh independent
running progress; preserve this actor and compare simultaneous retention/flight
in subsequent full pairs before a controlled repair. Last6144 block gives
35.64k active/30.35k worker-effective samples/s; minimumGPU1036MiB,host4764MiB,
zero pre-update KL and command mismatch. CPU-only status evidence/report:
E:/Aditya/robot_sim/physical_v29/training/diagnostics/status_review_20261004_2238/REPORT.md.


## October 4: next-check full-body GUI recording requested

Queued one-time priority in the same adaptive three-hour heartbeat: temporarily
pause/drain training, render all compatible original CAD visuals, record ten
seconds of normal-time motion per speed0.15/0.35/0.65/1.0/1.5 plus newly demonstrated
higher speeds, decode/review frames, deliver MP4s and resume the same owned run.
Prepared visual asset still requires actual conversion/rendered alignment QA.
No training pause, simulator launch or new video occurred while scheduling.
Workflow: physical_v29/training/automation/NEXT_GUI_RECORDING.md; pending state
in NEXT_CHECK.json. Remove the one-time prompt priority only after checked video
delivery and documented resume. Training sources and all gates remain unchanged.


## October4 23:40UTC check - full CAD GUI capture and exact resume

Five actual GUI ten-second normal-time clips at0.15/0.35/0.65/1.0/1.5m/s are complete and delivered. All140 frozen-v29 original CAD visual meshes/source colors restored on matching links with unchanged physics/joints/caps. All2500 distinct encoded frames decode;25 timeline and9 takeoff-neighbor frames inspected, full body remains attached,0falls. Walking baseline and preserved checkpoint038 running-case actor are explicitly separate. Tilted head/crouched/asymmetric pose remains actual policy output; full-CAD collision fidelity is still unqualified. Rejected white/band and instance-binding previews preserved. Owned R34 supervisor14864 drained atcheckpoint065/1025updates, resumed exact state after1458.4s; one CUDA worker and46hashes verify. Latest gates066/067 retain all prior walks but running flight0%, signed1.457/1.460. New CPU three-actor phase/clipped-target comparison reconstructs actions within4.77e-6; loaded leg persists in intended flight. Heavy clipping also exists in successful actor, so no causal diagnosis or cap change. No training runtime/reward/policy update. Next focused normal-limit full-objective/phase-target and actual PPO/prior gradient experiments if plateau continues; no another waiting-only check. Same adaptive three-hour heartbeat active, fulfilled video priority removed; no5m/s/ceiling endpoint. Report/QA/checkpoint/resume/CPU evidence: E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_2340/REPORT.md. Videos: training/videos/full_body_gui_20261004_2340.


Final post-resume evidence at2026-10-05T00:16:20.878561+00:00: same R34 remains run_150 at1089 completed main updates; block068 completed64 actual PPO updates at42737 active/36632 worker-effective samples/s, minimumGPU1017MiB andhost5721MiB, all pre-update KL/command mismatches0. Matching069nominal/070pushed on exact068 achieve signed1.479/1.479m/s,100% survival,29/30 alternating landings, but geometric flight0%/0.00694%, below1.5%. Retained0.35 narrowly fails nominal speed at0.449m/s; all pushed priors pass. Thus the earlier065 all-prior pass did not establish sustained retention. Current block071 runs6144 with one actual CUDA worker and46hashes verified. Read resume_learning_verification.json and final_owned_observation.json. No new source/reward change; preserved038 flight actor and qualified walking baseline remain. CPU history inventory:34physical runs,97legacy launches,61original RSL sessions,31CAD groups,1164operations,8historical missing manifest artifacts,0differing hashes,0read warnings. Three-hour heartbeat remains active with completed video priority removed and controlled diagnosis priorities.


## October 04, 2026 21:11 EDT: R35 head/arm refinement, exact learned-state continuation

User requests natural arms/head-forward while preserving ongoing running. R34 is preserved at1473 completed main updates and exact086 full checkpoint6611cb45edce83bf91abe42e971f8413716f77001d8732b7ff07c49c22f85702. R35 retains actor/critic/Adam/exploration and continues1.5 stage; no from-scratch or actor-only reset. Training-only bounded upper_output_posture_v1 changes only eight final output rows; prior lower derivative identical, upper copying excluded. Original rewardv8/verifierv6/actions/history/commands/physics/CAD/motor limits and iteration4 unchanged.33 CPU tests; actual8-updatePPO/full-state reload;192-robot20s pre-learning replay39 arrays/cases bitwise; matched control and refinement both show small-batch speed drift, refinement narrowly worsens0.65, all survive. Posture remains unqualified and physical coupling/normalPPO can affect locomotion. Exact original seed, never smoke actor, deployed.47 source hashes include a generated metadata supplement; frozen launcher unchanged, canonical future inventory fixed. Preparation/path/metadata guard failures preserved. One actual worker, resource guards and mandatory matching gates remain. First main evidence:1 block(s), latest10 observed updates, exact full load, KL/command mismatch0; no posture/running qualification. Report:E:\Aditya\robot_sim\physical_v29\training\diagnostics\upper_body_refinement_20261004\REPORT.md. Same3h heartbeat now tracks per-command upper posture/frame QA and simultaneous flight/retention.


Verified actual main continuation at 2026-10-05T01:12:13.258378+00:00: 8192 environments, 26 observed running PPO updates, exact full actor/critic/Adam/exploration load, zero pre-update KL/command mismatch and20 upper-refinement minibatches per update. One actual CUDA worker and47 hashes verify. Latest free GPU485MiB/host6075MiB. The same three-hour automation is verified ACTIVE with head/arm diagnostics added and cadence/thread/endpoint preserved. This is partial main learning, not independent posture or running qualification. CPU history inventory35 physical runs,1192 operations,0 differing hashes/0 read warnings;8 historical missing manifest artifacts remain. Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\upper_body_refinement_20261004\REPORT.md.


<!-- adaptive_check_20261005_0240 -->
## October 04 23:08 EDT — R35 head progress, grounded running and head/body input diagnosis

Read [the full report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_0240/REPORT.md). Same R35 checkpoint054,960 new updates; head RMS51.71→3.37deg; arms compact but not fully natural. Matched192-robot20s nominal055/pushed056 survive but do not qualify running(0/0.0208% flight) or all prior speeds.1024 updates observed later, new gates pending. No source/reward/physical change or learner reset.128/64-robot normal-limit phase/upper decomposition probes, exact4-update64-env PPO/full load, six gradient samples, CPU head-history counterfactuals and400-frame full-CAD GUI QA are preserved. No hybrid/pulse/input spoof deployed. Old running actor has substantial head-feature dependence; mechanics/control effects remain to isolate. Same owned supervisor resumed after472.2s.47 hashes verify;6144 production,target8192/reserve300/180healthy ramp unchanged. Throughput~44.9k active/38.7k worker-effective/24k full chain excluding this pause. Next closed-loop head/body dependency and1.5 seam/command tests, then a justified full-state matched-control continuation if stalled. All gates/endpoint remain, computer on.


<!-- adaptive_check_20261005_0540 -->
## October 05 02:16 EDT — R35 exact resume, causal head-feedback diagnosis, rejected running exposure

Read [the dated report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_0540/REPORT.md). R35 at2688 new updates preserves exact135 checkpoint788837ad7b431073575a2e3aa3ad0460fe1f97b3d480336a8e8c6e01576d06fb. No deployed source/reward/CAD/cap/optimizer reset. Head remains about3.4–3.8deg RMS at1.5, not fully natural arms or complete running.128-robot20s causal test: same native-zero head actions/targets in both neutral-head cohorts; original head-history feedback restores heading70.25→0.89deg and lateral1.409→0.109m/s with1.793% actual flight after5s. Diagnostic inputs only; no production spoof, no mechanic-only conclusion. Actual1.5 running commands occupy24% despite44% frontier; separately staged44% exposure passes55CPU and entire192-robot20s same-actor bitwise replay, exact4096env64update A/B plus4update full reload. Candidate loses nominal0.35 retention and has0/0.0208% flight versuscontrol0.656/0.757%; reject and discard both diagnostic actors. Current learner/supervisor45868 resumes after939.1s, one actual worker47hashes,6144production;8192target/300reserve/180healthy ramp unchanged. Reported throughput43811active/37734worker-effective/24077whole-chain completed-main samples/s excluding allR35diagnostic pauses. Research sources:Periodic Reward Composition andDAgger, hypotheses not guarantees. Next controlled posture-aware teacher/student transfer on current truthful observations and retained commands, before justified full-state continuation. All exact gates mandatory,5m/s/velocity-ceiling endpoint unproven, same3h automationactive, computeron.


Final check observation2026-10-05T06:17:45.734587+00:00: sameR35 at2752newmainupdates, oneactualCUDAworker/47hashes. Resumed138completed64updates, exact135actor/critic/Adam/exploration, allpreKL/action-commanddifferences0, minGPU1692MiB/host7626MiB/private+7MiB. Matching139/140 still do not qualify running/retention. Original5ftwalkerSHAreverified,0.9938/0.9946m/swith0/8falls; currentcompleted-mainactive43.8kversushistorical55.5kcollectionsteps/s(differentmodels/contracts,notmatchedbenchmark). See resumed_learning_verification.json, original_walker_comparison_refresh.json and final_owned_observation.json. Same3hautomationACTIVEverified; noendpoint.


Final live refresh 2026-10-05T06:24:44.313882+00:00: same R35 running, 2816 completed new main updates, 6144 training environments, one actual worker and all 47 frozen hashes verified. The latest matching142/143 remain unqualified; see final_pair_summary.json and final_live_status.json. CPU history refreshed to1325 operations, zero differing hashes and zero read warnings.


<!-- adaptive_check_20261005_0841 -->
## October 5, 04:41 EDT — R36 walking teacher-target correction

[Dated report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_0841/REPORT.md). R35 is preserved at 4,352 new main updates plus 1,473 inherited updates, checkpoint 213 SHA ec62b406f83b5a6f78a9b4979ccedfe4200b41bd1abc7c4d8cb35037ccd71c91. Normal-limit physical cohorts reproduce dependence of the older walking teacher on head feedback. Teacher-only phase context restores gait with a physically neutral head. Diagnostic head overrides and rewritten inputs were never deployed.

Three learning variants are preserved. Running-row imitation gained 0.490%/0.576% flight after 64 updates but lost it by 128 and was rejected. Partial-prior plus running transfer oversped at 0.15 and was rejected. The selected all-four walking-prior correction passed stopped and every prior case through 1.0 in a matched 4,096-environment, 64-update comparison and full 192-robot, 20-second nominal/pushed tests. Head RMS improved to about 3.1–3.5 degrees. Running flight remained 0%/0.0174%, so running is unqualified. Only teacher training labels change; the student receives truthful observations. The original 0.25 MSE formula, 23-output denominator, stopped teacher and head/arm loss remain. Numerical prior targets and gradients intentionally change.

Twenty-seven CPU counterexamples, twelve strict migration checks, real 64-environment PPO/full-state reload and whole-replay bitwise parity for 39 arrays pass. Three runtime files change; 44 protected files remain identical and 49 hashes verify. R36 continues the original 213 full actor, critic, Adam and exploration under the existing restart learning-rate ceiling; no diagnostic actor was deployed. The owned old supervisor was stopped only after a fresh source freeze. The 3,731.7-second diagnostic pause is documented. The first deployed observation has 21 genuine updates at 8,192, exact full-state load and zero KL/command mismatch. Minimum free GPU memory was 551 MiB and host available was 6,408 MiB. Later evidence belongs in the dated report and handoff.

R35 throughput was 43.53k active collection-plus-PPO, 37.49k worker-effective and 24.48k whole-chain samples/s excluding pauses; these denominators differ. New partial throughput is not a completed-block or whole-chain rate. Exact CAD motor samples do not establish a relevant velocity ceiling. Failed attempts and the cloned 0.15 scope annotation are preserved with an erratum. Full-CAD videos were already fulfilled; no new recording is claimed. Next verify retention stability and current-state flight/objective response. The same three-hour heartbeat remains active with target 8,192, reserve 300 MiB, host fallback, 180 healthy seconds and all independent gates. The computer stays on.


Final deployed observation 2026-10-05T10:13:30.497336+00:00: R36 completed its first 64 main updates at 8,192 with exact original full-state load, zero KL/command mismatch, minimum GPU 551 MiB and host 6355 MiB. Active/main-worker-effective throughput 46651/40376 samples/s. Matching 012/013 results are saved in deployed_gate_verification.json and REPORT.md. This is one actual learned block, not capacity or long-term stability. One worker and 49 hashes verify; same adaptive three-hour automation remains active and computer stays on.


<!-- adaptive_check_20261005_1142 -->
## October 5, 07:42 EDT — R36 constrained take-off diagnosis, same full-state continuation

[Report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_1142/REPORT.md). Latest saved observation: 960 R36 main updates plus 5,825 inherited; exact 053 checkpoint and matching 054/055 nominal/pushed tests preserve stopped and walking through 1.0. All 192 robots survive, 32 trials per command for 20 seconds. Running command 1.5 travels 1.371/1.375 m/s, but only 0%/0.0035% geometric flight; no running or 5 m/s qualification. Head about 3.43/3.43 degrees; natural arms not certified. Slow tracking drifts despite passing (0.15: 0.189→0.229; 0.35: 0.391→0.441).

Paused exact supervisor 19116 at 896-update full checkpoint 050, SHA 53d0eea47c3162f0c938fc01ef20dd3f736afd7de058c28c4e759403b391e5ac, drained worker and ran 12 cohorts × 24 robots × 20 seconds with truthful observations, free base, gravity and normal motor/CAD/joint/action constraints. Nine smooth retraction/timing/stance-extension variants all survive but none produces true flight; every intervention lowers full reward. Grounded force gaps are not flight. No pulse, root support, observation rewrite, diagnostic actor or changed reward is deployed. Exact 14,400-state target replay passes at 2.98e-8 rad maximum difference; ankle local target derivatives are zero in 79–88% of control samples from existing action/joint/slew constraints, no sampled CAD pair rejection. Sixteen CPU reward counterexamples pass, including actual v8 summed-force term reconstruction and redistribution invariance. Legacy mean_unload_score is explicitly corrected in interpretation only. Exact CAD 50 Hz motor mapping/unit checks pass, no relevant ceiling or substep attribution proven. Primary research lead: https://arxiv.org/html/2011.01387v2, periodic smoothed force/velocity composition; no research weights copied this check.

Same learner resumes after 431.7 seconds; actual 64-update block 053 preserves exact actor, critic, Adam and exploration, zero KL/command difference, at 8,192. No runtime source/weight/physics/cap/reset; 49 hashes and one actual CUDA worker. Rates 46.68k active / 40.39k worker-effective / 25.88k whole-chain excluding pause; 479 MiB minimum free GPU, 5,457 MiB minimum host available. Eval051 wrapper includes the pause. Initial slow CPU NPZ analysis and file-lock log-move attempt are preserved; diagnostic array caching repairs it, training untouched. Next bounded constraint-aware candidate versus unchanged control, phase/target latency and full-objective tradeoff, exact prior coverage/drift and matched independent gates before deployment. Same adaptive three-hour automation, 8,192 target/300 MiB reserve/180-second healthy ramp, endpoint save/pause/brief with computer on. All earlier evidence retained.


## Closing owned observation

At 2026-10-05T12:13:13.314974+00:00, the same R36 learner has 1,024 completed new main updates, with block `059_run_150_train_017` at 8,192 environments and 62 observed updates. GPU free is 481 MiB and host available 5729 MiB; current pre-update KL and command mismatch are zero. New matching `057_run_150_gate_017_nominal` / `058_run_150_gate_017_push` tests on `056_run_150_train_016` still pass stopped and all earlier walking commands through 1.0. At 1.5, signed speed is 1.399/1.401 m/s but true flight remains 0.0000%/0.0000%. Head RMS is 3.46/3.47 degrees. Slow speeds are 0.226/0.226 at 0.15 and 0.435/0.432 at 0.35, slightly down from the 960-update values but still above command. Tracking is nonmonotonic. One actual worker and all 49 hashes verify. No runtime change, qualification or endpoint is inferred. [Newer exact gates](closing_matching_gates.json), [owned state](closing_owned_observation.json), [closing metadata](closing_metadata_verification.json). History inventory has 36 physical runs, 1454 operations, zero differing hashes and zero read warnings; eight historical missing manifests remain recorded. Same automation is active and verified; computer stays on.


Timestamp clarification: the earlier closing note combined the owned snapshot with a later direct progress read. The following note uses the progress captured within the same CPU-owned observation; previous raw metadata is preserved in closing_metadata_before_timeclarification.json.


## Closing owned observation

At 2026-10-05T12:17:43.515439+00:00, the same R36 learner has 1,088 completed new main updates, with operation `062_run_150_train_018` at 8,192 environments. PPO progress scope: Latest completed PPO block 059_run_150_train_017; current operation is evaluation and has no PPO progress file.; its saved iteration is 63 at 8,192 environments. GPU free is 5591 MiB and host available 10542 MiB. Saved PPO pre-update KL and command mismatch are zero. New matching `060_run_150_gate_018_nominal` / `061_run_150_gate_018_push` tests on `059_run_150_train_017` still pass stopped and all earlier walking commands through 1.0. At 1.5, signed speed is 1.355/1.359 m/s but true flight remains 0.0000%/0.0000%. Head RMS is 4.02/4.04 degrees. Slow speeds are 0.220/0.220 at 0.15 and 0.448/0.443 at 0.35, slightly down from the 960-update values but still above command. Tracking is nonmonotonic. One actual worker and all 49 hashes verify. No runtime change, qualification or endpoint is inferred. [Newer exact gates](closing_matching_gates.json), [owned state](closing_owned_observation.json), [closing metadata](closing_metadata_verification.json). History inventory has 36 physical runs, 1456 operations, zero differing hashes and zero read warnings; eight historical missing manifests remain recorded. Same automation is active and verified; computer stays on.


## Closing exact-gate correction: 2026-10-05T12:17:43.515439+00:00

The latest matching nominal 060/pushed 061 pair on the 1088-update checkpoint 059 does **not** retain every earlier speed. Stopped, 0.15 and 1.0 pass both. Nominal 0.35 fails speed at0.448m/s, and pushed 0.65 fails speed at0.745m/s. Their counterpart tests pass. All robots survive. The 1.5 command gives1.355/1.359m/s with 0%/0% true flight; head RMS4.02/4.04 degrees. The previous 1024-update checkpoint 056 passed all earlier commands in matching 057/058 and is preserved with its exact SHA/gates. These qualifications belong to 056, not 059. [Latest exact failures](closing_matching_gates.json), [last prior-passing actor](last_all_prior_passing_actor.json).

The same R36 learner remains active with one actual CUDA worker and 49 hashes verified. The owned snapshot caught062 startup with no PPO progress yet; the63 iteration counter quoted in the first closing metadata was from completed059, not062 learning. Snapshot free GPU 5,591 MiB is startup capacity, not steady 8,192 training memory. Completed-main memory/throughput and exact resumed learning are recorded separately above. No sources, weights, physical caps, actor/critic/Adam/exploration or inference changed. The earlier closing prose assumed earlier-speed passes and is explicitly corrected here; raw gate results were preserved throughout. Failed diagnostic closing selection/progress reads are preserved in tool evidence and fixed by checkpoint-bound completed-pair selection and labeled fallback progress.

Next priority is per-command conditioned-target/gradient and tracking diagnosis, alongside feasible take-off timing. A new candidate requires an unchanged full-state control, 64/128-update comparison and all matching gates before deployment. Preserve current posture, 056, the qualified walking teacher and 038 flight actor. Keep the same adaptive three-hour check active, target8192/reserve300/180healthy ramp and computer-on endpoint. Running, 5 m/s and an actuator ceiling remain unqualified.


Final liveness: R36 completed 1,152 main updates and its same supervisor is actively running the 192-robot pushed evaluation064. One actual worker is verified. This is evaluation count, not PPO count; completed main blocks use8,192. The last complete pair discussed above remains checkpoint059; no newer qualification is inferred. Same automation is verified active and the CPU history inventory has36 physical runs/1458 operations, zero differing hashes and zero read warnings. See final_liveness.json.


<!-- adaptive_check_20261005_1444 -->
## October5,2026-10-05T15:52:49.806335+00:00 — exact R37 offline startup recovery; bounded clipping candidate rejected

[Report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_1444/REPORT.md). R36 online ground startup failure at081,1,536 main+5,825 inherited, no live supervisor;49 hashes and exact checkpoint080 preserved. Cached original NVIDIA ground and3textures, path-only spawner replacement, all51 arrays/cases in192-robot20s replay bitwise identical.10 CPU asset tests,11 strict migration checks,4 genuine4096-env PPO smoke updates, fresh physical/PPO/reload validation pass. Exact full actor/critic/Adam/exploration cpSHAe59b721540593883223ccc07d19aaca1d69cdd0b129f21b9aaecb37ff3a528ec continues R37 with7,361 prior main updates. Startup-only runtime:worker+local module+5asset/provenance files;46 protected authored source files identical;55 hashes. No gradient/reward/cap/PPO/head rollback.

Four actual PPO minibatches diagnose3.76–5.39% walking-prior derivative energy on identical clipped endpoints; no consistent running/teacher opposition.16 candidate CPU fixtures; matched4096-env64/128 full-state A/B and192-robot20s nominal/pushed tests retain walking but yield zero flight in both. Candidate rejected, all diagnostic actors discarded. Primary leads CAPG https://arxiv.org/abs/1802.07564 (not implemented) and periodic force/velocity composition https://arxiv.org/html/2011.01387v2. Phase requests42ms double swing; hip/knee target correlations40–60ms, descriptive only. New192-robot20s truthful speed probe saves4000 native5ms CAD motor substeps: no driving-speed clipping through2.0,1.5 grounded;2.5 falls and5–10ms clips do not prove a ceiling. No pulses/input rewrites/root support/hybrids deployed.

R3764 completed main updates; latest exact pair prior passesTrue, running passesFalse, head3.51/3.51deg. Rates46.66k active/40.43k worker-effective/13.19k whole-chain; denominators and earlier failure/downtime exclusion documented. Real resumed block exactload/zeroKL/commandmismatch, oneworker/55hashes. Preserve failed certificate launch/indexing/asset diagnostics and CPU failed-status recorder repair. Next constraint-aware full-return/timing search plus matched transfer, fresh prior/head/flight gates and memory/full-state recovery. Same3h automation,8192target/300MiBreserve/180healthy ramp; save,pause,brief endpoint with computeron. Existing videos/unsaved Fusion/earlier history retained.


<!-- adaptive_check_20261005_1746 -->
## October5,2026-10-05T18:10:54.098103+00:00 — bounded physical trajectory search rejected; exact R37 resumed

[Report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_1746/REPORT.md). Same full-state learner paused at060 SHA0d9eebbc204ed776681eda652b50a50d0db46a300c8e3efc72f15ec2e7063461,worker drained.128 smooth ten-parameter leg timing cohorts,8 robots/cohort,12s,followed by separate seed83,96 robots/20s. All survive;13 CPU checks; saved full term sums exactly reconstruct reward; observed/action commands match. Best-flight validation0.915% below1.5%,flight outside double swing,return−5.31,tracking loss−0.2069rate dominatesflight+0.00205. Best-return+1.95 remains0.0083%flight. Rejected for transfer,no diagnostic actor/offset/observation rewrite/root support/hybrid/head override deployed. Bounds are not proof of impossible running or actuator ceiling. Primary timing lead [Periodic Reward Composition](https://arxiv.org/html/2011.01387v2); no copied numerical weights.

Same supervisor resumes after342.2s; genuine64-update main full actor/critic/Adam/exploration equality,zero KL/command mismatch.55 frozen hashes,one actual worker,no runtime/reward/cap/optimizer change. Latest exact pair prior passTrue,running passFalse,head2.93/2.94deg; natural arms uncertified. R371216 main+7,361 inherited. Rates46.51k active/40.32k worker-effective/26.87k whole-chain excluding pause. Preserve initial1024 slow-speed regression,1152 recovery and checkpoint-bound newer tests; no monotonic claim. Failed inline quoting and startup progress read preserved. Next physical phase/objective timing,per-command retention,matched64/128 before any change. Same adaptive3h,target8192/reserve300/180healthy ramp,save/pause/brief computer-on endpoint; full-body videos fulfilled and unsaved Fusion retained.

<!-- adaptive_check_20261005_2046 -->
## 2026-10-05T21:00:49.915595+00:00 — timing and smoothed-force candidates rejected; action-bound diagnosis added

[Report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_2046/REPORT.md). The same R37 learner paused at full checkpoint 136, SHA 2c0689dfb08da6e251b8d00ee38fad5ae9ac021e5076a79250ed414ac05a1d04, after owned worker drain. Six fixed cohorts at seed 91 tested 192 robots for 20 seconds with truthful observations and native constraints. All survived; seven CPU-check categories, exact actor loading, matching commands and full reward reconstruction passed. Advancing the same leg offsets by 20/40/60 ms produced no flight and lower complete return. Original offsets gave 0.479% true flight and -4.274 return; best-return offsets gave 0.133% flight and +1.958 return. These offsets are rejected, and do not prove physical running impossible.

Sixteen CPU phase-smoothing categories and original-term reconstruction passed, but the full-return deficit persisted and unequal smoothed edge coefficients enabled load-redistribution credit. The box-smoothed candidate was rejected before a learning test; no 64/128-update comparison is claimed. A first floating binary-boundary periodicity fixture failed; its log and midpoint-sampling correction are preserved. Six synthetic clipped-Gaussian score cases and 24,032 truthful running states provide a new research lead: 14.53% raw means outside the action clamp and predicted ankle-pitch clipping of 57.8/63.3%. This is not actual PPO advantage-weighted variance or a deployed CAPG repair. Primary sources and limits are linked in the report.

The same supervisor resumed after 337.6 seconds. A genuine 64-update, 6,144-environment main block loaded actor, critic, Adam and exploration exactly, with zero pre-update KL and action-command mismatch. All 55 frozen hashes verify; no production source, reward, geometry, cap, actor or optimizer reset occurred. Exact checkpoint 139 fails 0.35 speed retention in both tests at 0.464/0.459 m/s; the other prior commands pass. Running flight remains unqualified, with head RMS about 3.08 degrees. Natural arm motion is not certified. Preserve checkpoint 133, walking teacher and earlier 038 flight actor.

The completed-operation count is 2,816, plus 58 verified retained updates from the 069 VRAM failure and exact 070 full reload; inherited count 7,361 remains declared rather than re-audited. All retained-main rates are 44.62k active / 38.60k worker-effective / 26.10k whole-chain samples/s, excluding both R37 diagnostic pauses. Current capacity is 6,144; target 8,192, 300 MiB reserve and 180 healthy training seconds remain. Next measure actual per-command advantages, teacher coverage, effective targets and CDF tail-score gradients, then require matched 64/128 learning and every independent gate before deployment. No diagnostic offsets, hybrids, spoofed observations, tilted-head rollback or increased caps are deployed. Keep the same three-hour automation active; at verified 5 m/s or a causal relevant velocity ceiling, save, pause and brief with the computer on. Videos remain fulfilled and unsaved Fusion work is preserved.


<!-- closing_2046 -->
Closing observation 2026-10-05T21:08:10.906596+00:00, owned snapshot 2026-10-05T21:06:43.628199+00:00 and actual PPO progress through 2026-10-05T21:08:06.357791+00:00: same supervisor, one actual worker and 55 hashes verify. Current `145_run_150_train_046` has 22 observed main updates at 6,144, zero KL/command mismatch, free GPU 1587 MiB and host 6754 MiB. Completed-operation main 2880, plus 58 verified recovered updates. Latest exact matching actor `142_run_150_train_045` still fails 0.35 speed retention at 0.452/0.448; all other earlier commands pass. Running command 1.5 reaches 1.415/1.419 m/s, true flight 0.0000%/0.0278%, head RMS 3.41/3.41 degrees. Running remains unqualified. These gates bind that saved actor, not current partial updates. Same heartbeat ID/thread/three-hour cadence/notification preference/durable scope are verified active. History inventory: 37 physical runs, 1620 operations, zero differing hashes and no read warnings; eight old missing manifests remain documented. [Final evidence](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_2046/final_liveness.json), [newer matching pair](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_2046/closing_newer_matching_gates.json).


<!-- adaptive_check_20261005_2347 -->
## 2026-10-06T00:02:22.187449+00:00 — independently qualified running1.5; improving2.0 and prior-teacher provenance audit

[Report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_2347/REPORT.md). Same R37 supervisor32492 and full-state learner continue unchanged. Exact199 SHA329d7dc0c6fe8a36a74d34924d90553a68bb57d7611ee2c8b4930a419e72d6f5 passes200/201 full nominal/pushed/stopped/prior gates: signed1.456/1.458m/s, true flight1.681/1.865%. The curriculum advances to2.0. Latest exact225_run_200_train_008 passes2.0 at1.898/1.901m/s, flight4.354/4.517%, all stopped/walking cases pass; retained1.5 flight1.017/1.090% still fails1.5% minimum. All224 survive20s;32 per command. Stage2 remains unqualified; no5m/s or ceiling. Head2.0 RMS3.35/3.36deg; natural arms uncertified.

New CPU exact-history replay11713 states each for199/222 agrees within1e-5action units. The old R34 head-context table is still applied after the prior teacher becomes neutral-head199. Conditioning creates self-teacher clipped lower-label RMS0.231–0.258 on certified199 states; truthful self-teacher error~1e-6. Current222 label changes0.236–0.279; student closer to conditioned labels. This identifies target/fixed-point mismatch, not physical causal repair;1.5 labels are unaffected. No actual PPO advantage capture, matched64/128 candidate or deployment is claimed. Preserve improving state; next test teacher-specific context provenance if retention stalls, with actual gradients and full-state matched physical/learning gates. Never spoof production inputs or remove the old original teacher's validated context indiscriminately.

Exact current CAD motor mapping uses full-population10Hz q/qd and64 finite-difference poses per pair; checks pass.2.0 ankle p99cap~0.88/0.84, sparse crossings do not prove driving clipping or a ceiling. Upper audit and inspected trend plot saved. CPU parser failures and repairs preserved; frozen runtime untouched.55hashes,one actual worker,4608completed+58recovered R37 updates,7361inherited separately; rates44.18k active/38.19k worker/25.62k whole chain, historical679.7s pauses excluded. No pause/source/reward/geometry/cap/optimizer/actor reset this check.6144production,target8192,reserve300,host guards/180healthy-training-second ramp retained. Same3h automation remains active with fresh priorities. Videos fulfilled; unsaved Fusion preserved; save/pause/brief computer-on endpoint remains.


<!-- closing_2347 -->
Closing observation 2026-10-06T00:08:40.359855+00:00, owned snapshot 2026-10-06T00:08:38.860711+00:00: same supervisor32492, one actual worker and55 frozen hashes. Stagerun_200, operation233_run_200_gate_011_push, count224, 4736completed main+58retained recovery. Latest exact actor228_run_200_train_009 matching nominal229_run_200_gate_010_nominal/pushed230_run_200_gate_010_push: 1.5 command: signed 1.465/1.467m/s, true flight 1.205/1.319%, pass False/False; 2 command: signed 1.942/1.941m/s, true flight 6.722/6.667%, pass True/True. Current tested stage qualified:False. Preserve improving learner. Owned main progress:None. Same heartbeat ID/thread/3hcadence/notification/durable scope verified ACTIVE. History inventory37physical runs/1707operations/zero hash differences/no read warnings;8old missing manifests remain documented. No source/reward/physics/cap/optimizer/actor reset or pause this check. No5m/s or persistent relevant velocity ceiling. [Final evidence](final_liveness.json), [latest exact pair](final_matching_gates.json).


## 2026-10-06T02:42:43.104317+00:00 — User five-hour cadence,5/4 endpoint and upper joint-angle time series

Updated existing native heartbeat to ACTIVE every5hours, same ID/thread/notification/GPT-6.1 Sol max. Verified5m/s remains primary; verified4m/s is fallback only with controlled proof native actuator limits block5; all exact nominal/pushed/stopped/prior gates remain. At endpoint save,pause training+automation,brief,computer on. Canonical workflow FIVE_HOUR_CHECK.md supersedes historical3h policy. No live training source/reward/physics/cap/optimizer/actor reset, pause or simulator launch.

CPU exact-pair review verifies R37 promotions2.0 and2.5; actor301 at3.0 is still unqualified (lateral and prior1.5 flight). New mandatory per-check angle-time diagnostic analyzes34 command/test records across301/293, actual head/shoulder/elbow angles, targets, velocities and arm/leg phase. Plot QA confirms asymmetry (1m/s left/right shoulder ranges38.8/24.9deg), gait-locked head oscillation and unequal3m/s arm/leg phase113/-176deg. Human reference is feature-based, exact speed-matched Unitree encoder curves unavailable; no human-like certification. Four metric fixtures pass. Sources/limitations, native update before/after and current exact gates: E:\Aditya\robot_sim\physical_v29\training\diagnostics\automation_five_hour_update_20261006\REPORT.md. Preserve progressing full learner and independently certified actors.


## 2026-10-06T07:51:55.504824+00:00 — Guard diagnostic1:source signature rejection

Changed baseenvironment source signature correctly rejected the old checkpoint. No learned state deployed. Added a strict checkpoint-SHA/source-inverse-patch bridge; no generic bypass. Evidence: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_0741\001_current_run_350_gate_nominal. Verified owned failed-worker cleanup is preserved; numerical rewards, classifiers, physics and caps unchanged.


## 2026-10-06T08:00:18.886153+00:00 — Guard diagnostic2:moving teacher signature rejection

Fresh3m physical teacher gates passed; PPO correctly rejected its original source signature. Added the same bounded explicit teacher migration and reran exact full-state PPO/reload. Failed actor was never deployed. Evidence: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_0741\104_full_state_ppo_smoke. Verified owned failed-worker cleanup is preserved; numerical rewards, classifiers, physics and caps unchanged.


## 2026-10-06T08:16:15.083282+00:00 — Guard continuation R38 startup:teacher certificate lookup failure

Fresh teacher nominal/pushed pass and current459 gates run, but qualification lookup expects `_gate_` filenames. Supervisor stops before main learning;0 new main updates. Frozen run/logs preserved: E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261006_r38_commanded_travel_budget. R39 uses the same original full459 learner and runtime bytes with a corrected supervisor gate label. Lookup regression and fifteen negative migration fixtures pass; fresh production startup/teacher/PPO repeats are required. No lost main learning or from-scratch actor.


## 2026-10-06T08:29:47.914798+00:00 — Corrected commanded60m termination, exact full-state R39 continuation

R37 newly qualified3m/s at349. All24 sampled later3.5 terminal events over12 pairs were solely excessive_drift near18s,upright: a60m absolute-travel guard falsely classified commanded motion. Repair integrates actual planar action-command travel and retains original60m excess margin; other physical faults/all independent classifiers unchanged.79 CPU checks,15 migration-rejection fixtures,117 bitwise array comparisons,real4-update PPO/full-load/reload pass. Failed signature/teacher-signature attempts and owned cleanup are preserved. Only environment/worker provenance+strict supervisor change;51 other runtime files identical,physics/CAD/native caps/reward/PPO/upper modules unchanged.

R39 preserves459 original actor/critic/Adam/exploration and16936 earlier actual updates including58 recovered; all55 hashes verify. Old supervisor retired after healthy startup; gap1029.1s recorded. Fresh349 priors through3 pass. Current459 baseline3.5 survives100%,3.530/3.535signedm/s,38.74/38.60%flight but fails lateral0.401/0.362; prior0.15net travel,1.5flight,2.5slip also fail. New completed main updates at observation0; no3.5/5/4 qualification or ceiling claim.

Mandatory reset-segmented per-speed upper angles/targets/velocities inspected: at1 shoulder ranges37.51/27.68deg,phase177/153; at3 unequal89/-177; at3.5small9.41/7.21deg ranges andheadRMS3.78deg/gait oscillation. Human comparison remains morphology-limited,Unitree speed-matched encoder curves unavailable. Sparse10Hz CAD ankle cap crossings are not sustained driving clipping;5ms/torque attribution needed before fallback. Report/evidence/decision: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_0741\REPORT.md. Existing same-thread heartbeat stays ACTIVE5h,verified5 orverified4+causal native limit,save/pause/brief,computer on.


## 2026-10-06T08:43:23.754762+00:00 — Closing R39 verification after128 main updates

Read E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_0741\REPORT.md closing addendum and E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_0741\final_gate_review.json. Exact actor014 nominal015/pushed016 survives320/320 but3.5 fails lateral; earlier0.15 net travel,nominal1.0 support,both3.0 lateral fail.1.5 flight and2.5 recover,so progress is nonmonotonic. Same full learner continues8192;128completed plus15 observed partial,55 hashes,oneworker,zero KL/command mismatch,stable memory. Latest actual upper angles/targets/velocities inspected in final_upper; asymmetric fast arms and gait-linked head motion remain,not human-like certification. Timeline correction: original supervisor retired afterR38 healthy startup,notR39; R38 subsequently failed before any main learning,thenR39 repeated startup. Gaps1742.3s toR39supervisor/2347.5s tofirstreported PPO documented. Native automation verifiedsameID/thread/notifications,ACTIVE5h.5/conditional4 endpoints unproven;computeron. No further runtime or policy reset since deployed guard repair.


## 2026-10-06T13:30:03.963362+00:00 — Truthful steering, actual arm response and rejected frontier lateral exposure

Same R39 full learner continues from132; no production source/reward/PD/cap/optimizer/actor change. Current3.5 lateral remains failed and retained0.15 net travel/1.5 flight/3.0 gait regress. Separately certified349 remains preserved. Mandatory per-speed head/shoulder/elbow actual angle-target-velocity plots inspected, including control/candidate128 and closing132; walking anti-phase improves but fast arms remain asymmetric and physical target response differs greatly. New200Hz normal-limit current-actor audit has zero straight1/3.5 driving clips; no native ceiling. Static arm mass/inertia is nearly symmetric, not a causal frequency-response model. Four actual PPO minibatches quantify teacher conditioning/upper gradients with stopped-hook accounting; no blanket weight increase justified.

Diagnostic uniform+/-0.1 frontier lateral exposure fails the exact-full-state4096-env64/128 comparison: candidate128 lateral0.773/0.740 versus control0.592/0.564 plus prior gait failures; REJECTED, all diagnostic actors discarded. CPU fixtures/load/commands and all matching gates preserved, including initial harness/CPU diagnostic failures. Same supervisor35688 resumes after1823.6s including worker drain;135 completes65 true6144 updates, exact full-state load/zeroKL/mismatch/stable memory/upper20 batches per update. Prepause2672 retained new plus16936 inherited; postresume2737 retained R39 updates. Target8192/reserve300/hostguards/180healthy ramp unchanged. Nominal clone reset diversity is now explicit and needs independently seeded initial-state trials before final5/conditional4 endpoint; existing classifiers unchanged.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_1243\REPORT.md. Latest pair E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_1243\closing_gate_review.json; upper data E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_1243\closing_upper\upper_joint_time_audit.json; resume E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_1243\resumed_learning_verification.json. Next controlled upper frequency/coupling and fixed-frame direction/objective tests; preserve head/gait/full learner and all rejected variants. Same active5h automation/verified5 or qualified4+causal native limit,save/pause/brief,computeron.


<!-- adaptive_check_20261006_1743_record -->
## 2026-10-06T18:20:26.764486+00:00 — Actual arm frequency response, world head axis and fixed-frame direction comparison

Same R39 full learner continues; Reject this fixed-frame kernel for immediate deployment; resume the same original full learner and existing upper refinement. At128 updates lateral improves about57%, but both3.5 direction tests still fail and a new2.5 no_hopping failure appears in both, while control passes2.5. Preserving gait outweighs isolated direction improvement. The frame hypothesis remains a research lead requiring focused retention/gradient evidence. No production source/reward/PD/cap/state reset or diagnostic actor deployed. Current3.5 direction/retention unqualified, separately certified349 preserved. Actual upper angle/target/velocity plots inspected for261,349,control/candidate128 and closing actor.320-robot free-base normal-limit response verifies actual shoulder35/1, correcting earlier35/4 default analogy; more damping attenuates fast pitch and is not selected. Zero native5ms driving clips across10 cohorts and low arm effort do not prove a motor ceiling. Modeled world head axis at261/3.5 gives16.11deg RMS despite smaller neck angles; CAD-face/Isaac-link parity unverified. No human/Unitree encoder-curve certification.

Both4096-env64/128 branches retain exact264 full state and all matched gates; diagnostic learning is excluded from main counts, runtime override explicitly uncertified by original signature. GroundUSD-copy startup failure and metadata lookup correction preserved. Same supervisor35688 resumes after1599.9s including drain;267_run_350_train_086 completes64 genuine6144-env updates, exact full load,zeroKL/mismatch,upper20 batches,stable memory. Prepause5528 retained R39+16936 inherited, target8192/reserve300/hostguards/180healthytraining ramp unchanged. Current5/conditional4 endpoint and initial-state diversity requirements unchanged,computeron.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_1743\REPORT.md. Matched: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_1743\matched_direction_review.json. Upper: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_1743\closing_upper\upper_joint_time_audit.json. Resume: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_1743\resumed_learning_verification.json. Next: Direction correction is a promising but undeployed hypothesis: exact4096-env128 comparison lowers3.5 lateral0.590/0.571 to0.254/0.242m/s, yet world-lateral still fails and retained2.5 hop fraction2.299% exceeds the unchanged2% cap in both tests;1.5 flight also fails. First analyze per-command actual advantages/PPO versus prior gradients and2.5 landing events on these saved actors. Require a bounded retention-preserving candidate, independently seeded replication and full-state64/128 control/all gates before deployment; original signatures do not certify diagnostic reward overrides. Do not blindly repeat the same kernel, permanently infer the frame hypothesis false, arbitrarily change weights, or deploy diagnostic actors. Every check inspect actual upper angle/target/velocity timelines. Actual shoulder35/1 free-base tests show high-frequency attenuation; more damping worsens pitch and is not selected. Before an upper repair, test bounded normal-limit stiffness/damping or feasible target-velocity/phase response with truthful feedback and motor-effort/base coupling, then narrow full-state matched control/retention tests. Preserve current head/gait/full learner. Cross-check the CPU modeled world head axis against actual Isaac head-link orientation and original CAD face axis before any world-forward posture label; native head zero alone is insufficient. No cap raise, spoof, hybrid or fabricated human/Unitree curves. Verify same owned worker55 frozen hashes/local ground/exact full-state recovery, completed plus33 proven recovered R39 updates and16936 inherited, moving6144/8192 target/reserve300/host fallback/180continuoushealthyTRAINING-second ramp. Exclude this1599.9s diagnostic pause and earlier1823.6s pause from clocks. All10 current normal-limit physical cohorts have zero native5ms driving-speed clips, not proof of a high-speed ceiling. Before final5 or conditional4 endpoint establish32 independently seeded nominal-physics initial-state trials per command with all unchanged classifiers; nominal clones alone are insufficient. Preserve useful actors, unsaved Fusion and fulfilled videos.


<!-- adaptive_check_20261006_1743_closing267 -->
Closing evidence 2026-10-06T18:26:41.110897+00:00: exact post-resume267 now passes all stopped and prior cases through3m/s in268 nominal/269 pushed, including recovered1.5 geometric flight.3.5 signed3.489/3.496 and lateral0.516/0.488 still fail direction. Preserve this improving original full learner. Fresh267 actual angle-target-velocity plots were saved and visually inspected; older264 review/plots/report remain in the dated folder. No diagnostic actor/source/PD/cap/reset deployed. Updated same native5h focus and current report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_1743\REPORT.md. Exact pair SHA20a133fdd4843118d816172239201a387ca947b38952fb38f094a156afa0a320. Newer partial actors are separately unqualified.

<!-- adaptive_check_20261006_2243_record -->
## October6 22:43UTC — actual gradients, shoulder response and head parity

Trigger: current3.5 direction and retained skills still unqualified. Read [adaptive_check_20261006_2243](E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261006_2243\REPORT.md), then freshACTIVE_RUN/state. SameR39/supervisor35688;55hashes,oneactualworker. No production source/reward/PD/cap/optimizer/actor reset. Exact402_run_350_train_131 nominal/pushed current signed3.462/3.467, lateral0.341/0.319; failures and all prior cases bind saved matching pairs. Preserve267 all-prior and separately certified349.

New192-robot20s K35/50/70,damping1 normal-limit shoulder test improves tracking but remains an undeployed candidate without64/128 learning comparison/all prior qualification. Allsurvive; no shoulder driving clips; sparse elbow clips no ceiling. Actual Isaac head-link/FK parity7.34e-5deg; worldnative-forward error larger than neckangles,CAD face/camera uncalibrated. Required current angle/target/velocity plots and coordination data visually inspected; no human/Unitree exact encoder certification. Four real4096 PPO minibatches captureactualadvantages/percommandmean gradients; mixedalignment,notproven globalopposition. Failed stopped-hook CPU subtraction explicitly retained. Earlier fixed-frame candidate2.5 startup pairedtouch0.90/0.94s reproduced; remains rejected,no gate relaxation.

Owned399 boundary pause167.4s including60s drain,then samefulllearner resumed: 402_run_350_train_131, 65updates6144,exactactor/critic/Adam/exploration,zeroKL/mismatch,upper20batches.12diagnosticchecks; premature before-drain harness failed safely and preserved. Through3998431retainedR39+16936inherited;44.89kactive/38.90kworker/26.36kchain with prior3423.4spauses excluded. Target8192/reserve300/hostfallback/180healthyTRAININGramp unchanged. Sameactive5h/verified5 orqualified4+native-limit endpoint,save/pauseboth/brief/computeron. Next controlled gain/retention/direction continuation tests; preserve full state,failures,267/349,unsavedFusion and fulfilledvideos.
<!-- /adaptive_check_20261006_2243_record -->


<!-- adaptive_check_20261007_0344_record -->
## 2026-10-07T04:32:18.678788+00:00 — Goal-budget recovery and rejected shoulder-gain continuation

R39 failed before the heartbeat when its 18-hour guard interrupted 501, not a physical failure/completion. Last full 498 SHA c04ae2f26310d5514905ce1a2fec9159594ec8ce8d5a19f67910291f497e2ae2 retained.32 unsaved updates excluded;10538 completed+33 recovered+16936 inherited=27507 actual prior main updates preserved. Supervisor-only R40 recovery adds explicit goal budget0 and finite saved-boundary budget handling, restoring exact actor/critic/Adam/exploration and both teacher/proof contexts. All 53 authored worker/physics/PPO/teacher files unchanged;55 frozen hashes verify.14 CPU checks, real 4-update, 64-env PPO/full reload, both320-robot, 20-second baseline arrays/cases bitwise identical. Existing no-progress/memory/native-limit guards retained.

Completed4096-env, 64/128 shoulder-pitch stiffness 70 versus35,damping 1 exact-full-state control experiment REJECTED: early drift improvement fades; both 3.5 direction gates fail; candidate 128 has 1.5 true flight0.795/0.899% versuscontrol 2.448/2.503%, introducing both retention failures. Diagnostic gain override not certified by original signature; all experimental actors remain outside production. Upper actual angle-target-velocity/arm-leg phase plots inspected for baseline/control/candidate and fresh saved R40 pair. Fast bias/asymmetry and gait-linked neck oscillation remain; no human/Unitree encoder likeness or world-gaze certification. Failed probes/accounting uncertainties preserved.

R40 supervisor 39000 has 64 completed plus 41 recovered = 105 retained new main updates, with 2 observed partial, one worker,55 hashes. Original498 exact full main load, zero KL/command mismatch, standing/moving/upper 20 batches. Last block 6144 envs; rates45.69k active/37.51k worker/18.82k chain incl startup/gates. MinimumGPU 1329MiB/host 4883MiB. Exact010_run_350_train_001_part_023.5 signed3.517/3.534, lateral0.433/0.405; no 3.5/4/5/ceiling qualification.267 all-prior/349 certified 3 retained separately. Target8192/reserve 300/host fallback/180 healthy TRAINING-second ramp remain. Same active5h/verified5 orqualified4+native-limit endpoint; save, pause both, brief, computer on. Videos fulfilled,unsaved Fusion untouched.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\REPORT.md. Decision: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\decision.json. Exact gates: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\closing_gate_review.json. Recovery: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\resumed_learning_verification.json. Upper: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\final_upper\upper_joint_time_audit.json. Next focused tests in NEXT_CHECK; full history preserved.
<!-- /adaptive_check_20261007_0344_record -->


<!-- closing_adaptive_check_20261007_0344 -->
Closing owned evidence 2026-10-07T04:34:26.307983+00:00: R40 supervisor 39000, one actual worker and all 55 frozen hashes verify. It has 128 completed-operation updates plus 41 hash-proven recovered updates = 169 retained new main updates, with 27,507 prior updates preserved. The second completed block loads actor/critic/Adam/exploration exactly; every retained pre-update KL and command mismatch is zero, with standing/moving/head-arm auxiliaries active at 20 batches/update. Its 6,144 environments have minimum free GPU 1385 MiB and host available 4972 MiB. Retained rates are 45.04k active / 37.69k worker-effective / 20.72k whole-chain samples/s; the last rate includes new startup, capacity and baseline gates through the recorded last PPO tick.

The closing snapshot is evaluating newer actor013 in nominal014; no new qualification is inferred while its matching pair is incomplete in that snapshot. The completed actor010 nominal011/pushed012 remains the reported independent pair: 3.517/3.534 m/s signed, 0.433/0.405 m/s lateral; current direction fails. Retained0.15 net travel and2.5 hopping/lateral fail both, with pushed3.0 lateral also failed. Eight actual joint-angle/target/velocity plots have been inspected; slow shoulder-bias reversal at2.5/3 and fast asymmetry remain. Stronger shoulder gain is rejected, original35/1 and full learned state preserved. Native automation update is confirmed ACTIVE on the same five-hour heartbeat/thread/notification preference and durable5 or qualified4+causal native-limit endpoint. No pending recording request, no shutdown, no unsaved Fusion work touched. Inventory40 physical runs, zero differing hashes/read warnings; eight historical missing manifests remain documented.

Fresh liveness: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\final_cpu\live_handoff.json. Full-state evidence: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\final_learning_verification.json. Exact completed pair and plots: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\closing_gate_review.json and E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\final_upper\upper_joint_time_audit.json. Native confirmation: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0344\automation_verification.json. Earlier observation counts above are retained with their timestamps.
<!-- /closing_adaptive_check_20261007_0344 -->

<!-- newer169_gates_20261007_0344 -->
Newer completed169-update actor013 at 2026-10-07T04:38:50.925211+00:00 now has matching nominal014/pushed015. Both 2.5 m/s cases recover all checks; 0.15 net travel still fails both and pushed3.0 lateral fails. Current3.5 direction remains unqualified at signed3.421/3.432 m/s, lateral0.407/0.383 m/s. All survive. Preserve this independent retention improvement; no source/gain/reward or actor reset. Next actual run016 is learning at6,144. Full upper timeline QA remains explicitly bound to105-update actor010, with earlier comparisons preserved; no new upper-motion certification is inferred for013. See physical_v29\training\diagnostics\adaptive_check_20261007_0344\closing_newer_pair.json.
<!-- /newer169_gates_20261007_0344 -->

<!-- adaptive_check_20261007_0844_record -->
## 2026-10-07T09:34:20.956238+00:00 — Teacher-label comparison, independent reset tests and direction objective audit

Same R40 learner/supervisor39000 is running; latest completed main block uses6144 environments, with one actual worker and55 verified frozen files. The owned snapshot operation is145_run_350_train_046 at6144 robots; evaluation count is not PPO capacity. 2905 completed +41 hash-proven recovered = 2946 retained current-run updates, plus27507 inherited. Exact 142_run_350_train_045 matching nominal/pushed signed3.5 speeds 3.476/3.487m/s, lateral 0.337/0.310; current stage remains unqualified. Gates bind this saved actor, never newer partial learning. Full-state teacher-context removal tested against unchanged4096-env64/128 control is REJECTED: candidate recovers0.15 but adds1.5 lateral and2.5 slip failures;3.5 remains unqualified. Extra push seed and32 independently generated q/qd reset states per command reproduce tradeoff. Truthful teacher349 requalifies all earlier cases. Four actual PPO minibatches and10 CPU ownership checks pass; no deployed source/reward/gain/cap/optimizer reset or diagnostic actor. Original signature does not certify overrides.

Required actual upper angle-target-velocity plots inspected at133,both128 diagnostic actors and142_run_350_train_045; fast asymmetry,slow shoulder bias drift and head oscillation remain. No human/Unitree encoder or optical gaze certification.108 synthetic direction fixtures reproduce high yaw-aligned reward for diagonal travel; no port error or complete-return proof. Same supervisor resumes after1918.3s; exact actor/critic/Adam/exploration and both teachers,zeroKL/mismatch,upper20batches preserved. Rates44.23k active/38.21k worker/25.40k chain with documented pause excluded. Initial failed pause bookkeeping recovered safely and is recorded separately. No5/4/ceiling endpoint. Target8192/reserve300/hostfallback/180healthyTRAINING ramp; ACTIVE5h,save/pause/brief/computeron endpoint unchanged.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0844\REPORT.md. Gates: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0844\closing_gate_review.json. Decision: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0844\decision.json. Upper: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0844\final_upper\upper_joint_time_audit.json. Resume: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0844\final_learning_verification.json. Next focused tests recorded in NEXT_CHECK; all earlier history preserved.
<!-- /adaptive_check_20261007_0844_record -->

<!-- adaptive_check_20261007_0844_completion -->
Closing confirmation 2026-10-07T09:36:28.015262+00:00: native app confirms same ACTIVEfive-hour heartbeat/thread/notification preference; durable scope/5-or-qualified4 endpoint prefix is byte-identical. No duplicate created. CPU history inventory has40 physical runs,2593 operations,zero differing hashes/read warnings;8 historical missing manifests remain documented. Eight angle/velocity images have visual QA. Exact resumed142 main64 updates and matching143/144 are verified; later145 partial learner remains separately unqualified. No production source/gain/reward/cap/state reset or diagnostic actor deployed. Reports and earlier history preserved. See E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0844\automation_verification.json and E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0844\inventory_verification.json.
<!-- /adaptive_check_20261007_0844_completion -->

<!-- adaptive_check_20261007_1344_record -->
## 2026-10-07T15:03:29.386289+00:00 — Course-feedback diagnosis and full-state comparison

The same R40 learner and supervisor 39000 are running, with one actual worker and 55 verified frozen files. It retains 5351 completed + 50 hash-proven recovered = 5401 current-run updates, plus 27,507 inherited. Exact actor 257_run_350_train_083 reaches 3.516/3.531 m/s in matching nominal/pushed 3.5-command tests, with 0.571/0.523 m/s lateral drift. The stage remains unqualified. Saved gates never qualify newer partial actors. The 2.5 m/s cases recovered in both tests; 0.15 net travel, 1.5 geometric flight and 3.0 lateral direction still fail, and nominal 1.0 support fails.

High-speed course-feedback candidate is REJECTED for immediate deployment. It passes both3.5 cases at64 updates(0.132/0.130m/s lateral),but loses both at128(0.185/0.179),weakens retained1.5 flight and retains0.15 travel failures. Extra push and independent small-reset cohorts reproduce the tradeoff. Preserve its promising physical direction evidence without declaring the law permanently ineffective; resume the exact original learner. Physical384robot20s heading/course probe confirms yaw response and0.128m/s course-feedback lateral speed,with zero5ms driving clips,all survivors and exact term sums. It is not independent qualification.60 command/tick fixtures and34 heading fixtures pass. Exact4096env64/128 controls,extra push and32 independent small q/qd resets/cmd complete; all experimental actors stay outside production. Mandatory actual upper angle-target-velocity plots inspected; asymmetry,bias and head oscillation remain,not human/Unitree certification. Same39000 resumes after3502.1s with full state/teachers/upper20.55hashes/oneworker,completed+50 recovered updates,27507 inherited and32 unsaved excluded. Rates41.10kactive/35.20kworker/22.93kchain,successful pauses excluded.8192target/reserve300/hostfallback/180healthyTRAINING ramp. ACTIVE5h and verified5 or verified4+causal limits endpoint unchanged; computeron.

Report:E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1344\REPORT.md. Decision:E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1344\decision.json. Comparison:E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1344\matched_heading_review.json. Upper:E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1344\final_upper\upper_joint_time_audit.json. Resume:E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1344\final_learning_verification.json. Earlier records preserved.
<!-- /adaptive_check_20261007_1344_record -->

Closing ownership observation 2026-10-07T15:04:52.077142+00:00: same supervisor39000, one actual worker and55 verified frozen files. Completed5415 +50 received recoveries = 5465 retained current-run main updates, plus27507 inherited. Last completed main uses4096 environments; current operation is261_run_350_gate_085_nominal with count320, which can be evaluation rather than PPO. Exact257 gates and upper QA remain the reviewed evidence; newer actors are not qualified by those results. Inventory40 physical runs,zero differing hashes/read warnings,eight historical missing manifests preserved. Same ACTIVEfive-hour heartbeat/thread/notification/durable endpoint verified. See closeout_verification.json.
Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1344\REPORT.md

<!-- fusion_vram_release_20261007_user_requested -->
## 2026-10-07T17:06:33.383997+00:00 — User-requested Fusion shutdown

Fusion and its preview process exited normally using Pause unstarted jobs and Update Later; no force-kill or unsaved-change discard. Global free VRAM initial773 MiB/post-exit5219 MiB; changing training phase prevents exact attribution. Same R40 supervisor39000/one worker/55 hashes continue. No training pause, restart, source or policy change. Existing target8192/reserve300/host guard/180healthyTRAINING ramp remains. Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\fusion_vram_release_20261007\REPORT.md.
<!-- /fusion_vram_release_20261007_user_requested -->

<!-- adaptive_check_20261007_1845_record -->
## 2026-10-07T19:24:49.761561+00:00 — Saved-actor response and passive gradient diagnosis

Same R40 learner/supervisor39000,one actual worker,55 frozen hashes. 7143 completed+50 hash-proven recovered=7193 retained new main updates,plus27507 inherited. Exact reviewed 338_run_350_train_110 reaches3.5-command signed3.423/3.436m/s with0.340/0.308m/s lateral;both current direction gates fail. Both0.15 net travel and1.5 geometric flight fail;other stopped/prior cases including recovered3m pass in this pair. Qualification never extends to newer partial actors.

Preserve and resume the exact original full-state learner. Saved-actor response and passive actual gradients narrow the next diagnosis; no new candidate has independent learning/retention evidence sufficient for deployment. Previous rejected candidates remain rejected. Paired384robots20s at0.15/1.5/3.5 show current course response0.368to0.129m/s lateral,versus rejected course1280.634to0.185;all survive,zero command mismatch and5ms driving clips. This is not qualification. Five passive-hook CPU fixtures and exact4-update4096-env PPO pass;before-standing capture resolves fresh subtraction within4.5e-10,alignment mixed. Upper actual angle-target-velocity/bias plots inspected:At3.5 nominal,head RMS3.43deg;physical pitch ranges14.23/7.52deg. Slow pitch means-15.99/6.40deg and faster residual ranges13.43/6.65deg. Arm/ipsilateral-hip phases147.0/162.7deg at2.67Hz;phase and amplitude are both diagnostic. Same39000 resumes after total681.4s across failed/successful diagnostics;no production change.55hashes/oneworker,6144,minGPU1940MiB,host8455MiB,rates37.68kactive/31.97kworker/20.28kchain.8192target/reserve300/180healthyTRAINING ramp and verified5 or conditional4 endpoint remain. Report:E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1845\REPORT.md. All failed/rejected evidence preserved.
<!-- /adaptive_check_20261007_1845_record -->

<!-- adaptive_check_20261007_1845_closing -->
### Closing exact evidence, 2026-10-07T19:28:21.723650+00:00

Closing owned evidence 2026-10-07T19:28:21.723650+00:00: the same R40 supervisor 39000 has one actual worker and all 55 hashes verified. It retains 7207 completed + 50 recovered = 7257 new main updates, plus 27,507 inherited. Exact actor 344_run_350_train_112 reaches 3.475/3.482 m/s signed at command3.5, but lateral 0.495/0.465 fails both direction tests. Both0.15 path-efficiency and1.5 geometric-flight tests fail; pushed3.0 lateral regresses. All survive. The earlier338 pair's3.0 recovery is preserved and is not attributed to344. The newer344 angle/target,velocity and slow-bias plots are also visually inspected. At3.5,head RMS3.87deg and shoulder pitch ranges13.09/6.76deg. Fast asymmetry,held biases and gait-linked head oscillation persist;these diagnostics do not certify human/Unitree motion or optical gaze. Fresh completed-main rates are 37.62k active/31.91k worker/20.24k whole-chain samples/s, with 6101.8s documented suspensions excluded. Resumed341 loads exact338 actor/critic/Adam/exploration;344 also loads its predecessor exactly. All KL/command checks and three20-batch auxiliaries pass. Last6144 block minimumGPU1937MiB/host9054MiB,private+12MiB,Torch allocation unchanged.

Report:E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1845\REPORT.md. Newer gates:E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1845\liveness_gate_review.json. Newer upper:E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_1845\closing_newer_upper\upper_joint_time_audit.json. No production change; same ACTIVE5h scope/verified5 or conditional4 endpoint.
<!-- /adaptive_check_20261007_1845_closing -->


## Five-hour check October7 23:46 UTC — timing parity and upper-target bias

Same R40 learner/supervisor39000,one actual worker and55 hashes at2026-10-08T00:04:22.417150+00:00. 9063 completed+50 recovered=9113 retained new main,+27507 inherited;0 observed partial separate. Reviewed exact428 nominal429/pushed430:3.5 signed3.395/3.401,lateral0.351/0.326m/s;current direction fails. Both0.15 net travel and1.5 geometric flight fail;other cases including3.0 pass. All survive. Saved gates never qualify newer actors. No production source/reward/gain/cap/optimizer/actor change. Same supervisor resumes after257.1s. 384-robot20s truthful5ms timing/filtered-course physical test,two seeds32 paired states per cohort,all survive,exact root/cache/directPhysX/control-command parity. Velocity-only smoothing is rejected versus instantaneous feedback. Mandatory upper timelines inspected:headRMS3.32deg,fast pitch ranges12.33/10.06deg,slow opposite biases and2.5/3m reversals remain;no human/Unitree/optical certification. Rates36.40k active/30.67k worker/19.35k chain,distinct denominators. Keep existing upper/both prior20-batch refinement,35/1 gains,full learner and all gates. Same ACTIVE5h endpoint;no5/4 or ceiling proof.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_2346\REPORT.md. Physical384/two independent seeds and5 quaternion fixtures pass;velocity120ms/cycle380ms variants rejected before learning. Three frozen upper-objective fixtures document the band/symmetry gap;no new posture candidate deployed. First zero-clip validator assumption failed and is preserved;isolated1.5 elbow clips are reported,no ceiling. Updated actual joint/target/velocity/bias plots and COM/contact audit;current full learner resumed. Next: bounded phase-balanced shoulder-bias continuation/control/all gates,retention-preserving direction diagnosis,and fresh resource/full-state audit.


### Closing431 pair — October7 23:46 check

Closing evidence 2026-10-08T00:08:38.898500+00:00:same R40/39000,one actual worker and55 hashes. 9063 completed+50 hash-proven recovered=9113 retained current-run,+27507 inherited;25 partial434 updates separate. Newest exact431 nominal432/pushed433 has3.5 signed3.443/3.452m/s,lateral0.270/0.242,improved from4280.351/0.326 but still fails world-lateral. Retained1.5 flight improves0.444/0.542% to1.187/1.208%,still below1.5%;0.15 signed travel worsens to0.058/0.062m/s and fails nettravel. All survive;every other saved case including3m passes. Preserve this useful nonmonotonic progress,no new qualification. New431 headRMS3.66deg,shoulder pitch ranges12.46/11.27deg and phases173/-145deg;angle/target/velocity/slow-bias plots inspected. Opposite held3.5 biases and2.5/3 slow reversals persist,no human/Unitree/optical-gaze certification. Same full learner retains all state,upper and both priors20batches/update. Resumed43164 real6144 updates has exact428 full load/zeroKL/command mismatch,minGPU1922MiB/host7191MiB,private+6MiB,Torch allocation unchanged. Completed-main rates36.40k active/30.67k worker/19.35k chain,6358.9s documented pauses excluded. No source/reward/gain/cap/optimizer/actor reset;same ACTIVE5h scope.5ms parity/filter experiments remain428-specific. Preserve improving gait while testing focused upper mean-bias/control and retention;do not restart merely from dated plateau.
Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_2346\REPORT.md

## Five-hour check October8 04:46 UTC — 2026-10-08T05:51:49.972831+00:00

The same R40 full learner and supervisor 39000 are running at 2026-10-08T05:51:11.727972+00:00, with one actual worker and 55 verified hashes. It retains 11111 completed + 50 verified recoveries = 11161 current-run updates, plus 27507 inherited; 0 partial updates are separately unqualified. Exact actor 524_run_350_train_172, nominal/pushed: 3.5 command signed speed 3.388/3.398, lateral 0.184/0.163 m/s; failures ['world_lateral_speed']/['world_lateral_speed']. Prior failures: {'nominal': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 2.5, 'failures': ['foot_slip']}, {'command': 3.0, 'failures': ['world_lateral_speed']}], 'pushed': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 2.5, 'failures': ['foot_slip']}, {'command': 3.0, 'failures': ['world_lateral_speed']}]}. The new phase-balanced two-shoulder-row mean candidate is REJECTED for immediate deployment. No production source, reward, gain, cap or learned-state reset; no diagnostic actor deployed. The same supervisor resumed after 2782.9s. All comparison components were independently validated, but the harness exited1 at its final 3008.8/2600s guard. Required actual upper joint, target, velocity and bias timelines were inspected; head RMS 3.68 degrees. Human/Unitree motion and optical gaze remain uncertified. Main rates: 35.59k active / 29.86k worker / 18.77k whole-chain samples/s. Denominators differ and documented suspensions are excluded. Existing upper and both prior auxiliaries remain at 20 batches/update, gains 35/1 and full learned state preserved. Same ACTIVE five-hour endpoint; no5/4 or relevant ceiling proof.

The phase-balanced shoulder-mean candidate is rejected for immediate deployment. At128 updates its standard3.5m/s nominal/pushed lateral speeds are0.158/0.138m/s;nominal fails and the standard pushed case passes. Both independently seeded current cases still fail direction,with lateral0.156/0.154m/s. The0.15 net-travel and1.5 geometric-flight failures persist,and nominal2.5 still slips. At64 it added2.5 slip and3m direction failures where the matched control passed. More decisively for its intended arm goal,the representative3.5 shoulder held-bias measure is19.60deg versus16.76deg in control. The noncancelling slow RMS confirms this is a real held offset,not just an averaging artifact. Some lower-speed bias,head and128-update3m retention results improve,so the hypothesis is not declared universally ineffective. It does not demonstrate the intended current-speed upper-motion benefit with complete retention. No diagnostic actor or new gradient enters production;the existing upper refinement and original full learner continue. Actual training-bin means differ from closed-loop output means;that is a distribution/feature-response lead,not a proved runtime fault. All planned components were independently validated despite the harness final overall-time failure,and the same supervisor resumed safely. Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261008_0446\REPORT.md. Controlled candidate, smoke, failed attempts, complete checkpoints and source hashes are preserved. No production change. Next: Upper motion remains a priority. The new phase-balanced ensemble shoulder-mean candidate is rejected after exact64/128 controls and independent states. At128 its3.5 held-bias measure19.60deg exceeds control16.76deg,even though some slower/head/3m cases improve. Compare actual per-robot cycle and slow target means with truthful training-state coverage,hidden-feature response and per-command auxiliary/PPO gradients. The last candidate minibatch3.25-3.5band mean(-17.72,+1.08deg) differs from final exact3.5 effective outputs(-29.10,+14.63deg);these are different states/times,not a proved bug. Do not blindly raise weights/caps or repeat the same ensemble band. Any individual-cycle or state-conditioned continuation needs narrow output-row derivatives,full-state64/128 controls,all independent prior/current/head/direction/flight gates and seeded replication. Preserve forward head,legs,both teachers,35/1 gains and full learner. Inspect actual angle/target/velocity plus noncancelling slow RMS/range;do not mistake slow reversals for natural swing or fabricate Unitree encoder curves. Compare newer exact original3.5 direction and all stopped/prior gates against524 and separately certified267/349. At524 signed3.388/3.398 and lateral0.184/0.163 remain failed;0.15 net travel,1.5 geometric flight,2.5 slip and3m lateral fail both. Direction improves from518 but retention is nonmonotonic. Use actual per-command advantage/teacher/effective-target and complete objective evidence to test a different bounded retention-preserving continuation before deployment. Existing course1,velocity/cycle filters,fixed-frame kernel,lateral exposure,teacher-context removal,stiffness70 and other rejected candidates stay rejected absent new evidence. No reward-identical waiting,spoof,hybrid,clock patch,tilted-head rollback or from-scratch reset. Verify same39000,one actual worker,55 hashes,local ground,full actor/critic/Adam/exploration and both teachers. Keep27507 inherited plus completed and50 hash-proven R40 recoveries;32 unsaved R39 updates excluded. Exclude this2782.9s suspension only through each timing endpoint;525 nominal wrapper includes it. The comparison finished all components but exited1 at3008.8/2600s final guard;retain failed log and plan future diagnostic budgets separately from saved-boundary waiting/drain rather than falsely claiming success. Keep8192target,300MiBreserve,host guards/fallback and180 continuous healthy TRAINING seconds;Fusion stays closed. Current6144 steady GPU is about1.8GiB free,so do not force8192 from evaluation memory. Only qualified5 or qualified4 plus controlled native limits proving5 blocked ends training/automation;every endpoint needs all unchanged gates and32 independently seeded states/cmd. Use actual actor-specific5ms CAD motor velocity/torque and sustained driving saturation if a high-speed plateau appears;isolated elbow clips or falls are not a ceiling.

## Five-hour check October 8, 09:46 UTC — 2026-10-08T10:19:00.156494+00:00

The same R40 learner and supervisor 39000 are running at 2026-10-08T10:08:27.898716+00:00, with one actual worker and 55 verified hashes. It retains 12,839 completed plus 50 recovered = 12,889 current-run main updates, plus 27,507 inherited; 0 partial updates are separate. Reviewed actor 605, nominal 606/pushed 607, passes both 3.5 m/s running cases: signed speed 3.472/3.474, lateral 0.117/0.120 m/s, genuine flight 37.72/37.49%. The whole stage remains unqualified because 0.15 net travel, 1.5 flight and 2.5 slip fail both tests. Saved gates never qualify newer partial actors. Required actor-605 upper timelines were inspected; head RMS 4.67 degrees, held shoulder biases and asymmetry remain. No human/Unitree motion or optical-gaze certification. Passive truthful-state and actual PPO captures establish a same-actor state-distribution/arm-feedback lead; no new learning candidate was deployed. All 51 unchanged physical replay arrays and cases are bitwise equal. Four diagnostic PPO updates pass full-load, KL and command checks. The same supervisor resumed after 110.1 seconds for the failed import attempt and 197.7 seconds for successful capture; failed evidence is preserved. Resumed block 608 completes 64 genuine 6144-environment updates with exact actor/critic/Adam/exploration. Retained rates are 35.17k active, 29.43k worker-effective and 18.48k whole-chain samples/s. Existing upper and both prior auxiliaries remain at 20 batches/update; 35/1 gains and all production sources, rewards, native caps and learned state remain unchanged. A CPU motor-case attribution error was corrected; no production fault or actuator ceiling is established. The same five-hour automation stays active; computer on.

Preserve the useful current-speed direction improvement. Full retention and upper motion remain incomplete. Four passive gradient batches show both signs of alignment; they do not justify a new deployed loss. Continue existing refinements and test truthful sequential individual-cycle/state-conditioned coverage before a narrow candidate.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261008_0946\REPORT.md. No production change. Passive physics/load/gradient evidence, failed-import cleanup, individual-cycle/state-response data and corrected motor-case attribution are preserved. Next focused tests:

1. Preserve actor 605, whose 3.5 m/s running case passes both tests, and compare newer exact actors with all stopped and earlier-speed gates. Actor 605 still fails 0.15 net travel, 1.5 flight and 2.5 slip. Audit their actual per-command advantages, teacher targets and landing/force events before a narrow retention continuation. Current-speed success alone does not promote the stage; preserve actor 267 and separately certified 349.
2. Continue mandatory upper angle, target, velocity, bilateral and arm-leg timelines. Same actor 605 gives phase-balanced 3.5 m/s shoulder outputs (-16.04, -28.52 degrees) on stable-command training states versus (-28.15, +9.90) on deterministic evaluation states. Measure truthful sequential per-robot cycle coverage, command transitions, exploration and arm-history/hidden-feature response. Stage an individual-cycle or state-conditioned two-final-row derivative only after reset-safe CPU fixtures, physical response, exact full-state 64/128 controls, all gates and independent seeds. The ensemble-mean candidate remains rejected. Preserve forward head, legs, teachers, 35/1 gains and all learned state; no observation substitution, hybrid or imposed reference curve.
3. Verify supervisor 39000, one worker, 55 hashes, local ground, exact full-state teachers and actual retained samples. Preserve 27,507 inherited updates plus completed and 50 verified R40 recoveries; exclude 32 unsaved R39 updates. Exclude this check's 110.1-second failed-import and 197.7-second successful-capture pauses through each timing endpoint. Keep target 8192, reserve 300 MiB, host guards and 180 continuous healthy training seconds; do not force a ramp from evaluation memory. Correctly mapped actor-605 motor samples show right ankle motor 2 at/above cap in 1.96% of 10 Hz samples. If a high-speed plateau appears, measure current-actor 5 ms requested/applied torque and sustained driving clips before any limit conclusion. Only qualified 5 m/s or qualified 4 m/s with causal limits blocking 5 ends the same five-hour heartbeat.

Closing evidence for the October 8 09:46 UTC check:

Closing owned evidence 2026-10-08T10:19:41.335832+00:00: same R40 supervisor 39000, one actual worker and 55 hashes. It retains 12,903 completed + 50 recovered = 12,953 current-run updates, plus 27,507 inherited. Newer exact actor 608, matching nominal 609/pushed 610, reaches 3.480/3.488 m/s at command 3.5, but lateral 0.265/0.245 fails both direction tests. Preserve actor 605's separately passing current-speed pair; improvement is nonmonotonic. Actor 608 recovers 2.5 slip in both tests; 0.15 net travel, 1.5 flight and 3.0 lateral still fail. All survive. Actor-608 upper angle/target/velocity/bias plots were also inspected: head RMS 4.00 degrees, shoulder pitch ranges 9.77/6.87 degrees, arm/hip phases 161/155 degrees. Held biases, slow 2.5/3 m/s reversals and fast asymmetry persist; natural motion remains uncertified. Latest completed block 611 has 64 genuine 6144-environment updates, exact full-state load, zero KL/command mismatch and all three auxiliaries at 20 batches/update. Minimum free GPU 1923 MiB and host 8391 MiB, private growth 9 MiB, Torch allocation unchanged. Retained rates 35.15k active / 29.41k worker-effective / 18.47k whole-chain samples/s. The 605 passive state/gradient and motor diagnoses remain actor-specific. No production change or diagnostic actor deployed; same active five-hour goal, computer on.

Evidence: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261008_0946\closeout_gate_review.json; E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261008_0946\closing_newer_upper.

## Five-hour check — October 8, 14:46 UTC — 2026-10-08T17:05:39.720551+00:00

The same R40 full learner and supervisor 39000 are running at 2026-10-08T17:01:51.212483+00:00, with one actual worker and 55 verified frozen hashes. It retains 15,207 completed plus 50 hash-proven recovered = 15,257 current-run main updates, plus 27,507 inherited; 14 partial updates are separate. Exact actor 719_run_350_train_237, nominal/pushed: command 3.5 reaches 3.351/3.358 m/s signed speed and 0.260/0.243 m/s lateral speed; failures ['world_lateral_speed']/['world_lateral_speed']. Prior failures: {'nominal': [{'command': 1.5, 'failures': ['geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}], 'pushed': [{'command': 1.5, 'failures': ['geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}]}. Saved gates do not qualify newer actors. The individual completed-cycle shoulder-mean candidate is rejected for deployment. At 64 updates it reduces fast shoulder bias but increases 3.5 m/s lateral drift to 0.461/0.442 m/s versus unchanged control 0.096/0.098, and adds a nominal 1 m/s support failure. At 128 updates its drift remains 0.381/0.358 versus control 0.078/0.109; the support failure persists and retained 1.5 m/s geometric flight still fails. Nominal fast shoulder slow RMS is 16.46/4.56 degrees versus control 15.71/6.17, so the earlier improvement does not remain superior on both sides. Actual 2.5 and 3 m/s target biases still reverse slowly. Independent initial-state tests preserve the failure evidence. Keep the exact original full-state learner, existing upper refinement and all gates; no diagnostic actor or new gradient enters production. This rejects the tested continuation, not every possible cycle-based refinement. Head RMS is 4.61 degrees. Actual upper angle/target/velocity, bilateral and arm-leg coordination, and noncancelling slow-bias timelines were inspected. Human/Unitree motion and optical gaze remain uncertified. The same supervisor resumed after 2936.3 seconds; both failed diagnostic attempts also resumed safely and their evidence is preserved. First resumed block 719 completed 64 genuine updates at 8,192 environments with exact full-state load, zero KL/command mismatch, stable memory and all three auxiliaries at 20 batches/update. Current operation is 722_run_350_train_238, count 8192. Main rates are 34.77k active, 29.03k worker-effective and 18.22k whole-chain samples/s. Original 35/1 gains, native caps, reward v8/verifier v6, truthful 400-input/23-output inference, full learned state and all three 20-batch auxiliaries are retained. No diagnostic actor is deployed. The same five-hour automation remains active; computer on.

The individual completed-cycle shoulder-mean candidate is rejected for deployment. At 64 updates it reduces fast shoulder bias but increases 3.5 m/s lateral drift to 0.461/0.442 m/s versus unchanged control 0.096/0.098, and adds a nominal 1 m/s support failure. At 128 updates its drift remains 0.381/0.358 versus control 0.078/0.109; the support failure persists and retained 1.5 m/s geometric flight still fails. Nominal fast shoulder slow RMS is 16.46/4.56 degrees versus control 15.71/6.17, so the earlier improvement does not remain superior on both sides. Actual 2.5 and 3 m/s target biases still reverse slowly. Independent initial-state tests preserve the failure evidence. Keep the exact original full-state learner, existing upper refinement and all gates; no diagnostic actor or new gradient enters production. This rejects the tested continuation, not every possible cycle-based refinement.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261008_1446\REPORT.md.

Next focused tests:

1. Read fresh exact current-speed and every prior gate first, preserving useful actors. The matched unchanged control passes 3.5 m/s but fails 0.15 net travel and 1.5 true flight; newer original 719 recovers 0.15 in both tests while 1.5 flight and 3/3.5 direction still fail. Keep exact actor scope and prioritize persistent failures. Prioritize their per-command actual advantages, teacher effective targets and contact/landing events. If those failures persist, test one bounded exact-command retention rehearsal on the CURRENT full state against unchanged 64/128 controls and independent training/initial-state seeds, retaining current direction, all other speeds and posture. The earlier 44% exposure, course, fixed-frame, teacher-context and other rejected variants remain rejected absent new evidence; do not change arbitrary weights or relax gates.
2. Inspect every actual head/shoulder pitch+roll/elbow angle, target and velocity timeline, bilateral and arm-leg phase, and noncancelling slow bias. The new individual-cycle mean candidate is rejected: early fast bias reduction weakens direction/support and is not sustained on both shoulders at 128. Quantify current lower-output sensitivity to arm-history variation on saved truthful states and distinguish physical arm/base coupling from learned feedback dependence in bounded normal-limit cohorts before another shoulder derivative. Latest cycle snapshots cover only a subset of robots/states; they do not certify deterministic evaluation coverage. Preserve head, gait, full state, both teachers, 35/1 gains and existing upper 20 batches/update. No production observation substitution, imposed reference curve, gain/cap increase or repeat of the same failed mean candidate without new evidence.
3. Verify the same supervisor 39000, one actual worker, 55 frozen hashes, local ground, exact actor/critic/Adam/exploration and both teachers after diagnostic resume. Count 27,507 inherited plus fresh completed and hash-proven recovered main updates; exclude unsaved R39, capacity, smoke and diagnostic learning. Exclude all three documented suspensions through each timing endpoint. Keep the 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy TRAINING seconds before ramps; Fusion stays closed. Final 5 m/s or conditional 4 m/s requires every unchanged gate and 32 independently seeded states per command, with actor-specific native 5 ms CAD speed/torque and sustained driving-saturation evidence proving 5 blocked for the 4 m/s fallback. Exact 716 nominal/pushed 10 Hz audits find left ankle motor 2 at/above cap in 2.031%/1.997% of 3.5 m/s samples (p99 1.639/1.652 times cap), with right motor 2 at 1.111%/0.833%. Attribute duration and requested/applied 5 ms driving speeds/torques on the newer plateau actor before inferring a ceiling; sampled overspeed alone is insufficient.

Closing owned evidence 2026-10-08T17:07:24.726091+00:00: same R40 supervisor 39000, one actual worker and 55 verified frozen hashes. It retains 15,271 completed + 50 recovered = 15,321 current-run main updates, plus 27,507 inherited. Both resumed blocks 719 and 722 complete 64 genuine updates at 8,192 environments with exact actor/critic/Adam/exploration, zero pre-update KL/command mismatch and all three auxiliaries at 20 batches/update. Last block minimum GPU 1673 MiB, host 6546 MiB, private growth 10 MiB and Torch allocation growth 0 MiB. The snapshot is evaluating newer actor 722; its matching pair is incomplete and no qualification is inferred. The reviewed gate/upper pair remains exact 719, all-command traces and motor audit remain exact 716. Current selected next training count is 8,192; evaluation's 320 robots/free memory do not establish training capacity. Read closing_owned_learning_verification.json and final_owned_cpu/live_handoff.json, then fresh ACTIVE_RUN/state. No production source/reward/gain/cap/actor/optimizer change or diagnostic actor is deployed. Same active five-hour endpoint; computer on.

<!-- adaptive_check_20261008_1946_record -->
## Five-hour check — October8,19:46UTC — 2026-10-08T23:01:28.782701+00:00

The same R40 full learner and supervisor 39000 are running at 2026-10-08T22:56:16.427327+00:00, with one actual worker and 55 verified frozen hashes. It preserves 16,807 completed plus 50 hash-proven recovered = 16,857 current-run main updates, plus 27,507 inherited. The 60 partial updates are separate. Reviewed exact actor 794 reaches 3.465/3.472 m/s signed speed at command 3.5, with 0.212/0.200 m/s lateral drift. Both direction tests fail. Retained 0.15 net travel and 1.5 flight/geometric flight fail both tests; all other stopped/prior cases, including recovered 3 m/s direction, pass. All robots survive. Saved gates do not qualify newer partial learning. The focused prior-bucket rehearsal is rejected after two independent learning seeds, exact full-state 64/128 controls and 32 individually seeded initial states per command. It lowers 1.5 flight in both seeds at 128 updates and adds 1 m/s support failures; 0.15 recovery is seed-dependent. No diagnostic actor or sampler enters production. A separate 448-robot, 20-second normal-limit comparison on exact 773 gives the qualified 349 teacher 1.83–1.86% genuine 1.5 flight and higher complete mean reward, versus current 0%. All 4,000 native 5 ms request/application samples have zero driving-speed clips in every cohort. Ankle overspeed occurs while braking. This is actor-specific evidence, not a 4/5 m/s endpoint. Actual head/shoulder/elbow angles, targets, velocities, arm-leg coordination and slow bias were inspected at 767, all four diagnostic 128-update actors, and 794 at every command. Actor 794 has 3.31-degree running head RMS, but held shoulder biases, unequal swing/elbows/roll and gait-linked neck oscillation remain. CPU arm-history substitutions change effective lower outputs by 0.324 RMS at 3.5, showing learned feedback dependence without identifying mechanical coupling. Human/Unitree encoder likeness and optical gaze remain uncertified. The same supervisor resumed after 5,405.4 seconds. Block 776 completed 64 genuine updates at 8,192 environments with exact actor/critic/Adam/exploration load, zero KL/command mismatch and all three auxiliaries at 20 batches/update. Retained rates are 35.10k active, 29.23k worker-effective and 18.44k whole-chain samples/s, excluding documented suspensions through each endpoint. The last actual main block kept at least 1,495 MiB GPU and 6,210 MiB host memory free, with private growth 7 MiB and unchanged Torch allocation. No production source, reward, gain, cap, actor or optimizer reset was deployed. Original 35/1 gains, truthful 400-input/23-output inference, teachers, upper refinement and iteration 4 remain. Keep the same active five-hour check and computer on.

The focused failed-prior rehearsal candidate is rejected after two independent learning seeds and exact full-state 64/128 controls. Only the existing 24% prior bucket changes: the expected 0.15/1.5 fractions rise from 3% to 4.8%, while standing 32% and frontier 44% remain. At 128 updates, candidate 1.5 genuine flight is 1.021/0.969% versus control 1.222/1.240% for seed 157, and 0.611/0.559% versus 1.069/0.986% for seed 167; all are below 1.5%. Seed 157 candidate loses 0.15 travel and both 1 m/s support passes. Seed 167 recovers 0.15 and 3 m/s direction but also fails both 1 m/s support tests, where its control pushed test passes. Extra individually seeded initial states reproduce the failures. Neither seed qualifies 3.5 direction. Preserve the original learner; no diagnostic actor or sampler enters production. This rejects the tested allocation, not every retention method or the possibility of convergence.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261008_1946\REPORT.md.

Next focused tests:

1. Preserve the current R40 full state and compare fresh matching 3.5 m/s direction, 0.15 m/s net travel, 1.5 m/s genuine flight and every prior gate. The focused prior-bucket reweighting is rejected at 64/128 updates across learning seeds 157/167 and paired individually seeded initial states; do not repeat it without new evidence. Under native limits, qualified actor 349 still gives 1.83–1.86% genuine 1.5 m/s flight and higher complete mean reward than exact 773. Resolve phase-specific teacher effective-target errors, post-clamp/joint/slew response and sole/contact timing on truthful settled and transitioning histories before one bounded dynamics-aware, command-local continuation. Require exact-current full-state 64/128 controls, two learning seeds and all unchanged gates. Positive normalized advantages, smaller teacher norms and mixed alignment do not prove a sampling defect or universal gradient opposition. Preserve 605 current-case evidence, 267 all-prior and separately certified 349.
2. Inspect every actual head/shoulder pitch and roll/elbow angle, target and velocity timeline, bilateral amplitude, arm-leg phase and noncancelling slow bias. In the first actual 773 minibatch, same-command/phase-bin donor arm-history substitutions change effective lower outputs by 0.324 RMS at 3.5 m/s. This CPU sensitivity does not identify mechanical coupling or a safe arm repair. Before another shoulder derivative, run bounded truthful free-base arm-target/base-response cohorts and reset-safe sequential state coverage to separate physical coupling from learned feedback dependence. Preserve the existing upper 20 batches, forward head, legs, both teachers and 35/1 gains. Prior mean, stiffness, exposure, course and teacher candidates remain rejected absent new evidence. No production observation substitution, imposed reference curve, hybrid or cap increase; mapped human/Unitree encoders and optical gaze remain unverified.
3. Verify supervisor 39000, one actual worker, 55 frozen hashes, local ground and exact actor/critic/Adam/exploration plus both teachers. Count 27,507 inherited updates plus fresh completed and 50 verified R40 recoveries; exclude 32 unsaved R39 and all diagnostic samples. Exclude the documented 5,405.4-second pause only through each timing endpoint. Keep the 8,192 target, 300 MiB reserve, host fallback and 180 continuous healthy TRAINING seconds; Fusion stays closed. Exact 773 native 5 ms requests/applications have zero driving-speed clips through 3.5 m/s; overspeed ankles are braking. Later 794 sparse 10 Hz motor samples cannot inherit that substep attribution. If a newer high-speed plateau persists, measure actor-specific 5 ms requested/applied torque, CAD speed, available torque and sustained driving saturation. Only fully qualified 5 m/s or qualified 4 m/s with causal native limits blocking 5 ends training/automation, with all gates and 32 independently seeded states per command.
<!-- /adaptive_check_20261008_1946_record -->

<!-- adaptive_check_20261008_1946_closing_owned -->

Closing owned evidence 2026-10-08T23:02:37.264481+00:00: the same R40 supervisor 39000 has one actual worker and 55 verified frozen hashes. It retains 16,871 completed plus 50 recovered = 16,921 current-run main updates, plus 27,507 inherited; 13 observed partial updates remain separate. Exact actor 797, matching nominal 798/pushed 799, reaches 3.461/3.469 m/s signed at command 3.5, with 0.151/0.146 m/s lateral drift. Both direction tests still fail. Retained 0.15 net travel, 1.5 geometric flight and 2/3 m/s direction now fail both tests; all other saved cases pass and all robots survive. Retained 1.5 true flight is 0.0000/0.0729%. Improvement is nonmonotonic: preserve 794's separately recovered 2/3 m/s cases. Four newer 797 angle/target, velocity and slow-bias plots were visually inspected; all-command population statistics are saved. Running head RMS is 3.27 degrees, but held shoulder bias and unequal swing/elbow/roll remain. Full all-command nominal/pushed trace QA stays explicitly bound to 794; the 773 native 5 ms motor attribution stays separate. Block 797 completed 64 genuine updates at 8,192 environments, loaded exact 794 actor/critic/Adam/exploration, had zero KL/command mismatch and all three auxiliaries at 20 batches/update. Minimum free GPU/host memory was 1493/6010 MiB, private growth 9 MiB and Torch allocation growth 0 MiB. Retained rates are 35.11k active / 29.24k worker-effective / 18.45k whole-chain samples/s, with 18662.7s documented pauses excluded. No diagnostic actor, sampler, reward, gain, cap or optimizer reset entered production. The current operation is 800 at 8,192. Inventory: 40 physical runs, zero differing hashes/read warnings; eight historical missing manifests remain documented. Read final_owned_gate_review.json, final_owned_learning_verification.json, final_owned_cpu/live_handoff.json and closing_newer_upper, then fresh ACTIVE_RUN/state. The same five-hour check remains active toward qualified 5 m/s or qualified 4 m/s with causal native limits blocking 5; computer on.

Evidence: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261008_1946\closing_owned_addendum.json.
<!-- /adaptive_check_20261008_1946_closing_owned -->

<!-- adaptive_check_20261009_0046_record -->
## Five-hour check — October 9, 00:46 UTC — 2026-10-09T01:51:43.073430+00:00

Same R40 full learner/supervisor 39000 is running at 2026-10-09T01:50:08.566848+00:00, with one actual worker and 55 verified frozen hashes. It retains 17,831 completed + 50 recovered = 17,881 current-run main updates, plus 27,507 inherited; 0 partial updates are separate. Exact 839_run_350_train_277 at command 3.5 gives 3.410/3.414 signed and 0.221/0.205 lateral m/s; failures ['world_lateral_speed']/['world_lateral_speed']. Prior failures: {'nominal': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['flight', 'geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}], 'pushed': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['flight', 'geometric_flight']}]}. Saved gates bind that exact actor, not newer partial learning. No new production source,reward,gain,cap or optimizer change is justified by these controlled diagnostics. Existing upper fine-tuning continues at20 batches/update. The improved neutral head and learned gait are preserved; no from-scratch restart or diagnostic learned actor is deployed. Exact839:1088 robots20s,two seeds32 paired initial states/cohort. Controls,truthful command transitions and own-feedback shoulder offsets all survive. Added768-robot sham shows all zero-offset body-replay cohorts fall,so replay falls cannot identify mechanical arm coupling. True-feedback5deg shoulder shifts have command-dependent direction tradeoffs and alter lower outputs; no uniform correction is selected. Passive qualified349 target reconstruction gives about2.27/2.25deg lower-target RMS at0.15/1.5,25%/33% joint clipping,about4.4%1.5 slew limits and no sampled CAD rejection.4000 native5ms baseline samples have zero driving-speed clips. Four actual PPO captures have mixed prior/upper alignment; no universal opposition is inferred. No64/128 candidate learning comparison is claimed this check. Required actual head/shoulder/elbow angle, target, velocity and slow-bias timelines were inspected at initial 827 and closing 839_run_350_train_277 across every command in both tests. Head RMS 3.81 degrees; held arm biases and unequal motion remain. No human/Unitree encoder or optical-gaze certification. Two failed probe interfaces resumed safely after 275.0/291.8 seconds; main diagnosis and sham resumed the same supervisor after 450.7/246.8 seconds. First resumed block 842_run_350_train_278 completes 64 genuine updates at 8192 with exact actor/critic/Adam/exploration, zero KL/command mismatch and all three auxiliaries at 20 batches/update. Rates 35.26k active / 29.33k worker-effective / 18.59k whole-chain samples/s; documented pauses excluded through each endpoint. No production source, reward, gain, cap, actor or optimizer reset and no diagnostic controller deployed. Original 35/1 gains, truthful 400/23 inference, teachers, upper refinement and iteration 4 remain. Same active five-hour endpoint; computer on.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261009_0046\REPORT.md.

1. Preserve the current full learner and compare exact3.5 direction,0.15 net travel,1.5 genuine/geometric flight and all prior gates,keeping833 recovered3m,605 current-case evidence,267 and separately certified349. The present true-feedback arm cohorts are stable but neither offsets nor passive labels qualify a repair. Run a bounded normal-limit diagnostic at0.15/1.5 comparing current,qualified349,and teacher body with the current learned upper controller under truthful feedback; these diagnostic hybrids must never enter production. Test whether349 labels remain physically useful with current arm histories before any arm-context/dynamics-aware command-local training labels.3.5 teacher output is outside its certificate and must not supply a label. Only stage a justified narrow current-full-state continuation after physical validation,64/128 unchanged controls,two learning seeds and all independent prior/current/head/direction/flight gates. Do not repeat focused-prior bucket reweighting or other rejected variants without new evidence.
2. Mandatory actual head/shoulder pitch+roll/elbow angle,target,velocity,arm-leg coordination and noncancelling bias timelines remain.839 closed-loop5deg offsets change effective lower outputs about0.23-0.25RMS at3.5; one sign lowers3.5 drift but worsens1.5 drift,the other reverses that tradeoff. A uniform shoulder shift is not selected.768-robot replay sham shows every zero-offset replay cohort falls too,with pre-offset divergence; do not repeat body-action replay as mechanical isolation or infer an arm ceiling from it. Use truthful closed-loop,command-specific physical response and sequential per-robot target coverage before a narrow output-row upper refinement. Preserve forward head,35/1 gains,legs,teachers,full state and existingupper20 batches; require64/128 controls and every gate. No production observation substitution,hybrid,imposed encoder curve,cap increase or repeated rejected mean/stiffness changes.
3. Verify same39000,one actual worker,55 frozen hashes,local ground,exact actor/critic/Adam/exploration and both teachers. Count27507 inherited plus completed and50 recovered; exclude32 unsavedR39 and diagnostic/smoke samples. Exclude274.97/291.84s failed-probe pauses,450.65s main diagnosis and246.77s replay-sham pause only through each rate endpoint. Keep8192 target,reserve300,host guards/fallback and180 continuous healthy training seconds;Fusion stays closed.839 baseline native5ms requests/applications have zero driving-speed clips at0.15/1.5/3.5; ankle overspeed is braking and this is actor-specific. Later10Hz audits cannot inherit that attribution. Final5 or conditional4 still needs every unchanged gate and32 independently seeded initial states per command,with causal native limits blocking5 for the4 fallback.
<!-- /adaptive_check_20261009_0046_record -->

<!-- adaptive_check_20261009_0547_record -->
## Five-hour check — October 9,05:47 UTC — 2026-10-09T06:52:04.651766+00:00

The same R40 full learner and supervisor 39000 are running at 2026-10-09T06:44:58.142747+00:00, with one actual worker, 55 verified frozen hashes and 53 protected authored worker files. It retains 19,559 completed + 50 hash-proven recovered = 19,609 current-run main updates, plus 27,507 inherited; 0 partial updates are separate. Reviewed exact actor 920_run_350_train_304 reaches 3.479/3.484 m/s signed at command 3.5, with 0.153/0.145 m/s lateral drift; both direction gates fail. Prior failures: {'nominal': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 0.35, 'failures': ['bilateral_stance']}, {'command': 1.5, 'failures': ['flight', 'geometric_flight']}, {'command': 2.0, 'failures': ['world_lateral_speed']}, {'command': 2.5, 'failures': ['world_lateral_speed']}, {'command': 3.0, 'failures': ['world_lateral_speed']}], 'pushed': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['flight', 'geometric_flight']}, {'command': 2.0, 'failures': ['world_lateral_speed']}, {'command': 2.5, 'failures': ['world_lateral_speed']}, {'command': 3.0, 'failures': ['world_lateral_speed']}]}. Actors 908/917 separately passed both current 3.5 cases but not every prior; the stage remains unqualified. Saved gates never qualify newer partial actors. Exact 917 has 1152 robots over 20 seconds in the compatibility test and 704 in the head/arms split, each with two seeds and 32 individually seeded paired states per cohort; all survive. Qualified 349 gives about 1.81% genuine flight at 1.5 m/s. Teacher body with current whole upper gives about 0.94%, current arms only about 1.075%, and current head only 1.478/1.502%. At 0.15 m/s, current arms on teacher body/head reach about 0.285 m/s versus the teacher's 0.184; head only stays near 0.184. Context-conditioned whole-upper transfer is also unsafe. These are closed-loop component effects, not pure mechanical isolation. Every diagnostic composite remains outside production. Four actual PPO captures show mixed prior/upper alignment. No new 64/128-update candidate learning comparison or qualification is claimed. Actual upper angle, target, velocity and bias timelines were inspected at 908 and 920, both tests and all commands. Actor 920 head RMS is 4.38 degrees; shoulder pitch ranges are 11.74/10.25 degrees. Held biases, unequal roll/elbows and slow reversals remain. Human/Unitree encoder likeness and optical gaze remain uncertified. The same supervisor resumes after 126.955-second failed-metadata, 476.254-second main compatibility and 371.378-second component-test pauses; the preboundary lookup attempt never paused training. Resumed blocks 920/923 each finish 64 genuine updates at 8192, loading exact 917/920 actor, critic, Adam and exploration, with zero KL/command mismatch and all three auxiliaries at 20 batches/update. Last actual main minimum free GPU/host memory is 1454/6228 MiB, private growth 10 MiB and Torch allocation growth 0 MiB. Retained rates are 35.50k active / 29.47k worker / 18.78k chain samples/s; documented suspensions are excluded through each endpoint. No production source, reward, gain, cap, actor or optimizer reset; no diagnostic controller or learned actor deployed. Existing upper and both teacher auxiliaries at 20 batches/update, 35/1 gains, truthful 400/23 inference, native CAD caps and iteration 4 remain. Same active five-hour endpoint; computer on.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261009_0547\REPORT.md.

1. Preserve the current full learner, the separately passing 3.5 m/s pairs 908/917 and 605, actor 267 with all prior cases passing, and separately certified 349. Compare fresh 3.5 m/s direction, 0.15 m/s net travel, 1.5 m/s genuine/geometric flight and every earlier gate. Reviewed 920 failed 2/2.5/3 m/s lateral direction; newer 923 recovers all three in both tests. Both 0.35 m/s stance and 0.65 m/s speed now fail, alongside the persistent 0.15 path and 1.5 flight failures. Preserve that actor-specific recovery and audit these fresh slow cases too. Two-seed component tests on exact 917 show that teacher legs with current arms overspeed at 0.15 and lose flight at 1.5; head changes also affect flight. This identifies closed-loop coordination, without isolating mechanical coupling. Next physically test the current body with qualified 349 arms only while retaining the CURRENT head, under each controller's truthful feedback. Resolve phase-specific effective leg targets, sole unloading and torso response on settled and transitioning histories. The inverse whole-upper test reached only about 0.9% flight; the arms-only inverse is untested. Never use 349 labels above its 3 m/s certificate. Select a command-local, state-conditioned arm-row or coupled transfer only after physical validation, exact current full-state 64/128-update controls, two learning seeds, all unchanged gates and independently seeded initial-state replication. Keep prior reweighting and rejected mean/gain/context/course candidates rejected without new evidence.
2. Every check inspect actual head yaw/pitch, shoulder pitch/roll and elbow angle, physical target and velocity timelines at every speed, including bilateral amplitude, arm-leg phase and noncancelling slow bias. Reviewed 920 has 4.38-degree head RMS and 11.74/10.25-degree shoulder pitch ranges, with opposite held fast biases and slow 2.5/3 m/s reversals. The component split attributes most slow overspeed and the larger 1.5 m/s flight loss to the current arm controller, but current head alone yields 1.478/1.502% flight and is not qualified. Do not deploy diagnostic composites or infer a purely mechanical cause. Preserve upper refinement at 20 batches/update, forward head, 35/1 gains, legs, both teachers and full learned state. Collins 2009 Figure 6 provides human shoulder/elbow time plots, but numerical curves and exact speed, axes, rest poses and morphology mapping remain unverified. Mapped Unitree encoder curves and calibrated optical gaze remain unavailable. Do not fabricate curves, substitute production observations, raise caps or restart from scratch.
3. Verify supervisor 39000, one actual worker, 55 frozen hashes, 53 protected authored files, local ground, exact actor/critic/Adam/exploration and both teachers. Count 27,507 inherited updates plus fresh completed and 50 hash-proven R40 recoveries; exclude 32 unsaved R39 updates and all smoke, capacity and diagnostic samples. Exclude the 126.955-second failed-metadata, 476.254-second compatibility and 371.378-second component-test pauses only through each timing endpoint. The canceled preboundary lookup never paused training. Resumed 920/923 load exact 917/920 respectively, with zero KL/command mismatch and all three auxiliaries at 20 batches/update. Keep target 8192, reserve 300 MiB, host guards/fallback and 180 continuous healthy TRAINING seconds; Fusion stays closed. Recorded 917 native 5 ms motor samples have zero driving clips and braking overspeed. The later 920 10 Hz ankle cap crossings cannot inherit that attribution. Only qualified 5 m/s, or qualified 4 m/s with causal native limits proving 5 blocked, ends training and automation; every endpoint needs all unchanged gates and 32 independently seeded initial-state trials per command.
Closing owned observation 2026-10-09T06:54:13.012311+00:00: the same R40 supervisor 39000 has one actual worker and 55 verified hashes. It retains 19,623 completed + 50 recovered = 19,673 current-run main updates, plus 27,507 inherited. Newer exact actor 923, matching nominal 924/pushed 925, reaches 3.527/3.532 m/s signed at command 3.5 with 0.180/0.165 m/s lateral drift; both direction gates still fail. Both 2/2.5/3 m/s cases recover all checks; preserve that improvement separately from 920. Both 0.15 net-travel, 0.35 bilateral-stance, 0.65 speed and 1.5 flight/geometric-flight tests fail. All robots survive. Full all-command upper timeline QA remains bound to 920, while native 5 ms teacher/component diagnostics remain bound to 917; no newer posture or motor qualification is inferred. Completed block 926 has 64 genuine updates at 8192, loads exact 923 actor/critic/Adam/exploration, and has zero pre-update KL/command mismatch with all three auxiliaries at 20 batches/update. Its minimum free GPU/host memory is 1454/6452 MiB, private growth 10 MiB and Torch allocation growth 0 MiB. Retained rates are 35.51k active / 29.48k worker-effective / 18.79k whole-chain samples/s. No production source, reward, gain, cap, actor or optimizer reset and no diagnostic controller deployed. Same active five-hour endpoint; computer on. Read closing_newer_pair.json, closeout_verification.json and fresh ACTIVE_RUN/state.
<!-- /adaptive_check_20261009_0547_record -->

<!-- adaptive_check_20261009_1047_record -->
## Five-hour check — October9,10:47UTC — 2026-10-09T13:21:24.476433+00:00

Closing owned observation 2026-10-09T13:24:37.792646+00:00: the same R40 full-state learner and supervisor 39000 are running, with one actual worker, 55 verified frozen hashes and 53 unchanged authored worker files. It retains 21,351 completed plus 50 hash-proven recovered = 21,401 current-run main updates, plus 27,507 inherited; 7 observed partial updates remain separate. Exact actor 1007_run_350_train_333, SHA 4c44fbd01761e257a8f901c33054295fb0cbfaaa253e64d4696d2807fac67689, passes both current cases at command 3.5: nominal/pushed signed speed 3.490/3.497 m/s, lateral speed 0.137/0.133 m/s. The complete stage remains unqualified: 0.15 m/s path efficiency and 1.5 m/s geometric flight fail both matching tests. Every other stopped/prior case passes both. Saved gates apply only to this actor. Full all-command upper angle/target/velocity and bias QA remains explicitly bound to actor 1004; no newer posture or natural-motion certification is inferred. Its head RMS is 3.10 degrees, with held arm biases and unequal motion.

The tolerance-width slow reward is REJECTED after exact full-state 64/128-update comparisons, two learning seeds and independently generated initial states. Both ordinary 128-update control nominal/pushed pairs pass 3.5 m/s direction; all four candidate cases fail, with 0.26–0.33 m/s lateral speed versus 0.12–0.13 for controls. Explicit current-actor 4, 4.5 and 5 m/s tests fail gait/direction. The actor-specific 5 ms motor probe has zero driving-speed clips, so those failures do not establish a velocity ceiling or qualify the 4 m/s fallback. Paired PPO captures reproduce a local cross-command gradient change through global advantage normalization, even with identical fast-command returns. This is a research lead, not proof of the full cause of long-term regression. The inverse current-head/body plus qualified-arms physical test also fails the bounded screen. No diagnostic actor, composite, reward or optimizer reset entered production.

The original supervisor resumed after the 5,449.8-second reward comparison and 1,028.5-second higher-speed diagnosis. Genuine main blocks load exact actor, critic, Adam and exploration; both teachers and upper refinement remain at 20 batches per update. The latest completed 64-update block at 8,192 environments keeps at least 1348 MiB GPU and 6051 MiB host memory free. Retained active/worker-effective/whole-chain rates are 35.71k/29.61k/18.94k samples/s; the denominators differ and documented suspensions are excluded through each endpoint. Original 35/1 gains, truthful 400-input/23-output inference, native CAD caps and iteration-4 execution remain. The CPU numeric-order reporting repair passes five checks and does not alter frozen training sources. Keep the same active five-hour heartbeat and verified 5 m/s or qualified 4 m/s plus causal-limit endpoint. Fusion stays closed, queued work and videos are preserved, and the computer stays on.

Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261009_1047\REPORT.md.

1. The tests now explicitly extend to 4, 4.5 and 5 m/s, but current-policy extrapolation does not qualify those speeds. Preserve the full R40 learner and separately passing 992/605/908/917 current-speed actors, 267 all-prior and certified 349. Compare fresh exact 3.5 direction and every retained speed, especially slow tracking and 1.5 genuine/geometric flight. Freshly requalify the earlier neutral-head all-prior actor 267 in the current runtime and compare its effective targets, full physical return and compatibility with the current upper controller against qualified 349 at 0.15/1.5. Neither earlier actor supplies labels above its independently reverified certificate. Use that previous-checkpoint evidence and the new exact paired reward/critic/global-advantage evidence to select a retention-preserving continuation. The narrow instantaneous slow kernel is rejected in both learning seeds; do not repeat it, reset the learner, relax gates or simply skip retention. Any new reward must have explicit versioned provenance, finite/counterexample checks, unchanged full-state 64/128 controls and independent seeds.
2. The inverse current-body/current-head with qualified-arm cohorts do not meet the bounded physical screen. Do not copy those arm rows into production. Resolve phase-specific leg targets, sole unloading and torso/arm feedback at 0.15/1.5 using the qualified teacher and current truthful states before a coupled command-local training target. Never use 349 labels above its 3 m/s certificate or deploy diagnostic composites. Every check still saves and inspects actual head, shoulder pitch/roll and elbow angles, targets, velocities, bilateral and arm-leg phase and noncancelling slow bias. Preserve forward head, both teachers, upper 20 batches/update and original 35/1 gains. No fabricated human/Unitree encoder curves, production observation substitution, imposed trajectory, hybrid or cap increase.
3. Verify the same owned supervisor 39000, one actual CUDA worker, all 55 frozen hashes/53 authored files, original local ground and exact actor/critic/Adam/exploration plus teachers. Count 27,507 inherited plus completed and 50 proven R40 recoveries; exclude diagnostic/smoke/capacity samples and 32 unsaved R39 updates. Exclude both current bounded pauses only through each timing endpoint. Keep target 8,192, reserve 300 MiB, host guards/fallback and 180 continuous healthy training seconds; Fusion stays closed. The higher-speed 5 ms motor attribution excludes terminal/reset samples. Turning, gait failure and brief clips are not a causal limit. Only qualified 5 m/s, or qualified 4 m/s with controlled native limits proving 5 blocked, all gates and 32 individually seeded states per command, ends training.
<!-- /adaptive_check_20261009_1047_record -->

<!-- adaptive_check_20261009_1548_record -->

## Humanoid five-hour check — October 9, 2026, 15:48 UTC

Closing owned observation 2026-10-09T18:23:43.255727+00:00: the same R40 full learner and supervisor 39000 are running, with one actual worker, 55 frozen hashes and 53 protected authored files. It retains 22,567 completed plus 50 hash-proven recovered = 22,617 current-run main updates, plus 27,507 inherited; 0 partial updates remain separate. Exact actor 1061_run_350_train_351, SHA d9970618093e7df8248453a9aa7d8e196a3dd2a03980c8ce37ec6fc436418e4d, reaches 3.544/3.548 signed and 0.100/0.106 lateral m/s at command 3.5. Current failures []/[]; prior failures {'nominal': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 2.5, 'failures': ['foot_slip']}], 'pushed': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 2.5, 'failures': ['foot_slip']}]}. Saved gates apply only to this actor; no complete 3.5, 4, 5 m/s or causal ceiling endpoint is established. Earlier neutral-head actor 267 is freshly requalified through 3 m/s under current physics in both ordinary and 32-individually-seeded-state tests. Truthful two-seed physical diagnostics support coupled body/arm targets with the current head at 0.15/1.5; the whole-upper composite loses flight. Composites are diagnostic only. The new settled-exact-command coupled target is learning-tested at 64/128 with unchanged full-state controls, two seeds and independent initial states. It remains rejected for immediate deployment because the full gates fail; partial target gains and head/arm tradeoffs are preserved. No production source, reward, gain, cap, actor or optimizer reset. Required all-command head/shoulder/elbow angle/target/velocity and slow-bias timelines are inspected for this exact closing actor and every matched 128 actor. Closing native neck RMS at 3.5 is 3.53 degrees; held bias, unequal swing/roll/elbows remain. Human/Unitree encoder likeness and optical gaze are uncertified. The false-positive URDF-order CPU review and safely resumed aborted evaluation are corrected and preserved; the experimental runtime head mapping was correct. Main rates are 35.83k active / 29.68k worker-effective / 19.05k whole-chain, with documented pauses excluded through each endpoint. Last genuine 8192-environment main block minimum GPU/host memory is 1400/5269 MiB; exact full loads, zero KL/command mismatch and all three 20-batch auxiliaries pass. Same active five-hour check, computer on, Fusion closed, queued work/videos preserved. The latest completed original main block is 1064_run_350_train_352; its full-state proof is separate from the fully plotted 1061 actor. Any newer incomplete or complete gate pair must be read separately before attributing qualification or posture.

Measured since previous check: 1,216 retained updates / 239,075,328samples; active/worker/chain rates37.56k/30.75k/20.75k with explicit pause exclusions.

Report: `E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261009_1548\REPORT.md`. Evidence/decision/matched physical/learning/upper/load proofs and corrected reviewer are nearby.

1. Preserve the current full learner, separately passing current-speed actors 1058/1061, the fresh neutral-head 267 all-prior certificate and certified 349. Compare fresh exact 3.5 direction, 0.15 path efficiency, 1.5 genuine flight and every retained gate. The coupled 267 target improves flight in both seeds but remains unqualified; seed 241 loses 3.5 direction at 64 and worsens slow path/adds 3 m/s direction failures at 128. Do not deploy those actors or extend the identical target solely for flight. Measure whether a narrow final-output-row derivative protects useful current-speed outputs, then compare it with the shared-feature route and unchanged control. Fixed-state head preservation does not guarantee gait retention. Any continuation needs exact full-state 64/128 controls, two learning seeds, independent initial states, every gate and versioned provenance. Keep existing weights and truthful inputs; never use 267/349 labels above their freshly verified 3 m/s certificate.
2. Inspect actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets and velocities at every speed, including bilateral amplitude, arm-leg phase and noncancelling slow bias. This coupled target has zero direct head derivative but changes shared features; an isolated 1e-4 SGD probe slightly changes fixed-state head means, whereas final-row-only derivatives leave them unchanged. Neither result proves the cause of physical neck regression or certifies a repair. Measure per-command effective targets, shared-feature/PPO/upper gradients and free-base response before another continuation, with current full-state controls, two learning seeds and all independent gates. Preserve forward head, gait, both teachers, upper refinement at 20 batches/update and 35/1 gains. Human/Unitree encoder likeness and calibrated optical gaze remain unavailable; no production hybrid, observation substitution, imposed curve or cap raise.
3. Verify supervisor 39000, one actual CUDA worker, 55 frozen hashes/53 protected authored files, local ground and exact actor/critic/Adam/exploration plus both teachers. Count 27,507 inherited updates plus completed and 50 verified R40 recoveries; exclude 32 unsaved R39 and all 512 matched, 12 smoke/reload and four passive diagnostic updates. Exclude the four saved diagnostic pauses only through each timing endpoint, including the reviewer interruption and remainder. Preserve the 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy TRAINING seconds; Fusion stays closed. This check’s native 5 ms probes cover 0.15/1.5 only and find no driving clips; later 10 Hz current-speed samples cannot inherit that attribution. Earlier 4/4.5/5 extrapolation fails gait/direction and proves no ceiling. Only qualified 5 m/s, or qualified 4 m/s with causal native limits blocking 5, ends training; every endpoint needs all unchanged gates and 32 independently seeded states per command.


<!-- adaptive_check_20261009_1548_closeout -->

Closing newer observation 2026-10-09T18:29:17.614731+00:00: the same R40 supervisor 39000 has one actual worker and all 55 frozen hashes/53 protected files. It retains 22,567 completed plus 50 recovered = 22,617 current-run main updates, plus 27,507 inherited; 44 observed partial updates in 1067 remain separate. Exact actor 1064, SHA 0ad6a1cc1c018951b82d05c67d02ca7a2663eb93845030e255a04503c60a61d0, passes both 3.5 m/s cases at 3.494/3.502 signed and 0.076/0.086 lateral m/s. Both retained 0.15 net travel, 1.5 geometric flight and 2.5 foot-slip tests still fail; 1.5 genuine flight is 0.132%/0.208%. Every other saved case passes and all survive. The stage remains unqualified. Full all-command upper plots and 10 Hz CAD motor/COM-contact audits stay bound to 1061, not 1064. This check’s native 5 ms attribution stays bound to the earlier 0.15/1.5 physical cohorts. Original 1064 loads exact 1061 actor/critic/Adam/exploration, zero KL/command mismatch and all three auxiliaries at 20 batches/update; current 1067 preserves exact 1064 loading. No experimental actor, target, reward, gain, cap or optimizer reset is deployed. Same active five-hour endpoint; computer on.

Read `closing_newer_pair.json` and `closeout_learning_verification.json` beside the dated report. Upper timeline QA remains explicitly bound to 1061.


<!-- adaptive_check_20261009_2048_record -->
## October9 20:48UTC five-hour check

The same R40 full learner and supervisor 39000 have one actual CUDA worker, 55 verified frozen hashes and 53 unchanged protected files. 23,655 completed + 50 recovered = 23,705 retained current-run main updates, plus 27,507 inherited; 0 partial updates remain separately unqualified. Exact actor 1115_run_350_train_369, SHA a69c23ddfa404584752537e2867102d7b04e1e0e4e08fed8d6fc748df6c55314, reaches 3.412/3.421 m/s signed forward and 0.204/0.187 m/s lateral at command 3.5. Current nominal/pushed passes: [False, False]. Saved failures: {'nominal': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 0.65, 'failures': ['bilateral_stance']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}, {'command': 3.5, 'failures': ['world_lateral_speed']}], 'pushed': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}, {'command': 3.5, 'failures': ['world_lateral_speed']}]}. No complete 3.5, 4 or 5 m/s stage or causal native-limit endpoint is established.

Final-row coupled target tested with exact full-state64/128 controls and shared-route control in two seeds; deployment rejected. Current true-feedback physics still supports qualified267 body/arms with current head at1.5. Fixed-state head preservation does not protect fast outputs. No production reward/gain/cap/optimizer reset. All mandatory upper timelines and independent gates inspected; human/Unitree/gaze unverified. Main resumed after8053.1s with exact full state. Since previous snapshot:201,326,592 genuine samples/1024 updates. [Report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261009_2048/REPORT.md).

1. Compare fresh exact 3.5 m/s direction and every retained command, especially 0.15 m/s path efficiency, 1.5 m/s true flight, the new nominal 0.65 m/s bilateral-stance failure and both retained 3 m/s direction failures. Preserve the current full learner, separately passing current-speed actors, and the neutral-head actor 267 with its reverified all-prior certificate. The selected final-row restriction does not protect other commands, and neither tested route qualifies for deployment. Next run a bounded complete-state continuation from neutral-head 267 against an unchanged current-state control, at 64/128 updates in two learning seeds with every nominal, pushed, prior, head, direction and flight gate plus independently seeded initial states. Preserve both complete actors, critics, Adam and exploration; this is an earlier learned-state branch, not a from-scratch restart or tilted-head rollback. Check its exact teacher/context/load provenance first. Promote or deploy only after the full contract passes. Do not repeat the rejected target route, copy diagnostic controllers into inference, or use teacher labels above their 3 m/s certificate. Any later reward change needs explicit versioned provenance, full-objective counterexamples and matched retention tests.
2. Inspect actual head yaw/pitch, shoulder pitch/roll and elbow angle, physical target and velocity timelines at every speed, including bilateral amplitude, arm-leg phase, slow bias and saturation. Fixed-state head preservation under an isolated selected derivative does not preserve closed-loop head or gait during full PPO learning. Compare the earlier neutral-head branch's upper coordination with the current learner and complete per-command returns before another upper refinement. Keep the existing 20-batch upper refinement, 35/1 gains, truthful observations, forward head, useful legs and both teachers. Collins 2009 measures walking at 1.25 m/s and provides shoulder/elbow time plots; numerical axes/rest/morphology mapping remains unverified. The inspected official Unitree material did not provide mapped speed-matched encoder curves. Human/Unitree likeness and calibrated optical gaze remain uncertified; no imposed reference curve, production hybrid, cap raise or repeated rejected gain/mean candidate.
3. Verify supervisor 39000, one actual CUDA worker, all 55 frozen hashes and 53 protected authored files, local ground and exact full-state recovery with both teachers. Count 27,507 inherited updates plus current completed and 50 hash-proven recoveries; exclude 32 unsaved R39 updates and all diagnostic samples. Exclude the documented suspension only through each timing endpoint. Keep target 8,192, reserve 300 MiB, host guards/fallback and 180 continuous healthy training seconds; Fusion stays closed. This check's native 5 ms motor attribution covers 0.15/1.5 m/s only, and sparse 3.5 m/s crossings cannot inherit it. A later high-speed plateau needs actor-specific requested/applied torque, CAD motor speeds and sustained driving saturation. The endpoint still requires every unchanged gate and 32 independently seeded states per command: qualified 5 m/s, or qualified 4 m/s with causal native limits proving 5 is blocked.


<!-- plateau_recommendation_20261010_0059 -->
## Plateau recommendation — 2026-10-10T01:01:17.287979+00:00

User asks why 3.5 m/s is not exceeded and whether to end training. Fresh exact actor 1139 (SHA b3a52c00e34a2ff922c96327c1746f9e226e974f88fe72d89d37e57a981116c4) passes both current cases at 3.4506/3.4610 m/s but fails both retained 0.15 m/s net-travel and 1.5 m/s geometric-flight cases. Frozen supervisor blocks promotion until every current/prior case passes; 4/4.5/5 stages exist. Recommend pausing/ending the unchanged production iteration while preserving full state, then a bounded earlier-neutral-head full-state comparison and separately justified objective revision. Specific learning cause unresolved; no motor ceiling or successful endpoint inferred. No source/reward/physics/native limit/learned-state changes, pause or automation change performed: this is a recommendation in response to a question. Existing learner remains active. [Review](E:/Aditya/robot_sim/physical_v29/training/diagnostics/plateau_review_20261010_0059/REPORT.md).


<!-- adaptive_check_20261010_0149_record -->
## October 10, 01:49 UTC five-hour check

Same R40 full learner and supervisor 39000 at 2026-10-10T03:42:41.288235+00:00, with one actual worker, 55 frozen hashes and 53 protected files. 24,615 completed + 50 recovered = 24,665 retained current-run updates, plus 27,507 inherited; 0 partial updates are separate. Exact actor 1157_run_350_train_383, SHA f0c727f41c551e08452e6956c2b330f37523956e33dff101152052440baf5224: command 3.5 gives signed speed 3.420/3.434 and lateral speed 0.451/0.422 m/s. Current nominal/pushed passes: [False, False]. Failures: {'nominal': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}, {'command': 3.5, 'failures': ['world_lateral_speed']}], 'pushed': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}, {'command': 3.5, 'failures': ['world_lateral_speed']}]}. Saved gates never qualify newer actors. No 5/4 m/s or causal ceiling endpoint is established.

Reject the tested earlier complete-state continuation for deployment. Both learning seeds retain useful 1.5 m/s flight but fail the full direction and prior-speed contract. The unchanged current-state controls also remain unqualified. Preserve both complete learned states and resume the original full learner; prioritize a measured, versioned objective/update that protects each command's retained behavior. At 128 updates, earlier-state ordinary nominal/pushed 1.5 m/s geometric flight is 2.226/2.132% in seed 317 and 1.875/1.861% in seed 331; corresponding controls have 0.000/0.035% and 0.000/0.049%. The earlier branches nevertheless have 3.5 m/s lateral speeds 0.623/0.599 and 0.451/0.421 m/s, versus control 0.148/0.137 and 0.373/0.349. Both earlier seeds fail 2.5 m/s slip and retained 3 m/s direction at 128; seed 331 adds retained 1 m/s direction failure, and slow path efficiency is not reliably recovered. Individually seeded initial states reproduce these failures. All robots survive, so survival alone is insufficient. Fresh actor 267 passed all priors through 3 m/s before this continuation, making its subsequent retention loss reproduced evidence. The complete checkpoint, including critic/Adam/exploration, is the branch difference; this does not isolate a reward term, actor feature or mechanical arm cause. Running neck RMS increases from matched controls 2.40 to 3.92 degrees and 4.55 to 5.65 degrees in the earlier branches; smaller shoulder offsets alone do not certify a better upper controller. This bounded failure does not prove every earlier-state approach ineffective, 5 m/s impossible or a native motor ceiling. No new reward, CPO optimizer, teacher target, diagnostic actor, gain or cap is deployed.

Two complete-state branches, two learning seeds and 64/128-update controls with all gates and diverse resets were tested; 512 diagnostic updates are excluded from main progress. Required initial, earlier, matched and closing upper timelines were inspected. The exact original main learner resumes after 5693.6 seconds. Since the previous snapshot it adds 188,743,680 actual samples / 960 retained updates. No production reward, gain, cap or optimizer reset. [Report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261010_0149/REPORT.md).

1. Prioritize a versioned retention-aware objective/update, rather than repeating the unchanged earlier-state continuation. First read fresh exact current/prior gates and preserve the current complete learner, fresh all-prior actor 267 and separately passing fast actors. Capture complete per-command returns, advantages, critic/bootstrap accounting, teacher gradients and actual path/sole-force/clearance events on truthful settled and transitioning histories. Derive prospective path-efficiency and geometric-flight costs from unchanged classifiers, keeping commands separate so improvements elsewhere cannot cancel a violation. Stage one narrow training-only candidate only after finite/reset/transition/clipping/ratio and stationary, diagonal/sliding, hopping and early-exit counterexamples plus normal-limit physical/full-load checks. Then run exact full-state 64/128 controls, two learning seeds and every ordinary/diverse nominal, pushed, stopped, prior, head, direction and flight gate. CPO is an undeployed research lead, not a tested optimizer or locomotion guarantee. No arbitrary weight sweep, gate relaxation, diagnostic actor deployment or from-scratch reset.
2. Inspect actual head yaw/pitch, shoulder pitch/roll and elbow angle, target and velocity timelines at every command, with bilateral amplitude, arm-leg phase and noncancelling slow bias. Compare the preserved earlier branch's flight advantage with its direction/slip losses and current measured motion; smaller held bias alone does not qualify a repair. Measure phase-conditioned effective upper/lower targets and auxiliary/PPO gradients before another upper derivative. Preserve forward head, useful legs, both teachers, original 35/1 gains and existing upper 20 batches per update. Keep rejected row, coupled-target, shoulder-mean, stiffness, exposure, teacher/context, course/filter and reward candidates rejected absent new evidence. No production hybrid, observation substitution, imposed reference trajectory or cap raise. Human/Unitree encoder likeness and calibrated optical gaze remain uncertified.
3. Verify the same owned supervisor, one actual CUDA worker, all 55 frozen hashes/53 protected authored files, local ground and exact actor/critic/Adam/exploration plus both teachers. Count 27,507 inherited updates, actual completed main updates and 50 hash-proven R40 recoveries; exclude unsaved R39, capacity, smoke and diagnostic samples. Exclude this documented pause only through each timing endpoint. Keep target 8,192, reserve 300 MiB, host guards/fallback and 180 continuous healthy training seconds; Fusion stays closed. This check adds no native 5 ms motor request/application probe. Sparse newer 10 Hz cap crossings cannot inherit older actor-specific driving/braking attribution. A high-speed plateau needs actual CAD motor speeds, requested/applied torque, available torque and sustained driving saturation before a limit claim. Only fully qualified 5 m/s, or qualified 4 m/s with causal native limits blocking 5, all unchanged gates and 32 individually seeded states per command, ends training and this same five-hour heartbeat.


<!-- final_check_user_budget_20261010 -->
## October 10 user budget: one final check and synchronized motor exports

At 2026-10-10T04:02:56.610993+00:00, the user authorizes exactly one remaining scheduled check in five hours and ends this training attempt after it. Native target slot: 2026-10-10 05:03 AM UTC-04:00 (2026-10-10T09:03:00+00:00). Read `training/automation/FINAL_CHECK_REQUEST.json` and `FINAL_CHECK_AND_MOTOR_CAPTURE.md` first. This instruction supersedes every older requirement to continue until qualified5/4m/s or to skip new videos. Pause the same automation at final entry, save complete learned state, stop only owned training/supervisor without restart and leave the computer on. Record a user-budget stop separately from locomotion success or an actuator ceiling.

If progress is insufficient, NEW normal-time full-CAD walking-to-running videos and synchronized actual5ms installed-motor position/velocity/requested+applied torque/signed mechanical power are pending. Provide per-motor torque/RPM tables at1m/s and maximum demonstrated stable speed. Use exact CAD differential motor states and external reductions before the anatomical pivots; do not fabricate internal rotor torque or electrical power. Save episode versus population scopes,32 independent initial states,frame/sample timing,video decode/visual QA and rawCSV/NPZ. Earlier fulfilled videos remain preserved but do not satisfy this new request. Current healthy training remains unchanged until this final check; no reward,cap,gain,actor,optimizer or production source change is made by scheduling. The unchanged iteration has not shown reliable progress beyond3.5m/s with complete retention; a breakthrough in five hours is uncertain and not promised.

[Final capture and stop protocol](E:/Aditya/robot_sim/physical_v29/training/automation/FINAL_CHECK_AND_MOTOR_CAPTURE.md).


<!-- substantial_iteration_20261010_0405_started -->
## October10,04:05UTC — substantial PPO update iteration started


## Simulator smoke and reload verified

The new candidate completes4 genuine64-environment PPO updates and then reloads its complete checkpoint for4 more. The unchanged control does the same. Every load preserves actor,critic,Adam and exploration exactly. Pre-update KL and command mismatch are zero, and both teachers and upper refinement run20 batches/update. GPU/host guards and stable allocations pass.

The paired first rollout has bitwise-equal truthful observations,actions,rewards,dones,GAE returns,critic values and raw advantages. Only the declared candidate normalized advantages differ. This verifies that the implemented update is executed; it does not prove that it improves locomotion. The frozen-source baseline evaluation has all physical arrays and every case bitwise identical to the original.

The4096-environment64/128 tests in two learning seeds are now running sequentially with one actual CUDA worker. Full results and ordinary/diverse gates remain pending. Three CPU completion-helper setup failures (access to protected system process metadata, Windows virtualenv shim identification and JSON self-reference) were repaired; logs are preserved. They did not alter or interrupt simulator learning. The completion helper is CPU-only and cannot deploy a policy or extend the final-check budget.

The existing arm/head refinement remains active. The next check must inspect the new actual joint/target/velocity and slow-bias plots rather than infer natural motion from reward. No speed-matched mapped Unitree encoder curve or calibrated optical-gaze certification is claimed.

[Report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/substantial_iteration_20261010_0405/REPORT.md). Final check remains09:03UTC; stop afterwards and complete new synchronized motor-shaft videos/tables if improvement is insufficient.


<!-- substantial_iteration_20261010_0405_complete -->
## October10 substantial command-conditioned PPO iteration





Reject candidate for ordinary deployment: complete independent gate contract does not pass. Substantial iteration was executed, with evidence preserved.

Comparison completed=True. Exact changed-update validation, full-load/KL/command/auxiliary and memory checks are saved per operation. Candidate diagnostic updates are excluded from main learner totals. Source hashes verify. Original supervisor resumes after 5533.0 seconds; final owned liveness must still be read fresh.

The matched128 ordinary and individually seeded case comparisons are in live_comparison.json; decision.json lists gained and lost passing commands. New actual upper angles/targets/velocities and noncancelling bias plots are saved for every128 actor. Visual inspection is pending the final check; no human/Unitree or optical-gaze certification is inferred.

Your single final check remains09:03UTC, then stop this attempt and the automation. No new scheduled prompt or extension was created. The final video/motor-shaft data/table request remains pending if progress is insufficient.

[Evidence](E:/Aditya/robot_sim/physical_v29/training/diagnostics/substantial_iteration_20261010_0405/REPORT.md).


<!-- final_check_20261010_0905_completed -->
## October 10 final authorized check — training and automation stopped

At 2026-10-10T09:58:27.983572+00:00, the single final check is complete under the user budget. R40 supervisor 39000 retired after full saved boundary 1226 at 09:09:47 Z; no learning resumed during exports. Same native automationPAUSED, zero remaining prompts, no automatic restart. Final actor 3.5339/3.5410 m/s still fails direction and prior gates; preserved 1064 fresh 32-state currentcase 3.4945/3.4921 passes but 0.15 path/1.5 flight/2.5 slip retention fails. No reliable breakthrough,4/5 qualification or causal ceiling. Substantial command-conditioned PPO iteration(two 64/128 learning seeds)executed/rejected; all 20 upperplots now visually inspected. Full actor/critic/Adam/exploration, both teachers and all 55 hashes/53 authoredfiles preserved. 26,023 completed+50 recovered=26,073 current retained,+27,507 inherited=53,580;4,293,328,896 current samples. Rates 36.063 k active/29.807 k worker/19.201 kchain through 09:06:06 Z;53,218.311 sdocumented suspensions excluded.

NEW 10 normal-time full-CAD videos and synchronized native 200 Hz 23 installed-motor angle/velocity/requested+appliedtorque/signed mechanicalpower exports are complete.32 independent states per 1/3.5 command provide populationtables; separate single-video episodes have exactframe/samplemaps. Full decode, visible 140 meshes,500 frames/10 s, timing/power and actualupperangle/target/velocity/biasQA pass; heldbias/asymmetry remain, human/Unitree/opticalgaze uncertified. Archived setupfailures fixed without learning; discarded evaluation overlap explicitly recorded. All earlier evidence/videos/Fusion queued work preserved, computeron.

[Final report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/final_check_20261010_0905/REPORT.md), [new videos](E:/Aditya/robot_sim/physical_v29/training/videos/final_motor_capture_20261010_0905/README.md), [installed-shaft table](E:/Aditya/robot_sim/physical_v29/training/diagnostics/final_check_20261010_0905/MOTOR_TABLES.md), [full checkpoint](E:/Aditya/robot_sim/physical_v29/training/diagnostics/final_check_20261010_0905/final_full_state_checkpoint.pt). No next training tests or scheduled continuation are authorized; later work requires new explicit authorization.


<!-- running_torque_p99_20261010 -->
## 2026-10-10T17:41:41.661506+00:00 — Running torque p99 and shaft/pivot clarification

User requested actual running maximum and 99th-percentile N·m after challenging the previously reported 4.25 N·m knee value. CPU-only analysis of preserved actor1064 (SHA 0ad6a1cc1c018951b82d05c67d02ca7a2663eb93845030e255a04503c60a61d0), selected nominal 3.5-command native200Hz capture, confirms measured signed fixed-frame speed3.494529m/s. There are32 individually initialized trials,18s after2s settling,115200 valid samples/motor; no falls/resets/invalid samples. P99 uses absolute applied torque pooled across those samples. All23 installed-shaft mappings verify against named raw pivot efforts or bitwise exact CAD individual motor states; old p95/max summaries reproduce. Original summary gave p95, not requested p99. Its4.25N·m left-knee maximum was at1m/s at the bare motor shaft. At running speed, left/right bare knee shafts p99=4.674/4.313Nm,max=5.126/4.844Nm; raw knee-pivot p99=71.043/65.554Nm,max=77.921/73.632Nm. Frozen16:1 external reduction and95% efficiency explain the reference difference. Separate single-video results remain separate from population.

New reproducible script,JSON,CSV,report and SHA256 receipts: [running_torque_p99_20261010 report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/running_torque_p99_20261010/REPORT.md). Values are modeled applied efforts, not hardware measurements or continuous-duty ratings. No physics,reward,gate,inference,cap,actor,critic,Adam,exploration,teacher or frozen source changed; all earlier files preserved. Training remains stopped_by_user_budget, automation remains paused; no new simulator or learning process was started. The new diagnostic does not change qualification or establish a ceiling.
<!-- /running_torque_p99_20261010 -->


<!-- running_output_torque_p99_20261010 -->
## 2026-10-10T17:46:21.250407+00:00 — Final output torque after all reductions

User clarified that requested running torques must be AFTER all reductions, superseding the earlier installed-shaft reporting scope. CPU analysis directly reads actor1064's preserved native200Hz applied_pivot_torque_Nm arrays at measured signed3.494529m/s,32 individually initialized nominal trials,2–20s,115200 valid samples/output. All23 final pitch/roll/yaw output p99/max are exported; no additional reduction is multiplied into these already-final efforts. Three exact CAD differential output reconstructions,efficiency×J-transpose×individual motor torque,match within5.46e-6Nm;17 non-differential statistics exactly reproduce prior pivot conversion. Hip-pitch p99/max left124.768/132.041Nm,right127.708/135.189Nm; knee left71.043/77.921,right65.554/73.632. Waist and hipyaw/elbow outputs reach configured caps; lower torque than a limit alone does not establish a data error,simulation realism or a hardware ceiling. These are modeled applied actuator torques,not physical measurements.

[Final output report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/running_output_torque_p99_20261010/REPORT.md), CSV/JSON/reproducible script/SHA receipts nearby. Original motor-shaft table and raw evidence remain preserved with their different shaft reference. No new physics,reward,controller,gain,cap,actor,critic,optimizer,teacher,inference or frozen source change. Training and the same automation remain stopped;no simulator or learning was started. Full checkpoint and upper/gait state preserved.
<!-- /running_output_torque_p99_20261010 -->


<!-- running_rpm_utilization_20261010 -->
## 2026-10-10T19:35:57.378395+00:00 — Running RPM utilization percentages

User requested actual RPM as percent of maximum,continuing the final-output after-reduction scope. CPU-only native200Hz actor1064 capture at signed3.494529m/s,32 independent nominal trials,2–20s/115200 samples per axis. Report exports absolute p99 and peak RPM plus100×observedRPM/matched configuredRPM for23 final outputs and23 installed shafts. Constant-ratio speed and limits scale equally,so utilization percent is invariant under reduction;efficiency is not applied to speed. Knees output cap207.9RPM: left p99/peak70.972/74.729%,right65.327/70.164%;right elbow peak100%. All pose-dependent CAD J×joint-velocity/motor-velocity and fixed-ratio mappings pass. Shoulder executable120 outputRPM(480 pre4:1) is used instead of metadata500RPM.

Ankle/waist final pitch/roll denominators are explicitly per-axis PhysX BOUNDS,not independently attainable coupled motor limits. Their actual individual motor-cap percentages use400RPM. Ankle peak motor usage reaches left1=146.153%,left2=163.312%,right1=99.382%,right2=148.278%. All75/609/1502 above-cap samples are braking,zero driving;above-cap fractions0.0651/0.5286/1.3038%. This records observed passive overspeed,not allowed powered speed,hardware validation or a causal locomotion ceiling. Configuration caps are operating assumptions,not measured loaded/no-load hardware capability.

[Report](E:/Aditya/robot_sim/physical_v29/training/diagnostics/running_rpm_utilization_20261010/REPORT.md),separate output/motor CSVs,JSON,reproducible script and source/artifactSHA receipts nearby. No source,physics,gain,reward,limit,inference,actor,critic,Adam,exploration,teacher or checkpoint changed. Training/automation stay stopped;no simulator or learner restarted.
<!-- /running_rpm_utilization_20261010 -->


<!-- wrist33_20261023 -->
## October 23, 2026 — differential wrist CAD v33, no training migration

User requested head-type servos in a narrow differential wrist with a spur idler to stagger motor placement. Preserved latest v32 version 4 and built both wrists in v33 using 48mm motor spacing, two/four spur meshes, 24T bevel prototypes, sleeved M3 idlers, centrally supported cross sleeve, paired output bearings and existing hands. Initial connector/skin interference, datum lookup and report-only count failures were preserved and corrected. Eight hand mesh groups remain unchanged; source cached poses normalized during Fusion recompute and the exact initial archive remains preserved. Original 65 joint definitions plus 22 new constraints reopen. Neutral solid audit, 12 native poses, 18 selected solid sweeps and 13 bevel rolling samples pass; not continuous mesh-hand, hardware/load or dynamics qualification. [Full report](E:/Aditya/robot_sim/outputs/shin_cad_20260912/wrist33/REPORT.md). No simulation/training source, reward, policy, cap or FOC file changed; training remains outside this task.


<!-- compact_wrists_v34_20261023 -->
## October 23, 2026 — compact differential wrists v34

User requested one fewer spur per path, 17–18 teeth with larger module, and shorter hand output lever. Both wrists now use18T/module1.5 spurs, zero/two idlers,37.5mm lever and157.5mm hand-plane-to-cassette end (previous54/193). Exact head-type cases, hand mounting planes and24T/module1 mitres remain; two688 bearings per wrist replace608. Spur ratio stays1:1; no torque/load qualification is claimed.

Final native document Humanoid_compact_wrists_v34_release v3; archive SHA256 `1f3c045dc28e415f9e0beeede2f95bc4431cc776fec50e7418913c6d1f6c23e3`.72 visible solids pass neutral checks;12 native poses,18 actual-forearm selected solid sweeps,52 gear-period samples,28 component/all83 joint native reopen and29-solid-per-side STEP checks pass.65 original joint types/owners/validity retained. Continuous/mesh-hand clearance, print fit and load remain unqualified.

Fusion UI/bridge recovery, initial ring/strap interference, inappropriate absolute STEP volume threshold, native root-name assignment failure, shell quoting preflight, missing FreeCAD runtime, VTK renderer stall, memory-exhausting old-component removal and transient-import placement reset are preserved as failed/repaired evidence. Final build uses preserved v32-v4 baseline plus v34 geometry, rigid-link grouping and explicit world-frame body placement; angle-only pre-placement checks are superseded. Original v33 and recovery files retained; startup manifest restored; workbench stopped after inventory; computer/Fusion remain on. No simulation/training/automation/controller/actuator-limit change or commit/push.

[Report and exact evidence](E:/Aditya/robot_sim/outputs/shin_cad_20260912/wrist34/REPORT.md).

## October 23, 2026 — XDrive placements v35 (xdrive35_20261023)

User requested nine Mini controllers behind six GWK6010 and three GWK7010 motors.
Found the library import incomplete; preserved it and used the documented 144-solid
reference STEP. Initial clearance studies produced prototype enclosure moves and
reliefs. User then explicitly requested controller placement only and would adjust
the mechanics. All those changes were undone; no proposed brackets were executed.
Nine linked references with original rear magnets, all 83 native joints and
1733 original body bounds survive independent native reopen.
Supplied rear plate included; axial spacing/encoder package/thermal stack remain
unverified, and neighboring intersections intentionally remain. Failed import,
name/capture and packaging probes are preserved. Current native/source/hash and
previews: E:/Aditya/robot_sim/outputs/shin_cad_20260912/xdrive35/REPORT.md.
No training, motor caps, firmware or custom FOC PCB changes.

### October 23, 2026 — Direct-solder steering (xdrive35_terminal_disable_20261023)

User subsequently requested disabling the green connectors to reduce casing/plastic
interference and will solder wires directly. The two screw-terminal assemblies
(17 supplied solids per board) are hidden reversibly in all nine linked boards.
PCB, signal connectors, capacitors, supplied rear plates and all original robot
mechanics remain. Native reopen verifies the disabled state and original bounds.
Wire/solder/strain-relief volume is unmodeled; no interference-free fit is claimed.
The prior connector-populated v2 archive and proof are preserved separately.
Updated source, export hashes and views: xdrive35/REPORT.md. No training or custom
FOC PCB changes.


## 2026-10-23 — SteadyWin enclosure CAD v36

2026-10-23T23:36:01.905875+00:00: User rejected oversized rectangular XDrive packaging and requested circular SteadyWin drivers with attached cases for all 6010/7010 except ankles. Preserved the unsaved v35 shoulder concept input; saved new v36 with nine separate case/lid/mount prototypes, four hip clamp caps, local support recesses and narrowed 140 mm bay. Motors, magnets, original CNC crossheads and 83 joint definitions retained; 3096 original solids preserved. Official motor/cap STEP and manual downloaded; no standalone PCB STEP available, so a labelled GDS68 dimensional envelope and provisional holes are used. Exact 7010 firmware/voltage/encoder/thermal match remains unverified. Neutral clearance, independent full native reopen and 15 STEP/28 solid roundtrips pass. Actual motion has 19/21 clear samples and two unresolved inward-arm/waist-driver-envelope conflicts; inward +/-15 samples are clear and original limits/mechanics are retained. Initial combined concepts, Boolean/interface failures and zero-angle computeAll test were preserved and corrected; no strength/continuous-motion/material/mass qualification claimed. Training stays stopped, custom PCB0255 unchanged. Source urn:adsk.wipprod:dm.lineage:L5mGVY-GSc6xk-57pUTA2A version2; F3D SHA cd19c3318afe919a04f1024fbd1974c7fc8df03a1ad58747503a2c1ec5a3e112. [Report](E:/Aditya/robot_sim/outputs/shin_cad_20260912/steadywin36/REPORT.md). No commit/push.


## October23local/24UTC2026 — CAD v37 six leg-driver mounts/four ankle encoders, deferred adapter

Recorded 2026-10-24T05:43:31.670591+00:00. User requested same driver package behind six10018s and real encoder boards behind fixed four ankle7010s. Nativev37 saves6integral driver mounts,10018 axial solid-set advances8.5/14.5/14mm,4independent visible officialAS5047P19-solid boards with1mm stock-magnet CAD gaps,integrated S06 frames/pads/servicebores and hardware references.3,221originalsolids,original occurrenceposes and83joints/axes/limits retained. Neutral1,324Boolean tests pass;28actualmotion samples isolate added contacts to twoalready-invalid30degcrossed-legposes;firstsampled original/current cross-legcontact11deg bothsides,clear through10deg. FullF3D and14STEP/22bodies independently reopen. No physics/training/reward/cap/optimizer/actor changes. Exact boardholes,10018electrical/thermal/firmware/realgearbox,material/mass/load,magnet/interface/tolerance remainunqualified. Failed BOMrequest,grounding/MoveFeature/sharedvisibility/support/onsetprobes arepreserved and corrected,not counted aspasses. See E:/Aditya/robot_sim/outputs/shin_cad_20260912/drivers_encoders37/REPORT.md; archiveSHA2f8bfc1ccb58a6b4d630d10e472fc7c3bc78cac14912977ad10a93c1323f0fb2. CustomPCB0255 unchangedSHAd6136915f8c79c7ba11bcbf8f39d75439624b727fdfe39c7c402193b874154fc. Sameone-timeCADfollow-upOct24,01:41local/05:41UTC includesexplicitlydeferredQ02adapterrefinement;skilladdsfeatureefficiency/load-pathguidance. Noongoingtrainingautomation orshutdown.


## October 24, 2026 — scheduled V38 Q02 cradle refinement

Recorded 2026-10-24T05:55:44.662620+00:00. User explicitly deferred this geometry task to the single four-hour CAD follow-up. Preserved V37 full archive and branched Humanoid_refined_hip_cradles_v38 v2, lineage urn:adsk.wipprod:dm.lineage:nMJF4CLITJ-yHUUmsHzqhw. Each Q02 receives two editable nativeR8 concave cheek/base roots and fourR2base corners in separate grounded child components. Original10mm cheeks/8mm base/5mm blind caps,39cylindrical surfaces per part,all motor/V37driver/encoder geometry and83joint axes/limits retained. Net volume+0.787%, no strength rating. Neutral86Boolean comparisons and16actual hip samples have no added contact; all original occurrence geometry preserved. F3D independent reopen 3795bodies/503occurrences/83joints/nativefeatures passes; twoSTEP solids independently reopen. ArchiveSHAc9d78092743d41365f1eaca517cd8c5ac1147eaf8235102e2effeb81336dd6a6. Initialunsupportededgeproperty,coincident-copy source alteration andcomplex-endpoint fillet failures retained;onlyownedunsavedbranch restored fromsavedbaseline,originalvolumeerrors0,independentcomponentsrepairverified. No failedfeature deployed. See E:/Aditya/robot_sim/outputs/shin_cad_20260912/adapter38/REPORT.md. Documentation/BOM/skill updated, one-time automation consumed and removed atcloseout. Training/customPCBunchanged,computeron,unrelatedunsavedFusionworkpreserved. Material/load/fatigue/manufacturing/continuousclearance and inheritedV37physics/electronics limitations remain.


## October24,2026 — V39 actual-chain prototype rejected by user

Recorded 2026-10-24T17:10:05.384093+00:00. RestorednativecomputeruseandShinWorkbenchwithharmlessinventory. BranchedsavedV38v2toV39. Built38full12.7mm-pitchchains,190solids/knee at159.4mmcentres;bothspansbow17.29mm. Nativepitch/seating/100toothBooleans,3795sourcebody/83jointpreservation,F3D4175body/505occurrence and2STEP190bodyreopenpass.57falsejointflagsweretuple/listcomparisonformat,notgeometry;exactserializeddifferences0. User explicitly rejects excessive oval/undertension appearance. V39isrejected,notcurrentorapproved. ArchiveSHAc299a0592541b52aae749e50113da050d88c9bb89d52c145bcfae00a67e0a156retained. See E:/Aditya/robot_sim/outputs/shin_cad_20260912/chains39/REPORT.md.

## October24,2026 — corrected straight-run knee chains V40

Recorded 2026-10-24T17:10:05.384093+00:00. User prefersmotorloweringoverraising. Removedoneinner/outerpairperchain:36full12.7mmpitches,18inner/18outer,180activepieces/knee. Loweredcomplete22motor/case/driver/output/support/hardwaregroups168solids7.83755mm,centres151.56245mm,straight12-pitchspans. No stretched/offsetlink. All4175sourcebodiesretained;20surpluspieceshidden.94healthyMovefeatures;132toothBooleanchecks0intersections;6fixed-neighborcounterfactualchecks0addedcontact. Actualallbodyplacementsverified;83joints/axes/limitsunchanged,lowerknees/rails/anklesfixed. Fullnative4175bodies/505occurrences/83joints andtwo180-bodySTEProundtrips pass. SHA05fed973cccead4a3ffb32617f5b43df3334f943aa5d98a5c5fc4af91dfd6e17. Staticgeometryonly;nochainpreload/dynamic/load/manufacturequalification. Useracceptedadapteroverlapleftunchanged. SourceHumanoid_tensioned_knee_chains_v40v2,urn:adsk.wipprod:dm.lineage:AhUM06-hR5u9R_-EdNKUbg;reportE:/Aditya/robot_sim/outputs/shin_cad_20260912/chains40/REPORT.md. Skill/build/BOM/historyupdatedlocally,CPUinventoryrefreshedatcloseout. Training/controllerPCB/automationunchanged,computeron,unsavedFusiondocspreserved.


## 2026-10-24 — V41 user-requested clearance repairs and adapter redesign

Both yaw-to-hip-roll adapters are redesigned in **Humanoid_clearance_adapter_redesign_v41**, saved Fusion version 2. They use tapered 8 mm cheeks, an 8.5 mm base and R12 primary root blends, with measured local clearance pockets instead of thin right-angle attachments. Both yaw bearing seats and their bearings/retainers are lowered 6.3 mm axially; motor placements and all 83 joints remain unchanged. The right output hub is realigned to its motor axis as explicitly requested; its shape is preserved. The electronics holder is shortened for a 2 mm battery gap, shoulder fasteners are recessed and both inner waist caps/tube tips are trimmed for triangular clearance. The V40 chains are preserved.

Final visible-assembly checks pass 17 hip poses with zero absolute adapter contacts and 15 shoulder/waist poses with no added interference. All 4,175 original bodies remain recoverable; the full F3D and 25 single-solid STEP files independently reopen. This is a CAD prototype: material choice, tolerances, strength, fatigue, fastening, manufacture and continuous whole-body clearance remain unqualified.

[V41 CAD, images and measured checks](E:/Aditya/robot_sim/outputs/shin_cad_20260912/clearance41/REPORT.md). Recorded 2026-10-24T21:49:15.360237+00:00. Source SHA256 `4f483cdbe0bc0822123c1c274f3deb8997367863f1290f9c94620733025a0c48`. Rejected nominally clear adapter proposals and restricted-visibility diagnostics remain preserved; final suites reran after fixing proxy bulbs. All current source geometry, save/export/reopen and limits are documented. No learning, FOC or automation change.


## 2026-10-25T02:57:50.185398+00:00 — V42 FOC controller CAD integration, incomplete

Actual populated0255controller imported;19mount positions,12mm upper lift,83active/84totaljoint scheme and neutral intended-geometry checks implemented. All27native samples inspected;6extremes already blocked by retained robot parts, no full-motion claim. Final actual encoder/hole metrology rejects moved-upper shared-instance offsets; independent-copy repair is unresponsive. Saved full input and rejected checkpoint, all failures retained. Native restart approval pending due unrelated unsaved documents; polling stopped to prevent replay. No final export/reopen or finished board-placement claim. [Report](E:/Aditya/robot_sim/outputs/shin_cad_20260912/foc68_42/REPORT.md). PCB/STEP hashes unchanged; no training/automation/shutdown/electrical modifications.


### 2026-10-25T03:45:10.594271+00:00 — V42 actual controller integration completed after authorized recovery

The actual foc-controller-68 populated STEP is installed at all19 BLDC motors: six6010, seven7010 including all four fixed ankles, and six10018. Fusion source **Humanoid_foc68_upper_clearance_v42**, saved version **4**. Waist-yaw fixed motor and the complete upper body are raised12mm. Approved V41 adapters and every leg motor placement remain. The failed shared-hierarchy placement is replaced by verified flat exact-source geometry. All19 sensor/hole placements,27 sampled motor-following poses,4233 original bodies,83 active joints and the full native archive,30STEP parts and2native knee-part reopens pass; two kneeSTEP files remain diagnostic. Whole-body continuous motion and structural, thermal/electrical and physical encoder qualification remain open.

Full archive SHAb5299d4d78224ea9d73a83ed34eeb1f9ea18b1af898ad954b87a8bf06b649d36 ;30acceptedSTEPparts plus2exact native knee-part archives;2diagnosticSTEPs retained. Thirteen recovery files copied before authorized Fusion restart; nested shared imported children failed actual motion, resolved with exact flat251-solid geometry. Old version-handle, external-reference import, wait-budget and duplicate partial-count failures preserved in foc68_42. Report: E:/Aditya/robot_sim/outputs/shin_cad_20260912/foc68_42/REPORT.md. No PCB/training/automation/shutdown change.


2026-10-25T03:48:21.353302+00:00 — Delayed bearing-seat reply: user says lower5mm in answer to the earlier fixed-seat/base question. Current V42already retains V41seats6.3mm below original. Asked whether5mm is total or additional; no further displacement applied. Controller releaseV42version4 remains verified and preserved. See foc68_42/pending_bearing_datum.json.


2026-10-25T03:49:39.752747+00:00 — User canceled the clarified total5mm bearing-seat revision before any CAD operation. Keep the verified working geometry and avoid unnecessary or worse geometry/aesthetics. No V43branch, bearing displacement or adapter edit occurred. V42version4 and its verified exports remain unchanged.


### 2026-10-25T08:49:18.985381+00:00 — Saved shin ODrive correction and wrist/mount refinement in existing V42 version5

The four fixed shin motors are restored to two ODrive boards and four independent AS5047 P encoder/carrier assemblies. The custom foc-controller-68 remains at 15 non-shin motors (six6010, three7010 yaw, six10018). Both wrists now use16 T/module1.5 bevel sets, with the existing24 mm pitch diameter,1:1 ratio,18 T/module1.5 spur gears and hand interfaces retained. Knee tie-bolt ears move3 mm inward (60 to57 mm radius); R3 boss roots and R2 oval-holder transitions replace abrupt connections. Eight hip cap/support bodies have R3 boss roots, with local relief where added material touched the approved hoods. Motor positions, approved adapters,12 mm upper/waist rise and83 active joints remain unchanged.

Full local archive SHA256 d745db6f7c3e570b7ba4b0b1439010a13bebcefee99935399ea8b57297b15ee0; one final cloud save, same lineage, no new design. Full native reopens4635unique bodies/956occurrences;9474 original occurrence/body checks and all83active/84total joint definitions preserve sources.42native wrist poses and15scoped mount poses pass; earlier8new contacts resolved by local pockets/added-root relief.18native parts and12STEP pass,6STEP diagnostic with native alternatives. Bilateral filename overwrite repaired with2targeted reexports and unique indexed paths. All rejected selectors, failed import/API/vector accesses, pre-relief collisions and export evidence preserved. Prototype only; no load/manufacture/continuous whole-body/electrical qualification. Four-shinFOC installation was a scope error and is superseded by restoredODrives/independentencoders;15non-shinFOC remain. No training/cap/PCB/firmware change, purchase, commit or push. One-time followup paused/consumed; authorized non-forcedshutdown after saved-source/stop receipts. Report E:/Aditya/robot_sim/outputs/shin_cad_20260912/shin_odrive43/REPORT.md.


<!-- shoulder44_verified -->
## Oval shoulder crossmember and continuous neck cradle — October25,2026

Saved **Humanoid_foc68_upper_clearance_v42, version6** in the same V42 lineage, one cloud save. Two76mm18×10mm obround tubes and separate central/end adapters replace the wide shoulder beam. New tapered10mm neck cheeks/R3 transitions give9.8mm bearing-root engagement. Existing motor interfaces and83active joints remain; full robot covers/hands restored. Knee boards were hidden, not absent; both hip-roll boards fit without motor moves or redesign.15FOC boards and shin ODrives/separate encoders retained.

17 scoped native poses pass;20native/20STEP parts and the full assembly independently reopen. 4635 original unique BReps preserved. Final stance is neutral; the initial exact fractional-degree pose remains in its input archive after native drive/snapshot restoration failed. Prototype only: physical strength/manufacture/continuous clearance and electrical qualification remain open. [Report and CAD](E:/Aditya/robot_sim/outputs/shin_cad_20260912/shoulder44/REPORT.md). Training stays stopped; no new automation or shutdown.


### 25 October 2026 — casing redesign continuation and native recovery (in progress)

The user retained knee 0–120° and elbow 0–135° limits. Staged neutral fit passed with zero unresolved contacts/kernel errors, but the full right-side 5° motion screen exposed 33 knee and36 elbow contacts at deep flexion. These are failed gates, not completed casing qualification. Native dense sweep33 failed ASM_INCONS_REL at the fixed hood and its cleanup stalled Fusion; its cache/offcuts and reports are retained. Original edited input F3D and12:31 crash recovery were copied before a normal-close attempt and owned Fusion restart. Reopened SAME V42 lineage, now version7,1020occurrences including hidden C45 tooling. Original baseline body names and available native bounds/volumes passed recovery checks; exact input archive remains authoritative for final independent preservation proof.

Replacement diagnostic36 projected actual clipped silhouettes in20 axial bands in3.2seconds. CPU37 builds1° swept polygons with outward guards and simplified-profile containment proof (zero uncovered area); eight self-touching native outline rings required only zero-area numerical repair. Native38 is applying these simpler profiles to staged forearm/thigh skins. It is still running; no C45 production parts installed, no final verification/export claimed. Current source mechanics/axes/limits/controllers stay retained, training stopped, no shutdown/new automation.


<!-- electronics46_verified -->
## Electronics and waist packaging E46 — October 28, 2026

Saved **Humanoid_electronics_waist_E46_20261028, version 2**: one derivative with two meaningful versions, including the final USB-C access repair. The reinforced 140 × 86 × 82 mm box holds real Waveshare P4/C6 and Adafruit BNO085 models. A retained 6808 bearing, continuous 99 mm waist case and rounded/blended adapter are installed. The oval shoulder tubes, motor positions, 12 mm upper lift and 83 active joints remain. Protected original bodies are preserved; 15 FOC boards, two ODrives and four independent AS5047P boards remain.

Neutral scoped checks, both declared connector corridors, 64 native parts and the full native assembly pass their reopen/fit checks. Of 26 native motion samples, 22 requested poses clear; two combined requests clear only at their actual 12.66064° pitch rather than 15°; two inward-arm extremes are already blocked. First inward contact remains 12.5° on both sides at 0.5° sampling. The strict full requested-motion gate remains false. STEP comparison is diagnostic after ASM_BAD_CONTAINMENT. **C45 shells remain unfinished**, and load, manufacturing, electrical and continuous-clearance qualification remain open. [Report](E:/Aditya/robot_sim/outputs/shin_cad_20260912/electronics46/REPORT.md), [electronics guide](E:/Aditya/robot_sim/outputs/shin_cad_20260912/electronics46/ELECTRONICS_QUICK_GUIDE.md), [Claude handoff](E:/Aditya/robot_sim/outputs/claude_robot_handoff_20261028/START_HERE.md). Training, PCB and automations were not resumed; computer on.



<!-- electronics_submission_review_20261028 -->
## Electronics submission inventory — October 28, 2026

Recorded 2026-10-29T00:30:55.156194+00:00. Read-only review of current E46/BOM and available user-visible BOM/journal/CAD chat records. Documented 19 BLDC plus24servo motors,15customFOC boards,2dual-axis ankle ODrives,4AS5047P boards,1P4/C6 and1BNO085. Confirmed36cells/12S3P, but no final brand/model: current BOM still marks AmpaceJP50/MolicelP60C as candidates. User journal200A burst/70A-per-cell recollection does not identify or qualify a cell. Remaining power, BMS/charger, regulator, physicalCAN/servo interface, encoder magnet/reference, cooling/firmware and optional camera-host decisions are explicit. Final purchased BOM is deferred until electronics are finalized, excluding printed/CNC parts and oval tubes while retaining purchased hardware. [Review](E:/Aditya/robot_sim/outputs/shin_cad_20260912/electronics_submission_review_20261028/REVIEW.md). FOC log0275 appended; exact0255PCB hash unchanged. No CAD/PCB/firmware/training/automation/shutdown/purchase/submission action.
