# Training follow-up — October 2, 08:45 EDT

The 12:45 UTC heartbeat found r15 alive at 8192 environments, but walking and
standing retention still failed. No shutdown is warranted. Its worker was
drained at 691 local + 612 inherited PPO updates, preserving
`runs/locomotion_20261002_r15_motion_discovery/072_walk_015_train_011_part_03/resource_recovery.pt`
and all prior logs, source, gates and trusted checkpoints.

## Evidence and diagnosis

The last matching nominal gate, from `068_walk_015_gate_011_nominal`, reported
100% walking survival, 0.2481 m/s at a 0.15 command, 84.89 left and 14.97 right
qualifying landings per robot in the measured 18 seconds, 2.774 Hz cycle rate,
9.10% flight and failed bilateral/cadence/hop checks. Standing had 0.4355 m
final drift, only 31.91% double support and failed anchor/support gates. The
pushed gate also failed; its standing survival was 79.69%. These counts are
the old v2 counter's output, not a verified natural gait.

The live discovery bonus averaged 1.315 units/s: accepted landings replenished
the ostensibly initial discovery bank, paying extra rewards throughout rapid
contact cycles. The independent counter also lacked an order check and
double-counted overlapping hop pairs: its fraction exceeded one.
[Identical CPU trajectory replays](checks/gait_integrity_20261002_1245/counterexamples.json)
confirmed 16 valid alternating steps paid 16 discovery units under v5, versus
two under v6. Three overlapping hop impacts gave a 1.333 fraction and one
credited step under v2; the repaired counter records 1.0 and zero steps.

The [joint diagnosis](checks/landing_audit_20261002_1245/diagnosis.json) again
finds all 23 CAD/URDF axes, origins and limits agree, with both leg Jacobians
rank six. No sampled leg hard-limit or leg motor-speed violation was found.
Rare elbow passive speed exceedances were measured. Frequent ankle target
clipping remains a concern; hardware limits were not enlarged. Normalized
input groups were bounded during learning. Those checks cannot prove adequate
balance authority or the accuracy of provisional mass/contact geometry.

## Focused reward and verifier repair

`gait_integrity_v6` caps first-step discovery at one unit per foot per physical
episode. Accepted steps and command switches cannot renew it. Landing travel
is frozen at touchdown, before settling can add fictitious sliding distance.
Credit waits 100 ms through the complete 80 ms paired-impact window. Walking
bilateral flight invalidates that swing's landing credit. Credit interval
bounds limit rapid event farming without prescribing exact joint trajectories.
Positive walking progress/swing terms require support, and walking flight has
an explicit cost. Running still permits phase-appropriate alternating flight.

Standing adds continuous costs for COM outside the conservative sole rectangle,
missing double support and drift while stopped. Its anchor starts at the
stopping location, so a stop after walking does not demand returning to the
episode origin. Common balance incentives remain unchanged.

`physical_landing_v3` independently measures raw physical landings without the
training cadence filter. Repeated same-foot landings remain counted and cause
an order failure, so deleting bad events cannot create an acceptable cadence.
Each hop-associated touchdown counts once. Hop measurements include the whole
rollout to avoid warmup-boundary denominator errors. Existing numerical gate
thresholds are identical by AST verification; two added checks require
alternating landing order and less than 2% walking flight. Running flight
thresholds and all nominal, pushed and prior-skill retention gates remain.

Every evaluation now saves bounded `gait_diagnostics_50hz.npz` traces for 16
robots: raw/clamped actions, targets, q/qd/torque, root states, foot positions,
loads, contacts, clearance, landings and pre-reset failure flags. No camera is
instantiated. Poses on a done tick may already be reset; flags identify failures.

## Verification and current run

[119 CPU tests passed](checks/gait_integrity_20261002_1245/cpu_tests.log), including
real alternating walking/running trajectories and counterexamples for rapid
steps, repeated bank payout, overlapping/staggered hops, sliding during landing,
stop-anchor semantics, strict gate migration and preserved checkpoint learning.
Reward migration verifies immutable CAD, mass, contact, hardware, reference,
action/observation, reset/randomization and PPO code. Reward-only state is
explicitly excepted. The explicit v2-to-v3 gate migration never copies policy
qualification; trusted prior skills receive fresh independent evaluations.

The first replacement was `runs/locomotion_20261002_r17_gait_integrity`, warm-
starting the saved 1303-update actor and normalizer, with fresh critic/Adam for
the changed rewards. R16 is preserved preflight-only evidence: its four-update
smoke completed, then review corrected the stop anchor before a full training
block. Its artifacts were drained without changing the saved R15 candidate.

R17's fresh four-update 64-environment Isaac PPO smoke and checkpoint reload
completed. The smoke confirmed actor tensors were identical before training,
with old critic/Adam not loaded for the new reward. Reload behavior remained
unqualified. All three capacity benchmarks completed: 4096 / 6144 / 8192
delivered 14354 / 17577 / 19449 environment steps/s, with minimum free VRAM
4319 / 3437 / 2571 MiB. At this short benchmark capacity, 6144 gave 9.6% less
total throughput than 8192. Long-block recycling overhead is excluded.

Fresh trusted standing gates passed: nominal survival 100%, pushed 92.1875%,
with every standing check passing. Pushed mean final drift was 0.0608 m,
double support 99.91% and COM/support outside fraction 1.24%. These results
qualify only the trusted standing prior, not the retained walking candidate.
The v3 reload trace was saved at 50 Hz (1000 ticks, 16 robots, about 9.4 MB).

R17 completed a 64-update block at 8192, recycling after updates 17, 34 and 51
when host memory crossed the guard. The saved [Adam audit](
checks/gait_integrity_20261002_1245/optimizer_continuation.json) confirms
544 to 576 minibatch steps and retained 1e-5 learning rate at the first restart.
The allocation leak remains unresolved, but continuation is verified.

Its matching post-block independent nominal test **failed**: standing survival
was 0%, with backward drift, while the walking case survived 95.3125% but
averaged only 0.00527 m/s and had zero qualifying steps. Reduced flight was
not walking progress. Standing rehearsal/reward alone did not prevent loss of
the prior skill. R17's later candidate was drained and preserved; no gate or
checkpoint has been overwritten.

## Training-only standing retention repair

The active replacement is `runs/locomotion_20261002_r19_standing_retention`.
The v6 rewards and v3 independent gates are unchanged. It preserves R17's
latest actor, critic, Adam and learning rate. An additional stopped-command
mean-action MSE (weight 0.25) keeps the learned actor near the verified standing
actor during PPO optimization. The frozen teacher uses at most 256 samples
per minibatch. It never blends or sends simulator actions; inference and
promotion evaluation instantiate no teacher. The target is clamped to the
existing action envelope. This is a retention mechanism, not a guarantee of
walking or a recovery controller.

[128 CPU tests passed](checks/gait_integrity_20261002_1245/cpu_tests_retention.log).
New tests verify equality with the exact supervised gradient, no added gradient
on moving rows, unchanged inference, a bounded sample count, frozen teacher
parameters and reduced stop error under a supervised update. Mandatory teacher
asset/observation metadata and exact checkpoint SHA are checked. The qualified
legacy checkpoint predates optional motion/actuator metadata; present extra
hashes must match, and fresh current-asset standing qualification remains
mandatory. R18 is preserved failed preflight evidence for this compatibility
fix; it completed no learning block.

R19's actual four-update 64-environment PPO smoke completed with the additional
loss active in all 32 minibatches of the last update. Saved standing-prior MSE
was 0.2710 and weighted loss 0.06776. This confirms execution, not restoration
of independent standing or successful walking. Fresh checkpoint reload also
completed. Capacity benchmarks passed at 4096 / 6144 / 8192 environments,
delivering 14157 / 17262 / 19033 environment steps/s with minimum free VRAM
4295 / 3347 / 2419 MiB. At this short benchmark capacity, 6144 was 9.3% slower
in total throughput than 8192; these figures exclude long-block recycling.

R19's freshly checked trusted standing prior passed both gates: nominal
survival 100%, pushed survival 92.1875%. Nominal had both feet loaded throughout,
zero COM/support outside samples and 0.0749 m mean final drift. Pushed final
drift was 0.0608 m and COM/support outside fraction 1.24%. This qualifies only
the standing prior; the checkpoint SHA is bound in `reward_migration.json`.

The retained candidate's baseline nominal gate failed with 0% survival in
both stopped and 0.15 m/s cases, and zero qualifying steps. It is the saved
1418-update R17 actor before R19's retention training, not a result of the new
loss. Its stopped lateral speed was 1.064 m/s; under the walking command it
was 1.745 m/s, with excessive slip and COM outside support. The pushed baseline
also failed with 0% survival and zero steps in both cases. All prior
actor/critic/Adam state is retained because
v6 rewards are unchanged; no qualification is transferred to this candidate.

At **2026-10-02 14:29:13 UTC / 10:29:13 EDT**, the full block
`010_walk_015_train_001` was running at **8192 environments**, with both
supervisor and worker PID/command lines verified. Its zero-based iteration 8
means nine completed PPO updates; chain block accounting is updated separately
when a block or recycle ends. VRAM was 5742 MiB used / 2215 MiB free, and host
RAM had 6883 MiB available. The additional loss was active in all 32 minibatches
of the latest update, with standing-prior MSE 0.2673 and weighted loss 0.06682.
The candidate retains the 1e-5 learning rate and prior optimizer state.

Early training telemetry still reports falls and very few valid steps
(0.0203 steps/s). Its moving speed of 0.156 m/s does not qualify walking.
No improved gait or restored candidate standing is claimed. No allocation
failure occurred in these first nine full updates. A startup graphics-cache
initialization error appears in stderr, but the worker continued through PPO;
it is not evidence of a training crash. Host allocation growth still requires
the verified checkpoint-recycling guard. The run remains active for its next
independent candidate and retention checks.

Read the [fresh machine-readable status](checks/gait_integrity_20261002_1245/status.json),
`ACTIVE_RUN.json` and the indicated run state for actual startup, benchmark,
trusted-prior, current-command and retention results. A smoke or benchmark
pass never qualifies walking. The supervisor targets 8192 environments, keeps
the 300 MiB reserve, falls back on allocation pressure and ramps one tier after
180 seconds of healthy live-training capacity. Host RAM guards checkpoint and
recycle owned workers, preserving Adam within v6. The underlying allocation
growth is still unresolved; maintenance is containment, not a leak fix.

No later walking or running qualification is claimed. The computer remains on
unless the entire chain through run_250 independently qualifies and all work
is saved. This audit did not issue a shutdown.
