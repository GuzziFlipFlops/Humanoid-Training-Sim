# Four-hour training audit — October 2, 2026

The scheduled check fired at 04:45 EDT. The computer remains on because walking
and the later running stages have not passed. The source-frozen repaired run is
`runs/locomotion_20261002_r15_motion_discovery`. Read `ACTIVE_RUN.json` and fresh
state/processes for current progress; this report is a dated audit.

## Confirmed failure

R14 was alive at 8192 environments, without an OOM crash. Its nominal 0.15 m/s
walking test survived all 20 seconds but averaged 0.03505 m/s, zero left steps
and 0.140625 right steps per robot. The pushed test also survived all trials,
averaging 0.03665 m/s and 0 / 0.34375 steps. Cadence and bilateral step checks
failed. Standing retention failed the conservative COM/support check. High
survival is real progress from the earlier falling policies, but is not walking.

The supervisor was drained before editing. It preserved
`r14/057_walk_015_train_009_part_02/resource_recovery.pt`, all logs and the trusted
standing prior. There were 548 R14 learning updates plus 64 inherited updates.
Resource recycling preserved learned actor/critic/Adam state; the repeated
workers were not restarting learning from scratch.

The reward still allowed up to 1.75 posture reward per second under a movement
command, while stepping guidance expired after about 2.56 seconds at 0.15 m/s
if no real landing occurred. The small missing-swing penalty did not offset
posture income. This supports a stationary local optimum rather than success.

## Focused repair

`motion_deficit_discovery_v5` retains full posture gradients. A moving policy
pays up to 1.8 per second when it has no validated forward activity. Activity
requires new territory, actual foot activity or a recent accepted landing,
traction, commanded speed/direction accuracy and reward eligibility. Ineligible
states cannot switch off that cost. At rest, maximum moving posture income
minus this cost is -0.05 per second before other existing costs. Genuine stop
commands retain their previous standing rewards.

Measured unloading, sole lift and forward foot/body placement now earn positive
increments from a finite bank, capped at one unit per foot. Holding an unloaded
pose, rocking, repeated kicks, sliding, positive command resampling and stop/
restart do not replenish it. Only a physical episode reset or an accepted real
alternating landing renews that foot's bank. The old load-transfer rate now
requires a recent accepted landing. The conservative total free discovery bound
stays below the eight-unit fall cost, including the old temporary gait/reference/
flight shaping. This is a bounded incentive repair, not a guarantee of learning.

Reward migration preserves the latest learned actor and its observation
normalizer. The changed reward requires fresh value estimates and optimizer
state. Immutable assets, CAD hardware/limits, references, PPO configuration,
observation/action mapping, contacts, reset/randomization and the independent
verifier are compared before migration. Trusted prior skills are independently
requalified in the new source. Current walking and all retention cases must
still pass; no old walking qualification is copied.

Running retains physical alternating landing checks, legal phase-specific
flight, clearance/cadence/net-travel incentives and hop rejection. The new
motion/discovery correction also applies to running commands. The chain cannot
reach running until all earlier walking gates pass. No cameras were added.

## Verification and remaining limits

All 102 CPU tests passed, including trajectory counterexamples for stationary
income, repeated rocking/kicks, planted sliding, command/reset farming,
control-step invariance, accepted-landing renewal and preserved candidate
migration. A fresh four-update 64-environment Isaac PPO smoke completed and
confirmed exact actor tensors before training, with fresh critic/optimizer for
the new reward. Checkpoint load completed; its behavior gate failed. Fresh
4096/6144/8192 PPO capacity benchmarks completed with 14566 / 17052 / 19316
environment steps/s and minimum free VRAM of 4349 / 3401 / 2475 MiB. These are
four-update measurements, not long-block throughput including worker recycling.
At this measured capacity, 6144 delivered 11.7% fewer total steps/s than 8192.
Checkpoint reload, capacity benchmarks,
trusted prior and current/retention gates are saved in the repaired run.
The latest runtime outcomes are recorded in the linked machine-readable audit;
smoke or benchmark completion alone is never behavior qualification.

R15 freshly requalified the trusted standing policy: 100% nominal and 92.1875%
pushed survival, with every standing check passing. The preserved 612-update
candidate's baseline walking cases survived 100% but averaged 0.09994 nominal /
0.10145 pushed m/s, zero left steps and 2.03125 / 2.171875 right steps. Bilateral
steps and cadence still failed, and candidate standing retention still failed
COM/support. These gates precede learning under v5; improvement from R14's
earlier 576-update gate cannot be attributed to the new reward.

The supervisor then launched `010_walk_015_train_001` at 8192. At its fifth
completed PPO update (zero-based iteration 4), live logs showed 2437 MiB free
VRAM, 11771 MiB available host RAM, 0 falls and 0 invalid differential states.
New finite discovery and motion-deficit terms were active in the logs. Actor
normalization p99 values were bounded, with the largest sampled group at 3.58.
This confirms startup and learning updates, not a successful walking repair.
The checkpoint-based RAM guard, reserve, fallback and three-minute ramp remain
active after this check ends. The chain stays on; no shutdown was issued.

The first full worker subsequently saved 18 complete updates at 1601 MiB
available host RAM and 1981 MiB free VRAM. The supervisor resumed
`011_walk_015_train_001_part_02` at the same 8192 environments for the remaining
46 updates. Five more real updates were observed after restart, with host
headroom restored above 11 GiB. The [saved optimizer audit](
checks/four_hour_audit_20261002/optimizer_continuation.json) confirms Adam
advanced from 576 to 608 minibatch updates on the first continued PPO update,
retaining the saved 1e-5 adaptive learning rate. This verifies continuation
within v5; it does not establish that the memory leak is fixed.

The new CPU audit again finds all 23 saved native CAD frames/axes/limits agree
with the training URDF, with both leg Jacobians rank six. Actual gate joint
ranges are saved separately. No sampled pre-fall hard-limit (0.005 rad
tolerance) or motor-speed violation was found. Normalization analysis now uses
the checkpoint that actually produced the sampled gate instead of the older
standing prior. Its sampled q/qd p99 maxima were 4.41 / 3.24, without evidence
of the earlier exploding-scale fault. High ankle target clipping remains a diagnostic
concern; this repair does not enlarge travel or alter gains/geometry. Exact
kinematics does not prove adequate balance authority, realistic collisions or
correct provisional mass/inertia. The reference lacks an explicit lateral COM
plan; natural appearance still requires rollout/video review.

Target capacity remains 8192, with the existing 300 MiB VRAM reserve, automatic
fallback and a 180-second healthy-allocation ramp at checkpoint boundaries.
Host memory growth remains unresolved. The existing guard checkpoints/recycles
the worker after a complete PPO update below 2 GiB available RAM or above the
private-commit threshold, retaining optimizer state within the new reward run.

Evidence: [summary](checks/four_hour_audit_20261002/summary.json),
[kinematics/policy diagnosis](checks/four_hour_audit_20261002/diagnosis.json),
[measured joint ranges](checks/four_hour_audit_20261002/measured_joint_ranges.csv).
`checks/four_hour_audit_20261002/capture_summary.py` reproduces the file-based
status snapshot without starting a second simulator.
