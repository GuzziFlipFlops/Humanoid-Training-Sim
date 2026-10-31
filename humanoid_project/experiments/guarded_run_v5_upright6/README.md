# Guarded v5 implementation

Current design, evidence, budgets, and safeguards are in `DESIGN.md`. The material below is the retained v4 provenance explaining the code lineage.

## V4 source background

September 3, 2026. Current state: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v4_20260903_0305\state.json`.
Supervisor log: same path ending `.launcher.log` instead of `\state.json`.
This is a separate local experiment. Production robot/task files and archived policies are unchanged.

## Why v3 stopped

v3 was not a crash or memory failure. At 00:33:38 it exhausted 400 attempted target-stage iterations, rejecting both 200-iteration candidates for worse slow walking. No candidate with a failed walking check was promoted.

The preceding revision did help: balance passed in 300 iterations, improving the 1 m/s stance-asymmetry metric 5.73% nominal and 6.50% with lateral kicks versus the original walker. The transition stage then passed after 200 iterations with approximately 1.78 m/s actual speed at a 1.8 m/s command.

| Nominal evaluation | Torso tilt at .5 m/s | Torso tilt at 1 m/s | Outcome |
|---|---:|---:|---|
| Original walker | 6.63 deg | 4.53 deg | Reference |
| Passing transition checkpoint | 8.50 deg | 5.37 deg | Walking retained |
| First target candidate | 10.63 deg | 10.00 deg | Rejected |
| Lower-learning-rate retry | 9.66 deg | 7.36 deg | Rejected |

The .5 m/s nominal limit is reference +3 degrees, approximately 9.63 degrees. Even though the final nominal miss was small, both nominal and push tests failed that criterion. We did not relax it.

Source evidence: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v3_20260902_2244`, particularly `00_balance_1_0300_gate.json`, `01_transition_1.8_0200_gate.json`, both `02_targets_*_gate.json`, associated evaluation JSON/NPZ, and state/history logs.

## Diagnosis and narrow changes

1. **Already-complete stage criteria were not checked before training.** Replaying the unchanged gate functions against the last passing transition evaluation shows that both target and torso stages already passed. Forcing extra updates introduced a regression without being necessary to meet either stage's task criteria. v4 checks each incoming checkpoint first and advances immediately if both nominal/push batches satisfy walking retention and the requested stage. A stage budget is a maximum, not a training quota.
2. **Slow-speed torso protection was missing.** The original torso-specific cost had a factor `clamp(speed-1, 0, 1)`, making it zero for walking. Other upright rewards measured the root/pelvis, which does not necessarily keep the articulated torso upright. v4 adds a modest actual-torso penalty during target/torso/speed training. This closes an observed gap; it does not prove the target penalty alone caused the old regression.
3. **Resume only validated progress.** Resume `01_transition_1.8_0200_8192/candidate.pt`, not either rejected target candidate. `resume.py` verifies both pass flags and SHA256 against the checkpoint actually evaluated in both batches. The original assets/policies must also match their recorded hashes.
4. **Smaller, conservative updates.** Start at learning rate 2.5e-5, with checks every 100 iterations in target/torso stages and every 200 in speed stages. Walking regression still rolls back to the last passing checkpoint. Same network, .35 action scale, mixed lower-speed samples, ankle pairing, stance cost, and existing reward blending.

## Added torso cost

`retention.py:slow_torso_cost` uses actual torso tilt. The unpenalized cone is `10 - 4*min(abs(speed),1)` degrees: 8 degrees at .5 m/s and 6 at 1 m/s, above the measured original walker values. This is a tolerance cone, not an exact joint angle or posture trajectory. Normalized positive excess uses a continuous quadratic-near-zero/linear-tail cost, capped at 1.9, weight -0.50. It is fully active through 1 m/s and fades to zero at 1.8 m/s. The separate original high-speed torso cone remains unchanged. No new fixed knee angles, cadence, stride length, or equal instantaneous loading.

All original gate thresholds and the independent evaluator are byte-identical to v3. A good survival percentage still does not establish full task success. Torso, speed, stance, slip, crossings, and contact checks all remain required.

## Remaining chain

Target requests at 1.8 m/s (up to 400), torso at 1.8 (up to 400), then 2.3/2.6/3.0/3.5 m/s (up to 400 each), then 4.0 m/s (up to 600). Maximum 3,000 additional iterations. Target/torso may advance with zero new iterations if their incoming checkpoint passes. Speed increases require tests at the new command; no automatic calendar-based progression.

Each stage has a pretraining gate. If the checkpoint, speed, and deterministic evaluation conditions are unchanged, the exact measured rollout can be reused to check additional criteria (e.g. measured torso tilt from the target-stage evaluation). This does not substitute measurements from a different policy/speed. Each pretraining gate records `evaluated_as` for its evidence source. A new speed/checkpoint always requires a new evaluation.

At v3's recent roughly 4.7 seconds/iteration, 3,000 iterations alone would be about 3.9 hours, plus evaluation/startup overhead. Actual time depends on gate passage, fallback, and GPU workload; this is not a promise of success at 4 m/s.

## Preflight and audit trail

- Sixteen CPU tests passed before launch: ten stance/ankle/gate regression tests plus six torso/archived-stage tests.
- Fresh 64-environment 32-iteration smoke with the actual resumed checkpoint and new target reward; new-process checkpoint reload.
- Earlier 4,096/6,144/8,192 scaling checks reused only after unchanged original-file hashes and completed-result verification, plus a fresh 8,192 new-code benchmark. If that cannot maintain memory reserve, try 6,144 then 4,096. Reserve remains 512 MiB.
- Original walker baseline measurements reused only with checkpoint hash and byte-identical evaluator/contact tracker. Marked `reused_from.json`; not represented as fresh rollouts. The resumed candidate gets fresh nominal/push comparisons.
- Background worker ownership, finite checks, JSON/CSV logs, source snapshots, per-block YAML/config/commands/checkpoints, and rollback remain intact.
- Read live state before claiming main training or any stage has passed. Preflight launch alone is not proof.

Critical files: `candidate.py` (reward wiring), `retention.py` (new pure-torch cost), `resume.py` (checkpoint provenance), `chain.py` (pre-stage checks), `gates.py` / `evaluation.py` (unchanged scoring), `test_retention.py`, `monitor.py`. The existing v3 `monitor.ps1` command forwards to this run so the user's monitor command remains usable.

Simulator limitations remain: simplified collision geometry with disabled self-collision, proxy crossing checks, implicit-drive estimated torque, and approximate ankle transmission. No hardware certification or guarantee of natural running is implied.
