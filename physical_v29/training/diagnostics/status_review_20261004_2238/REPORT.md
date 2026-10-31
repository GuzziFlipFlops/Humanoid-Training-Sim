# R34 user-requested status review

Evidence observed 2026-10-04T22:45:17.260216+00:00. The supervisor is running, stage run_150,
with 449 completed new main updates.
Current operation 041_run_150_train_008; selected training count
6144. One actual CUDA worker and all
46 frozen hashes were independently verified at 22:42 UTC; the owned snapshot
is preserved separately. State and immutable saved test results are fresher.

The exact checkpoint 038_run_150_train_007/candidate.pt
(SHA256 d41aeaaef963952487f01917f7defba618cbad4709f94b2e09bfd779e9b5e821) now passes the running case in both
full 192-robot, 20-second nominal and pushed tests, 32 independent trials per
command. Signed speed is 1.475762/1.476740 m/s; geometric flight
is 1.885417%/1.756945%, above the unchanged 1.5%
minimum; survival is 100% throughout both test populations. Running stance
asymmetry is 0.179171/0.172767, below the
unchanged 0.18 maximum. Nominal is close to that boundary, so robustness is
not established by this single matched pair.

| Command m/s | Actual signed speed, nominal / pushed | Case result | Failed checks |
|---|---|---|---|
| 0.00 | 0.000 / 0.000 | pass / pass | none |
| 0.15 | 0.262 / 0.263 | fail / fail | speed, world_forward_speed |
| 0.35 | 0.455 / 0.451 | fail / fail | speed, world_forward_speed |
| 0.65 | 0.737 / 0.737 | pass / pass | none |
| 1.00 | 1.047 / 1.044 | pass / pass | none |
| 1.50 | 1.476 / 1.477 | pass / pass | none |

The complete 1.5 stage has **not** qualified: 0.15 and 0.35 m/s retained commands
overspeed. Stopped, 0.65 and 1.0 cases pass both. The independently qualified
baseline through walking 1.0 remains preserved. No 2.0 or 5.0 qualification is
claimed. Previous full pairs had zero/negligible flight; this pair is new
independent running progress, not a sustained trend yet.

Decision: leave rewards, PPO settings, retention gradients, commands, physical
model, motor caps and frozen sources unchanged now. Preserve this useful flight
actor and continue the already running chain, with promotion blocked by prior
speed gates. Compare the next matched pairs for simultaneous flight and slow
tracking. If the tradeoff persists, perform a controlled per-command effective
action/target and objective/retention-gradient diagnosis before a focused repair;
do not assume a weight increase or actor restart will solve it.

Resources: the 8192 operation034 hit the 300 MiB VRAM reserve, saved
memory_recovery.pt and resumed at6144. Last completed6144 block038 achieved
35,642.8 active collection-plus-PPO samples/s and30,352.2 worker-effective
samples/s (startup/exit included), with minimum1036 MiB GPU free and4764 MiB
host available. All recorded main-block pre-update KL and action/reward command
differences are zero. Capacity guards, full-state recovery, target8192 and the
180-second healthy-memory upward ramp remain. Eval/startup free memory is not
eligible for the ramp. The three-hour automation remains active; computer on.

This is a CPU-only status audit, with no simulator launch, live source edit,
training restart or parameter adjustment. Gate qualification pertains to the
unchanged frozen physical_v29 training contract; the separately documented
full-CAD visual/collision fidelity limitations remain.

Final ownership check at22:47:05 UTC: one actual CUDA worker PID17016, operation041 at6144, 449 completed main updates plus26 observed current-block updates. All46 frozen hashes verify; GPU used6954MiB/free1003MiB, host available5407MiB. Current pre-update KL0. CPU history inventory refreshed:1134 operations,0 differing hashes,0 read warnings;8 historical missing manifest artifacts remain recorded.
