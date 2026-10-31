# Recovery v3: collision-aware continuation, September 3

## Evidence

v2 fixed the inference-buffer push crash and restored floor-level height feedback.
The completed 200- and 400-iteration evaluations nevertheless had 0/128 nominal
and perturbed successes. Mean maximum height rose to .55 m and 1.11 m, respectively.
Every recorded unsafe termination in the nominal 400 evaluation crossed the
unchanged 15 mm non-neighbor penetration limit (3712 frames including subsequent
resets; 128 distinct first trials failed). Peak penetration was 32.65 mm; peak
vertical speed 2.61 m/s and root angular speed 10.56 rad/s. Height alone was
therefore a misleading progress signal. These are measured associations, not
proof that every upward movement caused the collisions.

Source evidence: E:/Aditya/robot_sim/humanoid_project/logs/guarded_getup_v2_20260903_103200, especially
00_supine_discovery_0400_g0_s47/telemetry.npz and evaluation.json. v2 was stopped
partway through its third block; its original checkpoints are preserved.

## Changes

- Failure penalty 3 -> 8, above the entire 4-point episode progress budget.
- Smooth approach-to-penetration cost starts at 3 mm and reaches a rate of -2
  at the existing 15 mm failure limit. The hard limit is not relaxed. Direct
  mechanical-neighbor exclusions and all collision geometry remain unchanged.
- Multiply new-best progress by exp(-.5*vz^2 - .05*root_omega^2 - .05*qdot_rms^2).
  A ballistic/violent rise receives less progress credit. Best-score memory
  still advances, so stopping afterward cannot collect the missing credit.
- Small rise-speed cost above 1 m/s vertical or 4 rad/s root angular speed.
  These are soft engineering regularizers, not mechanical velocity-limit changes.
- The retained earlier v2 actor had mean exploration std 1.1125 (range .6245–1.5415),
  equivalent to roughly .78 rad mean joint-target noise with the .7 action scale.
  Migration retains actor mean weights and observation normalization, resets
  critic/optimizer, resets std to .45 and bounds it to [.15,.60] after optimizer
  steps. Discovery entropy .008 -> .001; refinement .003 -> .0005.
- Stop after a gate if any batch has >=90% unsafe trials, even if height improves.
  This prevents another blind continuation justified solely by launching higher.

The two-foot 2+5-second hold, fixed anchors, post-success invalidation, contact
checks, motor limits, observation dimensions and independent evaluator are unchanged.
These changes address measured failures; they do not guarantee convergence or
make this model hardware-ready. No walking/running files or checkpoints changed.

## Lineage and operation

Main source: v2's 00_supine_discovery_0200_8192/candidate.pt, which contains the
earlier v1 200 iterations plus 200 additional v2 iterations. Do not use the later
1.11 m candidate or the interrupted third block. Main continuation discards smoke
updates. Ceiling: 600 supine, 800 four-pose, 400 refinement additional iterations;
independent paired gates every 200. Start requests 8192 environments with 512 MiB
reserve, falling back only for demonstrated memory pressure. v2 previously passed
8192 initially but fell back to 6144 during its second block.

Run chain.py with --output NEW_DIRECTORY --resume_checkpoint EXACT_CHECKPOINT;
monitor.py reads logs/guarded_getup_v3_*/state.json. Per-block results, snapshots,
checkpoint hashes, telemetry.npz and GPU samples remain archived in that run.
See v1/v2 README files for prior experiment details and primary research references.

## Launch verification

Run: E:/Aditya/robot_sim/humanoid_project/logs/guarded_getup_v3_20260903_115042.
28 CPU tests passed. The 64-iteration smoke saved and reloaded; its full 25-second
perturbed evaluation completed (0/16 strict recoveries, 12/16 unsafe trials,
mean maximum pelvis height .586 m). This is not a successful recovery policy.
4096/6144/8192 four-iteration benchmarks passed, sampled peaks 5804/6723/7556 MiB.
Main supine continuation began at 12:06 EDT. At 12:08:30, 7/200 main iterations
were confirmed with finite updates, 8192 environments, 7596 MiB used / 592 MiB
free, and sampled GPU utilization 41%. Memory reserve remains 512 MiB. These are
launch observations, not live status or a guarantee of later success. Read state.json.

Only the read-only monitor's stop-status handling changed after the initial
source snapshot, to recognize stopped_unsafe; training/reward sources did not.
