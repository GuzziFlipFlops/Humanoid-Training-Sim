# Adaptive humanoid training check — every five hours

User update recorded 2026-10-06 UTC (October 5 in America/New_York). This workflow supersedes the three-hour cadence and generic actuator-ceiling endpoint in `THREE_HOUR_CHECK.md`. Earlier records remain historical evidence.

## Identity and authority

Update the existing heartbeat `humanoid-training-research-and-repair-every-four-hours`, in the same thread, every five hours. Preserve its notification preference and GPT-6.1 Sol / max reasoning setting. Never create a duplicate. The user authorizes research, diagnosis, justified repairs and training continuation, and asks this automation to revise its next focused tests after each check. Preserve these durable instructions when updating the dated handoff.

Read the humanoid-robot-training skill, this file, `NEXT_CHECK.json`, fresh `ACTIVE_RUN.json` and actual run state. Verify owned supervisor/worker processes, authored source hashes, checkpoint lineage, independent gates, GPU and host RAM and synchronized joint/contact/COM/action telemetry. Dated notes never establish the current stage or qualification.

## Preserve and improve learned locomotion

Keep one actual Isaac/CUDA worker, no stereo, target 8,192 environments, the 300 MiB VRAM reserve, host-memory guards, fallback and 180 continuous healthy **training** seconds before upward ramps. Preserve iteration-4 execution optimizations and native CAD geometry, joints and motor constraints unless evidence justifies a tested change. Preserve unrelated unsaved work and all checkpoints, logs, rejected experiments and source hashes.

Compare actual samples/time and matching independent behavioral trends with earlier checkpoints. Report active PPO, worker-effective and whole-chain rates separately, excluding documented pauses. Capacity, smoke, evaluation startup and reward are not learned-policy progress. Check recovery preserves actor, critic, Adam and exploration exactly where compatible.

Preserve an improving actor. Promote only after matching independent nominal, pushed, stopped and every earlier-speed retention gate passes on the exact saved checkpoint. Audit running alternation, cadence, clearance, geometric flight, fixed-frame heading, signed speed, lateral travel and net travel. Do not edit live frozen sources, relax gates or restart useful learning just to change a future speed list.

When independent progress stalls or tests reveal a fault, diagnose the reproduced failure with primary research and controlled counterfactuals. Test the complete objective and physical response, not a single bonus. Require CPU counterexamples, a small physical/PPO/full-state-load smoke and an appropriate matched unchanged-control learning comparison before deploying a narrow continuation in a fresh frozen run. Preserve compatible learned state; no from-scratch restart or tilted-head rollback to repair posture. Do not deploy diagnostic pulses, observation rewrites, hybrid controllers or lifted motor caps. Do not repeat rejected candidates without new evidence or spend repeated checks only waiting.

## Required upper-body analysis in every check

The user explicitly requests **joint angles over time**, rather than relying on head RMS or a still image. Use completed matching nominal/pushed gates, identify their exact checkpoint and commands, and save time-series plots plus machine-readable data for:

- Actual head yaw/pitch, shoulder pitch/roll and elbow flexion; physical targets and joint velocities where available. Report forward alignment relative to the torso, not a claim of world gaze from native angles alone.
- Per-speed mean posture, range, speed, smoothness/jitter and saturation. Separate whole-population statistics from selected high-rate traces; segment resets and avoid treating 10 Hz sampling as evidence about faster jitter.
- Left/right shoulder and elbow coordination, simultaneous versus alternating flexion, and each shoulder's timing relative to the ipsilateral hip/leg and actual gait cadence. Report phase and amplitude together; near-static arms do not have a meaningful swing phase. Inspect plots, not just aggregate numbers.
- Trends across earlier certified actors and newer candidates, including slow-speed retention and running. Preserve the learned neutral-head posture and all useful leg/torso behavior during refinement.

Compare available primary human locomotion data and official Unitree joint-motion references at comparable speeds. Align coordinate axes, signs, rest poses, joint mappings, timing and morphology before interpreting angles. Human wrist IMUs, Unitree reward configurations, resting poses or dance motion do not establish an exact walking/running encoder reference. Record missing or unmatched reference trajectories explicitly; never invent reference curves or declare human/Unitree likeness from joint bounds, head neutrality or reward alone.

The reusable CPU diagnostic is `diagnostics/adaptive_check_20261006_0741/upper_joint_time_audit.py`; provide the current exact-pair gate-review JSON and a new dated output directory. It caches immutable NPZ arrays once, segments first episodes before terminal/reset ticks, and writes actual head/shoulder pitch+roll/elbow angle-target plots, separate velocity plots, CSV and per-command population/coordination statistics. Its phase and 5-degree head bands are **diagnostic**, not newly relaxed or substituted locomotion gates. Continue broader visual/full-CAD checks when needed. If motion remains unnatural, measure effective clipped upper targets and auxiliary/PPO gradients, test physical response and a narrow full-state refinement against control without erasing leg learning.

For slow shoulder drift, also use `diagnostics/adaptive_check_20261007_1344/upper_bias_audit.py` with the current gate-review JSON and dated output directory. It separates actual 10 Hz population angles into a descriptive 0.5 Hz slow component and faster residual, excludes three-second filter edges, uses the actual post-warmup 2.02-second time origin and never filters across resets. Inspect its plot alongside physical target and velocity timelines. The faster residual includes non-cycle motion; neither total angle span nor this filtering certifies periodic, human-like or Unitree-like movement. Four synthetic decomposition checks and the original October 7 attempts are preserved in that diagnostic folder. This tool changes no training source, physics, reward or classifier.

## Completion: verified 5 m/s, or verified 4 m/s when actuator limits block 5

Primary endpoint: actual fixed-frame 5 m/s locomotion, using the existing declared 1% measurement tolerance (at least **4.95 m/s** signed forward speed at the 5 m/s command), with matching independent nominal/pushed and stopped/all prior-speed retention gates. Use at least 32 independent trials per command. Keep geometric-flight, contact, direction, survival and all other gates unchanged.

Fallback endpoint authorized by the user: qualify actual **4 m/s** by the same standard, with at least **3.96 m/s** signed forward speed under that same declared 1% measurement tolerance, and matching nominal/pushed/stopped/prior-speed gates, **only when** repeated higher-speed tests and controlled analysis establish that native locomotion-relevant actuator limits prevent reaching 5 m/s. This adds a measured-speed condition; it does not loosen an existing gate.

Attribute the limitation using actual CAD differential Jacobians/reductions, native motor velocity and available torque, sustained phase-specific saturation/driving clipping and a reproducible speed plateau. Distinguish policy faults, turning, falls, timing and inadequate exploration from an actuator ceiling. Isolated cap spikes, a commanded speed, reward, elapsed budget or a crash do not establish a limit or completion. Do not increase limits to reach either endpoint. A limit observed below 4 m/s is not a verified 4 m/s finish; report the achieved speed and evidence without claiming the requested fallback was reached.

At either verified endpoint, preserve the exact final full checkpoint, hashes, all gates, motor/upper-body telemetry and a completion briefing. Safely pause owned training at a saved boundary and prevent its supervisor from restarting it; pause this same automation. Brief the user on achieved speed, retention and any limiting actuators. **Leave the computer on.** This replaces historical shutdown instructions. While qualification is incomplete, keep the same heartbeat active and leave healthy training running.

## Durable records and adaptive next focus

After every check, write a nearby dated evidence/decision report, append `ITERATION_HISTORY.md` and the robot journals, refresh the CPU-only history inventory and current handoff references. Update `NEXT_CHECK.json` with the actual run/checkpoint, exact gate trends, upper-body time-series results, measured throughput/memory and one to three next focused tests. Inspect and update the existing native automation's dated focus through `automation_update`, preserving the five-hour cadence, same thread, notification preference, model, authority and 5/4 endpoint. Full-body recordings already delivered remain complete; no repeat request is pending.
