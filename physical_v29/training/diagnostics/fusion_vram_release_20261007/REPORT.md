# Fusion VRAM release — October 7

Fusion and its preview process are absent after normal UI shutdown. No process was force-killed. Normal close encountered one pending save/upload job; Fusion's **Pause unstarted jobs** action preserves queued jobs for the next launch. **Update Later** was then selected; no update was installed. No unsaved-change discard prompt was presented and no Discard/Do not save action was selected. Earlier Strap cam unsaved status was dated context, not assumed current state.

Global VRAM was 773 MiB free at the initial reading and 5219 MiB free in the saved post-exit reading, on an 8188 MiB GPU. Training continues across these readings, so the difference is not an isolated Fusion memory measurement or a throughput benchmark.

Fresh owned observation 2026-10-07T17:05:20.537373+00:00 verifies the same R40 supervisor 39000, one actual worker and all 55 frozen hashes. Current operation `303_run_350_gate_099_nominal` has count 320; evaluation count is not PPO capacity. Training was neither paused nor restarted and no training source or policy parameter changed. Existing target 8192, 300 MiB reserve, host guard/fallback and 180 continuous healthy-training-second ramp remain. A startup or evaluation free-memory reading does not authorize an immediate capacity increase.

The user now explicitly authorized closing Fusion to release GPU memory. Do not reopen it merely for a routine training check. Preserve queued user work; future CAD/video tasks may require reopening only when necessary. [Resource receipt](resource_receipt.json) binds processes, memory and [fresh ownership evidence](E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261007_1344/fusion_close_owned/live_handoff.json).
