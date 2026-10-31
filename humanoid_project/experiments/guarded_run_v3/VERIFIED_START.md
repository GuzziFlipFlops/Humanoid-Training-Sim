# Launch verification — September 2, 2026

Run: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v3_20260902_2244` (folder label is an identifier; actual launch was 22:41:02 local).

- Ten CPU tests passed. Gates and evaluator hashes match v2 exactly.
- 64-environment, 32-iteration smoke completed in 94.735 seconds; checkpoint reloaded/evaluated successfully in a new process.
- Smoke balance signal: valid coverage 46.22%; mean weighted cost -0.07805. This is reward activation evidence, not behavioral success.
- Fresh 4,096 / 6,144 / 8,192 four-iteration checks all completed. Times were 19.36 / 20.578 / 20.828 seconds including first-iteration overhead. Sampled peak total GPU memory was 6,650 / 7,363 / 7,765 MiB. The final peak briefly left less than 512 MiB; it did not persist for the three consecutive samples required to trigger fallback.
- Both original reference self-checks and both fresh resumed-candidate walking-retention checks passed. Nominal resumed 1 m/s speed 1.00204 m/s, zero evaluation falls, stance asymmetry 0.34360. This still does NOT pass the required balance improvement.
- Main block `00_balance_1_0100_8192` launched at 22:56:37 local. Verified at least 12 PPO iterations advancing, with empty worker/supervisor stderr at inspection.
- Current training samples: approximately 5.0–5.2 seconds/iteration, GPU utilization around 38–44%, 61 C, memory 7,671–7,673 MiB / 8,188 MiB (515–517 MiB free). This is close to the reserve; 8,192 is not guaranteed to stay selected. Supervisor falls back to 6,144 then 4,096 on sustained memory pressure.
- GPU power draw is unavailable through this device's `nvidia-smi`; do not invent measured watts.
- All six protected original files still match their SHA256 hashes. No original policy, physical model, or production task was overwritten. Private file index remains ignored by Git.

Timing update: at the first main-block rate near 5 seconds, the 4,200-iteration ceiling alone would take about 5.8 hours, plus 1–2 hours of checks/startups if every block is used. This supersedes the historical faster-iteration estimate for this launch. Early gate passage or environment fallback changes actual time.

First main behavioral gate occurs at 100 iterations. No claim of improved balance or 4 m/s running has yet been made. The background supervisor, not an active Codex turn, owns subsequent execution; read live `state.json` for current status.
