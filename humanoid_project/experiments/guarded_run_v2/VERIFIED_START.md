# Verified startup — September 2, 2026, 21:16 EDT

The revised chain passed its seven CPU tests, 32-iteration / 64-environment smoke run, saved-checkpoint reload, fresh 8,192-environment check, reward-blend endpoint assertions, and archived-walker self-comparisons.

Its first real indexing-only block completed 50 iterations at 8,192 environments in 187 seconds. Both matched walking test batches passed the preset practical regression tolerances, without relaxing any gate. Nominal 1 m/s mean speed was 0.9997 m/s with no falls. Nominal stance asymmetry was 0.375 versus reference 0.338: within the predeclared +0.04 tolerance, **not an improvement claim**. The next balance stage must demonstrate an improvement relative to the original seed before promotion.

At 21:15:55 EDT the supervisor started `01_balance_1_0200_8192`, the first 200-iteration block of the 600-iteration maximum stance-balance stage. Subsequent state is in the live file, not this startup note.

- State: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v2_20260902_2103\state.json`
- First passed gate: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v2_20260902_2103\00_index_1_0050_gate.json`
- First candidate: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v2_20260902_2103\00_index_1_0050_8192\candidate.pt`
- Active block logs: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v2_20260902_2103\01_balance_1_0200_8192\`

All policies remain experimental except the user-approved archived walker. The supervisor will reject regressed checkpoints and stop advancement at failed stage gates. No original CAD, actuator limits, old rewards/task configuration, or archived checkpoint was modified.
