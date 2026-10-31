# Verified restart — September 3, 2026

Supervisor launched at 03:04:55 local. Run: `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v4_20260903_0305`.

- 16 reward/contact/gate tests and 6 monitor tests passed. Compilation passed.
- Original robot/task files and archived policies retain all six original SHA256 hashes.
- 64-environment 32-iteration smoke passed in 93.906 seconds. New low-speed torso cost was active (last episode contributions approximately -0.125 to -0.132); this is activation evidence, not proof of learned improvement.
- Smoke checkpoint reloaded/evaluated in a separate process.
- Fresh 8,192-environment four-iteration benchmark passed in 21.281 seconds, using the accepted transition checkpoint, native .35 scale, and new reward code. Reserve remained 512 MiB. Earlier full 4,096/6,144/8,192 scaling evidence was hash-verified before reuse.
- Fresh resumed-candidate nominal and push evaluations passed walking retention. At 1 m/s actual speeds were 0.997/0.998 m/s; at 1.8 m/s both approximately 1.780 m/s. No falls in either batch at these commands (24 rollouts per case per batch).
- At 03:14:08, the unchanged target and torso stage gates each passed before additional training. The supervisor correctly skipped their unnecessary PPO blocks and advanced to the 2.3 m/s evaluation.
- Last inspected phase: `evaluating`, `02_speed_2.3_0000_eval_47`, training environment setting 8,192. The evaluation worker uses 144 environments. No full-size main PPO block has been claimed to start yet: the chain first tests the new speed and trains only when its gates are not already satisfied.
- Supervisor and worker error logs inspected without errors. Existing `guarded_run_v3\monitor.ps1` command successfully displays this v4 run.

This is a launch verification, not a live status file or certification of 2.3/4 m/s running. The process continues independently of the assistant turn. Read `state.json` and gate/evaluation artifacts for newer results.
