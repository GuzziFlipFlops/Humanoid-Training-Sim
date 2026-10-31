# Preflight results — September 2, 2026

Seven CPU tests passed: initial partial stances excluded; brief dropouts bridged; completed uneven stances penalized; stale/reset state cleared; independent ankle grouping/mirror invariance; missing walking evidence rejected; and a 4 m/s command with 1 m/s tracking error rejected.

The active launch is `E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v1_20260902_1726`.

- 64 environments, eight PPO iterations: saved checkpoint and result; no Python errors. All added reward terms were instantiated.
- Separate-process checkpoint reload and 20-second evaluation: passed execution and telemetry finiteness checks. This was a harness/load check, not a behavioral promotion.
- 4,096, 6,144 and 8,192 environments: four PPO iterations each completed and saved checkpoints. These models are smoke artifacts only; none becomes the actual training seed.
- 8,192 block: 14.218 seconds for the four learning iterations including first-iteration overhead; approximately 55,313 environment steps/second. Short benchmark, not a sustained-training time guarantee.
- Sampled total GPU memory at 8,192: 7,144 MiB of 8,188 MiB, leaving 1,044 MiB. Reserve: 512 MiB. Future memory pressure is still monitored.
- GPU power reporting returned N/A; no measured electrical power figure is available.
- SHA-256 checks confirmed both original checkpoints, `humanoid_cfg.py`, `env_cfg.py`, `skill_mdp.py`, and the robot USD remained unchanged.

The earlier `_1723` launch was intentionally stopped during preflight to fix Windows process-tree cleanup; it was not a reward/physics crash. The supervisor now terminates only its own worker tree if a reserve or stall condition requires a retry/stop, avoiding orphaned simulator children.

Live progress and final behavioral decisions belong in `state.json`, `*_gate.json` and the launcher log, not this static preflight record.
