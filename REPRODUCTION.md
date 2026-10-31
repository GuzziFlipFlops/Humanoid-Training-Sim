# Training setup

This is a selected replication package, not the complete per-update archive. Both original projects were copied; neither was moved or resumed. The working files occupy less than 500 MB.

## Robot versions

| Folder | Robot |
|---|---|
| `physical_v29` | Later CAD-based 23-axis robot; physical actuator/differential models and running curriculum |
| `humanoid_project` | Earlier 1.524 m / 22-axis robot; standing, walking, running and get-up experiments |
| `final_continuation` | Self-contained final v29 learner, teachers, frozen source and portable continuation wrapper |

These models have different observation/action contracts. Their checkpoints are not interchangeable.

## Environment

Use Isaac Sim 5.1.0 and Isaac Lab commit `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`, with the matching Isaac Lab/Isaac Lab RL extensions installed in a CUDA Python environment. Recorded packages include PyTorch 2.7.0+cu128 and rsl-rl-lib 5.0.1; install NumPy, psutil and TensorBoard as well. See `replication/versions.json` and follow the Isaac Lab installation instructions for platform prerequisites. The simulator, virtual environment and NVIDIA runtime are external dependencies, not vendored here.

## Later robot

Start with [the continuation guide](final_continuation/README.md). It includes hash verification and explicit evaluation/training commands. Its frozen assets and source match the final full checkpoint; USD is regenerated from the bundled training URDF.

The original authored scripts, rewards, experiments and historical frozen sources remain under `physical_v29/training`. Each run's frozen `source` directory identifies the environment/reward implementation for that experiment. Saved `chain_config.json`, agents, reports and decisions retain the progression and rejected changes. Historical scripts and reports may contain original absolute paths. Use the portable final wrapper for a new machine rather than starting an old supervisor unchanged.

The final v29 actor reached about 3.53 m/s at command 3.5 but failed direction and retained-skill gates. Earlier complete qualification reached 3 m/s. It did not qualify at 4 or 5 m/s. See `final_continuation/results/gate_assessment.json` and the final diagnostic report under `physical_v29/training/diagnostics/final_check_20261010_0905`.

## Earlier five-foot robot

Task definitions and rewards are in `humanoid_project/source/humanoid_tasks/curriculum`; PPO settings are in its `agents` subfolder. `humanoid_project/experiments` contains the separate guarded-running and get-up iterations. Their original `DESIGN.md`, candidate, worker, evaluation and gate scripts are included. Robot URDF/USD assets are in `humanoid_project/assets`.

From the repository root, resume the reviewed walking policy with:

```sh
python humanoid_project/scripts/rsl_rl/train.py --task Isaac-Humanoid5ft-NaturalWalk-v0 --headless --num_envs 64 --max_iterations 64 --resume_path humanoid_project/logs/rsl_rl/humanoid5ft_natural_walk/2026-08-31_17-06-10/model_2098.pt
```

For the guarded running policies, use the worker belonging to their exact experiment. For example:

```sh
python humanoid_project/experiments/guarded_run_v8_posture6/worker.py --mode train --stage speed --ceiling 5 --scale 0.35 --checkpoint humanoid_project/logs/guarded_run_v8_posture6_20260905_0705/01_speed_5_0400_8192/candidate.pt --reference humanoid_project/logs/rsl_rl/humanoid5ft_natural_walk/2026-08-31_17-06-10/model_2098.pt --output new_runs/five_foot_continuation --num_envs 64 --iterations 64 --learning_rate 0.000025 --seed 47 --resume_optimizer --headless
```

These commands start new continuations. They do not promise bitwise replay of the original rollout/RNG state. No fresh simulator session was started to test this export. PowerShell launchers assume an `env_isaaclab` environment beside the project; adjust that path for your installation. Older checkpoints are loaded through the compatibility handling in the supplied workers.

## Checkpoints and evidence

`replication/selected_checkpoints.json` records each selected checkpoint's role and SHA256. It includes reviewed earlier walking/4/5-command policies, an intermediate walking checkpoint, the last saved running/get-up states, and v29 R40 blocks 305, 611 and 917. Final v29 actor/critic/Adam/exploration, two teachers and comparison actor 1064 are in `final_continuation/checkpoints`.

The earlier 4/5 m/s review labels refer to that older model and its historical tests; they do not inherit the later robot's qualification contract. Failed or incomplete saved states are identified in the selection index. Bulk rollout arrays, full telemetry, all intermediate checkpoints, videos, simulator caches and framework installs are omitted to meet the size limit.

`replication/file_manifest.json` records copied-file hashes. The applicable third-party notices are retained with the sources and in `licenses`.
