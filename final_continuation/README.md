# Isaac Lab training

Compact continuation package: the final learner, frozen training setup, two teachers, a comparison actor and final evaluation summaries. About 35 MB; no run history or videos.

## Final outcome

Training stopped on October 10, 2026. Final actor 1226 measured 3.534 m/s nominal and 3.541 m/s pushed at a 3.5 m/s command, but failed direction and lower-speed retention gates. The highest separately certified complete milestone was 3 m/s. Neither 4 nor 5 m/s was qualified.

The final checkpoint is a continuation starting point, not a deployable hardware controller. It uses the frozen v29 robot, not the latest Fusion assembly.

## Contents

- `checkpoints/final_1226.pt`: actor, critic, Adam optimizer and learned exploration.
- `checkpoints/standing_teacher.pt` and `qualified_349_teacher.pt`: original training auxiliaries.
- `checkpoints/selected_1064_video_actor.pt`: comparison actor used for the final videos.
- `source/physical_v29`: exact frozen environment, rewards, actuator limits, differential mapping, URDF, motion reference and local ground.
- `config`: PPO settings, joint order, curriculum, teacher certificate and dependency versions.
- `results`: independent final gates, sample accounting and simulated motor-use table.
- `manifest.json`: SHA256 checksums of the supplied data and frozen sources.

The actor maps 400 observations through ELU layers 512 / 256 / 128 to 23 actions. Observation construction, history and action scaling are in the environment source; the network weights alone are insufficient.

## Setup

Use a CUDA-capable computer with Isaac Sim 5.1.0, PyTorch 2.7.0+cu128 and rsl-rl-lib 5.0.1. Install Isaac Lab at commit `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` using its installation instructions. Install its Isaac Lab and Isaac Lab RL extensions plus NumPy, psutil and TensorBoard into the same Python environment. Exact recorded versions are in `config/versions.json`.

Run these commands from this folder with that environment's Python:

```sh
python resume.py --mode verify
python resume.py --mode eval --num-envs 320 --output runs/evaluation
python resume.py --mode train --num-envs 64 --iterations 64 --output runs/continuation
```

The wrapper preserves checkpoint bytes, relocates the teacher certificate into `runtime`, verifies its evidence and invokes the original worker. Training loads actor, critic, Adam and exploration and retains both teachers and upper-body refinement. Each invocation runs one block; it does not start the historical automatic supervisor. Use a new output directory per invocation. Evaluation above is nominal; the original worker also supports `--perturb` for pushed gates.

The original run used 8192 environments, 24 steps/environment, 64 updates/block and learning rate 0.0001. Start smaller and measure available VRAM before increasing capacity. The source retains memory guards and native motor limits.

## Verification

Checkpoint hashes, actor dimensions, CPU deserialization, frozen reward signature and relocated teacher qualification checks were verified when packaging. No new Isaac simulator session or PPO update was run for this export. A fresh-machine simulator/load test is still required. Original absolute paths in evidence are provenance; `resume.py` supplies local runtime paths without editing frozen source.

Training-only third-party code retains its copyright notices; the Isaac Lab license is included beside the source.
