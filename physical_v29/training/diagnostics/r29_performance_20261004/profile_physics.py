"""Bounded, unchanged-physics performance experiment; never a skill gate."""
import argparse
from collections import defaultdict
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import psutil
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=4096)
parser.add_argument('--ticks', type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
for process in psutil.process_iter(['name', 'cmdline']):
    if (process.info['name'] or '').lower() in ('python.exe', 'pythonw.exe'):
        assert not any(Path(a).name == 'worker_unitree.py' for a in (process.info['cmdline'] or [])), 'Drain training first'

run = Path(args.run).resolve()
source = run / 'source/physical_v29'
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
assert not (out / 'result.json').exists()
sys.path.insert(0, str(source))
sys.path.insert(0, str(source / 'training'))
app = AppLauncher(args)

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from environment_unitree import LocomotionCfg, LocomotionEnv
from cad_differential import CadDifferential


def main():
    torch.set_num_threads(4)
    state = json.loads((run / 'state.json').read_text())
    assert state['status'] == 'paused_for_performance_review'
    checkpoint = Path(state['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    actor_state = saved['actor_state_dict']
    indexes = sorted(int(k.split('.')[1]) for k in actor_state if k.startswith('mlp.') and k.endswith('weight'))
    layers = []
    for index in indexes:
        weight = actor_state[f'mlp.{index}.weight']
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if index != indexes[-1]:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict({k.removeprefix('mlp.'):v for k,v in actor_state.items() if k.startswith('mlp.')})
    actor = actor.to('cuda:0').eval()
    cfg = LocomotionCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot_usd = str(source / 'training/assets/usd/robot_training.usd')
    assert Path(cfg.robot_usd).exists()
    cfg.sim.device = 'cuda:0'
    cfg.stage_speed = .35
    cfg.evaluation = True
    cfg.seed = 901
    for name, value in dict(gpu_found_lost_aggregate_pairs_capacity=2**22,
            gpu_total_aggregate_pairs_capacity=2**20, gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**17, gpu_heap_capacity=2**26,
            gpu_collision_stack_size=2**25).items():
        setattr(cfg.sim.physx, name, value)
    raw = LocomotionEnv(cfg)
    raw.initialize_randomization()
    raw.configure_fixed_commands(torch.full((args.num_envs,), .35, device=raw.device),
                                 torch.zeros(args.num_envs, device=raw.device))
    env = RslRlVecEnvWrapper(raw, clip_actions=3.)
    env.reset()
    print('PROFILE_ENVIRONMENT_READY', flush=True)

    profiling = False
    samples = defaultdict(list)
    def wrap(original, name):
        @functools.wraps(original)
        def timed(*pos, **kw):
            if not profiling:
                return original(*pos, **kw)
            start = time.perf_counter()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            value = original(*pos, **kw)
            b.record()
            samples[name].append((time.perf_counter()-start, a, b))
            return value
        return timed
    CadDifferential.state = wrap(CadDifferential.state, 'cad_state')
    raw.actuator.compute = wrap(raw.actuator.compute, 'actuator_compute')
    raw._pre_physics_step = wrap(raw._pre_physics_step, 'pre_physics')
    raw._get_dones = wrap(raw._get_dones, 'dones_and_telemetry')
    raw._get_rewards = wrap(raw._get_rewards, 'rewards_and_audits')
    raw._get_observations = wrap(raw._get_observations, 'observations')
    raw.sim.step = wrap(raw.sim.step, 'physx_step')
    raw.scene.write_data_to_sim = wrap(raw.scene.write_data_to_sim, 'scene_write_with_actuation')
    raw.scene.update = wrap(raw.scene.update, 'scene_update')
    actor.forward = wrap(actor.forward, 'actor_inference')
    with torch.inference_mode():
        for _ in range(8):
            obs = env.get_observations()
            env.step(actor(obs['policy']))
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.ticks):
            obs = env.get_observations()
            env.step(actor(obs['policy']))
        torch.cuda.synchronize()
        normal_seconds = time.perf_counter()-start
        profiling = True
        start = time.perf_counter()
        for _ in range(8):
            obs = env.get_observations()
            env.step(actor(obs['policy']))
        torch.cuda.synchronize()
        profiled_seconds = time.perf_counter()-start
        profiling = False
    summary = {name:dict(calls=len(rows), calls_per_control_tick=len(rows)/8,
                        cpu_dispatch_ms_per_control_tick=sum(row[0] for row in rows)*1000/8,
                        stream_elapsed_ms_per_control_tick=sum(row[1].elapsed_time(row[2]) for row in rows)/8)
               for name, rows in samples.items()}
    result = dict(completed=True, scope='Short same-physics actor-inference microbenchmark. No PPO, qualification or production source changes.',
                  environments=args.num_envs, unprofiled_ticks=args.ticks, unprofiled_seconds=normal_seconds,
                  environment_plus_inference_steps_per_second=args.num_envs*args.ticks/normal_seconds,
                  profiled_ticks=8, profiled_seconds=profiled_seconds, sections=summary,
                  section_scope='Parent and child timings overlap; do not sum all rows. CUDA stream elapsed includes launch gaps and is not isolated kernel busy time.',
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  preserved_physics=dict(dt=cfg.sim.dt, decimation=cfg.decimation, position_iterations=8, velocity_iterations=2, self_collision=True),
                  private_mb=getattr(psutil.Process().memory_info(), 'private', 0)/2**20,
                  host_available_mb=psutil.virtual_memory().available/2**20,
                  torch_allocated_mb=torch.cuda.memory_allocated()/2**20,
                  cache_sizes={key:len(model._tensors) for key,_,model in raw.actuator.pairs})
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException as error:
        code = 1
        traceback.print_exc()
        (out / 'failure.json').write_text(json.dumps(dict(error=repr(error))))
    finally:
        app.app.close()
    raise SystemExit(code)
