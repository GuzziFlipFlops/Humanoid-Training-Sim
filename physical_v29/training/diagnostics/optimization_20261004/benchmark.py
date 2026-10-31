"""Bounded same-physics optimization experiment; never a qualification gate."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--output', required=True)
p.add_argument('--num_envs', type=int, default=4096)
p.add_argument('--windows', type=int, default=3)
p.add_argument('--ticks', type=int, default=64)
p.add_argument('--warmup', type=int, default=24)
p.add_argument('--replay', action='store_true')
p.add_argument('--actuator_parity', action='store_true')
p.add_argument('--ppo_updates', type=int, default=0)
p.add_argument('--reload', default='')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
here = Path(__file__).resolve().parent
experiment = json.loads((here / 'experiment.json').read_text())
source = Path(args.source).resolve()
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
assert not (out / 'result.json').exists(), 'Do not overwrite evidence'
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    if proc.info['pid'] == os.getpid():
        continue
    command = proc.info['cmdline'] or []
    if (proc.info['name'] or '').lower() in ('python.exe', 'pythonw.exe'):
        # A venv Python launcher has an actual interpreter child; exclude its parent.
        if proc.info['pid'] == psutil.Process().ppid():
            continue
        assert not any(Path(s).name in ('worker_unitree.py', 'profile_physics.py', 'benchmark.py') for s in command), 'Only one Isaac worker'
sys.path.insert(0, str(source))
sys.path.insert(0, str(source / 'training'))
app = AppLauncher(args)
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from environment_unitree import LocomotionCfg, LocomotionEnv


def memory():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,memory.used', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    value = dict(vram_free_mb=int(gpu[0]), vram_used_mb=int(gpu[1]),
                 host_available_mb=psutil.virtual_memory().available / 2**20,
                 private_mb=getattr(psutil.Process().memory_info(), 'private', 0) / 2**20,
                 torch_allocated_mb=torch.cuda.memory_allocated() / 2**20)
    assert value['vram_free_mb'] >= 300, value
    assert value['host_available_mb'] >= 2048, value
    return value


def main():
    start_worker = time.perf_counter()
    torch.set_num_threads(4)
    original_checkpoint = Path(experiment['checkpoint'])
    assert hashlib.sha256(original_checkpoint.read_bytes()).hexdigest() == experiment['checkpoint_sha256']
    checkpoint = Path(args.reload) if args.reload else original_checkpoint
    active = json.loads((here.parents[1] / 'ACTIVE_RUN.json').read_text())
    assert active['status'] == 'paused_for_performance_review'
    cfg = LocomotionCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot_usd = str(source / 'training/assets/usd/robot_training.usd')
    cfg.sim.device = 'cuda:0'
    cfg.stage_speed = .35
    cfg.evaluation = not bool(args.ppo_updates)
    cfg.episode_length_s = 20. if args.ppo_updates else 30.
    cfg.seed = 901
    for name, value in dict(gpu_found_lost_aggregate_pairs_capacity=2**22,
            gpu_total_aggregate_pairs_capacity=2**20, gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**17, gpu_heap_capacity=2**26,
            gpu_collision_stack_size=2**25).items():
        setattr(cfg.sim.physx, name, value)
    raw = LocomotionEnv(cfg)
    raw.initialize_randomization()
    speeds = torch.full((args.num_envs,), .35, device=raw.device)
    if args.replay:
        speeds = torch.tensor([0., .15, .35, .65, 1., 1.5], device=raw.device)[torch.arange(args.num_envs, device=raw.device) % 6]
    if not args.ppo_updates:
        raw.configure_fixed_commands(speeds, torch.zeros(args.num_envs, device=raw.device))
    env = RslRlVecEnvWrapper(raw, clip_actions=3.)
    env.reset()
    if args.actuator_parity:
        from actuator_parity import check
        print('ACTUATOR_PARITY ' + json.dumps(check(source,raw,out/'actuator_parity.json')),flush=True)
    agent_cfg = json.loads((original_checkpoint.parent / 'agent.json').read_text())
    runner = OnPolicyRunner(env, agent_cfg, log_dir=str(out / 'unused_logger'), device='cuda:0')
    runner.load(str(checkpoint), map_location='cuda:0')
    runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]['lr'])
    if args.ppo_updates:
        from ppo_update_audit import PpoUpdateAudit
        from standing_retention import StandingRetention, validate_teacher
        audit = PpoUpdateAudit(runner.alg)
        prior = json.loads((original_checkpoint.parent/'run_config.json').read_text())['arguments']
        teacher_file=Path(prior['standing_prior'])
        assert hashlib.sha256(teacher_file.read_bytes()).hexdigest()==prior['standing_prior_sha256']
        teacher_saved=torch.load(teacher_file,map_location='cuda:0',weights_only=False)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected_hashes={key:prior_info for key,prior_info in saved['infos'].items()
                         if key in ('asset_sha256','motion_sha256','actuator_sha256')}
        validate_teacher(teacher_saved['infos'],prior.get('contract',saved['infos']['contract']),expected_hashes)
        teacher=copy.deepcopy(runner.alg.actor)
        teacher.load_state_dict(teacher_saved['actor_state_dict'])
        retention=StandingRetention(runner.alg.actor,teacher,weight=.25,max_samples=256)
        rows=[]
        original_log=runner.logger.log
        def progress(**kwargs):
            with torch.no_grad():runner.alg.actor.distribution.log_std_param.clamp_(np.log(.04),np.log(.7))
            original_log(**kwargs)
            collected=gc.collect() if kwargs['it']%4==0 else 0
            row=dict(iteration=kwargs['it'],collect_seconds=kwargs['collect_time'],learn_seconds=kwargs['learn_time'],
                     memory=memory(),ppo=audit.latest,retention=retention.read_audit(),
                     behavior=raw.read_reward_audit(),garbage_collected=collected)
            rows.append(row)
            (out/'progress.json').write_text(json.dumps(row,indent=2))
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('OPTIMIZATION_PPO_UPDATE '+json.dumps(row),flush=True)
        runner.logger.log=progress
        before=memory()
        ppo_start=time.perf_counter()
        runner.learn(args.ppo_updates,init_at_random_ep_len=True)
        elapsed=time.perf_counter()-ppo_start
        infos=saved['infos'].copy()
        infos['optimization_experiment']=dict(source=str(source),original_checkpoint_sha256=experiment['checkpoint_sha256'],
                                               independent_qualification=False)
        runner.save(str(out/'candidate.pt'),infos=infos)
        reload_saved=torch.load(out/'candidate.pt',map_location='cpu',weights_only=False)
        before_actor={k:v.detach().cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
        runner.load(str(out/'candidate.pt'),map_location='cuda:0')
        for key,value in runner.alg.actor.state_dict().items():assert torch.equal(before_actor[key],value.detach().cpu()),key
        collection=sum(row['collect_seconds'] for row in rows)
        learning=sum(row['learn_seconds'] for row in rows)
        result=dict(completed=True,scope='Bounded actual PPO benchmark with full checkpoint/Adam and standing retention, not qualification',
                    source=str(source),num_envs=args.num_envs,updates=args.ppo_updates,rows=rows,
                    active_ppo_samples_per_second=args.num_envs*24*args.ppo_updates/(collection+learning),
                    effective_ppo_samples_per_second=args.num_envs*24*args.ppo_updates/elapsed,
                    collection_seconds=collection,learning_seconds=learning,ppo_seconds=elapsed,
                    worker_seconds=time.perf_counter()-start_worker,memory_before=before,memory_after=memory(),
                    full_checkpoint_reload=True,optimizer_state_saved=bool(reload_saved['optimizer_state_dict']['state']),
                    checkpoint_sha256=experiment['checkpoint_sha256'])
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print('OPTIMIZATION_COMPLETE '+json.dumps(result),flush=True)
        retention.close();audit.close();env.close()
        return
    policy = runner.get_inference_policy(device='cuda:0')
    obs = env.get_observations()
    trace = {}
    terminations = 0
    sampled_ids = torch.arange(args.num_envs if args.replay else min(args.num_envs,24), device=raw.device)

    def step(tick, sample=False):
        nonlocal obs, terminations
        actions = policy(obs)
        before_obs = obs['policy'][sampled_ids].clone() if sample else None
        obs, reward, dones, _ = env.step(actions)
        if sample:
            d, m = raw.robot.data, raw.telemetry()
            values = dict(tick=torch.tensor(tick), policy_observation=before_obs,
                          q=d.joint_pos[sampled_ids], qd=d.joint_vel[sampled_ids],
                          effort=raw.actuator.applied_effort[sampled_ids], computed_effort=raw.actuator.computed_effort[sampled_ids],
                          root=d.root_state_w[sampled_ids], actions=actions[sampled_ids],
                          targets=raw.targets[sampled_ids], command=raw.commands[sampled_ids],
                          action_command=raw.action_commands[sampled_ids], reward=reward[sampled_ids], done=dones[sampled_ids],
                          contacts=m['contact'][sampled_ids], foot_force=m['foot_force'][sampled_ids],
                          feet=m['foot_position'][sampled_ids], com=m['com'][sampled_ids], invalid=m['invalid'][sampled_ids])
            for name, tensor in values.items():
                trace.setdefault(name, []).append(tensor.detach().cpu().numpy().copy())
        terminations += int(dones.sum()) if args.replay else 0

    windows, mem = [], []
    print('OPTIMIZATION_ENVIRONMENT_READY', flush=True)
    with torch.inference_mode():
        warmup_start = time.perf_counter()
        for tick in range(args.warmup):
            step(tick)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_start
        mem.append(memory())
        for window in range(args.windows):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for tick in range(args.ticks):
                absolute = window * args.ticks + tick
                if args.replay and absolute == 100:
                    raw._reset_idx(torch.arange(0, args.num_envs, 7, device=raw.device))
                    obs = env.get_observations()
                if args.replay and absolute == 150:
                    ids = torch.arange(2, args.num_envs, 7, device=raw.device)
                    velocity = raw.robot.data.root_vel_w[ids].clone()
                    velocity[:, 1] += .15
                    raw.robot.write_root_velocity_to_sim(velocity, env_ids=ids)
                step(absolute, sample=args.replay or tick % 16 == 0)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            windows.append(dict(window=window, ticks=args.ticks, seconds=elapsed,
                                samples_per_second=args.num_envs * args.ticks / elapsed))
            mem.append(memory())
            print('OPTIMIZATION_WINDOW ' + json.dumps(windows[-1]), flush=True)
    np.savez_compressed(out / 'trace.npz', **{k: np.stack(v) for k, v in trace.items()})
    caches = {}
    for key, _, model in raw.actuator.pairs:
        caches[key] = dict(constants=len(model._tensors), graphs=len(getattr(model, '_state_graphs', {})))
    result = dict(completed=True, scope='Bounded unchanged-physics inference benchmark, not PPO throughput or locomotion qualification',
                  source=str(source), num_envs=args.num_envs, windows=windows,
                  total_ticks=args.windows * args.ticks,
                  samples_per_second=args.num_envs * args.windows * args.ticks / sum(w['seconds'] for w in windows),
                  warmup_seconds=warmup_seconds, worker_seconds=time.perf_counter()-start_worker,
                  memory=mem, caches=caches, replay=args.replay, replay_terminations=terminations,
                  checkpoint_sha256=experiment['checkpoint_sha256'], seed=901,
                  physics=dict(dt=.005, decimation=4, self_collision=True, position_iterations=8, velocity_iterations=2),
                  actor_normalization=agent_cfg['actor']['obs_normalization'], history_preserved=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print('OPTIMIZATION_COMPLETE ' + json.dumps(result), flush=True)
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
