"""Read-only multi-policy gait diagnostics; never updates weights or training config.

Records every physics step before automatic resets. Torque fields are the
ImplicitActuator PD estimates, NOT measured drive torques. See manifest.json.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
parser.add_argument('--duration', type=float, default=24.0)
parser.add_argument('--replicas', type=int, default=8)
parser.add_argument('--randomized', action='store_true')
parser.add_argument('--clamp-waist-target', action='store_true', help='Diagnostic-only hard-limit clamp for waist pitch target.')
parser.add_argument('--seed', type=int, default=4217)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply_inverse, yaw_quat
from humanoid_tasks.curriculum.env_cfg import HumanoidNaturalRunEnvCfg, HumanoidNaturalWalkEnvCfg
from humanoid_tasks.curriculum.agents.rsl_rl_cfg import HumanoidRunFastPPORunnerCfg
import humanoid_tasks

CHECKPOINTS = {
    'walk': ROOT / 'logs/rsl_rl/humanoid5ft_natural_walk/2026-08-31_17-06-10/model_2098.pt',
    'run': ROOT / 'logs/rsl_rl/humanoid5ft_natural_run_from_working_walk/2026-09-01_19-54-34/model_3800.pt',
}
CASES = [('walk_1p0', 'walk', 1.0), ('run_0p0', 'run', 0.0), ('run_1p0', 'run', 1.0),
         ('run_1p8', 'run', 1.8), ('run_2p3', 'run', 2.3), ('run_2p6', 'run', 2.6),
         ('run_3p0', 'run', 3.0), ('run_4p0', 'run', 4.0)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Choose a fresh diagnostic output directory.')
    cfg = HumanoidNaturalRunEnvCfg()
    cfg.scene.num_envs = n = len(CASES) * args.replicas
    cfg.seed = args.seed
    cfg.curriculum = None
    cfg.observations.policy.enable_corruption = False
    cfg.events.push = None
    cfg.episode_length_s = args.duration + 5.0
    if not args.randomized:
        cfg.events.torso_mass = None
        cfg.events.torso_com = None
        cfg.events.actuator_gains = None
        cfg.events.physics_material.params.update(static_friction_range=(0.95, 0.95),
            dynamic_friction_range=(0.8, 0.8), restitution_range=(0., 0.), num_buckets=1)
    cfg.events.reset_base.params['pose_range'] = {k: (0., 0.) for k in ('x','y','z','roll','pitch','yaw')}
    cfg.events.reset_base.params['velocity_range'] = {k: (0., 0.) for k in ('x','y','z','roll','pitch','yaw')}
    cfg.events.reset_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    cmdcfg = cfg.commands.base_velocity
    cmdcfg.ranges.lin_vel_x = (1., 1.)
    cmdcfg.ranges.lin_vel_y = cmdcfg.ranges.ang_vel_z = (0., 0.)
    cmdcfg.rel_standing_envs = 0.
    cmdcfg.resampling_time_range = (1000., 1000.)
    cmdcfg.debug_vis = False
    agent = handle_deprecated_rsl_rl_cfg(HumanoidRunFastPPORunnerCfg(), metadata.version('rsl-rl-lib'))
    env = RslRlVecEnvWrapper(gym.make('Isaac-Humanoid5ft-NaturalRun-v0', cfg=cfg), clip_actions=agent.clip_actions)
    raw = env.unwrapped
    robot = raw.scene['robot']
    sensor = raw.scene.sensors['contact_forces']
    device = raw.device
    names = robot.joint_names
    policies, runners = {}, {}
    for key, ckpt in CHECKPOINTS.items():
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=device)
        runner.load(handle_deprecated_rsl_rl_checkpoint(str(ckpt), metadata.version('rsl-rl-lib')))
        policies[key] = runner.get_inference_policy(device=device)
        runners[key] = runner

    scales = {'walk': float(HumanoidNaturalWalkEnvCfg().actions.joint_pos.scale),
              'run': float(cfg.actions.joint_pos.scale)}
    case_manifest = []
    speed = torch.zeros(n, device=device)
    scale = torch.zeros((n, len(names)), device=device)
    for i, (label, policy, value) in enumerate(CASES):
        lo, hi = i * args.replicas, (i+1) * args.replicas
        speed[lo:hi] = value
        scale[lo:hi] = scales[policy]
        case_manifest.append(dict(name=label, policy=policy, target_mps=value,
                                  env_ids=list(range(lo, hi)), action_scale=scales[policy]))
    # Per-environment affine scaling preserves each checkpoint's native action interpretation.
    action_term = raw.action_manager.get_term('joint_pos')
    action_term._scale = scale
    if args.clamp_waist_target:
        waist_id = names.index('waist_pitch')
        original_process = action_term.process_actions
        def process_with_waist_clamp(actions):
            original_process(actions)
            bounds = robot.data.joint_pos_limits[:,waist_id]
            action_term._processed_actions[:,waist_id].clamp_(min=bounds[:,0], max=bounds[:,1])
        action_term.process_actions = process_with_waist_clamp
    command = raw.command_manager.get_term('base_velocity')
    def set_commands():
        command.vel_command_b[:] = 0.
        command.vel_command_b[:, 0] = speed
        command.is_standing_env[:] = speed == 0.

    # Identical initial conditions across cases; paired left/right mirrored small perturbations.
    # Reflection of an axial rotation about X/Z reverses its scalar angle; Y preserves it.
    swap = [names.index(name.replace('left_', 'right_')) if name.startswith('left_') else
            names.index(name.replace('right_', 'left_')) if name.startswith('right_') else i
            for i, name in enumerate(names)]
    sign = torch.tensor([-1. if ('roll' in name or 'yaw' in name) else 1. for name in names], device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    perturb = (torch.rand((args.replicas, len(names)), generator=gen, device=device) - .5) * .025
    rolls = (torch.rand(args.replicas, generator=gen, device=device)-.5) * .03
    for i in range(0, args.replicas-1, 2):
        perturb[i+1] = perturb[i, swap] * sign
        rolls[i+1] = -rolls[i]
    if args.replicas >= 2:
        perturb[:2] = 0.
        rolls[:2] = 0.
    qinit = robot.data.default_joint_pos.clone() + perturb.repeat(len(CASES), 1)
    rstate = robot.data.default_root_state.clone()
    rstate[:, :3] += raw.scene.env_origins
    roll = rolls.repeat(len(CASES))
    rstate[:, 3:7] = quat_from_euler_xyz(roll, torch.zeros_like(roll), torch.zeros_like(roll))
    robot.write_root_state_to_sim(rstate)
    robot.write_joint_state_to_sim(qinit, torch.zeros_like(qinit))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    set_commands()
    obs = raw.observation_manager.compute(update_history=False)
    foot_ids = [robot.find_bodies(s)[0][0] for s in ('left_foot','right_foot')]
    foot_sensor = [sensor.find_bodies(s)[0][0] for s in ('left_foot','right_foot')]
    torso_id = robot.find_bodies('torso')[0][0]
    nonfeet = [i for i,s in enumerate(sensor.body_names) if s not in ('left_foot', 'right_foot')]
    masses = robot.root_physx_view.get_masses().to(device)
    body_com = robot.root_physx_view.get_coms().cpu().numpy()
    reward_resolution = {}
    for name in raw.reward_manager.active_terms:
        term = raw.reward_manager.get_term_cfg(name)
        for key, val in term.params.items():
            if hasattr(val, 'joint_ids') and val.joint_names is not None:
                ids = list(range(len(names)))[val.joint_ids] if isinstance(val.joint_ids,slice) else val.joint_ids
                reward_resolution[f'{name}.{key}'] = [names[x] for x in ids]
    arrays = dict(masses=masses.cpu().numpy(), body_com_local=body_com,
                  inertias=robot.root_physx_view.get_inertias().cpu().numpy(),
                  stiffness=robot.data.joint_stiffness.cpu().numpy(),
                  damping=robot.data.joint_damping.cpu().numpy(),
                  armature=robot.data.joint_armature.cpu().numpy(),
                  effort_limits=robot.data.joint_effort_limits.cpu().numpy(),
                  velocity_limits=robot.data.soft_joint_vel_limits.cpu().numpy(),
                  position_limits=robot.data.joint_pos_limits.cpu().numpy(),
                  initial_q=qinit.cpu().numpy(), initial_root=rstate.cpu().numpy())
    np.savez_compressed(out / 'runtime_parameters.npz', **arrays)
    specs = [('q',22),('qd',22),('target',22),('torque_est',22),('torque_requested_est',22),
             ('foot_pos_h',6),('foot_vel_w',6),('foot_force_w',6),('foot_quat_w',8),
             ('root_pos_w',3),('root_vel_b',3),('root_ang_b',3),('root_quat_w',4),
             ('torso_quat_w',4),('com_h',3),('nonfoot_force_max',1)]
    offsets, width = {}, 0
    for name, size in specs:
        offsets[name] = [width, width+size]
        width += size
    assert len(names) == 22
    physics_dt, control_dt = float(raw.physics_dt), float(raw.step_dt)
    frames = round(args.duration/physics_dt)
    record = np.lib.format.open_memmap(out / 'physics.npy', mode='w+', dtype='float32', shape=(frames,n,width))
    first_fall = np.full(n, np.nan)
    dones_log = np.zeros((round(args.duration/control_dt), n), dtype=bool)
    cursor = 0
    original_update = raw.scene.update
    active = False
    def record_update(dt):
        nonlocal cursor
        original_update(dt)
        if not active:
            return
        data = robot.data
        heading = yaw_quat(data.root_quat_w)
        foot_rel = data.body_pos_w[:, foot_ids] - data.root_pos_w[:, None]
        foot_h = quat_apply_inverse(heading[:,None,:].expand(-1,2,-1).reshape(-1,4), foot_rel.reshape(-1,3)).reshape(n,6)
        com_w = (data.body_com_pos_w * masses[:,:,None]).sum(1) / masses.sum(1,keepdim=True)
        com_h = quat_apply_inverse(heading, com_w-data.root_pos_w)
        forces = sensor.data.net_forces_w
        vals = [data.joint_pos, data.joint_vel, data.joint_pos_target,
                data.applied_torque, data.computed_torque, foot_h,
                data.body_lin_vel_w[:,foot_ids].reshape(n,6), forces[:,foot_sensor].reshape(n,6),
                data.body_quat_w[:,foot_ids].reshape(n,8), data.root_pos_w,
                data.root_lin_vel_b, data.root_ang_vel_b, data.root_quat_w,
                data.body_quat_w[:,torso_id], com_h,
                torch.linalg.vector_norm(forces[:,nonfeet],dim=-1).max(1,keepdim=True).values]
        record[cursor] = torch.cat(vals,dim=1).detach().cpu().numpy()
        cursor += 1
    raw.scene.update = record_update
    manifest = dict(cases=case_manifest, num_envs=n, duration_s=args.duration, physics_dt=physics_dt,
        control_dt=control_dt, fields=offsets, joint_names=names, body_names=robot.body_names,
        seed=args.seed, startup_randomization=args.randomized, pushes=False, observation_noise=False,
        clamp_waist_target=args.clamp_waist_target,
        checkpoint_paths={k:str(v) for k,v in CHECKPOINTS.items()},
        checkpoint_sha256={k:sha(v) for k,v in CHECKPOINTS.items()},
        source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'humanoid_cfg.py',
            ROOT/'source/humanoid_tasks/curriculum/env_cfg.py',ROOT/'assets/usd/humanoid_5ft.usd']},
        reward_joint_resolution=reward_resolution, versions={k:metadata.version(k) for k in ('torch','rsl-rl-lib')},
        notes=['No training or reward changes. No cameras. All physics samples captured before automatic reset.',
               'Torque estimates are implicit PD approximations from immediately before each physics substep, not measured motor torque.',
               'Joint/pose/contact samples are after that same physics substep. Targets held at 50 Hz.',
               'Nominal mode has constant symmetric material/gains/mass/COM; initial small mirrored perturbations shared across cases.',
               'Randomized mode keeps independent startup material/gain/torso mass/COM variation; not matched material between cases.',
               'First two nominal replicas are identical initial states, not independent random trials.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    started = time.monotonic()
    with torch.inference_mode():
        for k in range(len(dones_log)):
            if k == 0: print('[DIAGNOSTIC] first observation',flush=True)
            set_commands()
            # The observation command remains consistent even after a reset resamples it.
            obs = raw.observation_manager.compute(update_history=False)
            actions = torch.zeros((n,22),device=device)
            if k == 0: print('[DIAGNOSTIC] first policy call',flush=True)
            actions[:args.replicas] = policies['walk']({key: value[:args.replicas] for key,value in obs.items()})
            actions[args.replicas:] = policies['run']({key: value[args.replicas:] for key,value in obs.items()})
            active = True
            if k == 0: print('[DIAGNOSTIC] first physics step',flush=True)
            obs, _, dones, extras = env.step(actions)
            active = False
            done = dones.detach().cpu().numpy().astype(bool)
            timeouts = extras.get('time_outs',torch.zeros_like(dones)).detach().cpu().numpy().astype(bool)
            fell = done & ~timeouts
            first_fall[np.isnan(first_fall) & fell] = (k+1)*control_dt
            dones_log[k] = done
            for key, sl in [('walk',slice(0,args.replicas)),('run',slice(args.replicas,None))]:
                if hasattr(policies[key],'reset'):
                    policies[key].reset(dones[sl])
            if (k+1) % 100 == 0:
                record.flush()
                print(f'[DIAGNOSTIC] sim={(k+1)*control_dt:.1f}/{args.duration}s wall={time.monotonic()-started:.1f}s first_falls={np.isfinite(first_fall).sum()}/{n}',flush=True)
    assert cursor == frames, (cursor,frames)
    record.flush()
    np.savez(out/'episodes.npz', first_fall_s=first_fall, dones=dones_log)
    manifest['completed'] = True
    manifest['elapsed_wall_s'] = time.monotonic()-started
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    raw.scene.update = original_update
    env.close()
    print(f'[DIAGNOSTIC] COMPLETE {out}',flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.app.close()
