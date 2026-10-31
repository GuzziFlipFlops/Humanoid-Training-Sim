"""Bounded diagnostic arm/base coupling and exact teacher target reconstruction.

Truthful student feedback, free base, original gains and CAD limits. External
arm offsets / body action replay are diagnostic controllers, never production.
"""
import argparse, copy, hashlib, json, sys, traceback
from pathlib import Path
OUT = Path(__file__).resolve().parent
pause = json.loads((OUT / 'diagnostic_pause.json').read_text())
TRAIN = OUT / 'control_source/physical_v29/training'
sys.path[:0] = [str(TRAIN), str(TRAIN.parent)]
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
import numpy as np, torch, psutil
torch.set_num_threads(4)
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from curriculum_provenance import training_reward_signature
from standing_retention import validate_qualification
from evaluation import GATE_VERSION
from environment_unitree import LocomotionCfg, LocomotionEnv
from heading_control import yaw_from_quat
from teacher_head_context import context_inputs
from action_projection import project_targets, ramp


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert pause['worker_drained']
    for p in psutil.process_iter(['cmdline']):
        assert not any(Path(x).name == 'worker_unitree.py' for x in p.info['cmdline'] or []), 'Live Isaac worker exists'
    out = OUT / 'arm_target_response'
    assert out.is_dir() and not (out / 'result.json').exists()
    cp = Path(pause['checkpoint'])
    assert sha(cp) == pause['checkpoint_sha256']
    configuration = json.loads((cp.parent / 'run_config.json').read_text())['arguments']
    cert = json.loads(Path(configuration['locomotion_prior_certificate']).read_text())
    teacher_cp = Path(cert['checkpoint'])
    signature = training_reward_signature(TRAIN)
    validate_qualification(cert, str(teacher_cp), signature, GATE_VERSION)
    converter = UrdfConverter(UrdfConverterCfg(asset_path=str(TRAIN / 'assets/robot_training.urdf'),
        usd_dir=str(TRAIN / 'assets/usd'), usd_file_name='robot_training.usd', fix_base=False,
        root_link_name='pelvis', merge_fixed_joints=True, make_instanceable=True,
        collision_from_visuals=False, self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type='force', target_type='none',
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.))))
    per = 32
    groups = []
    for seed in [4321, 4331]:
        for speed in [.15, 1.5, 3.5]:
            for mode, sign in [('control', 0), ('body_replay_positive', 1),
                    ('body_replay_negative', -1), ('closed_loop_positive', 1), ('closed_loop_negative', -1)]:
                groups.append(dict(name=f'{seed}_{speed}_{mode}', seed=seed, command=speed,
                    initial_command=speed, mode=mode, sign=sign, transition=False))
        for speed in [.15, 1.5]:
            groups.append(dict(name=f'{seed}_3.5_to_{speed}', seed=seed, command=speed,
                initial_command=3.5, mode='transition', sign=0, transition=True))
    n = len(groups) * per
    cfg = LocomotionCfg(); cfg.scene.num_envs = n; cfg.scene.env_spacing = 3.
    cfg.evaluation = True; cfg.stage_speed = 3.5; cfg.episode_length_s = 30.
    cfg.robot_usd = converter.usd_path; cfg.sim.device = args.device or 'cuda:0'; cfg.seed = 71
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**22
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**20
    from offline_ground import install_local_ground
    install_local_ground()
    raw = LocomotionEnv(cfg); raw.initialize_randomization()
    env = RslRlVecEnvWrapper(raw, clip_actions=3.)
    config = json.loads((cp.parent / 'agent.json').read_text())
    runner = OnPolicyRunner(env, copy.deepcopy(config), log_dir=None, device=raw.device)
    saved = torch.load(cp, map_location=raw.device, weights_only=False)
    assert saved['infos']['reward_signature'] == signature
    runner.load(str(cp), load_cfg={'actor': True, 'critic': False, 'optimizer': False, 'iteration': False})
    assert all(torch.equal(v, runner.alg.actor.state_dict()[k]) for k, v in saved['actor_state_dict'].items())
    policy = copy.deepcopy(runner.get_inference_policy(raw.device))
    teacher_runner = OnPolicyRunner(env, copy.deepcopy(config), log_dir=None, device=raw.device)
    teacher_saved = torch.load(teacher_cp, map_location=raw.device, weights_only=False)
    teacher_runner.load(str(teacher_cp), load_cfg={'actor': True, 'critic': False, 'optimizer': False, 'iteration': False})
    assert all(torch.equal(v, teacher_runner.alg.actor.state_dict()[k]) for k, v in teacher_saved['actor_state_dict'].items())
    teacher = copy.deepcopy(teacher_runner.get_inference_policy(raw.device))
    context = json.loads((TRAIN / 'teacher_head_context.json').read_text())
    tables = {float(k): torch.tensor(v, device=raw.device) for k, v in context['contexts'].items() if float(k) <= 1.}
    speeds = torch.tensor([g['initial_command'] for g in groups], device=raw.device).repeat_interleave(per)
    raw.configure_fixed_commands(speeds, yaw_from_quat(raw.robot.data.root_quat_w).clone()); raw.reset()
    q0, qd0 = [], []
    for g in groups:
        for robot in range(per):
            rng = np.random.default_rng(np.random.SeedSequence([g['seed'], int(g['initial_command'] * 100), robot]))
            q0.append(rng.uniform(-.008, .008, 23)); qd0.append(rng.uniform(-.01, .01, 23))
    q = raw.stand.expand(n, -1).clone() + torch.tensor(np.asarray(q0), device=raw.device, dtype=raw.stand.dtype)
    qd = torch.tensor(np.asarray(qd0), device=raw.device, dtype=q.dtype)
    assert ((q >= raw.target_lo) & (q <= raw.target_hi)).all()
    raw.robot.write_joint_state_to_sim(q, qd); raw.targets[:] = q; raw.previous_targets[:] = q; raw.previous_vel[:] = qd
    raw.scene.update(raw.step_dt)
    raw.history.zero_(); raw.critic_history.zero_(); raw._command_tick = -1; raw._last_history_tick = -1
    observation = raw._get_observations(); reference_heading = raw.heading_target.clone()
    initial_q = q.cpu().numpy().copy(); initial_qd = qd.cpu().numpy().copy()
    for i in range(len(groups)):
        ids = slice(i * per, (i + 1) * per)
        assert len({r.tobytes() for r in np.concatenate([initial_q[ids], initial_qd[ids]], 1)}) == per
    names = list(raw.robot.joint_names)
    shoulder = [names.index(x) for x in ['left_shoulder_pitch', 'right_shoulder_pitch']]
    nonshoulder = [i for i in range(23) if i not in shoulder]
    upper = [names.index(x) for x in ['head_yaw', 'head_pitch', 'left_shoulder_pitch', 'right_shoulder_pitch',
        'left_shoulder_roll', 'right_shoulder_roll', 'left_elbow_pitch', 'right_elbow_pitch']]
    lower = [i for i in range(23) if i not in upper]
    group_index = {(g['seed'], g['initial_command'], g['mode']): i for i, g in enumerate(groups)}
    baseline_ids = torch.tensor([r for i, g in enumerate(groups) if g['mode'] == 'control'
        for r in range(i * per, (i + 1) * per)], device=raw.device)
    transitions = torch.tensor([g['transition'] for g in groups], device=raw.device).repeat_interleave(per)
    final_speeds = torch.tensor([g['command'] for g in groups], device=raw.device).repeat_interleave(per)
    alive = torch.ones(n, dtype=torch.bool, device=raw.device); failed = torch.zeros(n, device=raw.device, dtype=torch.long)
    original_update_command = raw._update_observation_command
    switched = False
    def request_transition_before_observation():
        nonlocal switched
        if raw.common_step_counter >= 300 and not switched:
            raw.commands[transitions, 0] = final_speeds[transitions]
            raw.fixed_evaluation_speed[transitions] = final_speeds[transitions]
            switched = True
        original_update_command()
    raw._update_observation_command = request_transition_before_observation
    motor_names = []; pair_indices = set()
    for key, indices, model in raw.actuator.pairs:
        pair_indices.update(indices); motor_names += [key + '_motor_1', key + '_motor_2']
    ordinary = [i for i in range(23) if i not in pair_indices]
    motor_names += [raw.actuator.joint_names[i] for i in ordinary]
    native = {k: [] for k in ['cap_ratio', 'driving_clip', 'requested', 'applied', 'velocity', 'alive']}
    original_compute = raw.actuator.compute
    def passive_compute(control, joint_pos, joint_vel):
        result = original_compute(control, joint_pos, joint_vel)
        ratios, clips, requests, applications, velocities = [], [], [], [], []
        for key, indices, model in raw.actuator.pairs:
            state = raw.actuator.last_motor_states[key]
            cap = raw.actuator.records[raw.actuator.joint_names[indices[0]]]['motor_speed_cap_RPM'] * 2 * torch.pi / 60
            v, req, applied = state['motor_velocity'], state['motor_requested'], state['motor_applied']
            ratios.append(v.abs() / cap); requests.append(req); applications.append(applied); velocities.append(v)
            clips.append((v.abs() >= cap) & (req * v > 0) & (applied.abs() < 1e-8) & state['valid'][:, None])
        v = joint_vel[:, ordinary]; req = raw.actuator.computed_effort[:, ordinary]; applied = raw.actuator.applied_effort[:, ordinary]
        ratios.append(v.abs() / raw.actuator._speed[ordinary]); requests.append(req); applications.append(applied); velocities.append(v)
        clips.append((v.abs() >= raw.actuator._speed[ordinary]) & (req * v > 0) & (applied * v <= 0))
        for key, value in [('cap_ratio', torch.cat(ratios, 1)), ('driving_clip', torch.cat(clips, 1)),
                ('requested', torch.cat(requests, 1)), ('applied', torch.cat(applications, 1)), ('velocity', torch.cat(velocities, 1))]:
            native[key].append(value[baseline_ids].detach().cpu().numpy().copy())
        native['alive'].append(alive[baseline_ids].cpu().numpy().copy())
        return result
    raw.actuator.compute = passive_compute
    history = {}; labels = {}; terms = {}; parity = 0.; eligibility_count = 0
    for tick in range(1000):
        t = tick * raw.step_dt
        obs_command = observation['policy'].reshape(n, 5, 80)[:, -1, 6:9].clone()
        with torch.inference_mode(): student_action = policy(observation)
        action = student_action.clone()
        for i, g in enumerate(groups):
            ids = slice(i * per, (i + 1) * per)
            if g['mode'].startswith('body_replay'):
                control_i = group_index[(g['seed'], g['initial_command'], 'control')]
                source = slice(control_i * per, (control_i + 1) * per)
                action[ids, nonshoulder] = student_action[source, nonshoulder]
            if g['sign']:
                action[ids, shoulder] += action.new_tensor([1., -1.]) * g['sign'] * np.deg2rad(5.) / raw.scales[shoulder] * ramp(t)
        previous = raw.targets.clone()
        projected, clipped, slew, invalid = project_targets(action, raw.stand.expand(n, -1), raw.scales,
            raw.target_lo, raw.target_hi, previous, raw.slew, raw.step_dt, raw.actuator.pairs)
        phase = raw.phase.clone()
        if tick % 5 == 0:
            with torch.inference_mode():
                teacher_obs = observation.clone(); teacher_raw = teacher_obs['policy']; frames = teacher_raw.reshape(n, 5, 80)
                eligible_all = torch.zeros(n, dtype=torch.bool, device=raw.device)
                for command, table in tables.items():
                    eligible = ((frames[:, :, 6] - command).abs() < 1e-6).all(1)
                    eligible &= (frames[:, :, 78:80].square().sum(-1) > .5).all(1)
                    eligible &= (frames[:, -1, context['columns'][:2]].abs() < .174533).all(1)
                    ids = eligible.nonzero().flatten()
                    if len(ids): teacher_raw[ids] = context_inputs(teacher_raw[ids], table, context['columns'])
                    eligible_all |= eligible
                teacher_action = teacher(teacher_obs)
                truthful_teacher_action = teacher(observation)
                teacher_target, teacher_clip, teacher_slew, teacher_invalid = project_targets(teacher_action,
                    raw.stand.expand(n, -1), raw.scales, raw.target_lo, raw.target_hi, previous, raw.slew, raw.step_dt, raw.actuator.pairs)
                eligibility_count += int(eligible_all.sum())
                label_row = dict(time=t, phase=phase, command=obs_command, previous_target=previous,
                    student_raw=student_action, teacher_raw=teacher_action, truthful_teacher_raw=truthful_teacher_action,
                    student_projected=project_targets(student_action, raw.stand.expand(n, -1), raw.scales,
                        raw.target_lo, raw.target_hi, previous, raw.slew, raw.step_dt, raw.actuator.pairs)[0],
                    teacher_projected=teacher_target, teacher_joint_clipped=teacher_clip, teacher_slew_limited=teacher_slew,
                    teacher_CAD_invalid=teacher_invalid, context_eligible=eligible_all, alive=alive,
                    teacher_within_certified_command_range=(obs_command[:, :2].norm(dim=1) > .1)
                        & (obs_command[:, :2].norm(dim=1) <= cert['maximum_command_m_s'] + 1e-6),
                    pre_q=raw.robot.data.joint_pos, pre_qd=raw.robot.data.joint_vel,
                    observed_history=observation['policy'])
                for key, value in label_row.items(): labels.setdefault(key, []).append(value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value)
        observation, reward, done, _ = env.step(action)
        done = done.bool(); alive &= ~done; failed += done.long()
        if (~done).any(): parity = max(parity, float((projected[~done] - raw.targets[~done]).abs().max()))
        m = raw.telemetry()
        row = dict(time=t + raw.step_dt, phase=phase, reward_phase=raw.reward_proxy.cad_metrics['phase'],
            root=raw.robot.data.root_state_w, q=raw.robot.data.joint_pos, qd=raw.robot.data.joint_vel,
            tau=raw.robot.data.applied_torque, target=raw.targets, student_raw_action=student_action,
            applied_raw_action=action, effective_action=raw.actions, force=m['foot_force'], clearance=m['clearance'],
            com=m['com'], foot_position=m['foot_position'], foot_velocity=m['foot_velocity'],
            contact=m['contact'], alive=alive, failures=done, reward_rate=reward / raw.step_dt,
            observed_command=obs_command, action_command=raw.action_commands,
            target_clipped=raw.target_clipped, target_slew=raw.target_slew_limited)
        for key, value in row.items(): history.setdefault(key, []).append(value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value)
        for key, value in raw.last_reward_terms.items(): terms.setdefault(key, []).append(value.detach().cpu().numpy().copy())
        if tick % 100 == 0:
            print(json.dumps(dict(tick=tick, alive=int(alive.sum()), projected_map_error_rad=parity)), flush=True)
    data = {k: np.asarray(v) for k, v in history.items()}
    label_data = {k: np.asarray(v) for k, v in labels.items()}
    native_data = {k: np.asarray(v) for k, v in native.items()}
    assert len(native_data['cap_ratio']) == 4000
    command_error = float(np.abs(data['observed_command'] - data['action_command']).max())
    term_sum_error = float(np.abs(sum(np.asarray(v) for v in terms.values()) - data['reward_rate']).max())
    assert command_error == 0. and term_sum_error < 1e-3 and parity < 1e-6
    summaries = []
    for i, g in enumerate(groups):
        ids = slice(i * per, (i + 1) * per)
        window = data['time'] >= (8. if g['transition'] else 6.)
        good = data['alive'][window, ids]; root = data['root'][window, ids]
        actual = (data['clearance'][window, ids] > .001).all(-1) & (data['force'][window, ids] < 1.).all(-1) & good
        h = reference_heading[ids].cpu().numpy()[None, :]
        signed = root[:, :, 7] * np.cos(h) + root[:, :, 8] * np.sin(h)
        lateral = -root[:, :, 7] * np.sin(h) + root[:, :, 8] * np.cos(h)
        head = data['q'][window, ids][:, :, upper[:2]]
        summaries.append(dict(**g, robots=per, survival=float(alive[ids].float().mean()), resets=int(failed[ids].sum()),
            signed_forward_m_s=float(signed[good].mean()) if good.any() else None,
            absolute_lateral_m_s=float(np.abs(lateral)[good].mean()) if good.any() else None,
            genuine_flight_fraction=float(actual.mean()), mean_reward_rate=float(data['reward_rate'][window, ids][good].mean()) if good.any() else None,
            head_RMS_deg=float(np.rad2deg(np.sqrt(np.square(head).sum(-1)[good].mean()))) if good.any() else None,
            actual_shoulder_mean_deg=np.rad2deg(data['q'][window, ids][:, :, shoulder][good].mean(0)).tolist() if good.any() else None,
            physical_target_shoulder_mean_deg=np.rad2deg(data['target'][window, ids][:, :, shoulder][good].mean(0)).tolist() if good.any() else None,
            reward_terms={k: float(np.asarray(v)[window, ids][good].mean()) if good.any() else None for k, v in terms.items()}))
    np.savez_compressed(out / 'telemetry_50hz.npz', **data, **{'term_' + k: np.asarray(v) for k, v in terms.items()},
        groups=np.array([g['name'] for g in groups]), joint_names=np.array(names), robots_per_group=per,
        reference_heading=reference_heading.cpu().numpy(), initial_q=initial_q, initial_qd=initial_qd)
    np.savez_compressed(out / 'teacher_targets_10hz.npz', **label_data, joint_names=np.array(names),
        groups=np.array([g['name'] for g in groups]), robots_per_group=per, lower_indices=np.array(lower))
    np.savez_compressed(out / 'baseline_motors_200hz.npz', **native_data, motor_names=np.array(motor_names),
        baseline_robot_ids=baseline_ids.cpu().numpy(), dt=.005)
    result = dict(completed=True, checkpoint=str(cp), checkpoint_sha256=sha(cp), actor_loaded_exact=True,
        qualified_teacher_checkpoint=str(teacher_cp), teacher_checkpoint_sha256=sha(teacher_cp),
        teacher_maximum_certified_command_m_s=cert['maximum_command_m_s'],
        certified_teacher_used_only_for_passive_labels=True, reward_signature=signature, groups=summaries,
        independent_initial_states_per_cohort=per, seeds=[4321, 4331], maximum_action_command_difference=command_error,
        maximum_reward_term_sum_error=term_sum_error, maximum_physical_target_reconstruction_error_rad=parity,
        diagnostic_external_controllers_not_certified_by_original_reward_signature=True,
        normal_limits=True, original_gains=True, gravity=True, free_base=True, self_collision=True,
        observations_truthful=True, teacher_context_changes_only_existing_training_labels=True,
        runtime_pose_writes=False, root_support=False, diagnostic_controllers_deployed=False, locomotion_qualified=False,
        scope='1088 robots,20s,32 individually seeded nominal-physics initial states per cohort. Two signs of bounded opposite5deg shoulder offsets, with unchanged paired control body actions versus own truthful closed-loop action. Separate truthful3.5-to-.15/1.5 command histories. Exact passive teacher targets through original clamp/joint/slew/CAD map. Baseline native5ms requested/applied motor attribution. This is controlled diagnosis; neither replay nor arm offsets enter production or qualify locomotion.')
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(completed=True, target_parity=parity, command_error=command_error)), flush=True)
    env.close()


if __name__ == '__main__':
    try: main()
    except BaseException:
        (OUT / 'arm_probe_failure.json').write_text(json.dumps(dict(error=traceback.format_exc())), encoding='utf-8')
        raise
    finally: app.app.close()
