"""CPU-only audit of archived walking evidence versus the live R22 training.

Does not start Isaac, modify rewards, load a policy onto CUDA or overwrite
historical diagnostic data. Saved configs are parsed as data, without YAML
Python object constructors.
"""
from pathlib import Path
import ast, datetime, hashlib, importlib.util, json, math, sys, time
from types import SimpleNamespace
import numpy as np
import yaml
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'diagnostics/old_walker_comparison_20261002'
OUT.mkdir(exist_ok=True)
OLD = ROOT / 'humanoid_project'
WALK = OLD / 'logs/rsl_rl/humanoid5ft_natural_walk/2026-08-31_17-06-10'
active = json.loads((HERE / 'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
run = Path(active['run'])
state = json.loads((run / 'state.json').read_text())
frozen = run / 'source/physical_v29/training'

def data_yaml(path):
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def configured_rewards(cfg):
    return {name: dict(function=value['func'], weight=float(value['weight']), params=value.get('params', {}))
            for name, value in cfg['rewards'].items() if isinstance(value, dict) and 'weight' in value}

def checkpoint_summary(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    actor = saved['actor_state_dict']
    stats = {}
    for name, value in actor.items():
        if name.startswith('distribution.') and 'std' in name.lower() and isinstance(value, torch.Tensor):
            v = value.exp() if 'log_std' in name else value
            stats[name] = dict(min=float(v.min()), mean=float(v.mean()), max=float(v.max()))
    groups = saved.get('optimizer_state_dict', {}).get('param_groups', [])
    return dict(path=str(path), sha256=sha(path), iteration=saved.get('iter'),
        actor_tensor_shapes={k:list(v.shape) for k,v in actor.items() if isinstance(v,torch.Tensor)},
        exploration=stats, optimizer_learning_rates=[float(g['lr']) for g in groups],
        infos=saved.get('infos'))

walk_cfg = data_yaml(WALK / 'params/env.yaml')
walk_agent = data_yaml(WALK / 'params/agent.yaml')
stand_runs = sorted((OLD / 'logs/rsl_rl/humanoid5ft_natural_stand').glob('*/params/env.yaml'))
assert stand_runs
stand_cfg = data_yaml(stand_runs[-1])
walk_model = checkpoint_summary(WALK / 'model_2098.pt')
current_model = checkpoint_summary(Path(state['checkpoint']))

# Recompute the archived approved walker directly from the original 200 Hz
# recorder arrays, with the original first-episode and 3-second warmup mask.
old_cases = []
for pass_name in ['nominal', 'randomized']:
    folder = OLD / 'logs/gait_diagnostics_20260902' / pass_name
    meta = json.loads((folder / 'manifest.json').read_text())
    assert meta['completed'] and meta['checkpoint_sha256']['walk'] == walk_model['sha256']
    case = next(c for c in meta['cases'] if c['name'] == 'walk_1p0')
    ids = case['env_ids']
    raw = np.load(folder / 'physics.npy', mmap_mode='r')
    ep = np.load(folder / 'episodes.npz')
    fall = ep['first_fall_s'][ids]
    dt = meta['physics_dt']
    times = (np.arange(len(raw))+1)*dt
    valid = (times[:,None]>3) & (times[:,None] < np.where(np.isnan(fall),meta['duration_s']+dt,fall)[None,:])
    def field(name):
        lo, hi = meta['fields'][name]
        return np.asarray(raw[:,ids,lo:hi])
    velocity = field('root_vel_b')
    force = field('foot_force_w').reshape(len(raw), len(ids), 2, 3)
    q = field('q')
    assert np.isfinite(raw[:,ids,:]).all()
    params = np.load(folder / 'runtime_parameters.npz')
    positive_fz = np.maximum(force[...,2], 0)
    impulse = (positive_fz*valid[...,None]).sum((0,1))*dt
    summary = json.loads((folder / 'summary.json').read_text())
    recorded = next(c for c in summary['cases'] if c['case']=='walk_1p0')
    speed = float(velocity[...,0][valid].mean())
    assert abs(speed-recorded['mean_vx_mps']) < 1e-6
    old_cases.append(dict(dataset=pass_name, robots=len(ids), duration_s=meta['duration_s'],
        falls=int(np.isfinite(fall).sum()), mean_forward_speed_m_s=speed,
        speed_mae_m_s=float(np.abs(velocity[...,0]-1)[valid].mean()),
        lateral_speed_rms_m_s=float(np.sqrt(np.square(velocity[...,1][valid]).mean())),
        vertical_support_share=(impulse/impulse.sum()).tolist(),
        left_right_stance_s=recorded['left_right_stance_s'],
        left_right_stride_period_s=recorded['left_right_stride_period_s'],
        joint_names=meta['joint_names'], total_mass_kg=float(params['masses'][ids].sum(1).mean()),
        joint_angle_min_rad=q[valid].min(0).tolist(),joint_angle_max_rad=q[valid].max(0).tolist(),
        checkpoint_sha256_matches=True, all_walker_fields_finite=True))

# Save all independent R22 gate results, without treating PPO reward as success.
gates = []
for folder in sorted(run.glob('*_stand_gate_*_nominal')):
    report_path = folder / 'evaluation.json'
    if not report_path.exists(): continue
    report = json.loads(report_path.read_text())
    gates.append(dict(folder=folder.name, saved_at_utc=datetime.datetime.fromtimestamp(report_path.stat().st_mtime,datetime.timezone.utc).isoformat(),
        passed=report['passed'], cases=report['cases']))

rows = []
for folder in sorted(run.glob('*_stand_train_*')):
    if (folder / 'progress.jsonl').exists():
        rows.extend(json.loads(line) for line in (folder / 'progress.jsonl').read_text().splitlines())
assert rows and all(r['policy']['nominal_pose_max_abs_drift_rad']==0 for r in rows)
last = rows[-50:]
reward_keys = [k for k in last[-1]['behavior'] if k.startswith('reward_')]
reward_means = {k:float(np.mean([r['behavior'][k] for r in last])) for k in reward_keys}
regular = sum(v for k,v in reward_means.items() if k not in ['reward_living_clip_adjustment','reward_physical_termination'])
clipped = regular + reward_means['reward_living_clip_adjustment']
perf = dict(samples=len(last), median_collect_seconds=float(np.median([r['collect_seconds'] for r in last])),
    median_learn_seconds=float(np.median([r['learn_seconds'] for r in last])),
    mean_live_environment_steps_per_s=float(np.mean([r['num_envs']*24/(r['collect_seconds']+r['learn_seconds']) for r in last])),
    learning_rate_min=float(min(r['policy']['learning_rate'] for r in last)),
    learning_rate_max=float(max(r['policy']['learning_rate'] for r in last)),
    mean_exploration_std=float(np.mean([r['policy']['exploration_std'] for r in last])),
    regular_reward_rate_before_floor=regular, regular_reward_rate_after_floor=clipped,
    weighted_reward_rates=reward_means)

# Counterfactual reward signals use exact saved weights and the selected
# original function bodies. They demonstrate scoring behavior, not learnability.
old_source = OLD / 'source/humanoid_tasks/curriculum/skill_mdp.py'
tree = ast.parse(old_source.read_text())
selected = ['upright_exp','base_height_exp','planar_still_exp','gated_velocity_tracking_exp']
namespace = dict(torch=torch, SceneEntityCfg=lambda name:SimpleNamespace(name=name))
module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)] +
    [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in selected],type_ignores=[])
ast.fix_missing_locations(module)
exec(compile(module,str(old_source),'exec'),namespace)
spec = importlib.util.spec_from_file_location('current_comparison_reward_port',frozen/'unitree_reward_port.py')
port = importlib.util.module_from_spec(spec);sys.modules[spec.name]=port;spec.loader.exec_module(port)
g1 = port.profile('g1')
walk_terms = configured_rewards(walk_cfg)
stand_terms = configured_rewards(stand_cfg)
scenarios = []
for label, velocity_xy, tilt_deg, noise_std in [
    ('quiet_stand',[0.,0.],0.,0.),('sliding_stand',[0.,.15],0.,0.),
    ('lean_12_5_deg',[0.,0.],12.5,0.),('standing_action_noise_1',[0.,0.],0.,1.),
    ('standing_action_noise_045',[0.,0.],0.,.45),('standing_action_noise_020',[0.,0.],0.,.20)]:
    tilt = math.radians(tilt_deg)
    d = SimpleNamespace(projected_gravity_b=torch.tensor([[math.sin(tilt),0.,-math.cos(tilt)]]),
        root_pos_w=torch.tensor([[0.,0.,.90]]),root_lin_vel_b=torch.tensor([[*velocity_xy,0.]]))
    env = SimpleNamespace(scene={'robot':SimpleNamespace(data=d)},command_manager=SimpleNamespace(get_command=lambda name:torch.zeros(1,3)))
    original = {}
    for key in ['upright','height','planar_still','velocity_tracking']:
        term = stand_terms.get(key)
        if term is None:continue
        func = term['function'].split(':')[-1]
        if func not in namespace:continue
        params = {k:float(v) if k not in ['command_name'] else v for k,v in term['params'].items()}
        if key=='height':params['target_height']=.90 # relative height error, equal robot scale
        original[key] = float(namespace[func](env,**params))*term['weight']
    g1_track = g1['track_lin_vel_xy']['weight']*math.exp(-np.linalg.norm(velocity_xy)**2/g1['track_lin_vel_xy']['params']['std']**2)
    # Expected independent Gaussian innovation contribution only. Real policy
    # mean movement can make action-rate cost larger. No full-reward claim.
    scenarios.append(dict(name=label, old_standing_selected_terms=original,
        g1_zero_command_velocity_reward_rate=float(g1_track),
        g1_flat_orientation_reward_rate=g1['flat_orientation_l2']['weight']*math.sin(tilt)**2,
        independent_noise_expected_action_rate_old_22=float(walk_terms['action_rate']['weight']*2*22*noise_std**2),
        independent_noise_expected_action_rate_g1_23=float(g1['action_rate']['weight']*2*23*noise_std**2),
        scope='Isolated formula comparison at matching relative height; not the complete task reward'))

# TensorBoard reads are CPU-only and recover actual original training history.
event_stats = {}
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(WALK),size_guidance={'scalars':0});accumulator.Reload()
    for tag in accumulator.Tags()['scalars']:
        if any(k in tag.lower() for k in ['learning_rate','fps','total_time','mean_std','noise_std','action_rate','reward','episode_length']):
            values = accumulator.Scalars(tag)
            event_stats[tag] = dict(first_step=values[0].step,last_step=values[-1].step,
                samples=len(values), first=values[0].value,last=values[-1].value,
                last50_mean=float(np.mean([v.value for v in values[-50:]])),
                first_wall_utc=datetime.datetime.fromtimestamp(values[0].wall_time,datetime.timezone.utc).isoformat(),
                last_wall_utc=datetime.datetime.fromtimestamp(values[-1].wall_time,datetime.timezone.utc).isoformat())
except Exception as error:
    event_stats={'unavailable':repr(error)}

v4 = OLD / 'logs/guarded_run_v4_20260903_0305'
v4_gate = json.loads((v4/'06_speed_4_0600_gate.json').read_text())
retention = []
for folder in ['06_speed_4_0600_eval_47','06_speed_4_0600_eval_83']:
    result = json.loads((v4/folder/'evaluation.json').read_text())
    retention.append(dict(folder=folder, report=result))

record = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    automation=dict(id='humanoid-training-four-hour-check',deleted=not (Path('C:/Users/rshan/.codex/automations/humanoid-training-four-hour-check/automation.toml')).exists()),
    old_walking_checkpoint=walk_model,current_checkpoint=current_model,
    old_walking_recomputed_rollouts=old_cases, old_stand_config_path=str(stand_runs[-1]),
    old_stand_rewards=stand_terms,old_walk_rewards=walk_terms,
    old_walk_agent=walk_agent, old_walk_command=walk_cfg['commands'],
    old_walk_actions=walk_cfg['actions'],old_walk_observations=walk_cfg['observations'],
    old_walk_robot=walk_cfg['scene']['robot'],
    current_g1_rewards={k:dict(function=v['function'],weight=v['weight'],params={p:q for p,q in v['params'].items() if isinstance(q,(int,float,str))}) for k,v in g1.items()},
    current_state=state,current_nominal_gates=gates,current_last50_training=perf,
    isolated_scoring_examples=scenarios,original_tensorboard=event_stats,
    guarded_v4_gate=v4_gate,guarded_v4_evaluations=retention,
    limitations=['Saved replay analysis, no second Isaac worker.','Original model differs in morphology, observations, drives and contact physics.',
        'Original walking is asymmetric; speed success is not a human-gait certification.',
        'Isolated reward signal tests and causal hypotheses do not guarantee a new policy will learn.',
        'Positive reward flooring can flatten differences in returns; PPO still has likelihood-ratio gradients and does not require differentiating through the reward.'])
(OUT/'comparison.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(dict(output=str(OUT/'comparison.json'),automation=record['automation'],
    old_walking=[{k:v for k,v in r.items() if k not in ['joint_names','joint_angle_min_rad','joint_angle_max_rad']} for r in old_cases],
    old_actor_exploration=walk_model['exploration'], current_actor_exploration=current_model['exploration'],
    old_stand_terms={k:v['weight'] for k,v in stand_terms.items()},
    current_last50_training=perf,scoring=scenarios,
    original_tensorboard={k:v for k,v in event_stats.items() if k in ['Loss/learning_rate','Perf/total_fps','Policy/mean_noise_std','Policy/mean_std']},
    current_gates=[dict(folder=g['folder'],survival=g['cases'][0]['statistics']['survival'],
        drift_m=g['cases'][0]['statistics']['mean_final_drift_m'],failures=g['cases'][0]['failures']) for g in gates]),indent=2))
