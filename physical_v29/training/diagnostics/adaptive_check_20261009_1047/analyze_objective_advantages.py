"""Compare passive real PPO signals under two same-checkpoint objectives.

CPU only. Paired observation/action equality is measured, not assumed. These
captures can identify a local learning-signal change, never long-term causality.
"""
from pathlib import Path
import ast, hashlib, json, math
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'high_speed_extrapolation'
torch.set_num_threads(4)
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def cosine(a, b):
    if float(a.norm()) == 0 or float(b.norm()) == 0: return None
    return float((a.flatten() @ b.flatten() / (a.norm() * b.norm())).clamp(-1, 1))
def difference(a, b):
    assert a.shape == b.shape
    z = a.to(torch.float64) - b.to(torch.float64)
    return dict(exact_equal=bool(torch.equal(a, b)), max_absolute=float(z.abs().max()),
                RMS=float(z.square().mean().sqrt()))

worker = OUT / 'control_source/physical_v29/training/worker_unitree.py'
ppo_path = Path(torch.__file__).parents[1] / 'rsl_rl/algorithms/ppo.py'
worker_tree = ast.parse(worker.read_text(encoding='utf-8'))
agent = next(n for n in worker_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'agent_config')
explicit_options = [k.value for n in ast.walk(agent) if isinstance(n, ast.Call)
                    for k in n.keywords if k.arg == 'normalize_advantage_per_mini_batch']
ppo_tree = ast.parse(ppo_path.read_text(encoding='utf-8'))
cls = next(n for n in ppo_tree.body if isinstance(n, ast.ClassDef) and n.name == 'PPO')
ctor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
defaults = dict(zip([a.arg for a in ctor.args.args][-len(ctor.args.defaults):], ctor.args.defaults))
assert ast.literal_eval(defaults['normalize_advantage_per_mini_batch']) is False
assert not explicit_options, 'Recheck normalization under an explicit runtime setting'
source = ppo_path.read_text(encoding='utf-8')
assert 'if not self.normalize_advantage_per_mini_batch:' in source

proofs = {}
for branch in ['original', 'narrow']:
    folder = OUT / ('objective_' + branch)
    proof = read(folder / 'full_load_proof.json')
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    progress = [json.loads(s) for s in (folder / 'progress.jsonl').read_text().splitlines()]
    assert len(progress) == 4
    assert all(p['ppo']['pre_update_kl'] == 0 and p['behavior']['action_command_difference'] == 0 for p in progress)
    assert all(p[k]['batches'] == 20 for p in progress for k in ['retention','locomotion_retention','upper_refinement'])
    assert min(p['gpu']['free_mb'] for p in progress) >= 300
    provenance = read(folder / 'experimental_reward.json')
    assert provenance['applied'] == (branch == 'narrow')
    proofs[branch] = dict(exact_full_load=True, genuine_updates=4, minimum_GPU_free_MiB=min(p['gpu']['free_mb'] for p in progress),
                         capture_sha256={p.name: sha(p) for p in folder.glob('current_gradient_*.pt')}, provenance=provenance)

rows = []
for index in range(4):
    captures = {b: torch.load(OUT / ('objective_' + b) / f'current_gradient_{index:02d}.pt',
                             map_location='cpu', weights_only=False) for b in ['original','narrow']}
    a, b = captures['original'], captures['narrow']
    parity = {k: difference(a[k], b[k]) for k in ['raw_observations','mlp_input','mean','sampled_actions','old_log_prob','old_critic_values','ids']}
    parity['old_distribution_params'] = [difference(x,y) for x,y in zip(a['old_distribution_params'], b['old_distribution_params'])]
    parity['mean_network_state_exact'] = all(torch.equal(a['state'][k], b['state'][k]) for k in a['state'])
    parity['paired_first_rollout_exact'] = all(parity[k]['exact_equal'] for k in ['raw_observations','mlp_input','sampled_actions','old_log_prob'])
    signals = {}
    vectors = {}
    for branch, d in captures.items():
        model = nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                              nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
        model.load_state_dict(d['state'])
        means = model(d['mlp_input'])
        replay_error = float((means.detach()-d['mean']).abs().max())
        assert replay_error < 6e-5
        speed = d['raw_observations'][:,-80+6:-80+8].norm(dim=1)
        advantage = d['advantages'].flatten()
        returns = d['returns'].flatten()
        values = d['old_critic_values'].flatten()
        unnormalized = returns-values
        stats = d['rollout_return_value_stats']
        reconstructed_advantage=(unnormalized-stats['unnormalized_advantage_mean'])/(stats['unnormalized_advantage_std']+1e-8)
        advantage_error=float((reconstructed_advantage-advantage).abs().max())
        assert advantage_error < 2e-4, advantage_error
        mu, std = d['current_distribution_params']
        ratio = (torch.distributions.Normal(mu, std).log_prob(d['sampled_actions']).sum(-1)-d['old_log_prob'].flatten()).exp()
        deriv = d['incoming_PPO_gradient']
        assert all(torch.isfinite(v).all() for v in [advantage,returns,ratio,deriv])
        parameters = list(model.parameters())
        command_rows = []
        vectors[branch] = {}
        for command in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.,'frontier_above3','continuous_below3']:
            if command == 'frontier_above3': mask = speed > 3.+1e-6
            elif command == 'continuous_below3':
                mask = (speed>.1)&(speed<=3.+1e-6)
                for v in [.15,.35,.65,1.,1.5,2.,2.5,3.]: mask &= (speed-v).abs() >= 1e-6
            else: mask = (speed-float(command)).abs()<1e-6
            if not mask.any(): continue
            gradient = torch.cat([v.flatten() for v in torch.autograd.grad(means,parameters,
                grad_outputs=deriv*mask[:,None],retain_graph=True)])
            vectors[branch][str(command)] = gradient.detach()
            command_rows.append(dict(command=command,samples=int(mask.sum()),
                normalized_advantage_mean=float(advantage[mask].mean()),
                normalized_advantage_std=float(advantage[mask].std(unbiased=False)),
                positive_advantage_fraction=float((advantage[mask]>0).float().mean()),
                return_mean=float(returns[mask].mean()), ratio_mean=float(ratio[mask].mean()),
                old_critic_value_mean=float(values[mask].mean()),unnormalized_return_minus_value_mean=float(unnormalized[mask].mean()),
                ratio_outside_clip_fraction=float(((ratio[mask]<.8)|(ratio[mask]>1.2)).float().mean()),
                actual_PPO_mean_parameter_norm=float(gradient.norm())))
        signals[branch] = dict(mean_replay_max_error=replay_error,
            global_advantage_normalization_max_reproduction_error=advantage_error,global_rollout_value_statistics=stats,
            batch_normalized_advantage_mean=float(advantage.mean()),
            batch_normalized_advantage_std=float(advantage.std(unbiased=False)),
            commands=command_rows)
    command_comparisons = []
    for command in vectors['original'].keys() & vectors['narrow'].keys():
        x, y = vectors['original'][command], vectors['narrow'][command]
        command_comparisons.append(dict(command=command,PPO_parameter_cosine=cosine(x,y),
            parameter_difference_norm=float((x-y).norm()),
            original_norm=float(x.norm()),narrow_norm=float(y.norm())))
    rows.append(dict(batch=index,parity=parity,advantages=difference(a['advantages'],b['advantages']),
        returns=difference(a['returns'],b['returns']),incoming_PPO_mean_derivatives=difference(a['incoming_PPO_gradient'],b['incoming_PPO_gradient']),
        signals=signals,command_gradient_comparisons=command_comparisons,
        paired_learning_signal_attribution_valid=(index==0 and parity['paired_first_rollout_exact'] and parity['mean_network_state_exact'])))

result = dict(completed=True, checkpoint=read(OUT/'diagnostic_pause.json')['checkpoint'],
    checkpoint_sha256=read(OUT/'diagnostic_pause.json')['checkpoint_sha256'],proofs=proofs,batches=rows,
    advantage_normalization=dict(per_minibatch=False,scope='Across rollout before PPO minibatches',
        worker_sha256=sha(worker),installed_rsl_PPO_sha256=sha(ppo_path)),
    first_minibatch_paired_attribution_valid=rows[0]['paired_learning_signal_attribution_valid'],
    diagnostic_updates_excluded_from_main=8,diagnostic_actors_deployed=False,
    interpretation='Actual captured PPO gradients are before either prior hook, global norm clipping and Adam; actor mean-network only. Exact paired first-state evidence, if present, attributes a local objective-induced advantage/gradient change. Later minibatches also reflect prior optimizer updates. Neither a changed gradient nor global normalization proves the cause of long-term gait failure or validates a repair.')
encoded = json.dumps(result,indent=2,allow_nan=False)+'\n'
(ROOT/'objective_advantage_comparison.json').write_text(encoded,encoding='utf-8')
print(json.dumps(dict(completed=True,first_minibatch_paired_attribution_valid=result['first_minibatch_paired_attribution_valid'],
    first_advantage_difference=rows[0]['advantages'],diagnostic_updates_excluded_from_main=8),indent=2))
