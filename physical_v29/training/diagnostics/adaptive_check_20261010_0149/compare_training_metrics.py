"""Passive mixed-command learning metrics; independent gates determine behavior."""
import json,statistics
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_full_state'
read=lambda p:json.loads(Path(p).read_text())
records=[]
for seed in [317,331]:
 for branch in ['control','earlier']:
  for updates in [64,128]:
   folder=ROOT/f'seed{seed}_{branch}_{updates:03d}_train'
   if not (folder/'training_validation.json').exists():continue
   rows=[json.loads(v) for v in (folder/'progress.jsonl').read_text().splitlines()]
   assert len(rows)==64
   terms=sorted(k for k in rows[0]['behavior'] if k.startswith('reward_'))
   windows={}
   for name,window in [('first8',rows[:8]),('last8',rows[-8:]),('all64',rows)]:
    termmeans={k:statistics.mean(r['behavior'][k] for r in window) for k in terms}
    windows[name]=dict(weighted_term_means=termmeans,sum_logged_reward_terms= sum(termmeans.values()),
     bounded_PPO_sample_return_mean=statistics.mean(r['ppo']['return_mean'] for r in window),
     value_explained_variance=statistics.mean(r['ppo']['value_explained_variance'] for r in window),
     post_update_KL=statistics.mean(r['ppo']['post_update_kl'] for r in window),
     optimizer_learning_rate_min=min(r['policy']['learning_rate'] for r in window),
     optimizer_learning_rate_max=max(r['policy']['learning_rate'] for r in window),
     prior_loss_means={k:statistics.mean(r[k]['standing_prior_loss'] for r in window) for k in ['retention','locomotion_retention']},
     exact_command_fraction_means={k:statistics.mean(r['command_coverage']['exact_command_fractions'][k] for r in window) for k in rows[0]['command_coverage']['exact_command_fractions']},
     raw_Gaussian_std_min=min(x for r in window for x in r['policy']['exploration_std']),
     raw_Gaussian_std_max=max(x for r in window for x in r['policy']['exploration_std']),
     raw_Gaussian_std_at_upper_bound_fraction=statistics.mean(x>=.7-1e-6 for r in window for x in r['policy']['exploration_std']))
   records.append(dict(seed=seed,branch=branch,cumulative_updates=updates,train=str(folder),windows=windows))
source=OUT/'control_source/physical_v29/training/worker_unitree.py'
assert 'clamp_(np.log(.04),np.log(.5 if args.speed==0. else .7))' in source.read_text()
data=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),records=records,
 raw_Gaussian_std_bounds=dict(minimum=.04,moving_maximum=.7,stopped_stage_maximum=.5,source=str(source),
 interpretation='0.7 is an upper bound, not a forced exploration floor. Saturation of this learned parameter does not establish a locomotion defect or justify resetting exploration.'),
 scope='Mixed-command stochastic training populations and bounded PPO sample diagnostics. Logged weighted reward components do not reconstruct unlogged reward state, certify per-command complete returns, independent locomotion, teacher-gradient causality or a hardware ceiling. No parameter or source changes.')
(OUT/'training_metric_comparison.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(completed_blocks=len(records),summary=[dict(seed=r['seed'],branch=r['branch'],updates=r['cumulative_updates'],logged_reward_sum=r['windows']['last8']['sum_logged_reward_terms'],PPO_return=r['windows']['last8']['bounded_PPO_sample_return_mean']) for r in records])))
