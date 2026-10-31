import torch,json
from pathlib import Path
out=Path(__file__).resolve().parent;torch.set_num_threads(4)
for f in sorted(out.glob('current_gradient_*.pt')):
 d=torch.load(f,map_location='cpu',weights_only=False);y=d['mean'].requires_grad_(True);mu,std=d['current_distribution_params'];oldmu,oldstd=d['old_distribution_params']
 ratio=(torch.distributions.Normal(y,std).log_prob(d['sampled_actions']).sum(-1)-d['old_log_prob'].flatten()).exp()
 adv=d['advantages'].flatten();loss=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2)).mean();r=torch.autograd.grad(loss,y)[0];g=d['incoming_gradient']
 speed=d['raw_observations'][:,-80+6:-80+8].norm(dim=1)
 diff=g-r
 print(json.dumps(dict(batch=f.name,maximum_mean_difference=float((mu-y).abs().max()),incoming_max=float(g.abs().max()),reproduction_max=float(r.abs().max()),difference_max=float(diff.abs().max()),
  ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),difference_stopped=float(diff[speed<=.1].abs().max()),difference_moving=float(diff[speed>.1].abs().max()),
  incoming_reproduction_cosine=float((g*r).sum()/g.norm()/r.norm()),mean_adv=float(adv.mean()),std_adv=float(adv.std()),
  inferred_log_probability_error_max=float((torch.distributions.Normal(oldmu,oldstd).log_prob(d['sampled_actions']).sum(-1)-d['old_log_prob'].flatten()).abs().max()))))
