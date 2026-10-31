"""Parameter-space VJP from an actually collected PPO minibatch, CPU only."""
import json
from pathlib import Path
import torch
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent
d=torch.load(OUT/'gradient_batch.pt',map_location='cpu',weights_only=False)
model=torch.nn.Sequential(torch.nn.Linear(400,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,23))
model.load_state_dict(d['mlp_state']);x=d['observations'];mean=model(x)
error=float((mean-d['mean']).abs().max());assert error<.0005,error
parameters=list(model.parameters());run=d['latest_commands'][:,:2].norm(dim=1)>=1.5
prior=torch.zeros_like(mean);prior[d['ids']]=d['prior_derivative']
g_run=d['incoming_gradient']*run[:,None]
def vjp(g):return torch.cat([a.flatten() for a in torch.autograd.grad(mean,parameters,g,retain_graph=True)])
gr=vjp(g_run);gp=vjp(prior);gall=vjp(d['incoming_gradient'])
def pair(a,b):return {'norm_a':float(a.norm()),'norm_b':float(b.norm()),'dot':float(a@b),'cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0))}
r={'passed':True,'mean_reconstruction_max_error':error,'samples':len(x),'running_samples':int(run.sum()),'prior_samples':len(d['ids']),
   'running_ppo_vs_walking_prior':pair(gr,gp),'whole_minibatch_vs_walking_prior':pair(gall,gp),
   'running_rows_direct_prior_derivative_is_zero':bool(not prior[run].any()),
   'scope':'One real PPO minibatch; parameter-space gradient interaction, not a causal long-term ablation or proof of convergence. Passive capture leaves actual updates unchanged; actor discarded.'}
assert r['running_rows_direct_prior_derivative_is_zero']
(OUT/'gradient_analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
