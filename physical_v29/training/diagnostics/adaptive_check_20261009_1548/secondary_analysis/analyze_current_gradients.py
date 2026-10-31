import torch,json,sys
from pathlib import Path
from torch import nn
OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
sys.path.insert(0,str(OUT/'control_source/physical_v29/training'))
from upper_body_refinement import posture_loss
pause=json.loads((OUT/'diagnostic_pause.json').read_text())
smoke_arguments=json.loads((OUT/'passive_current_PPO/run_config.json').read_text())['arguments']
cert=json.loads(Path(smoke_arguments['locomotion_prior_certificate']).read_text())
teacher_state=torch.load(cert['checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict']
teacher_m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
teacher_m.load_state_dict({k.removeprefix('mlp.'):v for k,v in teacher_state.items() if k.startswith('mlp.')})
teacher_m.requires_grad_(False)
smoke_arguments=json.loads((OUT/'passive_current_PPO/run_config.json').read_text())['arguments']
stopped_state=torch.load(smoke_arguments['standing_prior'],map_location='cpu',weights_only=False)['actor_state_dict']
stopped_m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
stopped_m.load_state_dict({k.removeprefix('mlp.'):v for k,v in stopped_state.items() if k.startswith('mlp.')});stopped_m.requires_grad_(False)
rows=[]
for file in sorted(OUT.glob('current_gradient_*.pt')):
 d=torch.load(file,map_location='cpu',weights_only=False)
 m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
 m.load_state_dict(d['state']);y=m(d['mlp_input']);assert float((y-d['mean']).abs().max())<6e-5
 ids=d['ids'];target=d['teacher_raw_targets'].clamp(-3,3);difference=y[ids].detach()-target;difference[:,d['upper_indices']]=0
 derivative=.5*difference/difference.numel()
 same_executed=(y[ids].detach().clamp(-3,3)==target)&(y[ids].detach().abs()>3)
 plateau=derivative*same_executed
 remaining=derivative-plateau
 parameters=list(m.parameters());speed=d['raw_observations'][:,-80+6:-80+8].norm(dim=1)
 def vjp(gradient):return torch.cat([g.flatten() for g in torch.autograd.grad(y,parameters,grad_outputs=gradient,retain_graph=True)])
 prior=torch.zeros_like(y);prior[ids]=derivative;flat=torch.zeros_like(y);flat[ids]=plateau
 gp,gf=vjp(prior),vjp(flat)
 run_der=d['incoming_gradient']*(speed>=3.5-1e-6)[:,None];gr=vjp(run_der)
 prior_ppo=d['incoming_gradient']*((speed>.1)&(speed<=3.+1e-6))[:,None];gpp=vjp(prior_ppo)
 cos=lambda a,b:float((a@b/(a.norm()*b.norm()).clamp_min(1e-12)).clamp(-1.,1.))
 raw=d['raw_observations'][ids].reshape(-1,5,80);teacher=d['teacher_inputs'].reshape(-1,5,80)
 changed=(raw!=teacher).any(-1).any(-1)
 with torch.no_grad():
  truthful_target=teacher_m(d['raw_observations'][ids]).clamp(-3,3)
  conditioned_reproduction=teacher_m(d['teacher_inputs']).clamp(-3,3)
 assert float((conditioned_reproduction-target).abs().max())<6e-5
 correction=(truthful_target-target);correction[:,d['upper_indices']]=0
 truth_derivative=.5*(y[ids].detach()-truthful_target)/difference.numel()
 truth_derivative[:,d['upper_indices']]=0
 gt_der=torch.zeros_like(y);gt_der[ids]=truth_derivative;gt=vjp(gt_der)
 # Exactly reproduce upper-only detached-final-layer auxiliary, not a trunk VJP.
 stride=max(1,(len(y)+255)//256);urows=torch.arange(0,len(y),stride)[:256]
 u=y.detach()[urows][:,d['upper_indices']].requires_grad_(True)
 uloss,parts=posture_loss(u);ud=torch.autograd.grad(uloss,u)[0]
 hidden=m[:6](d['mlp_input']).detach()[urows]
 uw=torch.zeros_like(m[-1].weight);ub=torch.zeros_like(m[-1].bias)
 uw[d['upper_indices']]=ud.T@hidden;ub[d['upper_indices']]=ud.sum(0)
 gu=torch.cat([torch.zeros_like(p).flatten() for p in parameters[:-2]]+[uw.flatten(),ub.flatten()])
 # The stopped-prior hook was registered before this passive moving capture.
 # Reconstruct/subtract it for the whole-batch PPO comparison. Running/prior
 # moving masks already exclude its disjoint stopped rows.
 stopped_ids=(d['raw_observations'][:,-80+6:-80+9].norm(dim=1)<=.1).nonzero().flatten()
 ss=max(1,(len(stopped_ids)+255)//256);stopped_ids=stopped_ids[::ss][:256]
 stopped_der=torch.zeros_like(y)
 if len(stopped_ids):
  with torch.no_grad():
   stopped_difference=y[stopped_ids].detach()-stopped_m(d['raw_observations'][stopped_ids]).clamp(-3,3)
   stopped_difference[:,d['upper_indices']]=0
   stopped_der[stopped_ids]=.5*stopped_difference/stopped_difference.numel()
 gppo=vjp(d['incoming_gradient']-stopped_der)
 # Reproduce the actual advantage-weighted clipped PPO mean derivative.
 mu,std=d['current_distribution_params'];sampled=d['sampled_actions']
 assert float((mu-y.detach()).abs().max())<6e-5
 exact_mean=d['mean'].detach().requires_grad_(True)
 ratio=(torch.distributions.Normal(exact_mean,std).log_prob(sampled).sum(-1)-d['old_log_prob'].flatten()).exp()
 adv=d['advantages'].flatten()
 loss=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2)).mean()
 reproduced=torch.autograd.grad(loss,exact_mean,retain_graph=True)[0]
 ppo_der=d['incoming_PPO_gradient']
 reproduction_error=float((reproduced-ppo_der).abs().max())
 moving_mask=speed>.1
 moving_error=float((reproduced[moving_mask]-d['incoming_gradient'][moving_mask]).abs().max())
 moving_cosine=cos(reproduced[moving_mask].flatten(),d['incoming_gradient'][moving_mask].flatten())
 assert moving_cosine>.999,moving_cosine
 # Use captured exact means/advantages/Gaussian parameters to isolate PPO.
 # CPU/GPU score reductions near the clipped-surrogate boundary can switch a
 # few derivatives; do not claim bitwise reproduction. The old stopped-hook
 # subtraction audit is retained as a failed diagnostic, not a runtime fault.
 ppo_der=d['incoming_PPO_gradient'];gppo=vjp(ppo_der)
 command_gradients=[]
 for command in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.,'frontier_above3','continuous_below3']:
  if command=='frontier_above3':mask=speed>3.+1e-6
  elif command=='continuous_below3':
   mask=(speed>.1)&(speed<=3.+1e-6)
   for value in [.15,.35,.65,1.,1.5,2.,2.5,3.]:mask&=(speed-value).abs()>=1e-6
  else:mask=(speed-float(command)).abs()<1e-6
  if not mask.any():continue
  pder=(ppo_der if command==0. else d['incoming_gradient'])*mask[:,None];pgrad=vjp(pder)
  tder=prior*mask[:,None];tgrad=vjp(tder)
  own=mask[ids]
  command_gradients.append(dict(command=command,samples=int(mask.sum()),sample_fraction=float(mask.float().mean()),
   advantage_mean=float(adv[mask].mean()),advantage_std=float(adv[mask].std()),positive_advantage_fraction=float((adv[mask]>0).float().mean()),
   return_mean=float(d['returns'].flatten()[mask].mean()),importance_ratio_mean=float(ratio[mask].mean()),
   importance_ratio_clip_fraction=float(((ratio[mask]<.8)|(ratio[mask]>1.2)).float().mean()),
   raw_action_mean_clip_fraction=float((y[mask].abs()>3).float().mean()),
   sampled_action_clip_fraction=float((sampled[mask].abs()>3).float().mean()),
   actual_PPO_mean_parameter_norm=float(pgrad.norm()),teacher_parameter_norm=float(tgrad.norm()),
   PPO_teacher_cosine=cos(pgrad,tgrad) if own.any() else None,
   teacher_selected_samples=int(own.sum()),
   parameter_contribution_scope='Exact passively captured incoming PPO mean derivative before standing and moving hooks. Whole-minibatch scaling before global norm clipping/Adam; actor mean-network only, not exploration gradients.'))
 upper=[i for i in range(23) if i not in d['upper_indices']]
 commands=[]
 for command in [.15,.35,.65,1.,1.5,2.,2.5,3.]:
  mask=(raw[:,-1,6]-command).abs()<1e-6
  if not mask.any():continue
  rd=difference[mask][:,upper];ed=(y[ids][mask].detach().clamp(-3,3)-target[mask])[:,upper]
  commands.append(dict(command=command,selected=int(mask.sum()),conditioned_fraction=float(changed[mask].float().mean()),
   raw_lower_MSE=float(rd.square().mean()),effective_action_MSE=float(ed.square().mean()),
   same_endpoint_raw_error_fraction=float((plateau[mask].square().sum()/derivative[mask].square().sum().clamp_min(1e-12))),
   teacher_conditioning_lower_effective_RMS=float(correction[mask][:,upper].square().mean().sqrt()),
   truthful_teacher_effective_action_MSE=float((y[ids][mask].detach().clamp(-3,3)-truthful_target[mask])[:,upper].square().mean()),
   latest_head_RMS_deg=float(raw[mask][:,-1,[20,25]].square().mean().sqrt()*180/torch.pi)))
 rows.append(dict(batch=file.name,reproduction_max_error=float((y-d['mean']).abs().max()),
  batch_samples=len(y),selected=len(ids),per_command=commands,
  prior_parameter_norm=float(gp.norm()),plateau_parameter_norm=float(gf.norm()),
  plateau_prior_cosine=cos(gf,gp),running_PPO_norm=float(gr.norm()),prior_PPO_norm=float(gpp.norm()),
  running_teacher_cosine=cos(gr,gp),prior_PPO_teacher_cosine=cos(gpp,gp),
  truthful_teacher_parameter_norm=float(gt.norm()),conditioned_truthful_teacher_cosine=cos(gp,gt),
  upper_auxiliary_parameter_norm=float(gu.norm()),all_PPO_parameter_norm=float(gppo.norm()),
  upper_auxiliary_PPO_cosine=cos(gu,gppo),upper_losses={k:float(v) for k,v in parts.items()},
  stopped_hook_subtracted_for_whole_PPO_comparison=False,
  exact_PPO_captured_before_standing_hook=True,
  actual_standing_hook_norm=float(d['actual_standing_hook_derivative'].norm()),
  CPU_standing_subtraction_max_error=float((d['actual_standing_hook_derivative']-stopped_der).abs().max()),
  running_PPO_mask='Latest XY command norm >=3.5, exact current command; not all continuous frontier samples',
  actual_surrogate_mean_derivative_max_reproduction_error=reproduction_error,
  moving_surrogate_derivative_max_error=moving_error,
  moving_surrogate_derivative_cosine=moving_cosine,
  PPO_derivative_source='Actual incoming gradient passively captured before either prior hook; CPU surrogate is a cross-check only. Mean-network VJP before global clipping/Adam; exploration parameters omitted.',
  actual_per_command_gradients=command_gradients,
  plateau_teacher_derivative_energy_fraction=float(plateau.square().sum()/derivative.square().sum().clamp_min(1e-12)),
  running_raw_action_clip_fraction=float((y[speed>=3.5-1e-6].abs()>3).float().mean())))
result=dict(completed=True,teacher_checkpoint=cert['checkpoint'],teacher_sha256=cert['checkpoint_sha256'],batches=rows,scope='Four genuine PPO minibatches at 4096 environments. Passive labels and pre-auxiliary VJPs before clipping/Adam; exact upper-only final-row auxiliary. Truthful teacher is a CPU counterfactual, absent from production. Finite-state evidence is not a causal repair or proof of globally opposed gradients.')
(OUT/'current_gradient_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
