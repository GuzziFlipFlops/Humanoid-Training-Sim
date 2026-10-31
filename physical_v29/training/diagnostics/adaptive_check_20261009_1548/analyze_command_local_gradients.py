"""Student-state counterfactual gradients; no claims about learned outcomes."""
import json
from pathlib import Path
import torch
from torch import nn
from command_local_teacher_target import eligible,target_derivative
OUT=Path(__file__).resolve().parent;SECOND=OUT/'secondary_analysis';torch.set_num_threads(4)
cert=json.loads((OUT/'neutral_teacher_certificate.json').read_text())
saved=torch.load(cert['checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict']
teacher=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
teacher.load_state_dict({k.removeprefix('mlp.'):v for k,v in saved.items() if k.startswith('mlp.')});teacher.requires_grad_(False)
head_cols=json.loads((SECOND/'control_source/physical_v29/training/teacher_head_context.json').read_text())['columns'][:2]
records=[]
for file in sorted(SECOND.glob('current_gradient_*.pt')):
    d=torch.load(file,map_location='cpu',weights_only=False);ids=d['ids'];raw=d['raw_observations'][ids];mean=d['mean'][ids];upper=d['upper_indices']
    chosen=eligible(raw,head_cols);before=raw.clone();old=d['teacher_raw_targets'].clamp(-3,3)
    with torch.no_grad():neutral=teacher(raw).clamp(-3,3)
    target,difference,derivative,included=target_derivative(mean,old,neutral,chosen,upper,upper[:2])
    original_difference=mean-old;original_difference[:,upper]=0.;original_derivative=.5*original_difference/original_difference.numel()
    assert torch.equal(raw,before) and torch.equal(derivative[~chosen],original_derivative[~chosen])
    assert not derivative[:,upper[:2]].any()
    m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
    m.load_state_dict(d['state']);y=m(d['mlp_input']);parameters=list(m.parameters())
    def vjp(grad):
        full=torch.zeros_like(y);full[ids]=grad
        return torch.cat([p.flatten() for p in torch.autograd.grad(y,parameters,grad_outputs=full,retain_graph=True)])
    current_grad=vjp(original_derivative);candidate_grad=vjp(derivative)
    own=[];frames=raw.reshape(-1,5,80)
    for command in [.15,1.5]:
        mask=(frames[:,-1,6]-command).abs()<1e-6;selected=mask&chosen
        if not selected.any():continue
        pg=vjp(derivative*selected[:,None]);og=vjp(original_derivative*selected[:,None])
        own.append(dict(command=command,selected=int(selected.sum()),available=int(mask.sum()),
            teacher_changed_body_effective_RMS=float((neutral[selected][:,[i for i in range(23) if i not in upper]]-old[selected][:,[i for i in range(23) if i not in upper]]).square().mean().sqrt()),
            neutral_arm_target_RMS=float(neutral[selected][:,upper[2:]].square().mean().sqrt()),
            current_neutral_arm_effective_error_RMS=float((mean[selected].clamp(-3,3)[:,upper[2:]]-neutral[selected][:,upper[2:]]).square().mean().sqrt()),
            original_parameter_gradient_norm=float(og.norm()),candidate_parameter_gradient_norm=float(pg.norm()),
            original_candidate_cosine=float(torch.nn.functional.cosine_similarity(og,pg,dim=0)),
            target_or_derivative_not_deployed=True))
    records.append(dict(batch=file.name,eligible=int(chosen.sum()),selected_prior_samples=len(ids),per_command=own,
        other_selected_command_derivatives_bitwise_equal=True,direct_head_derivative_zero=True,student_inputs_unmodified=True,
        original_parameter_gradient_norm=float(current_grad.norm()),candidate_parameter_gradient_norm=float(candidate_grad.norm()),
        original_candidate_cosine=float(torch.nn.functional.cosine_similarity(current_grad,candidate_grad,dim=0))))
assert len(records)==4
result=dict(completed=True,current_checkpoint=json.loads((SECOND/'diagnostic_pause.json').read_text())['checkpoint'],
    neutral_teacher_checkpoint=cert['checkpoint'],neutral_teacher_sha256=cert['checkpoint_sha256'],records=records,
    scope='Four actual student-state PPO minibatches,CPU teacher-target counterfactual before global clipping/Adam. Settled exact0.15/1.5 targets may add six arm and fifteen body derivatives;direct head and other command derivatives are preserved. Shared-trunk influence on every command is a measured risk requiring the full matched learning trial. No natural-motion or safe-transfer conclusion from these local derivatives.')
(OUT/'command_local_gradient_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
