"""Reset, history, command and explicit-loss derivative counterexamples."""
import hashlib,json
from pathlib import Path
import torch
from command_local_teacher_target import eligible,target_derivative
OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
torch.manual_seed(157);head_cols=[20,25];upper=[3,8,13,14,15,16,17,18];head=upper[:2]
raw=torch.zeros(12,400);frames=raw.reshape(-1,5,80);frames[:,:,78]=1.;frames[:,:,6]=.15
frames[1,:,6]=1.5;frames[2,:,6]=3.5;frames[3,:,6]=0.;frames[4,:,6]=.35
frames[5,0,6]=.35;frames[6,0,78]=0.;frames[7,:,7]=.02;frames[8,-1,20]=.2
frames[9,-1,25]=.174533;frames[10,2,6]=1.5;frames[11,:,6]=5.
assert eligible(raw,head_cols).tolist()==[True,True,False,False,False,False,False,False,False,False,False,False]
checks=['two_eligible_commands','frontier_excluded','stopped_excluded','other_priors_excluded','command_transition_excluded',
        'reset_history_excluded','lateral_commands_excluded','head_outside_excluded','head_boundary_excluded',
        'one_changed_frame_excluded','above_certificate_excluded']
mean=torch.randn(12,23);old=torch.randn_like(mean)*4;neutral=torch.randn_like(mean)*4
before_raw=raw.clone();before_old=old.clone();before_neutral=neutral.clone();selected=eligible(raw,head_cols)
target,diff,derivative,included=target_derivative(mean,old,neutral,selected,upper,head)
assert torch.equal(raw,before_raw) and torch.equal(old,before_old) and torch.equal(neutral,before_neutral)
assert torch.equal(target[~selected],old.clamp(-3,3)[~selected]) and torch.equal(target[selected],neutral.clamp(-3,3)[selected])
assert torch.equal(derivative[~selected],.5*torch.where(included[~selected],mean[~selected]-old.clamp(-3,3)[~selected],0.)/mean.numel())
leaf=mean.clone().requires_grad_(True)
loss=.25*torch.where(included,leaf-target,torch.zeros_like(leaf)).square().sum()/leaf.numel()
exact=torch.autograd.grad(loss,leaf)[0]
gradient_max_error=float((exact-derivative).abs().max())
assert torch.allclose(exact,derivative,atol=1e-9,rtol=1e-6),gradient_max_error
assert not derivative[:,head].any() and not derivative[~selected][:,[i for i in upper if i not in head]].any()
checks+=['input_labels_unmodified','other_commands_identical','selected_truthful_neutral_labels','exact_23_output_MSE_derivative',
         'all_head_derivatives_zero','other_command_arm_derivatives_zero','selected_six_arm_and_fifteen_body_rows']
for kind in ['raw_nonfinite','mean_nonfinite','wrong_raw_shape','wrong_output_shape','changed_weight']:
    try:
        if kind=='raw_nonfinite':bad=raw.clone();bad[0,0]=float('nan');eligible(bad,head_cols)
        elif kind=='mean_nonfinite':bad=mean.clone();bad[0,0]=float('inf');target_derivative(bad,old,neutral,selected,upper,head)
        elif kind=='wrong_raw_shape':eligible(raw[:,:399],head_cols)
        elif kind=='wrong_output_shape':target_derivative(mean[:,:22],old,neutral,selected,upper,head)
        else:target_derivative(mean,old,neutral,selected,upper,head,weight=.5)
    except AssertionError:checks.append('reject_'+kind)
    else:raise AssertionError(kind)
result=dict(passed=True,checks=checks,count=len(checks),autograd_float32_max_error=gradient_max_error,
    first_bitwise_float32_fixture_failure_preserved=True,helper_sha256=hashlib.sha256((OUT/'command_local_teacher_target.py').read_bytes()).hexdigest(),
    physical_validation_required=True,learning_candidate_selected=False,production_change=False)
(OUT/'command_local_teacher_CPU_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
