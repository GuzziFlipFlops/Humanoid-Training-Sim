"""Undeployed diagnostic helper for a coupled, command-local teacher target.

This is a conditional training target and its derivative, never an inference
controller. Physical compatibility must pass before a learning trial is chosen.
"""
import torch
VERSION='neutral267_coupled_exact_command_candidate_v1'
COMMANDS=(.15,1.5)

def eligible(raw,head_columns):
    assert raw.ndim==2 and raw.shape[1]==400 and len(head_columns)==2
    assert torch.isfinite(raw).all()
    f=raw.reshape(-1,5,80)
    selected=torch.zeros(len(raw),dtype=torch.bool,device=raw.device)
    for command in COMMANDS:
        selected|=((f[:,:,6]-command).abs()<1e-6).all(1)&(f[:,:,7].abs()<1e-6).all(1)
    selected&=(f[:,:,78:80].square().sum(-1)>.5).all(1)
    selected&=(f[:,-1,list(head_columns)].abs()<.174533).all(1)
    return selected

def target_derivative(mean,original_target,neutral_target,selected,upper,head,weight=.25):
    assert mean.ndim==2 and mean.shape[1]==23 and mean.shape==original_target.shape==neutral_target.shape
    assert selected.dtype==torch.bool and selected.shape==(len(mean),)
    assert len(upper)==8 and len(set(upper))==8 and len(head)==2 and set(head).issubset(upper)
    assert all(torch.isfinite(x).all() for x in [mean,original_target,neutral_target])
    assert weight==.25,'No arbitrary retention-weight change'
    target=original_target.clamp(-3.,3.).clone()
    target[selected]=neutral_target[selected].clamp(-3.,3.)
    included=torch.ones_like(mean,dtype=torch.bool)
    included[:,list(upper)]=False
    for i in [x for x in upper if x not in head]:included[selected,i]=True
    difference=torch.where(included,mean.detach()-target,torch.zeros_like(mean))
    derivative=2.*weight*difference/difference.numel()
    assert not derivative[:,list(head)].any()
    return target,difference,derivative,included
