"""Diagnostic own-feedback teacher head versus arm separation."""
import torch
MODES=('current','teacher','teacher_body_current_upper','teacher_current_head','teacher_current_arms')
def cohorts():
    rows=[]
    for seed in [4421,4431]:
        for speed in [.15,1.5]:
            for mode in MODES:
                rows.append(dict(name=f'{seed}_{speed}_{mode}',seed=seed,command=speed,initial_command=speed,mode=mode,transition=False))
        rows.append(dict(name=f'{seed}_3.5_current',seed=seed,command=3.5,initial_command=3.5,mode='current',transition=False))
    return rows
def combine_actions(current,teacher,context_teacher,group_modes,per,upper,qualified):
    assert current.shape==teacher.shape and len(group_modes)*per==len(current)
    result=current.clone()
    for group,mode in enumerate(group_modes):
        rows=slice(group*per,(group+1)*per)
        assert mode in MODES
        if mode=='current':continue
        assert bool(qualified[rows].all())
        result[rows]=teacher[rows]
        selected=upper if mode=='teacher_body_current_upper' else upper[:2] if mode=='teacher_current_head' else upper[2:] if mode=='teacher_current_arms' else []
        if selected:result[rows,selected]=current[rows,selected]
    return result
