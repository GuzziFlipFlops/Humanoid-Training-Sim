"""External diagnostic composition; every controller reads its own true state."""
import torch

MODES = ('current', 'teacher', 'teacher_body_current_upper', 'current_body_teacher_upper',
         'context_teacher', 'context_teacher_body_current_upper')

def cohorts():
    result=[]
    for seed in [4401,4411]:
        for speed in [.15,1.5]:
            modes=MODES if speed==.15 else MODES[:4]
            for mode in modes:
                result.append(dict(name=f'{seed}_{speed}_{mode}',seed=seed,command=speed,
                    initial_command=speed,mode=mode,transition=False))
            for mode in ['current','teacher','teacher_body_current_upper'] + (['context_teacher_body_current_upper'] if speed==.15 else []):
                result.append(dict(name=f'{seed}_3_to_{speed}_{mode}',seed=seed,command=speed,
                    initial_command=3.,mode=mode,transition=True))
        result.append(dict(name=f'{seed}_3.5_current',seed=seed,command=3.5,initial_command=3.5,
            mode='current',transition=False))
    return result

def combine_actions(current, teacher, context_teacher, group_modes, per, upper, qualified):
    assert current.shape==teacher.shape==context_teacher.shape
    n,width=current.shape
    assert len(group_modes)*per==n and len(set(upper))==len(upper)
    assert all(0<=i<width for i in upper)
    lower=[i for i in range(width) if i not in upper]
    result=current.clone()
    for group,mode in enumerate(group_modes):
        rows=slice(group*per,(group+1)*per)
        assert mode in MODES
        if mode=='current':continue
        assert bool(qualified[rows].all()), 'No teacher action outside certificate'
        if mode in ['teacher','context_teacher']:
            result[rows]=context_teacher[rows] if mode=='context_teacher' else teacher[rows]
        elif mode=='current_body_teacher_upper':
            result[rows,upper]=teacher[rows,upper]
        else:
            selected=context_teacher if mode=='context_teacher_body_current_upper' else teacher
            result[rows,lower]=selected[rows,lower]
    return result
