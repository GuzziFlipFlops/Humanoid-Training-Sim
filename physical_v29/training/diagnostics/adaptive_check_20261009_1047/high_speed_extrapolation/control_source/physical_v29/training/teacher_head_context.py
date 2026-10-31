"""Training-only teacher context; student/inference inputs are untouched."""
import torch
VERSION='teacher_head_phase_context_v1'
UPPER=('head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch',
 'left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch')

def context_inputs(raw, table, columns):
    result=raw.clone().reshape(-1,5,80)
    phase=torch.remainder(torch.atan2(result[:,:,78],result[:,:,79])/(2*torch.pi),1.)
    location=phase*len(table); lo=location.floor().long()%len(table); hi=(lo+1)%len(table)
    blend=(location-location.floor()).unsqueeze(-1)
    result[:,:,columns]=table[lo]*(1.-blend)+table[hi]*blend
    return result.reshape(-1,400)

