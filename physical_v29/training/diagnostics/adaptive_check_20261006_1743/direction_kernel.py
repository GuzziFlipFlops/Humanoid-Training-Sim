"""Diagnostic fixed-reset-frame translation kernel; no weight/width change."""
import torch

def score(commands,velocity_yaw,heading_error,std):
    c,s=torch.cos(heading_error),torch.sin(heading_error)
    fixed=torch.stack([c*velocity_yaw[:,0]+s*velocity_yaw[:,1],-s*velocity_yaw[:,0]+c*velocity_yaw[:,1]],1)
    old=torch.exp(-((commands[:,:2]-velocity_yaw[:,:2])**2).sum(1)/std**2)
    new=torch.exp(-((commands[:,:2]-fixed)**2).sum(1)/std**2)
    return torch.where(commands[:,0]>3.+1e-6,new,old)
