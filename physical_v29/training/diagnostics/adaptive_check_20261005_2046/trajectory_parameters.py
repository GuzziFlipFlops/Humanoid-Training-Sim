"""Bounded smooth leg-only diagnostic offsets; never imported by training."""
import numpy as np

# Three extension and three retraction offsets in joint-target radians,
# followed by centers and widths in the half-cycle [0,.5).
LOW=np.array([-.25,-.35,-.20,-.25,-.30,-.25,.22,.34,.04,.03],np.float64)
HIGH=np.array([.25,.35,.20,.25,.40,.25,.41,.49,.12,.10],np.float64)
ORIGIN=np.array([0.,0.,0.,0.,0.,0.,.33,.445,.08,.065],np.float64)
INITIAL_STD=np.array([.12,.18,.10,.12,.20,.12,.045,.035,.025,.020],np.float64)
NAMES=['extension_hip_rad','extension_knee_rad','extension_ankle_rad',
       'retraction_hip_rad','retraction_knee_rad','retraction_ankle_rad',
       'extension_center_halfcycle','retraction_center_halfcycle',
       'extension_width_cycles','retraction_width_cycles']

def samples(rng,mean,std,count=32):
 x=np.clip(rng.normal(mean,std,(count,len(mean))),LOW,HIGH)
 x[0]=ORIGIN # Unchanged full-actor baseline in every generation.
 x[1]=np.clip(mean,LOW,HIGH)
 return x

def offsets_numpy(phase,parameters,t):
 """Offsets use the observed PRE-ACTION phase, not post-step reward phase."""
 phi=np.remainder(np.asarray(phase),.5)
 p=np.asarray(parameters)
 def bump(center,width):
  distance=np.abs(np.remainder(phi-center+.25,.5)-.25)
  return np.where(distance<width,.5*(1+np.cos(np.pi*distance/width)),0.)
 extension=bump(p[...,6],p[...,8]);retraction=bump(p[...,7],p[...,9])
 fade=np.clip((t-3.)/1.,0.,1.)
 return fade*(extension[...,None]*p[...,:3]+retraction[...,None]*p[...,3:6])

def update(parameters,fitness,elite_count=6):
 ids=np.argsort(fitness)[-elite_count:]
 selected=parameters[ids]
 return selected.mean(0),np.maximum(selected.std(0),INITIAL_STD*.15),ids
