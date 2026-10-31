"""Reference foot paths solved against this CAD's actual tilted hip joint axes.

Walking and running use smooth task-space alternating foot-path priors solved
through our FK/IK. The public Asimov recording is research material only;
no source joint angles, motion features or policy weights are copied.
All references are guidance for a trainable residual policy, not success evidence.
"""
import json,math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import find_peaks
HERE=Path(__file__).parent
meta=json.loads((HERE.parent/'assets/export_metadata.json').read_text())
J=meta['joints'];names=[j['name'] for j in J];lookup={n:i for i,n in enumerate(names)}
bychild={j['child']:j for j in J}

def fk(q,link):
 chain=[];a=link
 while a!='pelvis':chain.append(bychild[a]);a=bychild[a]['parent']
 p=np.zeros(3);r=np.eye(3)
 for j in reversed(chain):
  p+=r@j['origin_xyz_m'];r=r@Rot.from_rotvec(np.array(j['axis'])*q[lookup[j['name']]]).as_matrix()
 return p,r

def solve(side,target,rotation,q0):
 ns=[side+'_'+x for x in ['hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_pitch','ankle_roll']]
 ids=[lookup[n] for n in ns];lo=np.array([J[i]['limits_rad'][0]+.003 for i in ids]);hi=np.array([J[i]['limits_rad'][1]-.003 for i in ids])
 def error(v):
  q=q0.copy();q[ids]=v;p,r=fk(q,side+'_foot')
  return np.r_[(p-target)*12,Rot.from_matrix(rotation.T@r).as_rotvec()*1.2,(v-q0[ids])*.015]
 result=least_squares(error,np.clip(q0[ids],lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=65,ftol=1e-7,xtol=1e-7,gtol=1e-7)
 q=q0.copy();q[ids]=result.x;p,r=fk(q,side+'_foot')
 return q,float(np.linalg.norm(p-target)),float(Rot.from_matrix(rotation.T@r).magnitude())

def main():
 stand=np.zeros(23);height=meta['base_height_over_floor_m']-.038
 foot0={s:fk(stand,s+'_foot')[0] for s in ['left','right']}
 for side in ['left','right']:
  target=foot0[side].copy();target[2]+=.038
  # Place ankles slightly behind root to account for forward sole and body COM.
  target[0]-=.015
  stand[lookup[side+'_hip_pitch']]=-.25
  stand[lookup[side+'_knee_pitch']]=.50
  stand[lookup[side+'_ankle_pitch']]=-.25
  stand,err,angle=solve(side,target,np.eye(3),stand);assert err<.005,(side,err)
 stand[lookup['left_elbow_pitch']]=-.30;stand[lookup['right_elbow_pitch']]=-.30
 clips=[];diagnostics=[];heights=[];phases=120
 for speed in [0.,.35,.65,1.,1.5,2.,2.5]:
  duty=.62 if speed<=1.05 else .48 if speed<=1.5 else .43
  frequency=.90+.50*speed; period=1/frequency
  half_stride=min(.29,speed*period*duty/2)
  drop=.038+.022*min(speed,2.5)
  heights.append(meta['base_height_over_floor_m']-drop)
  clearance=.035+.018*min(speed,2.5)
  clip=np.zeros((phases,23));maxerr=0.;maxangle=0.;last=stand.copy()
   # A smooth task-space swing prior; no source joint-angle copying.
  for i in range(phases):
   phase=i/phases;q=stand.copy()
   for s,offset in [('left',0.),('right',.5)]:
    t=(phase+offset)%1.;target=foot0[s].copy();target[2]+=drop;target[0]-=.015
    foot_pitch=0.
    if speed>0:
     if t<duty:
      target[0]+=half_stride*(1-2*t/duty)
      foot_pitch=.36*(t/duty-.5)*min(speed,1.)
     else:
      u=(t-duty)/(1-duty);target[0]+=half_stride*(-1+2*(u*u*(3-2*u)))
      target[2]+=clearance*math.sin(math.pi*u)**2
      foot_pitch=.18*math.cos(math.pi*u)*min(speed,1.)
    q,err,angle=solve(s,target,Rot.from_euler('y',foot_pitch).as_matrix(),q if i==0 else np.where(np.isin(np.arange(23),[lookup[s+'_'+x] for x in ['hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_pitch','ankle_roll']]),last,q))
    maxerr=max(maxerr,err);maxangle=max(maxangle,angle)
   if speed>0:
    amp=min(.32,.12+.08*speed)
    q[lookup['left_shoulder_pitch']]=amp*math.sin(2*math.pi*phase)
    q[lookup['right_shoulder_pitch']]=-amp*math.sin(2*math.pi*phase)
   clip[i]=q;last=q
  clips.append(clip);diagnostics.append(dict(speed=speed,duty=duty,frequency=frequency,foot_clearance_m=clearance,max_foot_position_error_m=maxerr,max_orientation_error_rad=maxangle))
  print(diagnostics[-1],flush=True)
  assert maxerr<.025 and maxangle<.28,diagnostics[-1]
 np.savez(HERE/'assets/motion_reference.npz',speeds=np.array([0,.35,.65,1,1.5,2,2.5]),q=np.stack(clips).astype(np.float32),stand=stand.astype(np.float32),names=np.array(names),root_height=height,heights=np.array(heights))
 (HERE/'motion_validation.json').write_text(json.dumps({'reference_type':'CAD IK gait prior; no copied source joint angles; not dynamically validated motion','root_height':height,'stand':dict(zip(names,stand.tolist())),'clips':diagnostics},indent=2))

if __name__=='__main__':main()
