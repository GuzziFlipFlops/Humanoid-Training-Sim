"""Quaternion kinematics and frame/timestamp parity, CPU only."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;F=OUT/'timing_response'
def mul(a,b):
    w=a[...,:1]*b[...,:1]-np.sum(a[...,1:]*b[...,1:],axis=-1,keepdims=True)
    v=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])
    return np.concatenate([w,v],axis=-1)
def delta_velocity(q,dt):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    b=q[:-1].copy();b[...,1:]*=-1
    d=mul(q[1:],b);d=np.where(d[...,:1]<0,-d,d)
    length=np.linalg.norm(d[...,1:],axis=-1,keepdims=True)
    return d[...,1:]*(2*np.arctan2(length,d[...,:1])/np.maximum(length,1e-15))/dt
def yaw_pitch(q):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)),np.arcsin(np.clip(2*(w*y-z*x),-1,1))
def kinematic_heading(q,omega):
    yaw,pitch=yaw_pitch(q)
    return omega[...,2]+np.tan(pitch)*(omega[...,0]*np.cos(yaw)+omega[...,1]*np.sin(yaw))
# Independent analytic/finite-difference fixtures; include sign continuity and tilt.
t=np.arange(2001)*.0001
for tilt in [0.,.25,-.35]:
    q=Rotation.from_euler('ZYX',np.stack([.7*t,np.full_like(t,tilt),np.full_like(t,.2)],-1)).as_quat()[:,[3,0,1,2]]
    v=delta_velocity(q,.0001)
    assert np.max(np.abs(v[:,2]-.7))<1e-9 and np.max(abs(v[:,:2]))<1e-9
    assert np.max(abs(kinematic_heading(q[1:],v)-.7))<1e-9
q=Rotation.from_euler('ZYX',np.stack([.4*t+.1*np.sin(3*t),.2*np.sin(2*t),.3*np.cos(t)],-1)).as_quat()[:,[3,0,1,2]]
v=delta_velocity(q,.0001);fd=np.diff(np.unwrap(yaw_pitch(q)[0]))/.0001
assert np.max(abs(kinematic_heading(q[1:],v)-fd))<1e-4
flipped=q.copy();flipped[::2]*=-1
assert np.max(abs(delta_velocity(flipped,.0001)-v))<1e-9
with np.load(F/'root_kinematics_200hz.npz') as z:d={k:z[k].copy() for k in z.files}
with np.load(F/'telemetry.npz') as z:control={k:z[k].copy() for k in z.files}
result=json.loads((F/'result.json').read_text());root=d['root'].astype(float);dt=float(d['dt']);per=int(d['robots_per_group'])
assert len(root)==4000 and abs(dt-.005)<1e-12
assert np.allclose(np.diff(d['timestamp']),dt,atol=1e-10,rtol=0)
assert np.array_equal(np.diff(d['counter']),np.ones(3999,dtype=int))
parity=dict(root_link_pose_vs_direct_PhysX_max=float(abs(root[:,:,:7]-d['raw_pose']).max()),
    root_COM_velocity_vs_direct_PhysX_max=float(abs(root[:,:,7:]-d['raw_velocity']).max()),
    final_substep_vs_control_root_max=float(abs(d['root'][3::4]-control['root']).max()),
    repeated_action_command_per_four_substeps_max=float(abs(d['action_command'].reshape(1000,4,-1,3)-control['action_command'][:,None]).max()))
assert max(parity.values())<1e-6,parity
omega=root[:,:,10:13];q=root[:,:,3:7];rv=delta_velocity(q,dt)
yaw,pitch=yaw_pitch(q);heading_fd=np.diff(np.unwrap(yaw,axis=0),axis=0)/dt
analytic=kinematic_heading(q,omega);rows=[]
time=np.arange(1,4001)*dt;window=time[1:]>=5.
for i,name in enumerate(d['groups'].tolist()):
    ids=slice(i*per,(i+1)*per);post=omega[1:,ids];pre=omega[:-1,ids];mid=(pre+post)*.5
    lag=[]
    for offset in range(-4,5):
        j=np.arange(1000,3998);k=j+offset;valid=(k>=0)&(k<len(omega));j=j[valid];k=k[valid]
        lag.append(dict(offset_steps=offset,RMS_rad_s=float(np.sqrt(np.mean((rv[j,ids]-omega[k,ids])**2)))))
    sig=d['action_command'][1000:,ids,2].mean(1);f=np.fft.rfftfreq(len(sig),dt);power=abs(np.fft.rfft(sig-sig.mean()))**2
    rows.append(dict(group=name,world_rotation_delta_vs_post_gyro_RMS_rad_s=float(np.sqrt(np.mean((rv[window,ids]-post[window])**2))),
      world_rotation_delta_vs_pre_gyro_RMS_rad_s=float(np.sqrt(np.mean((rv[window,ids]-pre[window])**2))),
      world_rotation_delta_vs_midpoint_gyro_RMS_rad_s=float(np.sqrt(np.mean((rv[window,ids]-mid[window])**2))),
      heading_fd_vs_post_analytic_RMS_rad_s=float(np.sqrt(np.mean((heading_fd[window,ids]-analytic[1:,ids][window])**2))),
      heading_fd_vs_midpoint_analytic_RMS_rad_s=float(np.sqrt(np.mean((heading_fd[window,ids]-(analytic[:-1,ids]+analytic[1:,ids])[window]*.5)**2))),
      mean_heading_fd_rad_s=float(heading_fd[window,ids].mean()),mean_world_gyro_z_rad_s=float(post[window,:,2].mean()),
      dominant_yaw_command_Hz=float(f[np.argmax(np.where(f>.3,power,0))]),
      yaw_command_variation_RMS_rad_s=float(np.std(d['action_command'][1000:,ids,2],axis=0).mean()),
      quaternion_norm_max_error=float(abs(np.linalg.norm(q[:,ids],axis=-1)-1).max()),lag_sweep=lag))
    # The same trajectory sampled at20ms; compare endpoint versus actual interval samples.
    q20=q[3::4,ids];fd20=np.diff(np.unwrap(yaw_pitch(q20)[0],axis=0),axis=0)/.02
    a20=analytic.reshape(1000,4,-1)[:,:,ids]
    w20=control['time'][1:]>=5.
    rows[-1].update(heading_20ms_fd_vs_endpoint_gyro_RMS_rad_s=float(np.sqrt(np.mean((fd20[w20]-a20[1:,3][w20])**2))),
        heading_20ms_fd_vs_four_post_substep_gyro_mean_RMS_rad_s=float(np.sqrt(np.mean((fd20[w20]-a20[1:].mean(1)[w20])**2))))
diag=dict(completed=True,synthetic_fixtures_passed=5,checkpoint=result['checkpoint'],sha256=result['checkpoint_sha256'],
    dt_s=dt,substeps=len(root),parity=parity,groups=rows,
    actual_command_difference=result['maximum_command_difference'],
    interpretation='All states copied after the same5ms scene update; actor/link quaternion and world COM angular velocity checked against direct PhysX. Delta quaternion gives interval-average world angular velocity; endpoint/midpoint comparisons do not identify a policy delay or justify a deployed clock change. Angular velocity at the link and COM is physically equal; linear velocities differ by rigid offset. No production observations altered.',
    primary_documentation='https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html',
    local_source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT.parents[3]/'IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py',OUT.parents[3]/'IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py']})
(OUT/'timing_analysis.json').write_text(json.dumps(diag,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
for col,law in enumerate(['baseline','instant']):
    i=d['groups'].tolist().index('seed4191_3.5_'+law);robot=i*per;m=(time>=8)&(time<=10)
    axes[0,col].plot(time[m],analytic[m,robot],label='analytic heading rate from gyro')
    axes[0,col].plot(time[1:][m[1:]],heading_fd[m[1:],robot],alpha=.7,label='5ms quaternion heading difference')
    axes[1,col].plot(time[m],d['action_command'][m,robot,2],label='actual action yaw command')
    axes[1,col].plot(time[m],yaw[m,robot],label='actual quaternion heading(rad)')
    axes[0,col].set_title(law+'; one representative robot')
    for ax in axes[:,col]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[1,col].set_xlabel('Post-physics simulated time(s)')
axes[0,0].set_ylabel('rad/s');axes[1,0].set_ylabel('rad or rad/s as labelled')
fig.suptitle('Actual5ms heading/gyro/command alignment — diagnostic current actor')
fig.tight_layout();fig.savefig(OUT/'timing_alignment.png',dpi=160);plt.close(fig)
print(json.dumps(dict(parity=parity,synthetic_fixtures=5,groups=rows),indent=2))
