import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
with np.load(out/'arm_response/telemetry.npz') as d:z={k:d[k] for k in d.files}
xml=ET.parse(out/'control_source/physical_v29/training/assets/robot_training.urdf').getroot()
by_child={j.find('child').attrib['link']:j for j in xml.findall('joint')};link=str(z['head_body_name']);chain=[]
while link in by_child:
 j=by_child[link];chain.insert(0,j);link=j.find('parent').attrib['link']
assert link=='pelvis'
origin=lambda j:Rotation.from_euler('xyz',np.fromstring(j.find('origin').attrib.get('rpy','0 0 0'),sep=' ')).as_matrix() if j.find('origin') is not None else np.eye(3)
zero=np.eye(3)
for j in chain:zero=zero@origin(j)
forward=zero.T@np.array([1.,0.,0.]);names=z['joint_names'].tolist();shape=z['q'].shape[:2]
rot=np.broadcast_to(np.eye(3),shape+(3,3)).copy()
for j in chain:
 rot=rot@origin(j)
 if j.attrib['type']=='revolute':
  axis=np.fromstring(j.find('axis').attrib['xyz'],sep=' ');axis/=np.linalg.norm(axis)
  angle=z['q'][:,:,names.index(j.attrib['name'])]
  rot=rot@Rotation.from_rotvec((angle[:,:,None]*axis).reshape(-1,3)).as_matrix().reshape(shape+(3,3))
q=z['root'][:,:,3:7]
root=Rotation.from_quat(q[:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(shape+(3,3))
expected=root@rot
actual_q=z['head_quaternion']
actual=Rotation.from_quat(actual_q[:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(shape+(3,3))
error=np.rad2deg(Rotation.from_matrix((expected.swapaxes(-1,-2)@actual).reshape(-1,3,3)).magnitude()).reshape(shape)
axis=(actual@forward[...,None])[...,0]
yaw=np.arctan2(axis[:,:,1],axis[:,:,0]);heading=z['reference_heading'][None,:]
heading_error=np.rad2deg(np.arctan2(np.sin(yaw-heading),np.cos(yaw-heading)))
pitch=np.rad2deg(np.arctan2(axis[:,:,2],np.linalg.norm(axis[:,:,:2],axis=-1)))
rows=[];per=int(z['robots_per_group']);window=z['time']>=5
fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
for i,group in enumerate(z['groups']):
 ids=slice(i*per,(i+1)*per);valid=z['alive'][window,ids]
 h=heading_error[window,ids][valid];p=pitch[window,ids][valid]
 rows.append(dict(group=str(group),robots=per,FK_orientation_error_max_deg=float(error[:,ids].max()),world_heading_axis_error_RMS_deg=float(np.sqrt((h*h).mean())),
  world_heading_axis_error_p95_deg=float(np.quantile(abs(h),.95)),world_elevation_RMS_deg=float(np.sqrt((p*p).mean())),
  native_head_angles_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(z['q'][window,ids][:,:,[names.index('head_yaw'),names.index('head_pitch')]]**2,-1)))))))
 if i>=3:
  mask=(z['time']>=3)&(z['time']<=8);robot=i*per
  axs[0,i-3].plot(z['time'][mask],heading_error[mask,robot]);axs[1,i-3].plot(z['time'][mask],pitch[mask,robot]);axs[0,i-3].set_title(str(group))
for r in axs:
 for a in r:a.grid(alpha=.2);a.set_xlabel('Simulation seconds');a.set_ylabel('Degrees')
fig.suptitle('Actual Isaac head-link native-forward axis\nURDF FK parity checked; original face/camera optical axis remains uncalibrated')
fig.tight_layout(rect=[0,0,1,.9]);fig.savefig(out/'actual_head_world_axis.png',dpi=150);plt.close(fig)
record=dict(completed=True,head_body=str(z['head_body_name']),chain=[j.attrib['name'] for j in chain],native_forward_axis=forward.tolist(),
 FK_matches_actual_Isaac_head_orientation_within_point1_degree=bool(error.max()<.1),maximum_rotation_error_deg=float(error.max()),rows=rows,
 scope='192 physical robots/6 diagnostic cohorts, truthful measured joints and actual head-link quaternions at50Hz. Native forward calibrated to pelvis+X at zero URDF joints. CAD face/camera optical axis still unverified, no world-gaze or human-motion certification. Cohort clones are not independent initial-state trials.')
(out/'actual_head_axis_parity.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
