"""CPU FK native-forward axis audit; not an optical-camera gaze certification."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent;review=json.loads((out/'initial_gate_review.json').read_text());pair=review['latest']
root=ET.parse(Path(review['run'])/'source/physical_v29/training/assets/robot_training.urdf').getroot()
joints={j.find('child').attrib['link']:j for j in root.findall('joint')}
head=next(j for j in root.findall('joint') if j.attrib['name']=='head_pitch').find('child').attrib['link']
chain=[];link=head
while link in joints:
 j=joints[link];chain.insert(0,j);link=j.find('parent').attrib['link']
assert link=='pelvis'
def origin_rotation(j):return Rotation.from_euler('xyz',np.fromstring(j.find('origin').attrib.get('rpy','0 0 0'),sep=' ')).as_matrix() if j.find('origin') is not None else np.eye(3)
zero=np.eye(3)
for j in chain:zero=zero@origin_rotation(j)
native_forward=zero.T@np.array([1.,0.,0.])
assert np.allclose(zero@native_forward,[1.,0.,0.])
rows=[];fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True)
for label in ['nominal','pushed']:
 with np.load(Path(pair[label])/'gait_diagnostics_50hz.npz') as d:
  q=d['q'];names=d['joint_names'].tolist();shape=q.shape[:2];rot=np.tile(np.eye(3),shape+(1,1))
  for j in chain:
   rot=rot@origin_rotation(j)
   if j.attrib['type']=='revolute':
    a=np.fromstring(j.find('axis').attrib['xyz'],sep=' ');a/=np.linalg.norm(a)
    angle=q[:,:,names.index(j.attrib['name'])];rot=rot@Rotation.from_rotvec((angle[:,:,None]*a).reshape(-1,3)).as_matrix().reshape(shape+(3,3))
  r=d['root_state'];quat=r[:,:,3:7];world=Rotation.from_quat(quat[:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(shape+(3,3))@rot
  f=(world@native_forward[...,None])[...,0]
  target=d['heading_target'];yaw=np.arctan2(f[:,:,1],f[:,:,0]);error=np.arctan2(np.sin(yaw-target),np.cos(yaw-target));pitch=np.arctan2(f[:,:,2],np.linalg.norm(f[:,:,:2],axis=-1))
  for speed in d['commands']:
   ids=np.isclose(d['commands'],speed)
   if any(np.isclose(a['command'],speed) and a['test']==label for a in rows):continue
   mask=d['time']>=2;terminal=np.flatnonzero(d['done'][:,ids].any(1))
   if len(terminal):mask&=np.arange(len(mask))<terminal[0]
   if mask.sum()<10:continue
   e,p=np.rad2deg(error[mask][:,ids]),np.rad2deg(pitch[mask][:,ids]);rows.append(dict(test=label,command=float(speed),representatives=int(ids.sum()),samples=int(mask.sum()),world_heading_axis_error_rms_deg=float(np.sqrt((e**2).mean())),world_heading_axis_error_p95_deg=float(np.quantile(abs(e),.95)),world_pitch_rms_deg=float(np.sqrt((p**2).mean())),world_pitch_mean_deg=float(p.mean())))
  if label=='nominal':
   for col,speed in enumerate([1.,2.5,3.,3.5]):
    choices=np.flatnonzero(np.isclose(d['commands'],speed))
    if not len(choices):continue
    robot=choices[0];mask=(d['time']>=2)&(d['time']<=8)
    axes[0,col].plot(d['time'][mask],np.rad2deg(error[mask,robot]));axes[1,col].plot(d['time'][mask],np.rad2deg(pitch[mask,robot]));axes[0,col].set_title(f'{speed}m/s, representative{robot}')
for i in range(2):
 for ax in axes[i]:ax.grid(alpha=.2);ax.set_ylabel('Degrees');ax.set_xlabel('Simulation seconds')
axes[0,0].set_ylabel('World heading axis error (degrees)');axes[1,0].set_ylabel('World axis elevation (degrees)')
fig.suptitle('FK native-zero calibrated head forward axis, truthful measured joints\nIncludes base/waist/neck orientation. This is not calibrated optical-camera gaze.')
fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(out/'world_head_axis.png',dpi=160);plt.close(fig)
(out/'world_head_axis_audit.json').write_text(json.dumps(dict(completed=True,checkpoint=pair['checkpoint'],chain=[j.attrib['name'] for j in chain],native_forward_axis_in_head_link=native_forward.tolist(),rows=rows,scope='16 selected50Hz robots per gate; first episodes segmented. Native-zero forward is calibrated to pelvis+X at zero modeled joints. Does not calibrate CAD face/camera optical axis or prove human gaze; not a locomotion/posture gate.'),indent=2)+'\n')
print(json.dumps([a for a in rows if a['test']=='nominal' and a['command'] in [1.,3.5]],indent=2))
