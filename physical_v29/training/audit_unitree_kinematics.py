"""Static URDF FK/COM and world-axis audit; never changes physical assets."""
import json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
tree=ET.parse(HERE/'assets/robot_training.urdf').getroot()
motion=np.load(HERE/'assets/motion_reference.npz');names=motion['names'].tolist();stand=dict(zip(names,motion['stand']))
def vec(element,key,default='0 0 0'):
    return np.array([float(v) for v in element.get(key,default).split()]) if element is not None else np.zeros(3)
def rotate(axis,angle):
    a=np.asarray(axis);x,y,z=a;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
def origin(element):
    T=np.eye(4);T[:3,3]=vec(element,'xyz');r,p,y=vec(element,'rpy')
    T[:3,:3]=rotate([0,0,1],y)@rotate([0,1,0],p)@rotate([1,0,0],r);return T
def fk(q):
    transforms={'pelvis':np.eye(4)};axes={};pending=list(tree.findall('joint'))
    while pending:
        progressed=False
        for j in pending[:]:
            parent=j.find('parent').get('link');child=j.find('child').get('link')
            if parent not in transforms:continue
            T=transforms[parent]@origin(j.find('origin'));axis=vec(j.find('axis'),'xyz','1 0 0')
            axes[j.get('name')]=dict(point=T[:3,3].tolist(),axis=(T[:3,:3]@axis).tolist())
            R=np.eye(4);R[:3,:3]=rotate(axis,q.get(j.get('name'),0.))
            transforms[child]=T@R;pending.remove(j);progressed=True
        assert progressed,'URDF disconnected or cyclic'
    return transforms,axes
T,axes=fk(stand);total=0.;com=np.zeros(3)
for link in tree.findall('link'):
    inertia=link.find('inertial')
    if inertia is None:continue
    mass=float(inertia.find('mass').get('value'));total+=mass
    com+=mass*(T[link.get('name')]@np.r_[vec(inertia.find('origin'),'xyz'),1])[:3]
com/=total;com[2]+=float(motion['root_height'])
feet={};bounds=[]
mass_data=json.loads((HERE/'mass_estimate.json').read_text())
for s in ['left','right']:
    link=s+'_foot';box=mass_data['links'][link]['collision_box']
    corners=np.array([np.array(box['center'])+np.array(box['size'])*.5*np.array([x,y,z]) for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    world=(T[link]@np.c_[corners,np.ones(8)].T).T[:,:3];world[:,2]+=float(motion['root_height'])
    feet[s]=dict(origin=(T[link][:3,3]+[0.,0.,float(motion['root_height'])]).tolist(),normal=T[link][:3,2].tolist(),
                 lowest_z=float(world[:,2].min()),corners=world.tolist())
    bounds.append(world[:,:2])
box=np.concatenate(bounds);lo=box.min(0);hi=box.max(0)
jac={}
for name in names:
    if not any(k in name for k in ['hip','knee','ankle','waist']):continue
    altered=dict(stand);altered[name]+=1e-4;Tp,_=fk(altered)
    jac[name]={s:((Tp[s+'_foot'][:3,3]-T[s+'_foot'][:3,3])/1e-4).tolist() for s in ['left','right']}
report=dict(scope='Static nominal-pose FK/COM/axis audit; not dynamic balance or motor strength qualification',
            joint_count=len(names),root_height_m=float(motion['root_height']),mass_kg=total,com_world_m=com.tolist(),feet=feet,
            neutral_com_inside_combined_sole_AABB=bool(np.all(com[:2]>=lo)&np.all(com[:2]<=hi)),
            support_aabb_xy_m=[lo.tolist(),hi.tolist()],world_joint_axes=axes,foot_position_derivatives_m_per_rad=jac)
out=HERE/'diagnostics/unitree_port_20261002';out.mkdir(parents=True,exist_ok=True)
(out/'kinematic_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['world_joint_axes','foot_position_derivatives_m_per_rad','feet']},indent=2))
for name in ['left_hip_pitch','right_hip_pitch','left_knee_pitch','left_ankle_pitch','left_ankle_roll']:
    print(name,axes[name],jac[name])
