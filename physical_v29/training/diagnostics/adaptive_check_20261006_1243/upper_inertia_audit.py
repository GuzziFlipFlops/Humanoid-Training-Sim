"""URDF locked-subtree shoulder inertia; descriptive, not system ID."""
import json,xml.etree.ElementTree as ET,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
OUT=Path(__file__).resolve().parent;p=OUT/'control_source/physical_v29/training/assets/robot_training.urdf';root=ET.parse(p).getroot()
d=np.load(OUT/'steering_response/telemetry.npz');names=d['joint_names'].tolist();q=dict(zip(names,d['q'][0,0]))
links={l.attrib['name']:l for l in root.findall('link')};joints={j.attrib['name']:j for j in root.findall('joint')};children={}
def origin(e):
 t=np.eye(4)
 if e is not None:
  t[:3,3]=np.fromstring(e.attrib.get('xyz','0 0 0'),sep=' ');t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(e.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
 return t
for j in joints.values():children.setdefault(j.find('parent').attrib['link'],[]).append(j)
poses={'pelvis':np.eye(4)};jointposes={}
def walk(parent):
 for j in children.get(parent,[]):
  jt=poses[parent]@origin(j.find('origin'));jointposes[j.attrib['name']]=jt
  rot=np.eye(4)
  if j.attrib['type']=='revolute':
   ax=np.fromstring(j.find('axis').attrib['xyz'],sep=' ');rot[:3,:3]=Rotation.from_rotvec(ax*q[j.attrib['name']]).as_matrix()
  child=j.find('child').attrib['link'];poses[child]=jt@rot;walk(child)
walk('pelvis');assert len(poses)==len(links)
def subtree(name):
 return [name]+[k for j in children.get(name,[]) for k in subtree(j.find('child').attrib['link'])]
rows=[]
for side in ['left','right']:
 name=side+'_shoulder_pitch';j=joints[name];t=jointposes[name];a=t[:3,:3]@np.fromstring(j.find('axis').attrib['xyz'],sep=' ');a/=np.linalg.norm(a);m=0.;inertia=0.;details=[]
 for child in subtree(j.find('child').attrib['link']):
  i=links[child].find('inertial')
  if i is None:continue
  mass=float(i.find('mass').attrib['value']);ip=poses[child]@origin(i.find('origin'));v=i.find('inertia').attrib
  local=np.array([[float(v['ixx']),float(v['ixy']),float(v['ixz'])],[float(v['ixy']),float(v['iyy']),float(v['iyz'])],[float(v['ixz']),float(v['iyz']),float(v['izz'])]])
  world=ip[:3,:3]@local@ip[:3,:3].T;contribution=float(a@world@a+mass*np.linalg.norm(np.cross(ip[:3,3]-t[:3,3],a))**2)
  m+=mass;inertia+=contribution;details.append(dict(link=child,mass_kg=mass,shoulder_pitch_inertia_kg_m2=contribution))
 rows.append(dict(side=side,subtree_mass_kg=m,locked_subtree_pitch_inertia_kg_m2=inertia,uncoupled_35Nm_rad_PD_natural_frequency_Hz=float(np.sqrt(35./inertia)/(2*np.pi)),uncoupled_4Nms_rad_PD_damping_ratio=float(4./(2*np.sqrt(35.*inertia))),links=details))
result=dict(completed=True,urdf_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pose='Truthful sampled initial control pose at20ms; all descendants locked for the inertia sum.',rows=rows,mass_relative_difference=abs(rows[0]['subtree_mass_kg']-rows[1]['subtree_mass_kg'])/rows[0]['subtree_mass_kg'],inertia_relative_difference=abs(rows[0]['locked_subtree_pitch_inertia_kg_m2']-rows[1]['locked_subtree_pitch_inertia_kg_m2'])/rows[0]['locked_subtree_pitch_inertia_kg_m2'],scope='Provisional URDF mass/inertia only. Locked-subtree PD analogy excludes free elbow/roll/base coupling, controller history, actuator dynamics and gravity; descriptive frequencies do not establish a causal resonance or certify hardware. No runtime or gain change.')
(OUT/'upper_inertia_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
