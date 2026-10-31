"""Provisional component mass estimate and lightweight contact asset, never edits CAD.

CAD appearance/material placeholders are not trusted as analysis densities.
Purchased parts use catalog/planning masses; print volumes use explicit fill
fractions. Inertias use per-part box approximations and the parallel-axis theorem.
"""
import json, re, csv, copy, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

HERE=Path(__file__).parent
SOURCE=HERE.parent/'assets'
CAD=Path('E:/Aditya/robot_sim/outputs/shin_cad_20260912/v29')
meta=json.loads((SOURCE/'export_metadata.json').read_text())
rows=json.loads((CAD/'mesh_manifest.json').read_text())['bodies']
R=np.array(meta['coordinate_rotation_cad_to_isaac'])
origins={'pelvis':np.array(meta['base_native_mm'])}
pending=meta['joints'].copy()
while pending:
 for j in pending[:]:
  if j['parent'] in origins:
   origins[j['child']]=origins[j['parent']]+R.T@np.array(j['origin_xyz_m'])*1000
   pending.remove(j)

def owner(link):
 if not link.startswith('mechanism_'): return link
 if link.startswith('mechanism_wj_drive_'): return 'torso'
 if link.startswith('mechanism_wj_'): return 'waist_yaw_link'
 return ('left' if 'left_' in link else 'right')+'_shin'

def category(b):
 s=(b['part']+' '+b['body']+' '+b['appearance']).lower()
 p=b['part'].lower(); appearance=b['appearance'].lower()
 if 'battery' in s and 'envelope' in s: return 'battery'
 if b['mesh_body']: return 'hand'
 if 'rds51150_case' in s: return 'elbow_servo'
 if 'gearbox_shell' in p: return 'aluminum'
 if 'feetech_' in p and p.endswith('_case'): return 'motor'
 if any(k in p for k in ['dji10018','dji_10018','gwk7010','gwk6010','sts3215']): return 'motor'
 if 'vendor_output' in b['body'].lower(): return 'motor'
 if 'aiview_official' in s: return 'camera'
 if 'electronics' in s or 'odrive_board' in s: return 'electronics_envelope'
 if 'printed structural' in appearance: return 'printed_structure'
 if 'printed cosmetic' in appearance or 'pale shell' in appearance: return 'printed_shell'
 if 'machined aluminum' in appearance: return 'aluminum'
 if any(k in s for k in ['replaceable_sole','rubber']) or re.search(r'(^|[^a-z])tpu([^a-z]|$)',s): return 'rubber'
 if any(k in s for k in ['steel hardware','steel_crush','continuous_oval_steel','oval_steel','threaded_link_rod','sprocket','bolt','washer','nut_','fastener','bearing','shaft','pin_8','m3_output_reference']): return 'steel'
 if any(k in s for k in ['aluminum','aluminium','cnc','machined','motor_plate','foot_plate','metal_spine','spine_rod','oval_spine_rail']): return 'aluminum'
 if any(k in s for k in ['shell','c27_','c28_','pale','blue accent','cosmetic']): return 'printed_shell'
 if 'steel - satin' in b['appearance'].lower():
  if any(k in s for k in ['frame','waist','holder','base','cradle','housing']): return 'printed_structure'
  return 'aluminum'
 return 'printed_structure'

def main():
 # kg per cubic metre, with print effective solid fraction. No CAD default steel.
 densities={'steel':7850.,'aluminum':2700.,'printed_shell':1200.,'printed_structure':1150.*.85,'rubber':1150.}
 parts=[]; catalog=defaultdict(list)
 for b in rows:
  cat=category(b); link=owner(b['link']); bounds=np.array(b['bounds_mm'])
  center=R@((bounds[0]+bounds[1])/2-origins[link])/1000
  size=(bounds[1]-bounds[0])/1000;size=np.abs(R)@size
  if cat in densities:
   m=(b.get('volume_mm3') or 0)*1e-9*densities[cat]
   parts.append(dict(link=link,part=b['part'],body=b['body'],category=cat,mass_kg=m,center=center.tolist(),size=size.tolist(),basis='CAD solid volume x explicit intended density/fill'))
  else: catalog[(cat,b['part'])].append((b,link,center,size))
 for (cat,part),bs in catalog.items():
  if cat=='motor':
   p=part.lower()
   if '6010' in p: m=.03 if 'rotating_hub' in p else .27 if 'stator' in p else .30
   elif '7010' in p: m=.334
   elif '10018' in p: m=.765
   elif 'sts3215' in p or 'feetech_' in p: m=.055
   else: m=.03 # vendor moving output, split from stator allowance below
  elif cat=='battery': m=36*.072+.20
  elif cat=='elbow_servo': m=.320
  elif cat=='camera': m=.100
  elif cat=='hand': m=.160 # four material groups: printed, cosmetic, servos and fasteners per hand
  elif cat=='electronics_envelope':
   # Printed electronics box/lid is geometry, board envelopes are assumed purchase masses.
   if 'e24_electronics' in part.lower():
    vol=sum((b.get('volume_mm3') or 0) for b,_,_,_ in bs);m=vol*1e-9*1150*.85
   else:m=.045
  else: raise AssertionError(cat)
  # Purchased assemblies spanning output and stator remain distributed by modeled volume.
  v=np.array([max(b.get('volume_mm3') or 0,1.) for b,_,_,_ in bs]); v=v/v.sum()
  for w,(b,link,center,size) in zip(v,bs):
   parts.append(dict(link=link,part=part,body=b['body'],category=cat,mass_kg=float(m*w),center=center.tolist(),size=size.tolist(),basis='Catalog or explicit planning assembly mass; not solid metal volume'))
 # Additional unmodeled gears, electronics and wiring. Their uncertainty dominates.
 for side in ['left','right']:
  for link,m,part in [(side+'_hip_pitch_link',.35,'Hip roll cycloidal internals allowance'),('pelvis',.35,'Hip pitch cycloidal internals allowance'),(side+'_thigh',.25,'Knee planetary internals allowance'),(side+'_thigh',.10,'Knee chain and tensioner allowance'),('torso',.10,'Additional shoulder pitch reduction allowance'),(side+'_shin',.08,'Leg wiring and encoder allowance'),(side+'_upper_arm',.04,'Arm wiring allowance')]:
   parts.append(dict(link=link,part=part,body='unmodeled',category='allowance',mass_kg=m,center=[0.,0.,0.],size=[.06,.06,.03],basis='Unweighed planning allowance'))
 for link,m,part,c in [('torso',.60,'Host, motor driver boards, regulators and torso harness',[0,0,.20]),('head',.03,'Head wiring',[0,0,.04])]:
  parts.append(dict(link=link,part=part,body='unmodeled',category='allowance',mass_kg=m,center=c,size=[.12,.12,.06],basis='Unweighed planning allowance'))
 root=ET.parse(SOURCE/'humanoid_physical_v29.urdf').getroot()
 links={}
 for le in root.findall('link'):
  link=le.attrib['name'];p=[x for x in parts if x['link']==link]
  m=sum(x['mass_kg'] for x in p); assert m>.005,(link,m)
  com=sum(x['mass_kg']*np.array(x['center']) for x in p)/m
  I=np.zeros((3,3))
  for x in p:
   a=x['mass_kg'];s=np.array(x['size']);d=np.array(x['center'])-com
   I+=np.diag(a/12*np.array([s[1]**2+s[2]**2,s[0]**2+s[2]**2,s[0]**2+s[1]**2]))+a*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
  I+=np.eye(3)*1e-6
  inert=le.find('inertial');inert.find('mass').set('value',str(m));inert.find('origin').set('xyz',' '.join(map(str,com)))
  ie=inert.find('inertia')
  for k,v in [('ixx',I[0,0]),('iyy',I[1,1]),('izz',I[2,2]),('ixy',I[0,1]),('ixz',I[0,2]),('iyz',I[1,2])]:ie.set(k,str(v))
  links[link]={'mass_kg':m,'com_m':com.tolist(),'inertia_kg_m2':I.tolist()}
  for e in le.findall('visual')+le.findall('collision'):le.remove(e)
  # Geometry envelopes derived from appropriate named CAD skins/sole, not linkwide bbox.
  candidates=[b for b in rows if owner(b['link'])==link and (
   ('_foot' in link and 'replaceable_sole' in b['part'].lower()) or
   (link=='torso' and ('torso_' in b['part'].lower() and 'c27' in b['part'].lower())) or
   (link=='pelvis' and 'waist_' in b['part'].lower() and 'c28' in b['part'].lower()) or
   (link=='head' and ('head_shell' in b['part'].lower() or 'defined_vision_head' in b['part'].lower())) or
   (link.endswith(('_thigh','_shin','_upper_arm','_forearm')) and ('c27' in b['part'].lower()) and 'shell' in b['appearance'].lower())
  )]
  # C27 Pale shell appearance does not include literal shell; select component by link.
  if not candidates and link.endswith(('_thigh','_shin','_upper_arm','_forearm')):
   candidates=[b for b in rows if owner(b['link'])==link and b['part'].startswith('C27_')]
  if candidates:
   bb=np.array([b['bounds_mm'] for b in candidates]);mn=bb[:,0].min(0);mx=bb[:,1].max(0)
   c=R@((mn+mx)/2-origins[link])/1000;s=np.abs(R)@(mx-mn)/1000
   if link!='head' and not link.endswith('_foot'):
    s*=np.array([.78,.78,.87]) # conservative inner silhouette excludes seams and adjacent joint lips
   if link=='pelvis':s=np.minimum(s,[.20,.24,.12])
   if link=='torso':s=np.minimum(s,[.21,.29,.43])
   if link=='head':s*=.90
   rpy=[0,0,0]
  else:
   c=np.array([0,0,0.]);s=np.array([.036,.036,.025]);rpy=[0,0,0]
   if 'shoulder_pitch_link' in link:s=np.array([.075,.09,.07])
   # These are hinge-origin markers, not motor/housing envelopes. Pitch pivots
   # are close to the sagittal plane; large centered cubes collided across hips.
   # A small marker retains collision reporting without inventing broad geometry.
   if 'hip_pitch_link' in link or 'hip_roll_link' in link:s=np.array([.018,.018,.018])
  for typ in ['visual','collision']:
   e=ET.SubElement(le,typ,name=link+'_provisional_'+typ);ET.SubElement(e,'origin',xyz=' '.join(map(str,c)),rpy='0 0 0');g=ET.SubElement(e,'geometry');ET.SubElement(g,'box',size=' '.join(map(str,s)))
   if typ=='visual':
    mat=ET.SubElement(e,'material',name=link+'_mat');rgba=[.82,.86,.90,1.] if candidates else [.18,.24,.28,1.];ET.SubElement(mat,'color',rgba=' '.join(map(str,rgba)))
  links[link]['collision_box']={'center':c.tolist(),'size':s.tolist()}
 HERE.joinpath('assets').mkdir(exist_ok=True)
 urdf=HERE/'assets/robot_training.urdf';ET.indent(root);ET.ElementTree(root).write(urdf,encoding='utf-8',xml_declaration=True)
 categories=defaultdict(float)
 for p in parts:categories[p['category']]+=p['mass_kg']
 total=sum(x['mass_kg'] for x in links.values())
 report=dict(status='PROVISIONAL_COMPONENT_ESTIMATE_NOT_WEIGHED',total_mass_kg=total,planning_range_kg=[total*.85,total*1.22],links=links,categories=dict(categories),parts=parts,
             sources={'WGG6010_8_g':260.5,'GIM6010_8_g':370,'selected_6010_planning_g':300,'WGG7010_8_g':334,'10018_reference_g':765,'10018_source':'https://www.rexon.net/en/drone.php?act=view&no=2'},
             limitations=['GIM/GWK/WGG variant unresolved. 6010 planning mass 300g spans supplier 260.5-370g.','10018 765g is manufacturer-family reference, not a weighed modified motor.','Printed structures 85% effective solid, cosmetic skins solid thin-wall volume. No slicer mass.','Existing bearing envelopes treated as steel volume; exact supplier masses pending.','Gearset and electronics allowances are assumptions.','Part COMs and inertias use CAD bounding boxes and parallel axis; not native exact inertia.','Lightweight collisions underapproximate casing seam details; hardware contact qualification remains pending.'])
 (HERE/'mass_estimate.json').write_text(json.dumps(report,indent=2))
 with (HERE/'mass_by_link.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['link','mass_kg']);w.writerows((k,v['mass_kg']) for k,v in links.items())
 print(json.dumps(dict(total_mass_kg=total,categories=dict(categories),links={k:round(v['mass_kg'],3) for k,v in links.items()}),indent=2))

if __name__=='__main__':main()
