"""CPU-only visual/physics parity and saved actor selection for GUI playback."""
import json,hashlib,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def sig(e):return None if e is None else (e.tag,tuple(sorted(e.attrib.items())),(e.text or '').strip(),tuple(sig(c) for c in e))
run=Path(read(HERE/'ACTIVE_RUN.json')['run']);train=run/'source/physical_v29/training'
assert run.name=='locomotion_20261004_r34_bilateral_swing_unloading'
original=train/'assets/robot_training.urdf'
prepared=HERE/'diagnostics/video_body_review_20261004/playback_asset/robot_training_full_visuals.urdf'
a=ET.parse(original).getroot();b=ET.parse(prepared).getroot()
assert [sig(j) for j in a.findall('joint')]==[sig(j) for j in b.findall('joint')]
assert len(a.findall('link'))==len(b.findall('link'))==24
for old,new in zip(a.findall('link'),b.findall('link')):
    assert old.attrib==new.attrib
    assert [sig(c) for c in old if c.tag!='visual']==[sig(c) for c in new if c.tag!='visual']
meshes=[]
for link in b.findall('link'):
    for v in link.findall('visual'):
        m=v.find('geometry/mesh');assert m is not None
        p=Path(m.get('filename'));assert p.is_file()
        meshes.append({'link':link.get('name'),'visual':v.get('name'),'file':str(p),'sha256':sha(p),
                       'origin':dict(v.find('origin').attrib) if v.find('origin') is not None else {},'scale':m.get('scale')})
assert len(meshes)==140
asset=OUT/'playback_asset';asset.mkdir(exist_ok=True)
dest=asset/'robot_training_full_visuals.urdf';assert not dest.exists()
shutil.copy2(prepared,dest)
e=read(OUT/'evidence.json');gates=e['gates'];pairs=[]
for n in gates:
    if not n['folder'].endswith('nominal'):continue
    p=next((x for x in gates if x['folder']==n['folder'].replace('_nominal','_push') and x['checkpoint']==n['checkpoint']),None)
    if p:pairs.append((n,p))
good=[(n,p) for n,p in pairs if all(next(c for c in x['cases'] if c['command']==1.5)['passed'] for x in (n,p))]
assert good,'Preserved independently passed running actor required'
nom,push=good[-1];fast=Path(nom['checkpoint'])
slow=run/'002_ppo_smoke/transferred_actor.pt';assert slow.is_file()
clips=[]
for speed in [.15,.35,.65,1.,1.5]:
    checkpoint=slow if speed<1.5 else fast
    receipts=[str(run/n['folder']) for n in (nom,push)] if speed==1.5 else [str(run/'015_walk_100_gate_001_nominal'),str(run/'016_walk_100_gate_001_push')]
    for receipt in receipts:
        r=read(Path(receipt)/'result.json');cfg=read(Path(receipt)/'run_config.json')
        assert Path(cfg['arguments']['checkpoint'])==checkpoint
        case=next(c for c in r['cases'] if abs(c['statistics']['command_m_s']-speed)<1e-6)
        assert case['passed'] and case['statistics']['trials']>=32
    clips.append({'speed':speed,'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'gate_receipts':receipts,
                  'scope':'Qualified walking baseline' if speed<1.5 else 'Passed running case; whole stage unqualified due slow retention'})
proof={'recorded_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'training':str(train),
       'visual_urdf':str(dest),'visual_sha256':sha(dest),'training_urdf_sha256':sha(original),
       'joints_inertias_collisions_limits_equal':True,'mesh_count':len(meshes),'meshes':meshes,
       'clips':clips,'destination':str(HERE/'videos/full_body_gui_20261004_2340'),
       'no_higher_speed_independently_demonstrated':True,'gui_conversion_rendering_still_required':True}
(OUT/'capture_config.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in proof.items() if k!='meshes'},indent=2))
