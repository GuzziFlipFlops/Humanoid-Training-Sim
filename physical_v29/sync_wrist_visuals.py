"""Refresh only hand-containing visual STLs; preserve joints and actuator settings."""
import gzip,hashlib,json,struct,shutil
from pathlib import Path
import numpy as np
import trimesh

HERE=Path(__file__).resolve().parent
CAD=HERE.parent/'outputs/shin_cad_20260912/v29'
FIX=CAD/'wrist_review_20260930'
verification=json.loads((FIX/'verification.json').read_text())
assert verification['all_pass']
meta_path=HERE/'assets/export_metadata.json';meta=json.loads(meta_path.read_text())
before=json.loads((FIX/'mesh_manifest_before.json').read_text())
after=json.loads((FIX/'mesh_manifest_after.json').read_text())
old={r['i']:r for r in before['bodies']};new={r['i']:r for r in after['bodies']}
changed={i for i in new if old[i]['sha256']!=new[i]['sha256']};assert len(changed)==8
R=np.asarray(meta['coordinate_rotation_cad_to_isaac']);origins={'pelvis':np.asarray(meta['base_native_mm'])}
native=json.loads(Path(meta['native_joint_map_source']).read_text())
origins.update({j['moving_link']:np.asarray(j['origin_mm']) for j in native if j['anatomical']})
archive=HERE/'analysis/20260930_actuators/revisions/wrist_visuals_before';archive.mkdir(parents=True,exist_ok=True)
if not (archive/'export_metadata.json').exists():shutil.copy2(meta_path,archive/'export_metadata.json')
records=[]
for visual in meta['visuals']:
    if not changed.intersection(visual['sources']):continue
    path=HERE/'assets/meshes'/visual['filename']
    if not (archive/path.name).exists():shutil.copy2(path,archive/path.name)
    vv=[];ff=[];offset=0
    for i in visual['sources']:
        row=new[i];data=gzip.decompress((CAD/'mesh_cache'/row['cache']).read_bytes())
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        nv,nf=struct.unpack('<II',data[4:12])
        vertices=np.frombuffer(data,dtype='<f4',count=nv*3,offset=12).reshape(-1,3).astype(float)*10
        faces=np.frombuffer(data,dtype='<u4',count=nf*3,offset=12+nv*12).reshape(-1,3).astype(np.int64)
        assert max(abs(vertices.min(0)-row['bounds_mm'][0]).max(),abs(vertices.max(0)-row['bounds_mm'][1]).max())<.6
        vv.append((vertices-origins[visual['link']])@R.T/1000);ff.append(faces+offset);offset+=nv
    mesh=trimesh.Trimesh(vertices=np.vstack(vv),faces=np.vstack(ff),process=False)
    mesh.export(path)
    records.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sources':visual['sources'],'bounds_m':mesh.bounds.tolist()})
meta['wrist_alignment']={'native_archive':verification['archive'],'native_archive_sha256':verification['sha256'],'verification':str(FIX/'verification.json'),'eight_hand_meshes_aligned':True,'joint_frames_unchanged':True,'native_source_saved':True,'collision_proxy_note':'Baseline conservative boxes remain disabled in movement tests; contact qualification pending.'}
meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
(FIX/'simulation_visual_sync.json').write_text(json.dumps({'changed_mesh_sources':sorted(changed),'updated_STLs':records,'joint_frames_and_actuator_configuration_unchanged':True},indent=2),encoding='utf-8')
print(json.dumps({'changed_hand_meshes':len(changed),'updated_visual_STLs':len(records),'joint_frames_unchanged':True},indent=2))
