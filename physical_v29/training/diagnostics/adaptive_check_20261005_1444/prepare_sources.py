import json,shutil,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
record=read(OUT/'initial_evidence.json');run=Path(record['active']['run'])
source=OUT/'offline_source/physical_v29';assert not source.exists()
shutil.copytree(run/'source/physical_v29',source,ignore=shutil.ignore_patterns('__pycache__','*.pyc','usd'))
ground=source/'training/assets/ground';shutil.copytree(OUT/'ground_asset',ground)
files={p.relative_to(ground).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in ground.rglob('*') if p.is_file()}
(ground/'manifest.json').write_text(json.dumps(dict(origin=read(OUT/'ground_download.json')['url'],files=files,
 original_root_sha256=files['default_environment.usd'],mdl='OmniPBR.mdl resolves through bundled Isaac MDL search path; not a remote physical dependency'),indent=2)+'\n')
module='''"""Pin the original NVIDIA ground bytes locally; preserve the original spawner."""
import hashlib,json
from pathlib import Path

def install_local_ground():
    import isaaclab.sim as sim_utils
    root=Path(__file__).parent/'assets/ground'
    manifest=json.loads((root/'manifest.json').read_text())
    for relative,expected in manifest['files'].items():
        path=root/relative
        assert path.resolve().is_relative_to(root.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
    original=sim_utils.spawn_ground_plane
    def spawn(prim_path,cfg,*args,**kwargs):
        assert cfg.usd_path==manifest['origin'],'Unexpected ground asset; never silently substitute physics'
        # Only its resolver path changes. Mesh/collider bytes and every config
        # property, physics material and native Isaac spawner stay identical.
        return original(prim_path,cfg.replace(usd_path=str(root/'default_environment.usd')),*args,**kwargs)
    sim_utils.spawn_ground_plane=spawn
    return manifest
'''
(source/'training/offline_ground.py').write_text(module)
worker=source/'training/worker_unitree.py';data=worker.read_bytes()
token=b" envraw=LocomotionEnv(cfg);envraw.initialize_randomization();env=RslRlVecEnvWrapper(envraw,clip_actions=3.)"
assert data.count(token)==1
insert=b" from offline_ground import install_local_ground\n ground_manifest=install_local_ground()\n (out/'ground_asset_provenance.json').write_text(json.dumps(ground_manifest,indent=2))\n"
worker.write_bytes(data.replace(token,insert+token))
(OUT/'source_preparation.json').write_text(json.dumps(dict(source=str(source),checkpoint=record['checkpoint'],
 checkpoint_sha256=record['checkpoint_sha256'],source_change='worker startup local resolver only; environment/reward/history/physics source unchanged',files=files),indent=2)+'\n')
print(str(source))
