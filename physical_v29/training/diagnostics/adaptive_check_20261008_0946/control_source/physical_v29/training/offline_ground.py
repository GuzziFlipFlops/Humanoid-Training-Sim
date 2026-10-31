"""Pin the original NVIDIA ground bytes locally; preserve the original spawner."""
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
