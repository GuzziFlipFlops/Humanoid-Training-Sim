"""Create a separate USD layer; never edits the successful locomotion asset."""
import argparse
import hashlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
from pxr import Sdf, Usd, UsdPhysics, PhysxSchema
from geometry import Geometry, ROOT, URDF

try:
    out = Path(__file__).resolve().parent / 'assets'
    out.mkdir(exist_ok=True)
    dest = out / 'recovery.usda'
    if dest.exists(): raise FileExistsError('Recovery asset already exists; inspect before replacing')
    source = ROOT / 'assets/usd/humanoid_5ft.usd'
    g = Geometry()
    assert not g.penetrations(g.fk()), 'Neutral non-neighbor overlap requires a geometry repair'
    stage = Usd.Stage.CreateNew(str(dest))
    stage.GetRootLayer().subLayerPaths = [source.as_posix()]
    default = stage.GetDefaultPrim()
    if not default:
        base = Usd.Stage.Open(str(source)).GetDefaultPrim()
        default = stage.GetPrimAtPath(base.GetPath())
        stage.SetDefaultPrim(default)
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if set(g.links) != set(bodies): raise ValueError(f'USD/URDF body mismatch: {set(g.links)^set(bodies)}')
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(p).CreateEnabledSelfCollisionsAttr(True)
    for pair in sorted(g.filtered, key=lambda x: sorted(x)):
        a, b = sorted(pair)
        UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())
    stage.GetRootLayer().Save()
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(), output=str(dest),
                  enabled_self_collisions=True, filtered_pairs=[sorted(p) for p in g.filtered],
                  preserved_mass_kg=sum(x['mass'] for x in g.links.values()),
                  neutral_non_neighbor_penetrations=g.penetrations(g.fk()),
                  note='Same box geometry/inertia/limits. Only mechanical-neighbor filtering, all other pairs active.')
    (out / 'collision_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
finally:
    app.app.close()
