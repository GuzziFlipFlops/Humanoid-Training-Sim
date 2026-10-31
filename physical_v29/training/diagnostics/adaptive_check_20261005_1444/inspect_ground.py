import argparse,json,urllib.request,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(args).app
from pxr import Sdf,Usd,UsdPhysics,UsdGeom
OUT=Path(__file__).resolve().parent;root=OUT/'ground_asset/default_environment.usd'
layer=Sdf.Layer.FindOrOpen(str(root));assert layer
layer.Export(str(OUT/'ground_original.usda'))
stage=Usd.Stage.Open(str(root))
physics=[];assets=[]
for prim in stage.Traverse():
 if prim.HasAPI(UsdPhysics.CollisionAPI):
  physics.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),apis=list(prim.GetAppliedSchemas()),
    attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes()}))
 for a in prim.GetAttributes():
  v=a.Get()
  if isinstance(v,Sdf.AssetPath):assets.append(dict(prim=str(prim.GetPath()),attribute=a.GetName(),path=v.path))
base=json.loads((OUT/'ground_download.json').read_text())['url'].rsplit('/',1)[0]+'/'
downloads=[]
for path in sorted(set(v['path'] for v in assets)):
 if path.startswith('http') or not path:continue
 target=root.parent/path;target.parent.mkdir(parents=True,exist_ok=True)
 url=urllib.parse.urljoin(base,path)
 try:
  with urllib.request.urlopen(url,timeout=20) as r:data=r.read()
  target.write_bytes(data);downloads.append(dict(path=path,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 except Exception as e:downloads.append(dict(path=path,url=url,error=repr(e)))
record=dict(physics=physics,assets=assets,downloads=downloads,root_sha256=hashlib.sha256(root.read_bytes()).hexdigest())
(OUT/'ground_inspection.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2));app.close()
