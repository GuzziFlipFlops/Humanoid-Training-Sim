"""Verify path-only config replacement and refuse mismatched/corrupt assets."""
import sys,json,shutil,types,importlib.util,hashlib
from dataclasses import dataclass,replace
from pathlib import Path
OUT=Path(__file__).resolve().parent;source=OUT/'offline_source/physical_v29/training'
fixture=OUT/'cpu_ground_fixture';assert not fixture.exists();fixture.mkdir()
shutil.copytree(source/'assets',fixture/'assets');shutil.copy2(source/'offline_ground.py',fixture/'offline_ground.py')
origin=json.loads((fixture/'assets/ground/manifest.json').read_text())['origin']
@dataclass
class Cfg:
 usd_path:str=origin
 size:tuple=(20000.,20000.)
 physics_material:tuple=(.85,.8,0.)
 color:tuple=(0.,0.,0.)
 visible:bool=True
 def replace(self,**kwargs):return replace(self,**kwargs)
seen=[]
def original(path,cfg,*args,**kwargs):seen.append((path,cfg,args,kwargs));return 'original_spawner'
sim=types.ModuleType('isaaclab.sim');sim.spawn_ground_plane=original
package=types.ModuleType('isaaclab');package.sim=sim
sys.modules['isaaclab']=package;sys.modules['isaaclab.sim']=sim
spec=importlib.util.spec_from_file_location('fixture_offline_ground',fixture/'offline_ground.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
manifest=module.install_local_ground();cfg=Cfg();result=sim.spawn_ground_plane('/World/ground',cfg,(0,0,0),test_argument=5)
assert result=='original_spawner' and cfg.usd_path==origin
path,new,args,kw=seen[-1]
assert new.usd_path==str(fixture/'assets/ground/default_environment.usd')
assert new.size==cfg.size and new.physics_material==cfg.physics_material and new.color==cfg.color and new.visible==cfg.visible
assert path=='/World/ground' and args==((0,0,0),) and kw==dict(test_argument=5)
checks=['original_spawner','original_config_not_mutated','local_path_only','size_unchanged','friction_restitution_unchanged','color_visibility_unchanged','all_args_preserved']
try:sim.spawn_ground_plane('/World/other',Cfg(usd_path='different.usd'))
except AssertionError:checks.append('different_asset_rejected')
else:raise AssertionError('Unexpected asset accepted')
root=fixture/'assets/ground/default_environment.usd';data=root.read_bytes();root.write_bytes(data+b'corruption')
sim.spawn_ground_plane=original
try:module.install_local_ground()
except AssertionError:checks.append('corrupt_asset_rejected')
else:raise AssertionError('Corruption accepted')
root.write_bytes(data);module.install_local_ground();checks.append('exact_restoration_accepted')
record=dict(passed=True,checks=checks,check_count=len(checks),original_root_sha256=hashlib.sha256(data).hexdigest(),
 physics_source_unchanged=True,scope='CPU spawner delegate/config/hash rejection checks; real physics parity is separate.')
(OUT/'offline_ground_cpu_tests.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
