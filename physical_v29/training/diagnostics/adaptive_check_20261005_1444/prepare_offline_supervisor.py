"""Prepare a strict, evidence-pinned startup-only full-state migration."""
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
b=(HERE/'supervisor_unitree.py').read_text()
token=' if retention_proof:\n  assert not optimization_proof and not curriculum_proof'
assert b.count(token)==1
b=b.replace(token," if optimization_proof and json.loads(Path(optimization_proof).read_text()).get('schema')=='verified_offline_ground_v1':\n  assert not retention_proof and not curriculum_proof\n  return verify_offline_ground_migration(prior,new_source,optimization_proof)\n"+token)
function='''
def verify_offline_ground_migration(prior,new_source,proof_path):
 """Allow only the hashed original ground and its tested resolver shim."""
 from curriculum_provenance import training_reward_signature
 prior=Path(prior);new_source=Path(new_source);proof_path=Path(proof_path).resolve()
 proof=json.loads(proof_path.read_text());state=json.loads((prior/'state.json').read_text())
 assert proof['schema']=='verified_offline_ground_v1'
 assert state['status'] in ('failed','superseded_for_offline_ground') and state.get('child_pid') is None
 assert Path(proof['prior_run']).resolve()==prior.resolve()
 assert Path(proof['checkpoint']).resolve()==Path(state['checkpoint']).resolve()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert sha(state['checkpoint'])==proof['checkpoint_sha256']
 assert sha(__file__)==proof['supervisor_sha256']
 def inventory(root):
  return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()
          and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
 old,new=inventory(prior/'source/physical_v29'),inventory(new_source)
 changed={k for k in old.keys()|new.keys() if k not in old or k not in new or old[k].read_bytes()!=new[k].read_bytes()}
 ground={'training/assets/ground/'+p for p in ['default_environment.usd','manifest.json',
  'Materials/Textures/Wireframe_blue.png','Materials/Textures/WireframeBlur_blue.png','Materials/Textures/WireframeBlur_basecolor.png']}
 allowed=ground|{'training/worker_unitree.py','training/offline_ground.py'}
 assert changed==allowed and set(proof['implementation_hashes'])==allowed,changed
 for k,h in proof['implementation_hashes'].items():assert sha(new[k])==h,k
 assert sha(new['training/assets/ground/default_environment.usd'])=='8a21c317d638d33a4e6c20c958a7ed8f4d8c5195efd7e20f1b85178b4a15638a'
 token=b" envraw=LocomotionEnv(cfg);envraw.initialize_randomization();env=RslRlVecEnvWrapper(envraw,clip_actions=3.)"
 insert=b" from offline_ground import install_local_ground\\n ground_manifest=install_local_ground()\\n (out/'ground_asset_provenance.json').write_text(json.dumps(ground_manifest,indent=2))\\n"
 assert old['training/worker_unitree.py'].read_bytes().count(token)==1
 assert new['training/worker_unitree.py'].read_bytes()==old['training/worker_unitree.py'].read_bytes().replace(token,insert+token)
 assert training_reward_signature(new_source/'training')==proof['reward_signature']
 assert training_reward_signature(prior/'source/physical_v29/training')==proof['reward_signature']
 assert proof['actor_critic_Adam_exploration_retained'] and not proof['training_gradient_repair']
 assert not proof['qualifies_running'] and proof['same_stage']==state['stage']
 for key in ['cpu_tests','physical_validation','full_state_smoke']:
  record=proof[key];p=(proof_path.parent/record['path']).resolve()
  assert p.is_relative_to(proof_path.parent) and sha(p)==record['sha256']
  assert json.loads(p.read_text())['passed'],key
 return state

'''
token='def verify_teacher_context_migration('
assert b.count(token)==1;b=b.replace(token,function+token)
token="  shutil.copytree(HERE/'unitree_vendor',snapshot/'training/unitree_vendor')"
assert b.count(token)==1
b=b.replace(token,"  shutil.copy2(HERE/'offline_ground.py',snapshot/'training/offline_ground.py')\n  shutil.copytree(HERE/'assets/ground',snapshot/'training/assets/ground')\n"+token)
token="  frozen['vram_ramp.py']=hashlib.sha256((HERE/'vram_ramp.py').read_bytes()).hexdigest()"
assert b.count(token)==1
b=b.replace(token,"  frozen['offline_ground.py']=hashlib.sha256((snapshot/'training/offline_ground.py').read_bytes()).hexdigest()\n  for path in (snapshot/'training/assets/ground').rglob('*'):\n   if path.is_file():frozen[path.relative_to(snapshot/'training').as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()\n"+token)
(OUT/'tested_supervisor_unitree.py').write_text(b)
compile(b,str(OUT/'tested_supervisor_unitree.py'),'exec')
print('Prepared strict offline ground supervisor; not deployed')
