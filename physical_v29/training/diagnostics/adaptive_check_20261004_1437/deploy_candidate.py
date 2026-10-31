"""Deploy only the tested running feedback and stricter flight classifier."""
import hashlib,json,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(OUT/'cpu_tests.json')['passed']
smoke=read(OUT/'running_ppo_smoke/result.json')
reload=read(OUT/'running_checkpoint_reload/result.json')
assert smoke['completed'] and smoke['iterations']==4 and reload['completed']
progress=[json.loads(line) for line in (OUT/'running_ppo_smoke/progress.jsonl').read_text().splitlines()]
assert len(progress)==4
assert all(p['ppo']['pre_update_kl']==0. and p['behavior']['action_command_difference']==0. for p in progress)
assert read(OUT/'running_ppo_smoke/warm_start.json')['actor_tensors_equal_before_exploration_reset']
assert all(c['statistics']['survival']==1. for c in reload['cases'])
assert all('geometric_flight_fraction' in c['statistics'] for c in reload['cases'])
old=Path(read(OUT/'drain.json')['run'])/'source/physical_v29'
staged=OUT/'validation_source/physical_v29'
names={'training/stage_rewards.py','training/evaluation.py','training/reward_warm_start.py'}
differences=[];protected=0
for p in old.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.usd') or 'usd' in p.parts:continue
    rel=p.relative_to(old).as_posix();q=staged/rel
    assert q.is_file(),rel
    if p.read_bytes()!=q.read_bytes():differences.append(rel)
    else:protected+=1
assert set(differences)==names,differences
proof={'smoke':smoke,'reload_completed':True,'reload_diagnostic_only':'64 environments, 4 seconds; not qualification',
       'pre_update_kl':[p['ppo']['pre_update_kl'] for p in progress],
       'action_command_difference':[p['behavior']['action_command_difference'] for p in progress],
       'protected_authored_files_equal':protected,'changed_runtime_files':differences,'deployment':{}}
for rel in sorted(names):
    name=Path(rel).name;target=HERE/name;backup=OUT/'canonical_before'/name
    assert backup.is_file() and target.read_bytes()==backup.read_bytes(),'Canonical source unexpectedly changed: '+name
    assert target.read_bytes()==(old/rel).read_bytes(),name
    candidate=staged/rel
    proof['deployment'][name]={'before':sha(target),'after':sha(candidate)}
    shutil.copy2(candidate,target)
    assert sha(target)==proof['deployment'][name]['after']
(OUT/'deployment_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
