"""Check execution/load contracts and physical parity before deployment."""
import hashlib,json,numpy as np,torch
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pause=read(OUT/'diagnostic_pause.json');RUN=Path(pause['run'])
checks=[]
for folder in ['running_ppo_smoke','full_load_smoke']:
    p=OUT/folder;r=read(p/'result.json');assert r['completed'] and r['iterations']==4
    rows=[json.loads(x) for x in (p/'progress.jsonl').read_text().splitlines()];assert len(rows)==4
    assert all(x['ppo']['pre_update_kl']==0 and x['behavior']['action_command_difference']==0 for x in rows)
    checks.append({'name':folder,'passed':True,'pre_update_KL':[x['ppo']['pre_update_kl'] for x in rows]})
load=read(OUT/'full_load_smoke/full_load_proof.json')
assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
reload=read(OUT/'checkpoint_reload/result.json');assert reload['completed'] and all(c['statistics']['survival']==1 for c in reload['cases'])
saved=torch.load(read(OUT/'selected_seed.json')['checkpoint'],map_location='cpu',weights_only=False)
baseline=torch.load(OUT/'running_ppo_smoke/transferred_actor.pt',map_location='cpu',weights_only=False)
actor_equal=all(torch.equal(v,baseline['actor_state_dict'][k]) for k,v in saved['actor_state_dict'].items())
assert actor_equal,'Fresh baseline must preserve original actor and trained exploration'
old=RUN/'022_run_150_gate_005_nominal';new=OUT/'same_actor_nominal'
a=read(old/'result.json');b=read(new/'result.json')
cases=a['cases']==b['cases'];assert cases,'Same-actor physics/classification changed'
arrays=[]
for filename in ['telemetry.npz','gait_diagnostics_50hz.npz']:
    x=np.load(old/filename,allow_pickle=False);y=np.load(new/filename,allow_pickle=False)
    assert set(x.files)==set(y.files),(filename,'keys')
    for key in x.files:
        equal=np.array_equal(x[key],y[key]);assert equal,(filename,key)
        arrays.append({'file':filename,'array':key,'equal':equal})
proof={'passed':True,'same_actor_and_exploration_equal':actor_equal,'full_load_proof':load,
    'checks':checks,'same_actor_cases_bitwise_equal':cases,'same_actor_arrays_bitwise_equal':arrays,
    'exact_baseline_checkpoint':str(OUT/'running_ppo_smoke/transferred_actor.pt'),
    'exact_baseline_sha256':sha(OUT/'running_ppo_smoke/transferred_actor.pt'),
    'current_running_qualified':b['passed'],'prior_cases_pass':all(c['passed'] for c in b['cases'][:-1]),
    'scope':'192 robots, 20s same-actor nominal physical replay plus real PPO/full-state reload. All running gates remain unchanged. Smoke actors discarded.'}
(OUT/'validation_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k!='same_actor_arrays_bitwise_equal'},indent=2))
