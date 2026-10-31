import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_shoulder'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause=read(OUT/'diagnostic_pause.json');cp=Path(pause['checkpoint']);checks=[]
run=Path(pause['run']);nom=[]
for p in run.glob('*nominal/run_config.json'):
    if Path(read(p)['arguments']['checkpoint']).resolve()==cp.resolve():nom.append(p.parent)
assert nom,'Same actor original nominal evidence absent'
original=nom[-1];new=ROOT/'unchanged_actor_nominal'
ev_old=read(original/'evaluation.json');ev_new=read(new/'evaluation.json')
assert {k:v for k,v in ev_old.items() if k!='seconds'}=={k:v for k,v in ev_new.items() if k!='seconds'}
checks.append('same_original_actor_full_nominal_cases_bitwise_equal')
arrays=[]
for name in ['telemetry.npz','gait_diagnostics_50hz.npz','startup_diagnostics.npz']:
    with np.load(original/name) as a,np.load(new/name) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            assert np.array_equal(a[key],b[key],equal_nan=True) if a[key].dtype.kind in 'fc' else np.array_equal(a[key],b[key]),(name,key)
            arrays.append(dict(file=name,key=key))
checks.append('all_original_full_population_physical_arrays_bitwise_equal')
smoke=ROOT/'candidate_smoke';load=read(smoke/'full_load_proof.json')
assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
progress=[json.loads(v) for v in (smoke/'progress.jsonl').read_text().splitlines()]
assert len(progress)==4 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in progress)
reload=read(ROOT/'candidate_reload/full_load_proof.json');assert reload['actor_equal'] and reload['critic_equal']
checks.extend(['actual_four_PPO_updates_full_start_equal_KL_and_commands_zero','checkpoint_reload_actor_critic_equal'])
source=read(OUT/'protected_source_copy.json');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(OUT/'control_source/physical_v29'/k)==h for k,h in source['protected_hashes'].items())
checks.append('all_protected_authored_sources_identical')
review=read(OUT/'matched_shoulder_review.json');assert review['completed']
controls=[r for r in review['rows'] if r['branch']=='control'];candidates=[r for r in review['rows'] if r['branch']=='candidate']
for c,a in zip(controls,candidates):
    for label in ['diverse_nominal','diverse_push']:
        if label in c:assert c[label]['initial_state_sha256']==a[label]['initial_state_sha256']
checks.append('independent_initial_states_identical_between_control_and_candidate')
result=dict(passed=True,checks=checks,physical_arrays=len(arrays),array_details=arrays,original_nominal=str(original),
    protected_files=source['files'],no_diagnostic_actors_deployed=True,new_gradient_not_original_signature_certified=True)
(OUT/'experiment_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='array_details'},indent=2))
