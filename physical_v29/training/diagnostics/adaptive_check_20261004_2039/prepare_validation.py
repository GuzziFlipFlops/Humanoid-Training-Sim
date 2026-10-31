"""Freeze candidate source and strictly bind the physically audited v7 actor."""
import hashlib, json, shutil, sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run'])
assert read(OUT/'bilateral_counterfactuals.json')['passed']
assert read(OUT/'physical_response/result.json')['completed']
assert read(OUT/'gradient_analysis.json')['passed']
seed=run/'021_run_150_train_004_part_02/candidate.pt'
gates=[]
for name in ['022_run_150_gate_005_nominal','023_run_150_gate_005_push']:
    p=run/name;cfg=read(p/'run_config.json');r=read(p/'result.json')
    assert Path(cfg['arguments']['checkpoint']).resolve()==seed.resolve()
    assert r['completed'] and all(c['passed'] for c in r['cases'] if c['statistics']['command_m_s']<1.5)
    gates.append({'path':str(p),'current_running_case_passed':r['cases'][-1]['passed'],
        'genuine_flight_fraction':r['cases'][-1]['statistics']['geometric_flight_fraction']})
proof={'checkpoint':str(seed),'sha256':sha(seed),'prior_gate_receipts':gates,
       'latest_preserved':pause['checkpoint'],'latest_sha256':pause['checkpoint_sha256'],
       'reason':'R33 checkpoint021 retains every prior walking command in matching full gates and reproduces 1.377% genuine flight. Final actor has zero flight. Preserve exact mean and trained exploration; new objective receives fresh critic/Adam and entirely fresh qualification.',
       'qualification_reused':False,'helper_pulse_deployed':False}
(OUT/'selected_seed.json').write_text(json.dumps(proof,indent=2)+'\n')
target=OUT/'validation_source/physical_v29';assert not target.exists()
shutil.copytree(run/'source/physical_v29',target,ignore=shutil.ignore_patterns('__pycache__','usd','*.pyc'))
shutil.copy2(OUT/'candidate_source/training/stage_rewards.py',target/'training/stage_rewards.py')
stage=target/'training/stage_rewards.py'
s=stage.read_text().replace("running_unload_source='Local adaptation informed by Siekmann2011.01387 periodic force composition',",
    "running_unload_aggregation='Exponential of summed intended swing loads; no load redistribution credit',\n                running_unload_source='Local adaptation informed by Siekmann2011.01387 periodic force composition',")
stage.write_text(s)
warm=target/'training/reward_warm_start.py';s=warm.read_text()
s=s.replace("INITIAL_EXPLORATION_STD = .35", "FLIGHT_SOURCE_REWARD = 'cad_running_phase_unload_v7'\nFLIGHT_SOURCE_SIGNATURE = '68e32568f44b7b0a22271e175e523cf8a25990459c15b1807603ba61e411372f'\nINITIAL_EXPLORATION_STD = .35")
s=s.replace('(SOURCE_REWARD,GAIT_SOURCE_REWARD,RUNNING_SOURCE_REWARD)', '(SOURCE_REWARD,GAIT_SOURCE_REWARD,RUNNING_SOURCE_REWARD,FLIGHT_SOURCE_REWARD)')
s=s.replace("    elif source==RUNNING_SOURCE_REWARD:", "    elif source==FLIGHT_SOURCE_REWARD:\n        assert float(info.get('speed',-1))==1.5, 'Use the physically audited R33 running actor'\n        assert info.get('reward_signature')==FLIGHT_SOURCE_SIGNATURE, 'v7 source differs from audited R33'\n    elif source==RUNNING_SOURCE_REWARD:")
s=s.replace('source in (GAIT_SOURCE_REWARD,RUNNING_SOURCE_REWARD)', 'source in (GAIT_SOURCE_REWARD,RUNNING_SOURCE_REWARD,FLIGHT_SOURCE_REWARD)')
warm.write_text(s)
old=run/'source/physical_v29';differences=[];protected=0
for p in old.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc' or 'usd' in p.parts:continue
    rel=p.relative_to(old).as_posix();q=target/rel
    assert q.is_file(),rel
    if p.read_bytes()==q.read_bytes():protected+=1
    else:differences.append(rel)
assert set(differences)=={'training/stage_rewards.py','training/reward_warm_start.py'},differences
sys.path[:0]=[str(target/'training'),str(target)]
import torch,copy
from reward_warm_start import validate_actor_transfer
saved=torch.load(seed,map_location='cpu',weights_only=False)
assert saved['infos']['reward_signature']=='68e32568f44b7b0a22271e175e523cf8a25990459c15b1807603ba61e411372f'
transfer=validate_actor_transfer(saved,target/'training');assert transfer['preserve_trained_exploration']
negative=[]
for key in ['contract','asset_sha256','motion_sha256','actuator_sha256','reward_signature','reward_version']:
    bad=copy.deepcopy(saved);bad['infos'][key]='wrong'
    try:validate_actor_transfer(bad,target/'training')
    except AssertionError:negative.append(key)
    else:raise AssertionError('Invalid transfer accepted: '+key)
r={'passed':True,'selected_seed':proof,'transfer':transfer,'negative_transfer_checks':negative,
    'protected_authored_files_equal':protected,'changed_authored_files':differences,
    'training_retention_and_independent_gates_unchanged':True,'stage_rewards_sha256':sha(stage),'warm_start_sha256':sha(warm)}
(OUT/'candidate_proof.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
