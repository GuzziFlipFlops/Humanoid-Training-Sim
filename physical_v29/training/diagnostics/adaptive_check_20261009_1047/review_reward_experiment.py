"""CPU review of exact full-state controls, exposure and unchanged gates."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_slow_reward'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
rows=[]
def cases(ev):
    return [dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],trials=c['statistics']['trials'],
        survival=c['statistics']['survival'],signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],
        lateral_speed=c['statistics']['reset_frame_lateral_speed_m_s'],flight=c['statistics']['geometric_flight_fraction'],statistics=c['statistics']) for c in ev['cases']]
for seed in [179,191]:
    for branch in ['control','candidate']:
        for updates in [64,128]:
            prefix=f'seed{seed}_{branch}_{updates:03d}';train=ROOT/(prefix+'_train');nom=ROOT/(prefix+'_nominal');push=ROOT/(prefix+'_push')
            if not all((p/'result.json').exists() and read(p/'result.json').get('completed') for p in [train,nom,push]):continue
            cp=train/'candidate.pt';row=dict(seed=seed,branch=branch,updates=updates,checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(nom),pushed=str(push))
            for label,folder in [('nominal',nom),('pushed',push)]:
                assert Path(read(folder/'run_config.json')['arguments']['checkpoint']).resolve()==cp.resolve()
                ev=read(folder/'evaluation.json');row[label+'_all_pass']=ev['passed'];row[label+'_cases']=cases(ev)
                assert all(c['trials']>=32 for c in row[label+'_cases'])
            proof=read(train/'full_load_proof.json');assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
            progress=[json.loads(v) for v in (train/'progress.jsonl').read_text().splitlines()];assert len(progress)==64
            assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in progress)
            assert all(r[k]['batches']==20 for r in progress for k in ['retention','locomotion_retention','upper_refinement'])
            exposure=read(train/'experimental_reward.json');assert exposure['applied']==(branch=='candidate')
            row.update(exact_full_load=proof,zero_KL_command_mismatch=True,original_auxiliaries_batches=20,
                executed_reward_override=exposure,training_validation=read(train/'training_validation.json'))
            for suffix in ['diverse_nominal','diverse_push']:
                folder=ROOT/(prefix+'_'+suffix)
                if (folder/'result.json').exists() and read(folder/'result.json').get('completed'):
                    ev=read(folder/'evaluation.json');div=read(folder/'initial_diversity.json')
                    assert all(n>=32 for n in div['unique_initial_states_by_command'].values())
                    row[suffix]=dict(passed=ev['passed'],initial_state_sha256=div['initial_state_sha256'],
                        unique_initial_states_by_command=div['unique_initial_states_by_command'],cases=cases(ev))
            rows.append(row)
complete=len(rows)==8 and all('diverse_nominal' in r and 'diverse_push' in r for r in rows if r['updates']==128)
if complete:
    for seed in [179,191]:
        selected=[r for r in rows if r['seed']==seed and r['updates']==128]
        for suffix in ['diverse_nominal','diverse_push']:assert selected[0][suffix]['initial_state_sha256']==selected[1][suffix]['initial_state_sha256']
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),completed=complete,harness_completion_record_present=(ROOT/'completion.json').exists(),rows=rows,
    two_independent_learning_seeds=[179,191],experimental_reward_not_certified_by_original_signature=True,
    physics_caps_original_teachers_and_upper_unchanged=True,diagnostic_actors_deployed=False)
(OUT/'matched_reward_review.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
for r in rows:
    if r['updates']==128:
        review=read(OUT/'initial_gate_review.json');review['latest']=r
        (OUT/f'seed{r["seed"]}_{r["branch"]}_gate_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(seed=r['seed'],branch=r['branch'],updates=r['updates'],current={k:{v:r[k+'_cases'][-1][v] for v in ['signed_speed','lateral_speed','flight','failures']} for k in ['nominal','pushed']},
    prior_failures={k:[dict(command=c['command'],failures=c['failures']) for c in r[k+'_cases'][:-1] if not c['passed']] for k in ['nominal','pushed']}) for r in rows]))
