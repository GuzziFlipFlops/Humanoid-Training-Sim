"""Independent CPU review of saved components; never deploys diagnostic actors."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_sequence'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
rows=[]
for branch in ['control','candidate']:
    for updates in [64,128]:
        train=ROOT/f'{branch}_{updates:03d}_train';nom=ROOT/f'{branch}_{updates:03d}_nominal';push=ROOT/f'{branch}_{updates:03d}_push'
        if not all((p/'result.json').exists() and read(p/'result.json').get('completed') for p in [train,nom,push]):continue
        cp=train/'candidate.pt';row=dict(branch=branch,updates=updates,checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(nom),pushed=str(push))
        for label,folder in [('nominal',nom),('pushed',push)]:
            assert Path(read(folder/'run_config.json')['arguments']['checkpoint']).resolve()==cp.resolve()
            ev=read(folder/'evaluation.json');row[label+'_all_pass']=ev['passed']
            row[label+'_cases']=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],
                trials=c['statistics']['trials'],survival=c['statistics']['survival'],signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],
                lateral_speed=c['statistics']['reset_frame_lateral_speed_m_s'],flight=c['statistics']['geometric_flight_fraction'],statistics=c['statistics']) for c in ev['cases']]
        proof=read(train/'full_load_proof.json');assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
        progress=[json.loads(v) for v in (train/'progress.jsonl').read_text().splitlines()];assert len(progress)==64
        assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in progress)
        assert all(r[k]['batches']==20 for r in progress for k in ['retention','locomotion_retention','upper_refinement'])
        audit=[r['upper_refinement']['experimental_sequence_mean'] for r in progress]
        assert all(r['applied']==(branch=='candidate') for r in audit)
        active=[r for r in audit if r['batches']]
        assert active and all(r['statistics']['bounded_gradient_norm']<=.1*r['statistics']['actual_PPO_shoulder_norm']+1e-6 for r in active)
        row.update(exact_full_load=proof,zero_KL_command_mismatch=True,sequence_warmup_updates=sum(r['batches']==0 for r in audit),
            active_sequence_updates=len(active),actual_gradient_statistics_mean={k:sum(r['statistics'][k] for r in active)/len(active) for k in active[0]['statistics']},
            first_actual_batch=active[0]['first_actual_batch'],last_actual_batch=active[-1]['first_actual_batch'])
        for suffix in ['diverse_nominal','diverse_push']:
            folder=ROOT/f'{branch}_{updates:03d}_{suffix}'
            if (folder/'result.json').exists() and read(folder/'result.json').get('completed'):
                ev=read(folder/'evaluation.json');div=read(folder/'initial_diversity.json')
                assert all(n>=32 for n in div['unique_initial_states_by_command'].values())
                row[suffix]=dict(passed=ev['passed'],initial_state_sha256=div['initial_state_sha256'],
                    unique_initial_states_by_command=div['unique_initial_states_by_command'],
                    cases=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],
                        signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],lateral_speed=c['statistics']['reset_frame_lateral_speed_m_s'],
                        flight=c['statistics']['geometric_flight_fraction']) for c in ev['cases']])
        rows.append(row)
complete=len(rows)==4 and all('diverse_nominal' in r and 'diverse_push' in r for r in rows if r['updates']==128)
if complete:
    for suffix in ['diverse_nominal','diverse_push']:
        selected=[r for r in rows if r['updates']==128]
        assert selected[0][suffix]['initial_state_sha256']==selected[1][suffix]['initial_state_sha256']
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),completed=complete,
    harness_completion_record_present=(ROOT/'completion.json').exists(),rows=rows,
    experimental_new_gradient=True,training_only=True,original_signature_does_not_certify_override=True,
    inference_physics_rewards_native_motor_caps_unchanged=True,diagnostic_actors_deployed=False)
(OUT/'matched_sequence_review.json').write_text(json.dumps(result,indent=2)+'\n')
for branch in ['control','candidate']:
    selected=[r for r in rows if r['branch']==branch]
    if selected:
        review=read(OUT/'initial_gate_review.json');review['latest']=selected[-1]
        (OUT/(branch+'_gate_review.json')).write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps([dict(branch=r['branch'],updates=r['updates'],current={k:{v:r[k+'_cases'][-1][v] for v in ['signed_speed','lateral_speed','flight','failures']} for k in ['nominal','pushed']},
    prior_failures={k:[dict(command=c['command'],failures=c['failures']) for c in r[k+'_cases'][:-1] if not c['passed']] for k in ['nominal','pushed']}) for r in rows]))
