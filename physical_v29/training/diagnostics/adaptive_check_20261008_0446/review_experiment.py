import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_shoulder'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
rows=[]
for branch in ['control','candidate']:
    for updates in [64,128]:
        f=ROOT/f'{branch}_{updates:03d}_train';n=ROOT/f'{branch}_{updates:03d}_nominal';p=ROOT/f'{branch}_{updates:03d}_push'
        if not all((d/'evaluation.json').exists() for d in [n,p]):continue
        cp=f/'candidate.pt';row=dict(branch=branch,updates=updates,checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(n),pushed=str(p))
        for label,folder in [('nominal',n),('pushed',p)]:
            assert Path(read(folder/'run_config.json')['arguments']['checkpoint']).resolve()==cp.resolve()
            ev=read(folder/'evaluation.json');row[label+'_all_pass']=ev['passed']
            row[label+'_cases']=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],
                trials=c['statistics']['trials'],survival=c['statistics']['survival'],signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],
                lateral_speed=c['statistics']['reset_frame_lateral_speed_m_s'],flight=c['statistics']['geometric_flight_fraction'],statistics=c['statistics']) for c in ev['cases']]
        load=read(f/'full_load_proof.json');assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
        progress=[json.loads(v) for v in (f/'progress.jsonl').read_text().splitlines()];assert len(progress)==64
        assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in progress)
        audit=[r['upper_refinement']['experimental_shoulder_mean'] for r in progress]
        assert all(r['batches']==20 and r['applied']==(branch=='candidate') for r in audit)
        assert all(r['statistics']['bounded_gradient_norm']<=.10001*r['statistics']['actual_PPO_shoulder_norm']+1e-9 for r in audit)
        row['exact_full_load']=load;row['zero_KL_command_mismatch']=True
        row['actual_gradient_statistics_mean']={k:sum(r['statistics'][k] for r in audit)/len(audit) for k in audit[0]['statistics']}
        row['first_actual_minibatch']=audit[0]['first_actual_batch'];row['last_actual_minibatch']=audit[-1]['first_actual_batch']
        for suffix in ['diverse_nominal','diverse_push']:
            d=ROOT/f'{branch}_{updates:03d}_{suffix}'
            if (d/'evaluation.json').exists():
                ev=read(d/'evaluation.json');diversity=read(d/'initial_diversity.json')
                assert all(c>=32 for c in diversity['unique_initial_states_by_command'].values())
                row[suffix]=dict(passed=ev['passed'],initial_state_sha256=diversity['initial_state_sha256'],
                    unique_initial_states_by_command=diversity['unique_initial_states_by_command'],
                    cases=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],
                        signed=c['statistics']['reset_frame_forward_speed_m_s'],lateral=c['statistics']['reset_frame_lateral_speed_m_s'],
                        flight=c['statistics']['geometric_flight_fraction']) for c in ev['cases']])
        rows.append(row)
components_complete=len(rows)==4 and all('diverse_nominal' in r and 'diverse_push' in r for r in rows if r['updates']==128)
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),completed=components_complete,
    harness_completion_record_present=(ROOT/'completion.json').exists(),
    operational_status_explanation=('The harness exited1 after every component finished at3008.8s because its final2600s overall guard included saved-boundary waiting/drain. The finally handler resumed the original supervisor. This is a failed operational harness exit,not a production fault or a successful exit. Independent full-state/gate/diversity evidence is validated separately.'
        if 'AssertionError: Overall bounded comparison budget exceeded' in (OUT/'matched_harness.log').read_text()
        else 'Inspect original harness log separately from component evidence.'),
    completion_scope='Independent component/full-state/gate/diversity evidence;harness overall-time status is separate and must be reported.',rows=rows,
    experimental_new_gradient=True,training_only=True,original_signature_does_not_certify_override=True,
    no_inference_or_physical_or_reward_or_motor_change=True,diagnostic_actors_deployed=False)
(OUT/'matched_shoulder_review.json').write_text(json.dumps(result,indent=2)+'\n')
for branch in ['control','candidate']:
    selected=[r for r in rows if r['branch']==branch]
    if selected:
        review=read(OUT/'initial_gate_review.json');review['latest']=selected[-1]
        (OUT/(branch+'_gate_review.json')).write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps([dict(branch=r['branch'],updates=r['updates'],current={label:{k:r[label+'_cases'][-1][k] for k in ['signed_speed','lateral_speed','flight','failures']} for label in ['nominal','pushed']},
    prior_failures={label:[dict(command=c['command'],failures=c['failures']) for c in r[label+'_cases'][:-1] if not c['passed']] for label in ['nominal','pushed']},gradient=r['actual_gradient_statistics_mean']) for r in rows],indent=2))
