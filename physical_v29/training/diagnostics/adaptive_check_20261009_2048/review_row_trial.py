"""Read complete, exact checkpoint cases, including failures; no inference."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from runtime_joint_order import runtime_joint_order
OUT=Path(__file__).resolve().parent;TRIAL=OUT;ROOT=TRIAL/'matched_row_route'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
records=[]
for seed in [251,263]:
    for branch in ['control','shared','rows']:
        for updates in [64,128]:
            train=ROOT/f'seed{seed}_{branch}_{updates:03d}_train';checkpoint=train/'candidate.pt'
            if not checkpoint.exists() or not (train/'training_validation.json').exists():continue
            receipt=read(train/'training_validation.json');assert receipt['passed'] and receipt['actual_updates']==64
            assert read(train/'experimental_teacher_target.json')['applied']==(branch!='control')
            if branch!='control':
                joint_names=runtime_joint_order(train)
                expected_head=[joint_names.index('head_yaw'),joint_names.index('head_pitch')]
                progress=[json.loads(line) for line in (train/'progress.jsonl').read_text().splitlines()]
                assert all(r['locomotion_retention']['head_excluded']==expected_head for r in progress)
                assert all(r['locomotion_retention']['neutral_teacher_sha256']==read(TRIAL/'neutral_teacher_certificate.json')['checkpoint_sha256'] for r in progress)
                assert read(train/'command_local_target_validation.json')['passed']
            row=dict(seed=seed,branch=branch,updates=updates,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),training=receipt,tests={})
            for test in ['nominal','push','diverse_nominal','diverse_push']:
                folder=ROOT/f'seed{seed}_{branch}_{updates:03d}_{test}'
                if not (folder/'result.json').exists():continue
                result=read(folder/'result.json')
                if not result.get('completed'):continue
                cfg=read(folder/'run_config.json');assert Path(cfg['arguments']['checkpoint']).resolve()==checkpoint.resolve()
                ev=read(folder/'evaluation.json')
                assert len(ev['cases'])==10 and all(c['statistics']['trials']==32 for c in ev['cases'])
                cases=[]
                for c in ev['cases']:
                    z=c['statistics'];cases.append(dict(command=z['command_m_s'],passed=c['passed'],failures=c['failures'],survival=z['survival'],
                        signed_speed=z['reset_frame_forward_speed_m_s'],lateral_speed=z['reset_frame_lateral_speed_m_s'],
                        true_flight=z['geometric_flight_fraction'],force_threshold_flight=z['flight_fraction'],statistics=z))
                diversity=None
                if test.startswith('diverse'):
                    diversity=read(folder/'initial_diversity.json');assert len(diversity['unique_initial_states_by_command'])==10 and all(x==32 for x in diversity['unique_initial_states_by_command'].values())
                row['tests'][test]=dict(folder=str(folder),passed=ev['passed'],cases=cases,independent_state_diversity_receipt=str(folder/'initial_diversity.json') if diversity else None)
            records.append(row)
all_expected=len(records)==12 and all(set(row['tests'])==({'nominal','push'} if row['updates']==64 else {'nominal','push','diverse_nominal','diverse_push'}) for row in records)
candidates=[r for r in records if r['branch']=='rows']
all_candidate_gates_pass=all_expected and all(test['passed'] for row in candidates for test in row['tests'].values())
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),experiment_manifest=read(ROOT/'experiment_manifest.json') if (ROOT/'experiment_manifest.json').exists() else None,
    branch_all_gates_pass={b:all_expected and all(t['passed'] for r in records if r['branch']==b for t in r['tests'].values()) for b in ['control','shared','rows']},records=records,all_expected_components_complete=all_expected,all_candidate_gates_pass=all_candidate_gates_pass,
    complete_original_full_state_preserved=True,diagnostic_actors_deployed=False,
    original_signature_does_not_certify_executed_experimental_targets=True,
    decision='Incomplete controlled comparison' if not all_expected else 'Requires complete head/upper/versioned deployment review' if all_candidate_gates_pass else 'Reject tested continuation for immediate deployment; preserve current full learner and exact evidence',
    scope='Exact full-state64/128 controls at4096 environments,two learning seeds,unchanged320robot20s nominal/pushed gates and32 independently generated initial states per command at128. Original physics/rewards/caps/classifiers remain; experimental checkpoint inference has no teacher or target hook.')
(OUT/'matched_row_route_review.json').write_text(json.dumps(result,indent=2)+'\n')
compact=[]
for row in records:
    entry={k:row[k] for k in ['seed','branch','updates']};entry['tests']={}
    for test,r in row['tests'].items():
        entry['tests'][test]=dict(all_pass=r['passed'],failures=[dict(command=c['command'],failures=c['failures']) for c in r['cases'] if not c['passed']],
            current={k:r['cases'][-1][k] for k in ['signed_speed','lateral_speed','true_flight']},
            retained1_5={k:next(c for c in r['cases'] if c['command']==1.5)[k] for k in ['signed_speed','true_flight']})
    compact.append(entry)
print(json.dumps(dict(complete=all_expected,all_candidate_gates_pass=all_candidate_gates_pass,records=compact),indent=2))
