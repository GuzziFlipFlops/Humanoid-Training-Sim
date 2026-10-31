"""CPU review of completed, exactly paired whole-state tests; preserve failures."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_full_state'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
records=[]
for seed in [317,331]:
    for branch in ['control','earlier']:
        for updates in [64,128]:
            train=ROOT/f'seed{seed}_{branch}_{updates:03d}_train';cp=train/'candidate.pt'
            if not (train/'training_validation.json').exists():continue
            receipt=read(train/'training_validation.json');assert receipt['passed'] and receipt['actual_updates']==64
            row=dict(seed=seed,branch=branch,updates=updates,checkpoint=str(cp),checkpoint_sha256=sha(cp),training=receipt,tests={})
            for label in ['nominal','push','diverse_nominal','diverse_push']:
                folder=ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}'
                if not (folder/'result.json').exists() or not read(folder/'result.json').get('completed'):continue
                assert Path(read(folder/'run_config.json')['arguments']['checkpoint']).resolve()==cp.resolve()
                ev=read(folder/'evaluation.json');assert len(ev['cases'])==10 and all(c['statistics']['trials']==32 for c in ev['cases'])
                cases=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],
                    survival=c['statistics']['survival'],signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],
                    lateral_speed=c['statistics']['reset_frame_lateral_speed_m_s'],true_flight=c['statistics']['geometric_flight_fraction'],statistics=c['statistics']) for c in ev['cases']]
                if label.startswith('diverse'):
                    d=read(folder/'initial_diversity.json');assert len(d['unique_initial_states_by_command'])==10 and all(n==32 for n in d['unique_initial_states_by_command'].values())
                row['tests'][label]=dict(folder=str(folder),passed=ev['passed'],cases=cases,
                    independent_state_receipt=str(folder/'initial_diversity.json') if label.startswith('diverse') else None)
            records.append(row)
complete=len(records)==8 and all(set(r['tests'])==({'nominal','push'} if r['updates']==64 else {'nominal','push','diverse_nominal','diverse_push'}) for r in records)
passes={b:complete and all(t['passed'] for r in records if r['branch']==b for t in r['tests'].values()) for b in ['control','earlier']}
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),all_expected_components_complete=complete,records=records,
    branch_all_gates_pass=passes,earlier_all_gates_pass=passes['earlier'],diagnostic_actor_deployed=False,
    scope='Exact full-state64/128 comparisons in two learning seeds. Original runtime/objectives/teachers/caps/inference. Initial checkpoint including critic/Adam/exploration is the branch difference. Ordinary pairs plus32 individually seeded nominal/pushed initial states per command at128; no nominal-clone diversity inference.')
(OUT/'matched_full_state_review.json').write_text(json.dumps(result,indent=2)+'\n')
compact=[]
for r in records:
    row={k:r[k] for k in ['seed','branch','updates']};row['tests']={}
    for name,t in r['tests'].items():
        row['tests'][name]=dict(all_pass=t['passed'],failures=[dict(command=c['command'],failures=c['failures']) for c in t['cases'] if not c['passed']],
            current={k:t['cases'][-1][k] for k in ['signed_speed','lateral_speed','true_flight']},
            retained1_5={k:next(c for c in t['cases'] if c['command']==1.5)[k] for k in ['signed_speed','true_flight']})
    compact.append(row)
print(json.dumps(dict(complete=complete,branches_pass=passes,records=compact),indent=2))
