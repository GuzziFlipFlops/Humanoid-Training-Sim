"""CPU-only exact checkpoint/pair review across the full current curriculum."""
import json,hashlib,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
p=argparse.ArgumentParser();p.add_argument('--snapshot',default='initial_owned_observation.json');p.add_argument('--output',default='initial_gate_review.json');args=p.parse_args()
live=read(OUT/args.snapshot);run=Path(live['active_run']['run']);s=live['state']
assert s['status'] in ('running','failed','time_limit_current_stage_unpassed') and all(live['frozen_source_integrity'].values())
records=[]
for nom in sorted(run.glob('*_gate_*_nominal')):
    if not (nom/'result.json').exists() or not read(nom/'result.json').get('completed'):continue
    suffix=nom.name.split('_',1)[1].replace('_nominal','_push');push=next(iter(run.glob('*_'+suffix)),None)
    if not push or not (push/'result.json').exists() or not read(push/'result.json').get('completed'):continue
    cp=Path(read(nom/'run_config.json')['arguments']['checkpoint'])
    assert cp.resolve()==Path(read(push/'run_config.json')['arguments']['checkpoint']).resolve()
    row=dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(nom),pushed=str(push))
    for name,folder in [('nominal_cases',nom),('pushed_cases',push)]:
        ev=read(folder/'evaluation.json');row[name]=[]
        for c in ev['cases']:
            z=c['statistics']
            row[name].append(dict(command=z['command_m_s'],passed=c['passed'],failures=c['failures'],trials=z['trials'],
                survival=z['survival'],signed_speed=z['reset_frame_forward_speed_m_s'],lateral_speed=z['reset_frame_lateral_speed_m_s'],
                flight=z['geometric_flight_fraction'],alternation=z.get('alternating_landings_fraction'),cadence=z.get('cycle_hz'),
                statistics=z))
        row[name.replace('_cases','_all_pass')]=ev['passed']
    records.append(row)
assert records
for promoted in s['promotions']:
    if promoted.get('requalification'):continue
    match=[p for p in records if Path(p['checkpoint']).resolve()==Path(promoted['checkpoint']).resolve() and ('_'+promoted['stage']+'_gate_') in Path(p['nominal']).name]
    assert len(match)==1 and match[0]['nominal_all_pass'] and match[0]['pushed_all_pass']
    assert all(c['trials']>=32 for k in ['nominal_cases','pushed_cases'] for c in match[0][k])
    promoted['matching_pair']=match[0]
latest=records[-1]
heads={}
for label in ['nominal','pushed']:
    z=np.load(Path(latest[label])/'telemetry.npz');q=z['q'];names=z['joint_names'].tolist();ids=[names.index('head_yaw'),names.index('head_pitch')]
    heads[label]=[dict(command=c['command'],head_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(q[:,z['case_labels']==i][:,:,ids]**2,-1)))))) for i,c in enumerate(latest[label+'_cases'])]
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),owned_snapshot_utc=live['recorded_utc'],run=str(run),state=s,gate_pairs=records,promotions=s['promotions'],latest=latest,latest_head=heads,
    scope='Only completed exact-checkpoint nominal/pushed pairs. Promotion is validated against all case gates; newer learning actors are separately unqualified.')
(OUT/args.output).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(promotions=[{k:p[k] for k in ['stage','speed','checkpoint']} for p in s['promotions'] if not p.get('requalification')],latest=Path(latest['checkpoint']).parent.name,
    cases={k:[{a:c[a] for a in ['command','passed','failures','signed_speed','flight']} for c in latest[k]] for k in ['nominal_cases','pushed_cases']},heads=heads),indent=2))
