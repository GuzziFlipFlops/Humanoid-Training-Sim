import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run']);cp=Path(pause['checkpoint'])
folders={}
for label,pattern in [('nominal','*_nominal'),('pushed','*_push')]:
    matches=[p for p in run.glob(pattern) if (p/'result.json').exists() and read(p/'result.json').get('completed')
        and Path(read(p/'run_config.json')['arguments']['checkpoint']).resolve()==cp.resolve()]
    assert len(matches)==1;folders[label]=matches[0]
row=dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),**{k:str(v) for k,v in folders.items()})
for label,folder in folders.items():
    ev=read(folder/'evaluation.json');row[label+'_all_pass']=ev['passed']
    row[label+'_cases']=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],trials=c['statistics']['trials'],
        survival=c['statistics']['survival'],signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],
        lateral_speed=c['statistics']['reset_frame_lateral_speed_m_s'],flight=c['statistics']['geometric_flight_fraction'],statistics=c['statistics']) for c in ev['cases']]
review=read(OUT/'initial_gate_review.json');review['latest']=row;review['scope']='Exact owned diagnostic anchor; independently saved matching gates before candidate learning.'
(OUT/'anchor_gate_review.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({label:[{k:c[k] for k in ['command','passed','failures','signed_speed','lateral_speed','flight']} for c in row[label+'_cases']] for label in ['nominal','pushed']}))
