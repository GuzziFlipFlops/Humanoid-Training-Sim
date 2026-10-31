import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
out=Path(__file__).resolve().parent;root=out/'matched_direction';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=[]
for branch in ['control','candidate']:
 for updates in [64,128]:
  f=root/f'{branch}_{updates:03d}_train';n=root/f'{branch}_{updates:03d}_nominal';p=root/f'{branch}_{updates:03d}_push'
  if not all((x/'result.json').exists() and read(x/'result.json').get('completed') for x in [f,n,p]):continue
  cp=f/'candidate.pt';row=dict(branch=branch,updates=updates,checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),nominal=str(n),pushed=str(p),cases={})
  for label,folder in [('nominal',n),('pushed',p)]:
   assert Path(read(folder/'run_config.json')['arguments']['checkpoint']).resolve()==cp.resolve()
   ev=read(folder/'evaluation.json');row[label+'_all_pass']=ev['passed']
   cases=[]
   for c in ev['cases']:
    z=c['statistics'];cases.append(dict(command=z['command_m_s'],passed=c['passed'],failures=c['failures'],trials=z['trials'],survival=z['survival'],signed_speed=z['reset_frame_forward_speed_m_s'],lateral_speed=z['reset_frame_lateral_speed_m_s'],flight=z['geometric_flight_fraction'],statistics=z))
   row['cases'][label]=cases;row[label+'_cases']=cases
  row['full_load_proof']=read(f/'full_load_proof.json')
  assert all(row['full_load_proof'][k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
  progress=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
  assert len(progress)==64 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in progress)
  row['zero_KL_and_command_mismatch']=True
  rows.append(row)
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),complete=len(rows)==4,rows=rows,diagnostic_actors_deployed=False,scope='Executed diagnostic reward override is declared and hashed; original source signature is not certification of this override. Gates/classes remain unchanged and bind exact learned actors.')
(out/'matched_direction_review.json').write_text(json.dumps(record,indent=2)+'\n')
for b in ['control','candidate']:
 selected=[r for r in rows if r['branch']==b]
 if selected:
  review=read(out/'initial_gate_review.json');review['latest']=selected[-1];(out/(b+'_gate_review.json')).write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps([dict(branch=r['branch'],updates=r['updates'],running=[{k:c[k] for k in ['signed_speed','lateral_speed','flight','failures']} for label in ['nominal','pushed'] for c in r['cases'][label] if c['command']==3.5],prior_failures={label:[(c['command'],c['failures']) for c in r['cases'][label] if c['command']<3.5 and not c['passed']] for label in ['nominal','pushed']}) for r in rows],indent=2))
