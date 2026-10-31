"""CPU-only live view and exact per-command comparator."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_command_advantage'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
records=[]
for p in sorted(ROOT.glob('*/evaluation.json')):
 d=read(p);folder=p.parent
 records.append(dict(operation=folder.name,all_pass=d['passed'],cases=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],
  signed=c['statistics']['reset_frame_forward_speed_m_s'],lateral=c['statistics']['reset_frame_lateral_speed_m_s'],
  true_flight=c['statistics']['geometric_flight_fraction'],path_efficiency=c['statistics']['forward_path_efficiency'],
  failures=c['failures']) for c in d['cases']]))
pairs=[]
for seed in (347,359):
 for updates in (64,128):
  for test in ('nominal','push','diverse_nominal','diverse_push'):
   selection=[next((x for x in records if x['operation']==f'seed{seed}_{b}_{updates:03d}_{test}'),None) for b in ('control','candidate')]
   if not all(selection):continue
   a,b=selection
   pairs.append(dict(seed=seed,updates=updates,test=test,control_all_pass=a['all_pass'],candidate_all_pass=b['all_pass'],
    lost_passing_commands=[x['command'] for x,y in zip(a['cases'],b['cases']) if x['passed'] and not y['passed']],
    gained_passing_commands=[x['command'] for x,y in zip(a['cases'],b['cases']) if not x['passed'] and y['passed']],
    cases=[dict(command=x['command'],control=x,candidate=y) for x,y in zip(a['cases'],b['cases'])]))
ops=[]
for p in ROOT.glob('*/progress.jsonl'):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 if not rows:continue
 ops.append(dict(operation=p.parent.name,updates=len(rows),num_envs=rows[-1]['num_envs'],
  applied=rows[-1]['command_advantage']['applied'],min_GPU=min(x['gpu']['free_mb'] for x in rows),
  min_host=min(x['gpu']['ram_available_mb'] for x in rows),pre_KL_max=max(x['ppo']['pre_update_kl'] for x in rows),
  actual_samples=len(rows)*rows[-1]['num_envs']*24,actual_active_seconds=sum(x['collect_seconds']+x['learn_seconds'] for x in rows)))
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),records=records,paired_tests=pairs,training=ops,
 completed=(ROOT/'completion.json').exists(),failure=read(ROOT/'failure.json') if (ROOT/'failure.json').exists() else None)
(OUT/'live_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(completed=result['completed'],failure=result['failure'],training=ops,
 latest_evals=[dict(operation=x['operation'],failures=[dict(command=c['command'],failures=c['failures']) for c in x['cases'] if not c['passed']],fast=x['cases'][-1]) for x in records[-2:]],
 paired=[{k:v for k,v in x.items() if k!='cases'} for x in pairs]),indent=2))
