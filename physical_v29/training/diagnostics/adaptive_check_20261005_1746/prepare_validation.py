"""Freeze independent-seed validation of best return and best flight, not training."""
import json,hashlib
from pathlib import Path
from trajectory_parameters import ORIGIN,NAMES
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'trajectory_search/result.json').read_text()); assert r['completed']
rows=[]
for group in r['rounds']:
 for row in group['rows']:
  row=dict(row,generation=group['generation'],return_delta=row['mean_complete_return']-group['rows'][0]['mean_complete_return']);rows.append(row)
best_return=max(rows,key=lambda v:v['return_delta']);best_flight=max(rows,key=lambda v:v['geometric_flight_fraction'])
selection=dict(checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],best_return=best_return,best_flight=best_flight,
 parameters=[ORIGIN.tolist()]+[[v['parameters'][name] for name in NAMES] for v in [best_return,best_flight]],
 labels=['unchanged_control','best_complete_return','best_geometric_flight'],
 diagnostic_only=True,scope='Independent reset seed83,32 robots per cohort,20s,normal constraints; no new learned actor')
(OUT/'validation_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
text=(OUT/'trajectory_search.py').read_text()
replacements={"out=OUT/'trajectory_search'":"out=OUT/'trajectory_validation'",'per=8; count=32; n=per*count':'per=32; count=3; n=per*count',
 'cfg.seed=71':'cfg.seed=83','for generation in range(4):':'for generation in range(1):',
 'parameters=samples(rng,mean,std,count); repeated=np.repeat(parameters,per,axis=0)':"parameters=np.asarray(json.loads((OUT/'validation_selection.json').read_text())['parameters']); repeated=np.repeat(parameters,per,axis=0)",
 'for tick in range(600):':'for tick in range(1000):',
 "OUT/'trajectory_search_failure.json'":"OUT/'trajectory_validation_failure.json'",
 '4 populations of 32 trajectories,8 robots each,12s; zero-offset controls every round.':'Independent seed83:control,best return,best flight,32 robots each,20s.'}
for old,new in replacements.items(): assert old in text,old; text=text.replace(old,new)
(OUT/'trajectory_validation.py').write_text(text)
print(json.dumps({k:selection[k] for k in ['best_return','best_flight']},indent=2))
