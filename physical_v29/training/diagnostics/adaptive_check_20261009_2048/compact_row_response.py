import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'current_row_response.json').read_text())
rows=[]
for r in d['records']:
 s=r['responses']['shared'];v=r['responses']['rows']
 rows.append(dict(batch=r['batch'],eligible=r['eligible_states'],shared_head_RMS=s['head_raw_RMS_change'],rows_head_max=v['head_raw_max_change'],
  shared_trunk_norm=s['trunk_gradient_norm'],rows_trunk_norm=v['trunk_gradient_norm'],PPO=r['gradient_comparison'],
  frontier_shared=s['per_command'].get('frontier_3.25_3.5'),frontier_rows=v['per_command'].get('frontier_3.25_3.5'),
  slow_shared=s['per_command'].get('0.15'),slow_rows=v['per_command'].get('0.15'),running1_5_shared=s['per_command'].get('1.5'),running1_5_rows=v['per_command'].get('1.5')))
out=dict(actual_minibatches=len(rows),fixed_state_head_rows_only_preserved=True,isolated_SGD_step=1e-4,records=rows,not_Adam_global_clip_or_physical_retention=True)
(OUT/'current_row_response_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
