import json
from pathlib import Path
OUT=Path(__file__).resolve().parent;F=OUT/'motor_retention_response'
r=json.loads((F/'result.json').read_text());a=json.loads((F/'substep_motor_audit.json').read_text());rows=[]
for g,mg in zip(r['groups'],a['commands']):
 assert g['name']==mg['name']
 clipping=[m for m in mg['motors'] if m['speed_clipped_fraction']>0]
 x=dict(name=g['name'],command=g['command'],actor=g['actor'],signed_speed=g['mean_forward_world_speed'],lateral_speed=g['mean_absolute_world_lateral_velocity'],
  flight=g['unloaded_geometric_flight_fraction'],survival=g['survival'],resets=g['resets'],head_RMS_deg=g['head_RMS_deg'],reward_rate=g['mean_reward_rate'],
  driving_clip_motors=clipping,highest_p99_cap_ratio=max(m['p99_cap_ratio'] for m in mg['motors']))
 rows.append(x)
result=dict(completed=True,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],records=rows,
 max_reward_term_sum_error=r['maximum_reward_term_sum_error'],command_mismatch=r['maximum_command_difference'],
 scope='Two paired independent initial-state seeds,32 robots/cohort. Unchanged current and qualified teacher; normal limits. Not a final qualification or relevant actuator ceiling.',endpoint_proven=False)
(OUT/'motor_retention_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
for row in rows:print(json.dumps(row))
