import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
a=json.loads((OUT/'current_gradient_analysis.json').read_text())
summary=[]
for b in a['batches']:
 summary.append(dict(batch=b['batch'],running_teacher_cosine=b['running_teacher_cosine'],upper_PPO_cosine=b['upper_auxiliary_PPO_cosine'],
  standing_subtraction_error=b['CPU_standing_subtraction_max_error'],actual_PPO_norm=b['all_PPO_parameter_norm'],
  prior_norm=b['prior_parameter_norm'],upper_norm=b['upper_auxiliary_parameter_norm'],commands=[r for r in b['actual_per_command_gradients'] if r['command'] in [.15,1.5,3.,'frontier_above3']],
  target_errors=[r for r in b['per_command'] if r['command'] in [.15,1.5,3.]]))
(OUT/'passive_gradient_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
for r in summary:
 print(json.dumps(dict(batch=r['batch'],prior_norm=r['prior_norm'],PPO_norm=r['actual_PPO_norm'],upper_norm=r['upper_norm'],running_teacher_cosine=r['running_teacher_cosine'],
  commands=[{k:v for k,v in c.items() if k in ['command','sample_fraction','advantage_mean','actual_PPO_mean_parameter_norm','teacher_parameter_norm','PPO_teacher_cosine','teacher_selected_samples']} for c in r['commands']])))
