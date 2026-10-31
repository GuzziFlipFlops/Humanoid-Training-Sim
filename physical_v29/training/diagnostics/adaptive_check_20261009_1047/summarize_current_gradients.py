"""Compact passive per-command gradient evidence, not a repair claim."""
from pathlib import Path
import json
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
r=read(OUT/'current_gradient_analysis.json');batches=r['batches'];assert len(batches)==4
keys=['prior_parameter_norm','all_PPO_parameter_norm','upper_auxiliary_parameter_norm',
      'running_teacher_cosine','prior_PPO_teacher_cosine','upper_auxiliary_PPO_cosine',
      'CPU_standing_subtraction_max_error','moving_surrogate_derivative_cosine']
ranges={k:dict(minimum=min(b[k] for b in batches),maximum=max(b[k] for b in batches)) for k in keys}
commands=[]
for c in [0,.15,.35,.65,1,1.5,2,2.5,3,'frontier_above3','continuous_below3']:
    rows=[x for b in batches for x in b['actual_per_command_gradients'] if x['command']==c]
    commands.append(dict(command=c,actual_minibatches=len(rows),
        **{k:dict(minimum=min(x[k] for x in rows),maximum=max(x[k] for x in rows)) for k in ['sample_fraction','advantage_mean','positive_advantage_fraction','actual_PPO_mean_parameter_norm','teacher_parameter_norm','importance_ratio_clip_fraction']},
        teacher_PPO_cosines=[x['PPO_teacher_cosine'] for x in rows]))
summary=dict(completed=True,checkpoint=read(OUT/'diagnostic_pause.json')['checkpoint'],
    checkpoint_sha256=read(OUT/'diagnostic_pause.json')['checkpoint_sha256'],norm_and_reproduction_ranges=ranges,per_command=commands,
    exact_passive_incoming_PPO_before_both_hooks=True,
    scope='Four real4096-env PPO minibatches. Parameter contributions use whole-minibatch scaling before global clipping and Adam; exploration parameters omitted. Normalized advantage signs, norm size or mixed alignment do not prove a sampling defect, globally opposed gradients or a safe reward/teacher change.')
(OUT/'passive_gradient_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(ranges,indent=2))
