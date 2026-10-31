"""Bind non-cancelling upper motion measurements to matching diagnostic gates."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
records=[]
for seed in [179,191]:
 for branch in ['control','candidate']:
  f=OUT/f'seed{seed}_{branch}_upper'
  if not (f/'upper_joint_time_audit.json').exists():continue
  audit=read(f/'upper_joint_time_audit.json');bias=read(f/'upper_bias_audit.json');target=read(f/'upper_target_bias_audit.json')
  record=dict(seed=seed,branch=branch,updates=128,folder=str(f),checkpoint=bias['checkpoint'],sha256=bias['sha256'],visual_QA_saved=(f/'visual_QA.json').exists(),records=[])
  for a in audit['records']:
   if a['actor_tag']!='latest':continue
   b=next(r for r in bias['records'] if r['test']==a['test'] and abs(r['speed_m_s']-a['command_m_s'])<1e-6)
   t=next(r for r in target['records'] if r['test']==a['test'] and abs(r['speed']-a['command_m_s'])<1e-6)
   record['records'].append(dict(command=a['command_m_s'],test=a['test'],head_RMS_deg=a['head_rms_deg'],head_p95_deg=a['head_p95_deg'],
    shoulders={j:dict(population=a['joints'][j],slow_bias=b['joints'][j],representative_physical_target=t['joints'][j]) for j in ['left_shoulder_pitch','right_shoulder_pitch']},
    coordination=a['coordination'],representative_50Hz=a['representative_50hz']))
  records.append(record)
result=dict(completed=len(records)==4,records=records,scope='Actual128-update control/candidate upper angles,targets,velocities and slow bias. Diagnostic feature comparisons, no human or mapped Unitree encoder certification. Physical lower-body coupling can change despite unchanged upper auxiliary.')
(OUT/'upper_matched_comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(seed=r['seed'],branch=r['branch'],head=[x['head_RMS_deg'] for x in r['records'] if x['command']==3.5 and x['test']=='nominal'],visual_QA=r['visual_QA_saved']) for r in records]))
