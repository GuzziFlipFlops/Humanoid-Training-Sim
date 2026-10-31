"""Compact schema read; never print raw tensor arrays or full run history."""
import json
from pathlib import Path
out=Path(__file__).resolve().parent
files=['controller_cpu_tests.json','teacher_compatibility_analysis.json','teacher_target_history_analysis.json',
 'current_substep_motor_attribution.json','current_gradient_analysis.json','diagnostic_resume.json',
 'bounded_diagnosis/validation.json','teacher_smoke/result.json','teacher_compatibility/result.json',
 'head_arm_factorial/CPU_tests.json','head_arm_factorial/teacher_compatibility_analysis.json',
 'head_arm_factorial/bounded_diagnosis/validation.json','head_arm_factorial/diagnostic_resume.json',
 'head_arm_factorial/teacher_compatibility/result.json','primary_motion_reference_review.json']
for name in files:
 d=json.loads((out/name).read_text(encoding='utf-8-sig'))
 info={'file':name,'type':type(d).__name__}
 if isinstance(d,dict):
  info['keys']=list(d)
  for k,v in d.items():
   if isinstance(v,(str,int,float,bool)) or v is None:info[k]=v
   elif isinstance(v,list):
    info[k+'_count']=len(v)
    if v and isinstance(v[0],dict):info[k+'_first_keys']=list(v[0])
 print(json.dumps(info))
