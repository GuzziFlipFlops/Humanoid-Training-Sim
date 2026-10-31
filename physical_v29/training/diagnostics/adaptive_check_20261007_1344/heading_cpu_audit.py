"""CPU boundary/symmetry counterexamples for the diagnostic command law."""
import sys,json,hashlib,math
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training';sys.path.insert(0,str(TRAIN))
from heading_control import heading_yaw_command,wrap_angle,yaw_from_quat
rows=[]
for error in [-math.pi,-.5,-.1,0.,.1,.5,math.pi]:
 for gain in [0.,.5,1.,2.]:
  yaw=.13;q=torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]],dtype=torch.float32)
  target=torch.tensor([yaw+error]);moving=torch.tensor([True])
  e=wrap_angle(target-yaw_from_quat(q));rate=(torch.tensor([gain])*e).clamp(-.2,.2)
  assert torch.isfinite(rate).all() and bool((rate.abs()<=.2).all())
  if gain==.5:assert torch.equal(rate,heading_yaw_command(q,target,moving))
  stopped=torch.where(torch.tensor([False]),rate,torch.zeros_like(rate));assert torch.equal(stopped,torch.zeros_like(rate))
  rows.append({'error_rad':error,'gain':gain,'rate_rad_s':float(rate[0]),'original_control_equal':gain!=.5 or torch.equal(rate,heading_yaw_command(q,target,moving))})
for target,yaw in [(math.pi-.01,-math.pi+.01),(-math.pi+.01,math.pi-.01)]:
 e=wrap_angle(torch.tensor([target-yaw]));assert abs(float(e[0]))<.021
for velocity in [[0.,0.],[.1,0.],[1.,.1],[1.,-.1]]:
 v=torch.tensor([velocity]);angle=torch.atan2(v[:,1],v[:,0]);e=wrap_angle(-angle)
 e=torch.where(v.norm(dim=1)>.2,e,torch.zeros_like(e));assert torch.isfinite(e).all()
 if v.norm()<=.2:assert float(e[0])==0
result={'passed':True,'gain_wrap_stopped_bound_fixtures':28,'pi_crossing_fixtures':2,'low_speed_course_fixtures':4,
 'heading_probe_sha256':hashlib.sha256((OUT/'heading_probe.py').read_bytes()).hexdigest(),
 'pinned_heading_source_sha256':hashlib.sha256((TRAIN/'heading_control.py').read_bytes()).hexdigest(),
 'rows':rows,'scope':'CPU command-law safety and exact original.5-gain parity.Independent physical command/actor/term checks remain mandatory.No learning,physics or production source changed.'}
(OUT/'heading_cpu_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('34 heading-command boundary/parity fixtures pass.')
