"""Prepare complete bounded harness and CPU checks before launch."""
import json,hashlib
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261007_0844'
from heading_candidate import course_command,patch_environment
from heading_control import heading_yaw_command
from types import SimpleNamespace
checks=[]
for speed in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5,4.,5.]:
 for yaw in [-3.14,-.6,0.,.6,3.14]:
  q=torch.tensor([[torch.cos(torch.tensor(yaw/2)),0.,0.,torch.sin(torch.tensor(yaw/2))]])
  target=torch.tensor([0.]);velocity=torch.tensor([[speed,.25,0.]])
  class Fixture:
   def _update_observation_command(self):
    if self._command_tick==self.common_step_counter:return
    self.commands[:,2]=heading_yaw_command(q,target,self.commands[:,:2].norm(dim=1)>.1)
    self._command_tick=self.common_step_counter
  patch_environment(Fixture);f=Fixture();f._command_tick=-1;f.common_step_counter=0
  f.commands=torch.tensor([[speed,0.,0.]]);f.heading_target=target
  f.robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=q,root_lin_vel_w=velocity))
  old=heading_yaw_command(q,target,f.commands[:,:2].norm(dim=1)>.1)
  f._update_observation_command()
  if speed<=3.:assert torch.equal(f.commands[:,2],old)
  else:assert torch.equal(f.commands[:,2],course_command(q,target,velocity))
  held=f.commands.clone();f.robot.data.root_lin_vel_w=torch.tensor([[1.,-2.,0.]])
  f._update_observation_command();assert torch.equal(f.commands,held)
  assert float(f.commands[:,2].abs().max())<=.200001
  checks.append(dict(speed=speed,yaw=yaw,prior_parity=speed<=3.,consistent_tick=True))
(OUT/'heading_candidate_cpu.json').write_text(json.dumps(dict(passed=True,checks=checks,count=len(checks),wrapper_sha256=hashlib.sha256((OUT/'heading_candidate.py').read_bytes()).hexdigest()),indent=2))
script=(OLD/'matched_context_test.py').read_text()
script=script.replace('label-provenance test','course-command test').replace("root=out/'matched_context'","root=out/'matched_heading'")
script=script.replace("candidate='Disable tilted-head phase tables only on certified neutral-head locomotion teacher349',\n old_teacher_context_source='R34 tilted-head .15/.35/.65/1 histories',", "candidate='For x>3, add unit-gain fixed-frame velocity-course feedback to existing half-gain heading yaw command, clamp +/-0.2rad/s',\n prior_commands_through3_bitwise_unchanged=True,teacher_context_unchanged=True,")
script=script.replace('weights_physics_actions_gates_upper_PPO_unchanged','reward_weights_physics_actions_gates_upper_PPO_unchanged')
script=script.replace("['worker_unitree.py','teacher_context_candidate.py','capture_current_gradients.py','diverse_initial_eval.py']","['worker_unitree.py','heading_candidate.py','diverse_heading_eval.py']")
script=script.replace("wrapper=out/'teacher_context_candidate.py'","wrapper=out/'heading_candidate.py'")
a=script.index('# Actual current-state passive PPO labels;');b=script.index("smoke=root/'candidate_smoke'",a)
script=script[:a]+script[b:]
a=script.index("cert=read(args['locomotion_prior_certificate'])");b=script.index("for branch in ['control','candidate']:",a)
script=script[:a]+script[b:]
script=script.replace("str(out/'diverse_initial_eval.py')","str(out/'diverse_heading_eval.py')")
script=script.replace("'--seed','2301']+common,g)","'--seed','2301']+common+(['--candidate-course'] if branch=='candidate' else []),g)")
(OUT/'matched_heading_test.py').write_text(script)
review=(OLD/'review_context.py').read_text().replace("ROOT=OUT/'matched_context'","ROOT=OUT/'matched_heading'")
review=review.replace("  if branch=='candidate':assert row['conditioned_samples']==0\n",'')
review=review.replace("matched_context_partial.json","matched_heading_partial.json").replace("matched_context_review.json","matched_heading_review.json")
(OUT/'review_heading.py').write_text(review)
for p in ['heading_candidate.py','diverse_heading_eval.py','matched_heading_test.py','review_heading.py']:
 compile((OUT/p).read_text(),str(OUT/p),'exec')
print(json.dumps(dict(cpu_checks=len(checks),compiled=True)))
