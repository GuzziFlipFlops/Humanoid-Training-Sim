"""Counterfactual teacher-label/gradient ownership fixtures, CPU only."""
import torch,json,sys,hashlib
from pathlib import Path
from tensordict import TensorDict
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
from upper_body_refinement import LocomotionOnlyRetention,UPPER_JOINTS
torch.set_num_threads(4);torch.manual_seed(7)
names=list(UPPER_JOINTS)+['body_'+str(i) for i in range(15)]
class Actor(torch.nn.Module):
 def __init__(self):
  super().__init__();self.mlp=torch.nn.Sequential(torch.nn.Linear(400,23))
 def forward(self,obs):return self.mlp(obs['policy'])
state=Actor().state_dict();teacher_state=Actor().state_dict()
def run(raw,variant,command_range):
 actor=Actor();actor.load_state_dict(state);teacher=Actor();teacher.load_state_dict(teacher_state)
 before=raw.clone();obs=TensorDict({'policy':raw.clone()},batch_size=[len(raw)])
 hook=LocomotionOnlyRetention(actor,teacher,joint_names=names,command_range=command_range)
 if variant:hook.phase_tables={}
 means=actor(obs);means.retain_grad();means.sum().backward()
 assert torch.equal(obs['policy'],before),'Student observation modified'
 gradient=means.grad.clone()-1.;audit=hook.read_audit();hook.close()
 return means.detach(),gradient,audit
fixtures=[]
for command in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5]:
 raw=torch.randn(48,5,80)*.02;raw[:,:,6:9]=0.;raw[:,:,6]=command
 phase=torch.linspace(.03,.97,48)[:,None].expand(-1,5)
 raw[:,:,78]=torch.sin(phase*2*torch.pi);raw[:,:,79]=torch.cos(phase*2*torch.pi)
 raw=raw.reshape(-1,400)
 a,ga,aa=run(raw,False,(.1,3.));b,gb,ab=run(raw,True,(.1,3.))
 assert torch.equal(a,b),'Teacher labels changed actor forward'
 assert torch.equal(ga[:,:8],torch.zeros_like(ga[:,:8])) and torch.equal(gb[:,:8],torch.zeros_like(gb[:,:8]))
 if command not in [.15,.35,.65,1.]:assert torch.equal(ga,gb),'Changed unconditioned/current/stopped gradients'
 else:assert float((ga-gb).abs().max())>0.,'Fixture failed to exercise context correction'
 assert ab['teacher_phase_conditioned_samples']==0
 # Independently reconstruct unchanged .25/23-output MSE derivative.
 teacher=Actor();teacher.load_state_dict(teacher_state)
 with torch.no_grad():target=teacher(TensorDict({'policy':raw},batch_size=[len(raw)])).clamp(-3,3)
 expected=.5*(b-target)/b.numel();expected[:,:8]=0.
 if not .1<command<=3.:expected.zero_()
 assert torch.allclose(gb,expected,atol=7e-8,rtol=1e-4)
 sa,gsa,_=run(raw,False,None);sb,gsb,_=run(raw,True,None)
 assert torch.equal(sa,sb) and torch.equal(gsa,gsb),'Stopped teacher derivative changed'
 fixtures.append(dict(command=command,actor_forward_equal=True,truthful_gradient_reconstructed=True,
  student_observations_equal=True,stopped_teacher_equal=True,upper_direct_gradient_zero=True,
  current_unselected_samples_untouched=command>3. or command<=.1,
  teacher_gradient_difference_max=float((ga-gb).abs().max())))
result=dict(passed=True,fixtures=fixtures,fixture_count=len(fixtures),
 scope='Synthetic exact derivative and ownership checks; not physical gait or convergence proof',
 candidate_path=str(OUT/'teacher_context_candidate.py'),candidate_sha256=hashlib.sha256((OUT/'teacher_context_candidate.py').read_bytes()).hexdigest())
(OUT/'context_cpu_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=True,fixtures=len(fixtures))))
