"""Teacher labels queried on truthful current learner states; CPU only."""
import sys,json,copy
from pathlib import Path
import numpy as np,torch
from torch import nn
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];sys.path.insert(0,str(OUT))
from phase_transfer import context_inputs,UPPER
torch.set_num_threads(2)
context=json.loads((OUT/'phase_context.json').read_text());data=np.load(OUT/'prior_causal_response/telemetry.npz')
names=list(data['joint_names']);lower=[i for i,n in enumerate(names) if n not in UPPER]
def load(path):
    saved=torch.load(path,weights_only=False,map_location='cpu')
    model=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
    model.load_state_dict({k.removeprefix('mlp.'):v for k,v in saved['actor_state_dict'].items() if k.startswith('mlp.')});return model.eval()
pause=json.loads((OUT/'diagnostic_pause.json').read_text());current=load(pause['checkpoint'])
teachers={kind:load(HERE/('runs/locomotion_20261004_r34_bilateral_swing_unloading/'+folder+'/candidate.pt')) for kind,folder in [('flight','038_run_150_train_007')]}
teachers['walking']=load(HERE/'runs/locomotion_20261004_r34_bilateral_swing_unloading/002_ppo_smoke/transferred_actor.pt')
ticks=np.flatnonzero(data['time']>=5.)[::10];rows=[]
for i,speed in enumerate([.35,.65,1.,1.5]):
    obs=torch.tensor(data['observations'][ticks,i*5*32:i*5*32+32:4].reshape(-1,400))
    actual=torch.tensor(data['raw_action'][ticks,i*5*32:i*5*32+32:4].reshape(-1,23))
    teacher=teachers['flight' if speed>=1.5 else 'walking'];table=torch.tensor(context['contexts'][str(speed)])
    with torch.no_grad():
        mean=current(obs);assert (mean-actual).abs().max()<1e-4
        truthful=teacher(obs);conditioned=teacher(context_inputs(obs,table,context['columns']))
        difference=mean[:,lower]-conditioned[:,lower].clamp(-3.,3.)
        effective=mean[:,lower].clamp(-3.,3.)-conditioned[:,lower].clamp(-3.,3.)
        truth_delta=truthful[:,lower].clamp(-3.,3.)-conditioned[:,lower].clamp(-3.,3.)
        features=current[:6](obs);derivative=.25*difference.clamp(-1,1)/(len(obs)*23.)
        gradient=derivative.T@features
    rows.append(dict(command=speed,samples=len(obs),reconstruction_max_error=float((mean-actual).abs().max()),
      raw_lower_error_RMS=float(difference.square().mean().sqrt()),effective_clipped_lower_error_RMS=float(effective.square().mean().sqrt()),
      teacher_conditioning_effective_response_RMS=float(truth_delta.square().mean().sqrt()),
      nominal_target_error_RMS_rad=float(effective.square().mean().sqrt()*.25),nominal_target_error_scope='Before coupled CAD projection, slew and torque dynamics',
      auxiliary_final_weight_norm=float(gradient.norm()),
      largest_clipped_target_errors=[dict(joint=names[lower[j]],RMS_rad=float(effective[:,j].square().mean().sqrt()*.25)) for j in torch.argsort(effective.square().mean(0),descending=True)[:4]]))
result=dict(completed=True,rows=rows,scope='Frozen teacher labels on CURRENT physically recorded neutral-head learner observations; conditioning only a separate teacher input. No production rewrite or deployment.')
(OUT/'teacher_target_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
