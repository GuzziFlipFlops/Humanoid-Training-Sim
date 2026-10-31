"""CPU fixed-state feature substitutions; never production observations."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
pause=json.loads((OUT/'diagnostic_pause.json').read_text())
saved=torch.load(pause['checkpoint'],map_location='cpu',weights_only=False)
d=torch.load(OUT/'current_gradient_00.pt',map_location='cpu',weights_only=False)
state={k.removeprefix('mlp.'):v for k,v in saved['actor_state_dict'].items() if k.startswith('mlp.')}
assert all(torch.equal(state[k],v) for k,v in d['state'].items())
m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23));m.load_state_dict(state);m.eval()
with np.load(OUT/'truthful_nominal/truthful_policy_states.npz') as z:rep=z['representative_observations'];time=z['representative_time']
with np.load(OUT/'truthful_nominal/gait_diagnostics_50hz.npz') as z:names=z['joint_names'].tolist()
ids=[names.index(n) for n in ['left_shoulder_pitch','right_shoulder_pitch']]
upper=[names.index(n) for n in ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']]
def cols(joints):return [base+j for base in [9,32,55] for j in joints]
def phase(x):return torch.atan2(x[:,-1,78],x[:,-1,79])
rows=[]
for command in [1.,2.5,3.,3.5]:
    x=d['raw_observations'].reshape(-1,5,80)
    mask=(x[:,:,6]-command).abs().max(1).values<1e-6
    train=x[mask];assert len(train)>100
    original=torch.tensor(rep[(time>=5)&(time<=17)]).reshape(-1,5,80)
    emask=(original[:,:,6]-command).abs().max(1).values<1e-6
    eva=original[emask][::2]
    distance=(phase(eva)[:,None]-phase(train)[None,:]);distance=torch.atan2(distance.sin(),distance.cos()).abs()
    chosen=distance.argmin(1);matched=train[chosen]
    assert float(distance.min(1).values.max())<.1
    experiments={}
    for label,joints in [('pitch_history',ids),('arm_history',[j for j in upper if not names[j].startswith('head')]),('head_history',[names.index(n) for n in ['head_yaw','head_pitch']]),('lower_body_history',[j for j in range(23) if j not in upper])]:
        changed=eva.clone();changed[:,:,cols(joints)]=matched[:,:,cols(joints)]
        experiments[label]=changed
    with torch.no_grad():
        y=m(eva.reshape(-1,400)).clamp(-3,3);ty=m(train.reshape(-1,400)).clamp(-3,3)
        h=m[:6](eva.reshape(-1,400));th=m[:6](train.reshape(-1,400))
        interventions={}
        for label,changed in experiments.items():
            cy=m(changed.reshape(-1,400)).clamp(-3,3);lower=[j for j in range(23) if j not in upper]
            interventions[label]=dict(shoulder_effective_mean_deg=(cy[:,ids].mean(0)*.25*180/torch.pi).tolist(),
                shoulder_output_change_RMS_action_units=float((cy[:,ids]-y[:,ids]).square().mean().sqrt()),
                direct_lower_output_change_RMS_action_units=float((cy[:,lower]-y[:,lower]).square().mean().sqrt()),
                changed_input_scope='Only all5frames of named joints q-offset/qdscaled/previous-applied-action; real source values from phase-matched exact-command training states. Remaining physical state/history stays fixed; this is an artificial CPU intervention, not a physically self-consistent rollout or causal motor experiment.')
        # Equal phase bins for both populations; avoids sampling-frequency bias.
        def balanced(values,frames):
            angle=torch.remainder(phase(frames),2*torch.pi);bins=(angle*8/(2*torch.pi)).long().clamp(0,7)
            return torch.stack([values[bins==b].mean(0) for b in range(8)]).mean(0)
        er=balanced(y[:,ids],eva)*.25*180/torch.pi;tr=balanced(ty[:,ids],train)*.25*180/torch.pi
        rows.append(dict(command=command,training_states=len(train),evaluation_states=len(eva),same_exact_first_minibatch_actor=True,
            max_phase_pair_error_rad=float(distance.min(1).values.max()),
            evaluation_phase_balanced_effective_mean_deg=er.tolist(),training_phase_balanced_effective_mean_deg=tr.tolist(),
            hidden_feature_centroid_distance=float((balanced(h,eva)-balanced(th,train)).norm()),
            training_history_head_RMS_deg=float(train[:,-1,[9+names.index('head_yaw'),9+names.index('head_pitch')]].square().mean().sqrt()*180/torch.pi),
            evaluation_history_head_RMS_deg=float(eva[:,-1,[9+names.index('head_yaw'),9+names.index('head_pitch')]].square().mean().sqrt()*180/torch.pi),
            CPU_counterfactuals=interventions))
result=dict(checkpoint=pause['checkpoint'],sha256=pause['checkpoint_sha256'],first_actual_minibatch_actor_bitwise_identical=True,
    rows=rows,production_observations_truthful=True,production_change=False,
    conclusion='Same-command phase balancing does not remove the measured training/evaluation state distribution difference. Fixed-state substitutions quantify model sensitivity; they neither identify a production observation fault nor prove mechanical effects absent. Lower outputs also change, so no history substitution or hybrid is eligible for deployment. A truthful-state temporal shoulder continuation remains a hypothesis requiring controlled learning and retention tests.')
(OUT/'state_context_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([dict(command=r['command'],training=r['training_phase_balanced_effective_mean_deg'],evaluation=r['evaluation_phase_balanced_effective_mean_deg'],interventions=r['CPU_counterfactuals']) for r in rows],indent=2))
