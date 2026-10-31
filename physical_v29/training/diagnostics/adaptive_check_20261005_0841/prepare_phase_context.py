"""Construct diagnostic teacher contexts from synchronized baseline telemetry."""
import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
data=np.load(OUT/'prior_causal_response/telemetry.npz')
names=list(data['joint_names']);columns=[start+names.index(n) for start in [9,32,55] for n in ['head_yaw','head_pitch']]
speeds=[.35,.65,1.,1.5];bins=64;contexts={}
for i,speed in enumerate(speeds):
    ids=slice((i*5+1)*32,(i*5+2)*32)
    obs=data['observations'][data['time']>=5.,ids].reshape(-1,5,80)[:,-1]
    phase=np.remainder(np.arctan2(obs[:,78],obs[:,79])/(2*np.pi),1.)
    index=np.floor(phase*bins).astype(int)%bins
    table=np.full((bins,6),np.nan,dtype=np.float32)
    for b in range(bins):
        selected=index==b
        if selected.any():table[b]=obs[selected][:,columns].mean(0)
    populated=np.flatnonzero(np.isfinite(table).all(1))
    assert len(populated)>=30
    x=np.r_[populated-bins,populated,populated+bins]
    for j in range(6):table[:,j]=np.interp(np.arange(bins),x,np.tile(table[populated,j],3))
    contexts[str(speed)]=table.tolist()
result=dict(schema='diagnostic_teacher_head_phase_context_v1',bins=bins,columns=columns,contexts=contexts,
  source_trace=str(OUT/'prior_causal_response/telemetry.npz'),source_sha256=hashlib.sha256((OUT/'prior_causal_response/telemetry.npz').read_bytes()).hexdigest(),
  source_scope='Truthful simultaneous original teacher head q/qd/previous action, after5s. Diagnostic teacher only; never student or production observations.')
(OUT/'phase_context.json').write_text(json.dumps(result,indent=2)+'\n')
text=(OUT/'physical_prior_causal.py').read_text()
text=text.replace("out=OUT/'prior_causal_response'", "out=OUT/'phase_causal_response'")
text=text.replace("['latest','baseline','neutral_actual','neutral_reference','neutral_constant']", "['latest','baseline','neutral_actual','neutral_reference','neutral_phase_context']")
text=text.replace("    speeds=torch.tensor",'''    phase_context=json.loads((OUT/'phase_context.json').read_text())
    phase_tables={float(k):torch.tensor(v,device=raw.device) for k,v in phase_context['contexts'].items()}
    speeds=torch.tensor''')
text=text.replace("                    actor_input[:,:,head_channels]=head_context[(group[4],float(group[1]))]",'''                    table=phase_tables[float(group[1])]
                    angle=torch.remainder(torch.atan2(actor_input[:,:,78],actor_input[:,:,79])/(2*torch.pi),1.)
                    position=angle*64.; lower=position.floor().long()%64; upper=(lower+1)%64
                    fraction=(position-position.floor()).unsqueeze(-1)
                    actor_input[:,:,head_channels]=table[lower]*(1.-fraction)+table[upper]*fraction''')
text=text.replace('prior_causal_probe_failure.json','phase_causal_probe_failure.json')
(OUT/'physical_phase_causal.py').write_text(text)
resume=(OUT/'resume_after_diagnosis.py').read_text().replace("'physical_prior_causal.py'", "'physical_prior_causal.py','physical_phase_causal.py'")
(OUT/'resume_after_diagnosis.py').write_text(resume)
print(json.dumps(dict(bins=bins,commands=speeds,production_inputs_unchanged=True)))
