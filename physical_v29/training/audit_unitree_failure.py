"""CPU reconstruction of the exact transmission at recorded failure ticks."""
import json,sys
from pathlib import Path
import torch
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent))
from cad_differential import CadDifferential
out=HERE/'diagnostics/unitree_termination_20261002';out.mkdir(parents=True,exist_ok=True)
trace=json.loads((HERE/'diagnostics/unitree_port_20261002/r20_policy_failure_visuals/telemetry.json').read_text())
specs=json.loads((HERE.parent/'assets/differential_geometry.json').read_text())
names=trace['joint_names'];cases=[]
for case in ['policy_stand','policy_walk']:
    rows=[r for r in trace['rows'] if r['case']==case]
    fault=next(r for r in rows if r['done'])
    frames=[]
    for r in rows[:7]:
        states={}
        for key,spec in specs.items():
            ids=[names.index(n) for n in spec['joint_names']]
            target=torch.tensor([[r['target'][i] for i in ids]],dtype=torch.float32)
            q=torch.tensor([[r['q'][i] for i in ids]],dtype=torch.float32)
            theta,J,valid,err=CadDifferential(spec).state(q)
            _,_,target_valid,_=CadDifferential(spec).state(target)
            lohi=torch.tensor([m['motor_limits_rad'] for m in spec['motors']])
            states[key]=dict(joints=spec['joint_names'],q_rad=q[0].tolist(),target_rad=target[0].tolist(),
                theta_rad=theta[0].tolist(),motor_limits_rad=lohi.tolist(),J=J[0].tolist(),
                valid=bool(valid[0]),target_valid=bool(target_valid[0]),closure_error_mm=err[0].tolist(),
                motor_angle_margin_rad=torch.minimum(theta[0]-lohi[:,0],lohi[:,1]-theta[0]).tolist())
        frames.append(dict(t=r['t'],done=r['done'],failures=r['failures'],root_height_m=r['root'][2],states=states))
    cases.append(dict(case=case,first_failure_s=fault['t'],first_failure_causes=fault['failures'],frames=frames))
result=dict(checkpoint='r20 68-update candidate',scope='Saved 50 Hz actual joint poses; actuator validity uses last 200 Hz substep, so fault timestamps can precede the reconstructed end-of-step pose',cases=cases)
(out/'transmission_failure.json').write_text(json.dumps(result,indent=2))
for c in cases:
    print(json.dumps(dict(case=c['case'],first_failure_s=c['first_failure_s'],first_failure_causes=c['first_failure_causes'],
        first_invalid=[dict(t=r['t'],pair=k,**v) for r in c['frames'] for k,v in r['states'].items() if not v['valid']][:3]),indent=2))
