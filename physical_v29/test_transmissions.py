"""Independent finite-difference, shared-envelope, speed and virtual-work tests."""
import json
from pathlib import Path
import torch
from cad_differential import CadDifferential

HERE=Path(__file__).resolve().parent
OUT=HERE/json.loads((HERE/'assets/export_metadata.json').read_text())['hardware_validation_dir']
OUT.mkdir(parents=True,exist_ok=True)
torch.manual_seed(30)
specs=json.loads((HERE/'assets/differential_geometry.json').read_text())
report={'scope':'CPU float64 analytical transmission validation; not contact/dynamics/strength qualification','differentials':{}}
for key,spec in specs.items():
    model=CadDifferential(spec)
    q=torch.cat((torch.zeros(1,2,dtype=torch.float64),(torch.rand(120,2,dtype=torch.float64)-.5)*.3))
    theta,J,valid,error=model.state(q)
    assert valid.all(),key
    columns=[]
    eps=1e-6
    for i in range(2):
        delta=torch.zeros_like(q);delta[:,i]=eps
        plus=model.state(q+delta)[0];minus=model.state(q-delta)[0]
        columns.append((plus-minus)/(2*eps))
    fd=torch.stack(columns,-1)
    diff=float((J-fd).abs().max())
    assert diff<1e-6,(key,diff)
    theta32,J32,valid32,error32=model.state(q.float())
    errorJ32=float((J32.double()-J).abs().max())
    assert valid32.all() and errorJ32<.003 and error32.max()<.003,(key,errorJ32,error32.max())
    motor_limit=30 if key=='waist' else 45
    speed=400*2*torch.pi/60
    qd=(torch.rand_like(q)-.5)*2
    desired=(torch.rand_like(q)-.5)*1000
    efficiency=.95
    effort,state=model.limit(q,qd,desired,motor_limit,speed,efficiency=efficiency)
    assert state['motor_applied'].abs().max()<=motor_limit+1e-10
    work_error=float(((effort*qd).sum(1)-efficiency*(state['motor_applied']*state['motor_velocity']).sum(1)).abs().max())
    assert work_error<1e-8
    # Both separate pure-axis peak maxima together must exceed a motor limit.
    invt=torch.linalg.inv(J[:1].transpose(1,2))
    pure=efficiency*motor_limit/invt[0].abs().amax(0)
    both=pure[None]
    joint_applied,state=model.limit(q[:1],torch.zeros_like(q[:1]),both,motor_limit,speed,efficiency=efficiency)
    assert state['motor_requested'].abs().max()>motor_limit*1.5
    assert state['motor_applied'].abs().max()<=motor_limit+1e-10
    # Overspeed permits braking but removes accelerating effort.
    motor_vel=torch.tensor([[speed*1.1,speed*1.2]],dtype=torch.float64)
    fastqd=torch.linalg.solve(J[:1],motor_vel[...,None]).squeeze(-1)
    positive_motor=torch.tensor([[motor_limit,motor_limit]],dtype=torch.float64)
    fastdesired=(J[:1].transpose(1,2)@positive_motor[...,None]).squeeze(-1)
    _,fast=model.limit(q[:1],fastqd,fastdesired,motor_limit,speed,efficiency=efficiency)
    assert fast['motor_applied'].abs().max()<1e-10
    report['differentials'][key]={'samples':len(q),'max_J_vs_finite_difference_error':diff,'float32_J_error':errorJ32,'float32_closure_error_mm':float(error32.max()),'max_closure_error_mm':float(error.max()),'max_virtual_work_error_W':work_error,'shared_motor_limit_pass':True,'overspeed_motoring_pass':True,'neutral_J':J[0].tolist(),'simultaneous_pure_peaks_requested_motor_Nm':state['motor_requested'][0].tolist(),'simultaneous_pure_peaks_applied_joint_Nm':joint_applied[0].tolist(),'pass':True}
report['all_pass']=all(v['pass'] for v in report['differentials'].values())
(OUT/'transmission_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
