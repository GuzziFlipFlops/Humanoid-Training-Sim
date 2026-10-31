"""Saved synchronous COM/contact/joint/action data; diagnostic, not pressure."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--gate-review',default='initial_gate_review.json');p.add_argument('--output',default='gate_com_contact_audit.json');args=p.parse_args()
r=json.loads((OUT/args.gate_review).read_text());rows=[]
for suffix in ['nominal','pushed']:
    folder=Path(r['latest'][suffix]);f=folder/'gait_diagnostics_50hz.npz'
    with np.load(f) as z:
        com=z['com'];foot=z['foot_position'];contact=z['contact'];force=z['foot_force']
        root=z['root_state'];commands=z['commands'];time=z['time'];done=z['done']
        difference=float(abs(z['observed_command']-z['action_command']).max())
        finite=all(np.isfinite(z[k]).all() for k in ['com','q','qd','tau','target','root_state','raw_action','foot_force','foot_position'])
    assert finite and difference==0 and not done.any()
    cases=[]
    for speed in sorted(set(commands.tolist())):
        ids=np.flatnonzero(commands==speed);window=time>=2.
        xyz=com[window][:,ids];centroid=foot[window][:,ids].mean(2)
        cases.append(dict(command=float(speed),representative_robots=len(ids),
            COM_height_min_mean_max_m=[float(xyz[:,:,2].min()),float(xyz[:,:,2].mean()),float(xyz[:,:,2].max())],
            COM_pelvis_horizontal_offset_mean_m=float(np.linalg.norm(xyz[:,:,:2]-root[window][:,ids,:2],axis=-1).mean()),
            COM_feet_centroid_horizontal_offset_p95_m=float(np.quantile(np.linalg.norm(xyz[:,:,:2]-centroid[:,:,:2],axis=-1),.95)),
            two_loaded_contact_fraction=float(contact[window][:,ids].all(-1).mean()),
            unloaded_force_fraction=float((force[window][:,ids]<1.).all(-1).mean())))
    rows.append(dict(test=suffix,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),sample_rate_Hz=50.,
        first16_representatives=True,full_population_com_outside_proxy='Existing telemetry.metrics[:, :, 15] and unchanged gate,not raw full-population COM coordinates',
        finite_all_arrays=True,reset_ticks=0,observed_action_command_difference=difference,cases=cases))
result=dict(completed=True,checkpoint=r['latest']['checkpoint'],sha256=r['latest']['sha256'],tests=rows,
    scope='Actual mass-weighted COM from original telemetry,contact and synchronized joint/target/action data in exact matching gates. Moving COM offsets alone do not qualify or reject locomotion. No pressure certification,sole-margin change or new support classifier.',
    transformed_sole_context_preserved=True,physics_geometry_and_gates_unchanged=True)
(OUT/args.output).write_text(json.dumps(result,indent=2)+'\n')
print('Both exact matching gates: synchronous50Hz COM/contact/action data finite,zero command mismatch,no resets.')
