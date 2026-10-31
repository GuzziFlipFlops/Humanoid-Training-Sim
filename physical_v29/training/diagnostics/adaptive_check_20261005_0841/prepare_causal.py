"""Prepare diagnostic-only probes; no canonical or frozen runtime changes."""
from pathlib import Path
import shutil
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261005_0540'
for name in ['pause_for_diagnosis.py','resume_after_diagnosis.py']:
    target=OUT/name
    assert not target.exists()
    text=(OLD/name).read_text()
    if name.startswith('resume'):
        text=text.replace("'physical_head_causal.py','run_matched_exposure.py','worker_unitree.py'", "'physical_prior_causal.py','run_matched_transfer.py','worker_unitree.py'")
    target.write_text(text)
text=(OLD/'physical_head_causal.py').read_text()
start=text.index("    groups=[")
end=text.index("    per=32;",start)
text=text[:start]+'''    speeds_to_test=[.35,.65,1.,1.5]
    variants=['latest','baseline','neutral_actual','neutral_reference','neutral_constant']
    groups=[]
    for speed in speeds_to_test:
        old='flight' if speed>=1.5 else 'walking'
        for variant in variants:
            groups.append((str(speed)+'_'+variant,speed,0.,0.,'latest' if variant=='latest' else old,.40,0.))
'''+text[end:]
text=text.replace("out=OUT/'head_causal_response'", "out=OUT/'prior_causal_response'")
text=text.replace("'flight':HERE/'runs/locomotion_20261004_r34_bilateral_swing_unloading/038_run_150_train_007/candidate.pt'", "'flight':HERE/'runs/locomotion_20261004_r34_bilateral_swing_unloading/038_run_150_train_007/candidate.pt', 'walking':HERE/'runs/locomotion_20261004_r34_bilateral_swing_unloading/002_ppo_smoke/transferred_actor.pt'")
text=text.replace("    speeds=torch.tensor",'''    # Constant teacher-only head context is estimated from preserved exact
    # teacher nominal traces. Test it physically before proposing training labels.
    certificate=json.loads((checkpoint.parent/'qualified_prior_certificate.json').read_text())
    head_context={}
    head_channels=[9+raw.lookup[x] for x in ['head_yaw','head_pitch']]
    head_channels += [32+raw.lookup[x] for x in ['head_yaw','head_pitch']]
    head_channels += [55+raw.lookup[x] for x in ['head_yaw','head_pitch']]
    traces={'flight':HERE/'runs/locomotion_20261004_r34_bilateral_swing_unloading/039_run_150_gate_008_nominal/telemetry.npz'}
    # The certificate contains exact prior qualification references.
    teacher_path=files['walking']
    matching=[]
    for gatepath in RUN.glob('*prior*nominal/evaluation.json'):
        matching.append(gatepath)
    if not matching:
        matching=list(RUN.glob('*walk_100*nominal/evaluation.json'))
    assert matching,'Missing current qualified walking teacher trace'
    traces['walking']=matching[-1].parent/'telemetry.npz'
    for kind,path in traces.items():
        trace=np.load(path); jnames=list(trace['joint_names'])
        hids=[jnames.index(x) for x in ['head_yaw','head_pitch']]
        evaluation=json.loads((path.parent/'evaluation.json').read_text())
        for ci,case in enumerate(evaluation['cases']):
            speed=case['statistics']['command_m_s']
            mask=trace['case_labels']==ci
            q=trace['q'][trace['time']>=3.][:,mask][:,:,hids].mean((0,1))
            # Default head position is zero; current native action scale is .25.
            head_context[(kind,float(speed))]=torch.tensor(np.r_[q,[0.,0.],q/.25],device=raw.device,dtype=torch.float32)
    (out/'teacher_head_context.json').write_text(json.dumps({str(k):v.cpu().tolist() for k,v in head_context.items()},indent=2))
    speeds=torch.tensor''')
start=text.index("                if gi==3 and t>=3.:")
end=text.index("\n        action=action.clone()",start)
text=text[:start]+'''                variant=gi%5
                if variant==3 and t>=3.:
                    baseline=slice((gi//5*5+1)*per,(gi//5*5+2)*per)
                    actor_input[:,:,head_channels]=observed[baseline,:,head_channels]
                if variant==4 and t>=3.:
                    actor_input[:,:,head_channels]=head_context[(group[4],float(group[1]))]
                action[ids]=actors[group[4]]({'policy':actor_input.reshape(per,400)})
                if variant>=2 and t>=3.:
                    action[ids,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]=0.
'''+text[end:]
text=text.replace('head_causal_probe_failure.json','prior_causal_probe_failure.json')
text=text.replace('128 physical robots, 20s; bounded diagnostic policy-phase or joint-action changes after3s.', '640 physical robots, 20s at .35/.65/1./1.5 commands; neutral head and teacher-context interventions after3s.')
(OUT/'physical_prior_causal.py').write_text(text)
print('Prepared owned pause/resume and 640-robot diagnostic, with no training edits.')
