"""Validate saved diagnostic evidence and check protected checkpoint/source hashes."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from analyze_policy_gait import cleaned_contacts, intervals

root=Path(sys.argv[1])
reports=[]
for folder in ('nominal','randomized','waist_clamp_ablation'):
    path=root/folder
    if not (path/'episodes.npz').exists():
        continue
    meta=json.loads((path/'manifest.json').read_text())
    data=np.load(path/'physics.npy',mmap_mode='r')
    params=np.load(path/'runtime_parameters.npz')
    episodes=np.load(path/'episodes.npz')
    checkpoints={key:hashlib.sha256(Path(p).read_bytes()).hexdigest()==meta['checkpoint_sha256'][key]
                 for key,p in meta['checkpoint_paths'].items()}
    # The recorder can evolve between controlled tests; production sources must not.
    protected={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
               for p,h in meta['source_sha256'].items() if not p.endswith('diagnose_policy_gait.py')}
    finite=bool(np.isfinite(data).all())
    expected=round(meta['duration_s']/meta['physics_dt'])
    cases=[]
    t=(np.arange(len(data))+1)*meta['physics_dt']
    a,b=meta['fields']['foot_force_w']
    forces=data[:,:,a:b].reshape(len(data),meta['num_envs'],2,3)
    for case in meta['cases']:
        ids=case['env_ids']
        falls=episodes['first_fall_s'][ids]
        valid=(t[:,None]>3.) & (t[:,None]<np.where(np.isnan(falls),np.inf,falls)[None,:])
        fz=forces[:,ids][:,:,:,2].sum(-1)
        weight=np.broadcast_to(params['masses'][ids].sum(-1)*9.81,fz.shape)
        cases.append(dict(case=case['name'],mean_total_foot_fz_over_weight=float((fz/weight)[valid].mean()) if valid.any() else None))
    names=meta['joint_names']; bodies=meta['body_names']
    joint_sym={}
    for field in ('stiffness','damping','armature','effort_limits','velocity_limits'):
        diffs=[np.max(np.abs(params[field][:,i]-params[field][:,names.index(n.replace('left_','right_'))]))
               for i,n in enumerate(names) if n.startswith('left_')]
        joint_sym[field+'_max_abs_left_right_difference']=float(max(diffs))
    mass_diff=max(float(np.max(np.abs(params['masses'][:,i]-params['masses'][:,bodies.index(n.replace('left_','right_'))])))
                  for i,n in enumerate(bodies) if n.startswith('left_'))
    contact_sensitivity=[]
    selected=next(c for c in meta['cases'] if c['name']=='run_1p8')
    for filter_ms in (20,30,40,50):
        durations=[[],[]]
        for idx in selected['env_ids']:
            cut=episodes['first_fall_s'][idx]
            stop=len(data) if np.isnan(cut) else min(len(data),int(cut/meta['physics_dt']))
            for side in range(2):
                contacts=cleaned_contacts(forces[:stop,idx,side,2]>5.,round(filter_ms/1000/meta['physics_dt']))
                for lo,hi in intervals(contacts):
                    if lo*meta['physics_dt']>3. and hi<stop:
                        durations[side].append((hi-lo)*meta['physics_dt'])
        contact_sensitivity.append(dict(filter_ms=filter_ms,left_right_stance_s=[float(np.mean(v)) for v in durations]))
    report=dict(dataset=folder,completed=meta['completed'],shape=list(data.shape),expected_frames=expected,
        all_fields_finite=finite,checkpoints_unchanged=checkpoints,production_sources_unchanged=protected,
        total_mass_range_kg=[float(params['masses'].sum(1).min()),float(params['masses'].sum(1).max())],
        max_paired_link_mass_difference_kg=mass_diff,joint_symmetry=joint_sym,cases=cases,
        run_1p8_contact_filter_sensitivity=contact_sensitivity)
    assert meta['completed'] and len(data)==expected and finite
    assert all(checkpoints.values()) and all(protected.values())
    reports.append(report)
(root/'evidence_validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
print(json.dumps(reports,indent=2))
