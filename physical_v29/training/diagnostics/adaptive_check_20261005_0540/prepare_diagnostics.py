"""Stage isolated probes; never modify a live frozen run."""
import json,shutil,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
run=Path(json.loads((HERE/'ACTIVE_RUN.json').read_text())['run'])
assert run.name=='locomotion_20261004_r35_upper_body_refinement'
old=OUT.parent/'adaptive_check_20261005_0240'
text=(old/'physical_upper_decomposition.py').read_text()
text=text.replace("out=OUT/'hybrid_response'", "out=OUT/'head_causal_response'")
start=text.index('    groups=[');end=text.index('    cfg=LocomotionCfg()',start)
text=text[:start]+'''    groups=[('latest_run',1.5,0.,0.,'latest',.40,0.),
            ('preserved_flight_actor',1.5,0.,0.,'flight',.40,0.),
            ('old_body_neutral_head_actual_feedback',1.5,0.,0.,'flight',.40,0.),
            ('old_body_neutral_head_reference_head_feedback',1.5,0.,0.,'flight',.40,0.)]
    per=32;n=len(groups)*per
'''+text[end:]
start=text.index('                action[ids]=actors[group[4]]');end=text.index('\n        action=action.clone()',start)
text=text[:start]+'''                actor_input=observed[ids].clone()
                if gi==3 and t>=3.:
                    # Only a causal diagnostic: head q/qd/previous-action history
                    # from the simultaneous original-head cohort. Physical head
                    # control stays native zero in BOTH neutral-head cohorts.
                    columns=[9+raw.lookup[x] for x in ['head_yaw','head_pitch']]
                    columns += [32+raw.lookup[x] for x in ['head_yaw','head_pitch']]
                    columns += [55+raw.lookup[x] for x in ['head_yaw','head_pitch']]
                    actor_input[:,:,columns]=observed[per:2*per,:,columns]
                action[ids]=actors[group[4]]({'policy':actor_input.reshape(per,400)})
                if gi>=2 and t>=3.:
                    action[ids,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]=0.
'''+text[end:]
text=text.replace('for tick in range(500):','for tick in range(1000):')
text=text.replace("        summaries.append(stats)",'''        head=data['q'][window,ids][:,:,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]
        stats['head_combined_rms_deg']=float(np.sqrt((head**2).sum(-1).mean())*180/np.pi)
        stats['head_target_max_abs_rad']=float(np.abs(data['target'][window,ids][:,:,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]).max())
        stats['mean_abs_world_lateral_speed']=float(np.abs(data['root'][window,ids,8][good]).mean())
        quat=data['root'][-1,ids,3:7];w,x,y,z=quat.T
        stats['mean_abs_final_yaw_deg']=float(np.abs(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))).mean()*180/np.pi)
        summaries.append(stats)''')
text=text.replace('64 physical robots, 10s','128 physical robots, 20s')
text=text.replace("OUT/'hybrid_probe_failure.json'","OUT/'head_causal_probe_failure.json'")
(OUT/'physical_head_causal.py').write_text(text)
target=OUT/'validation_source/physical_v29'
assert not target.exists(),'Preserve staged attempts'
shutil.copytree(run/'source/physical_v29',target,ignore=shutil.ignore_patterns('__pycache__','usd','*.pyc'))
p=target/'training/command_curriculum.py';source=p.read_text()
source=source.replace("'exact_prior_frontier_rehearsal_v1'","'gait_consistent_running_frontier_v2'")
needle="    speed = torch.where(r < .15, 0., torch.where(r < .45, anchors[index],\n            torch.where(r < .75, upper, upper*(.80+.20*v))))"
replacement="""    # Keep frontier exposure inside the current running gait. At the first
    # running stage, the old 1.2-1.5 interval predominantly trained walking.
    # Draw count, stopped/prior cohorts and all <=1.0 commands stay identical.
    frontier_lower = max(1.5, upper*.80) if upper >= 1.5 else upper*.80
    frontier = (upper*(.80+.20*v) if frontier_lower == upper*.80
                else frontier_lower+(upper-frontier_lower)*v)
    speed = torch.where(r < .15, 0., torch.where(r < .45, anchors[index],
            torch.where(r < .75, upper, frontier)))"""
assert source.count(needle)==1;source=source.replace(needle,replacement)
p.write_text(source)
manifest={'run':str(run),'source_changes':{'training/command_curriculum.py':{
    'before':hashlib.sha256((run/'source/physical_v29/training/command_curriculum.py').read_bytes()).hexdigest(),
    'after':hashlib.sha256(p.read_bytes()).hexdigest()}},'deployed':False,'scope':'Training command exposure only; no inference, reward or physical change.'}
(OUT/'staging.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
