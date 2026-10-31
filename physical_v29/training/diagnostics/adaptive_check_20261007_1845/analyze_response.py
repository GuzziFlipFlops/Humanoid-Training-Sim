"""Complete-return and truthful yaw-response comparison, not qualification."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;folder=OUT/'heading_response'
read=lambda p:json.loads(Path(p).read_text())
r=read(folder/'result.json');z=np.load(folder/'telemetry.npz');dt=.02
names=z['joint_names'].tolist();groups=z['groups'].tolist();per=int(z['robots_per_group'])
rows=[]
for group,name in enumerate(groups):
    ids=slice(group*per,(group+1)*per);root=z['root'][:,ids];quat=root[:,:,3:7]
    qw,qx,qy,qz=[quat[:,:,i] for i in range(4)]
    yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    course=np.arctan2(root[:,:,8],root[:,:,7]);heading=z['reference_heading'][ids]
    wrap=lambda x:np.arctan2(np.sin(x),np.cos(x))
    windows=[]
    for lower,upper in [(2.,5.),(5.,10.),(10.,20.),(2.,20.)]:
        w=(z['time']>=lower)&(z['time']<=upper);good=z['alive'][w,ids]
        lateral=root[w,:,8];forward=root[w,:,7]
        flight=(z['clearance'][w,ids]>.001).all(-1)&(z['force'][w,ids]<1).all(-1)&good
        windows.append(dict(interval_s=[lower,upper],signed_forward_m_s=float(forward[good].mean()),
            absolute_lateral_m_s=float(abs(lateral)[good].mean()),
            yaw_command_saturation_fraction=float((abs(z['action_command'][w,ids,2])>=.199999).mean()),
            heading_abs_deg=float(np.rad2deg(abs(wrap(yaw[w]-heading))).mean()),
            course_abs_deg=float(np.rad2deg(abs(wrap(course[w]-heading))).mean()),
            mean_yaw_rate=float(root[w,:,12].mean()),
            mean_yaw_command=float(z['action_command'][w,ids,2].mean()),
            mean_heading_feedback_term_rad_s=float((.5*wrap(heading-yaw[w])).mean()),
            mean_course_feedback_term_rad_s=float(wrap(heading-course[w]).mean()),
            signed_body_heading_deg=float(np.rad2deg(wrap(yaw[w]-heading)).mean()),
            signed_course_heading_deg=float(np.rad2deg(wrap(course[w]-heading)).mean()),
            opposing_feedback_fraction=float(((.5*wrap(heading-yaw[w]))*wrap(heading-course[w])<0).mean()),
            unclipped_feedback_cancellation_ratio=float((abs(.5*wrap(heading-yaw[w])+wrap(heading-course[w]))/(abs(.5*wrap(heading-yaw[w]))+abs(wrap(heading-course[w]))+1e-8)).mean()),
            true_flight_fraction=float(flight.mean()),reward_rate=float(z['reward_rate'][w,ids][good].mean()),
            per_robot_absolute_lateral_p25_p50_p75=np.quantile(abs(lateral).mean(0),[.25,.5,.75]).tolist()))
    upper_indices=[names.index(n) for n in ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']]
    w=z['time']>=5.;actual=z['q'][w,ids][:,:,upper_indices];targets=z['target'][w,ids][:,:,upper_indices]
    upper=dict(actual_target_RMS_deg=float(np.rad2deg(np.sqrt(((actual-targets)**2).mean()))),
        measured_angles={names[j]:dict(mean_deg=float(np.rad2deg(z['q'][w,ids,j].mean())),
          median_p95_p05_deg=float(np.median(np.rad2deg(np.quantile(z['q'][w,ids,j],.95,axis=0)-np.quantile(z['q'][w,ids,j],.05,axis=0))))) for j in upper_indices})
    rows.append(dict(group=name,windows=windows,upper=upper))
by={g['name']:g for g in r['groups']};comparisons=[]
for actor in r['actor_checkpoints']:
    b=by[f'{actor}_3.5_baseline'];c=by[f'{actor}_3.5_course1']
    comparisons.append(dict(actor=actor,baseline_lateral=b['mean_absolute_world_lateral_velocity'],
      course_lateral=c['mean_absolute_world_lateral_velocity'],
      lateral_relative_change=(c['mean_absolute_world_lateral_velocity']/b['mean_absolute_world_lateral_velocity']-1),
      complete_reward_rate_change=c['mean_reward_rate']-b['mean_reward_rate'],
      head_RMS_change=c['head_RMS_deg']-b['head_RMS_deg'],
      flight_fraction_change=c['unloaded_geometric_flight_fraction']-b['unloaded_geometric_flight_fraction'],
      reward_term_changes={k:c['reward_terms'][k]-v for k,v in b['reward_terms'].items()},
      yaw_command_saturation_baseline=b['yaw_command_saturation_fraction'],
      yaw_command_saturation_course=c['yaw_command_saturation_fraction']))
fig,axes=plt.subplots(4,3,figsize=(17,14),sharex=True)
for i,actor in enumerate(r['actor_checkpoints']):
    for course_flag,color in [('baseline','tab:blue'),('course1','tab:orange')]:
        name=f'{actor}_3.5_{course_flag}';group=groups.index(name);ids=slice(group*per,(group+1)*per)
        root=z['root'][:,ids];qw,qx,qy,qz=[root[:,:,3+j] for j in range(4)]
        yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        axes[i,0].plot(z['time'],abs(root[:,:,8]).mean(1),color=color,label=course_flag)
        axes[i,1].plot(z['time'],np.rad2deg(yaw).mean(1),color=color,label=course_flag)
        axes[i,2].plot(z['time'],z['action_command'][:,ids,2].mean(1),color=color,label=course_flag)
    for j,title in enumerate(['Mean absolute world lateral speed (m/s)','Mean body yaw (deg)','Actual desired yaw rate (rad/s)']):
        axes[i,j].set_title(actor+' — '+title);axes[i,j].grid(alpha=.25)
        if i==3:axes[i,j].set_xlabel('Simulated time (s)')
    axes[i,0].legend()
fig.suptitle('Truthful paired-state physical response of saved actors\nDiagnostic baseline versus course feedback; no learning or qualification',fontsize=16)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'saved_actor_response.png',dpi=130);plt.close(fig)
motor=read(folder/'substep_motor_audit.json')
clips=[dict(group=g['name'],name=m['name'],fraction=m['speed_clipped_fraction'],max_contiguous_s=m['maximum_contiguous_speed_clip_seconds']) for g in motor['commands'] for m in g['motors'] if m['speed_clipped_fraction']>0]
result=dict(completed=True,actor_checkpoints=r['actor_checkpoints'],actor_sha256=r['actor_sha256'],
    groups=rows,course_comparisons=comparisons,actual_native_5ms_driving_clips=clips,
    qualifiers=dict(qualification=False,actors_deployed=False,physics_and_caps_unchanged=True,
      paired_independent_initial_states_per_cohort=per,diagnostic_initial_state_protocol='24 independently drawn q/qd offsets repeated exactly across groups',
      instantaneous_velocity_statistics_are_not_gate_finite_difference_statistics=True,
      limits_endpoint_proven=False,source_signature_does_not_certify_external_diagnostic_command_law=True))
(OUT/'saved_actor_response_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(course_comparisons=[{k:v for k,v in x.items() if k!='reward_term_changes'} for x in comparisons],native_driving_clip_events=len(clips)),indent=2))
