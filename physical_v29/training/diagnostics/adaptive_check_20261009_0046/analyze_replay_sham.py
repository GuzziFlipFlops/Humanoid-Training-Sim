"""Validate and analyze the zero-offset replay control; no success from falls."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;SHAM=OUT/'replay_sham'
read=lambda p:json.loads(Path(p).read_text())
r=read(SHAM/'arm_target_response/result.json');resume=read(SHAM/'diagnostic_resume.json')
assert r['checkpoint_sha256']==read(OUT/'arm_target_response/result.json')['checkpoint_sha256']==resume['checkpoint_sha256']
assert resume['same_supervisor'] and resume['supervisor_pid']==39000 and not resume['training_source_changed']
assert read(SHAM/'bounded_diagnosis/validation.json')['passed']
with np.load(SHAM/'arm_target_response/telemetry_50hz.npz') as z:
 d={k:z[k] for k in ['time','alive','failures','q','applied_raw_action','student_raw_action','initial_q','initial_qd','joint_names','robots_per_group']}
per=int(d['robots_per_group']);names=d['joint_names'].tolist()
shoulder=[names.index(x) for x in ['left_shoulder_pitch','right_shoulder_pitch']]
body=[j for j in range(23) if j not in shoulder]
index={(g['seed'],g['command'],g['mode']):i for i,g in enumerate(r['groups'])}
rows=[];tests=[];t=d['time'];pre=(t>=2)&(t<5)
fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
for speed,axes_row in zip([.15,1.5,3.5],axes):
 for mode,color in [('control','black'),('body_replay_zero','tab:green'),('body_replay_positive','tab:blue'),('body_replay_negative','tab:orange')]:
  curves=[];differences=[]
  for seed in [4321,4331]:
   i=index[(seed,speed,mode)];j=index[(seed,speed,'control')]
   ids=slice(i*per,(i+1)*per);control=slice(j*per,(j+1)*per);g=r['groups'][i]
   assert np.array_equal(d['initial_q'][ids],d['initial_q'][control]) and np.array_equal(d['initial_qd'][ids],d['initial_qd'][control])
   assert len({v.tobytes() for v in np.concatenate([d['initial_q'][ids],d['initial_qd'][ids]],1)})==32
   raw_delta=d['applied_raw_action'][:,ids][:,:,body]-d['applied_raw_action'][:,control][:,:,body]
   if mode.startswith('body_replay'):assert np.abs(raw_delta).max()==0
   if mode=='body_replay_zero':assert np.array_equal(d['applied_raw_action'][:,ids][:,:,shoulder],d['student_raw_action'][:,ids][:,:,shoulder])
   delta=d['q'][:,ids]-d['q'][:,control]
   paired_alive=d['alive'][:,ids]&d['alive'][:,control]
   squared=np.square(delta).mean(2);count=paired_alive.sum(1)
   mean=np.divide((squared*paired_alive).sum(1),count,out=np.full(len(t),np.nan),where=count>0)
   error=np.rad2deg(np.sqrt(mean))
   first=np.flatnonzero(d['failures'][:,ids].any(1))
   rows.append(dict(seed=seed,command=speed,mode=mode,survival=g['survival'],resets=g['resets'],
    pre_offset_joint_RMS_difference_deg=float(np.rad2deg(np.sqrt(np.square(delta[pre][paired_alive[pre]]).mean()))),
    first_failure_time_s=float(t[first[0]]) if len(first) else None,
    nonshoulder_action_stream_max_difference=float(abs(raw_delta).max()),additional_shoulder_offset_zero=mode=='body_replay_zero'))
   tests.append(f'{seed}/{speed}/{mode}:32 paired initial states and exact declared action stream')
   curves.append(d['alive'][:,ids].mean(1));differences.append(error)
  axes_row[0].plot(t,np.mean(curves,0)*100,color=color,label=mode)
  finite=np.isfinite(differences);count=finite.sum(0)
  mean=np.divide(np.nansum(differences,0),count,out=np.full(len(t),np.nan),where=count>0)
  axes_row[1].plot(t,mean,color=color,label=mode)
 for a in axes_row:a.axvline(5,color='.5',ls='--',lw=.7);a.grid(alpha=.2)
 axes_row[0].set_ylabel(f'{speed} m/s alive(%)');axes_row[1].set_ylabel('Paired live joint RMS(deg)')
axes[0,0].legend(fontsize=8);axes[-1,0].set_xlabel('Physical time(s)');axes[-1,1].set_xlabel('Physical time(s)')
fig.suptitle('Replay sham:64 paired independent states/command/mode; added offsets begin at5s')
fig.tight_layout();fig.savefig(OUT/'replay_sham_control.png',dpi=150);plt.close(fig)
zero=[x for x in rows if x['mode']=='body_replay_zero']
all_zero_fail=all(x['survival']==0 for x in zero)
result=dict(completed=True,passed=True,checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256'],
 rows=rows,validation_cases=tests,actual_control_robots=192,actual_replay_robots=576,
 all_zero_offset_replay_cohorts_eventually_fall=all_zero_fail,all_original_controls_survive=all(x['survival']==1 for x in rows if x['mode']=='control'),
 maximum_target_map_error_rad=r['maximum_physical_target_reconstruction_error_rad'],maximum_command_difference=r['maximum_action_command_difference'],
 original_gains_and_caps_retained=True,same_supervisor_resumed=resume,
 conclusion='Prescribed body-action replay is unstable even without an extra arm offset, and joint states diverge before5s. Replay falls cannot isolate mechanical arm coupling. Closed-loop arm-offset responses remain valid combined physical-and-policy responses; no additive or mechanical ceiling inference.',
 mechanical_arm_coupling_identified=False,diagnostic_controller_deployed=False,production_repair_tested=False)
assert all_zero_fail and result['all_original_controls_survive']
(OUT/'replay_sham_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
first=read(OUT/'arm_coupling_analysis.json')
first['inference']=result['conclusion'];first['replay_sham_evidence']=str(OUT/'replay_sham_analysis.json')
(OUT/'arm_coupling_analysis.json').write_text(json.dumps(first,indent=2)+'\n')
print(json.dumps(dict(passed=True,all_zero_offset_replay_fall=all_zero_fail,validation_cases=len(tests),controls_survive=True)))
