"""Measure upper motion and enforce exact pre-learning replay/full-state lineage."""
import hashlib,json,numpy as np,xml.etree.ElementTree as ET
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run'])
matches=[]
for p in run.glob('*_gate_*_nominal/run_config.json'):
 cfg=read(p)['arguments']
 if Path(cfg['checkpoint']).resolve()==Path(pause['checkpoint']).resolve() and cfg['speed']==1.5 and cfg['seed']==901 and cfg['duration']==20.:
  matches.append(p.parent)
assert len(matches)==1,matches
old=matches[0]
baseline=read(old/'result.json');same=read(OUT/'same_actor_nominal/result.json')
assert baseline['cases']==same['cases'],'Pre-learning same-actor physical cases changed'
arrays=[]
for name in ['telemetry.npz','gait_diagnostics_50hz.npz']:
 x=np.load(old/name);y=np.load(OUT/'same_actor_nominal'/name)
 assert set(x.files)==set(y.files)
 for key in x.files:
  assert np.array_equal(x[key],y[key]),(name,key)
  arrays.append(dict(file=name,array=key,equal=True))
smoke=OUT/'running_fullstate_smoke';r=read(smoke/'result.json')
rows=[json.loads(x) for x in (smoke/'progress.jsonl').read_text().splitlines()]
assert r['completed'] and r['iterations']==8 and len(rows)==8
load=read(smoke/'full_load_proof.json')
assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
assert load['source_sha256']==pause['checkpoint_sha256']
assert all(x['ppo']['pre_update_kl']==0. and x['behavior']['action_command_difference']==0.
 and x['upper_refinement']['batches']==20 and x['locomotion_retention']['batches']>0 for x in rows)
reloaded=read(OUT/'checkpoint_reload/result.json')
assert reloaded['completed'] and all(c['statistics']['survival']==1. for c in reloaded['cases'])
joint_audit=[]
for j in ET.parse(OUT/'validation_source/physical_v29/training/assets/robot_training.urdf').getroot().findall('joint'):
 if j.attrib['name'] in ['head_yaw','head_pitch']:
  assert j.find('origin').attrib['rpy']=='0 0 0'
  joint_audit.append(dict(name=j.attrib['name'],origin=j.find('origin').attrib,axis=j.find('axis').attrib,
   zero_orientation='Identity relative to torso; native head +X remains torso +X'))
assert len(joint_audit)==2
def upper_report(folder):
 data=np.load(folder/'telemetry.npz');names=list(data['joint_names']);q=data['q'];v=data['qd']
 result=[]
 for label,speed in enumerate(data['case_speeds']):
  ids=np.flatnonzero(data['case_labels']==label);a=q[:,ids];vel=v[:,ids]
  stat=dict(command_m_s=float(speed),trials=len(ids),joints={})
  for name in ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']:
   z=np.rad2deg(a[:,:,names.index(name)])
   stat['joints'][name]=dict(mean_deg=float(z.mean()),rms_deg=float(np.sqrt(np.mean(z*z))),
    p95_abs_deg=float(np.quantile(np.abs(z),.95)),std_deg=float(z.std()),
    velocity_rms_rad_s=float(np.sqrt(np.mean(vel[:,:,names.index(name)]**2))))
  h=a[:,:,[names.index('head_yaw'),names.index('head_pitch')]]
  stat['head_combined_rms_deg']=float(np.rad2deg(np.sqrt(np.mean(np.sum(h*h,axis=-1)))))
  pair=np.stack([a[:,:,names.index('left_shoulder_pitch')]+a[:,:,names.index('right_shoulder_pitch')],
   a[:,:,names.index('left_shoulder_roll')]+a[:,:,names.index('right_shoulder_roll')],
   a[:,:,names.index('left_elbow_pitch')]-a[:,:,names.index('right_elbow_pitch')]],-1)
  stat['arm_symmetry_rms_deg']=float(np.rad2deg(np.sqrt(np.mean(pair*pair))))
  result.append(stat)
 return result
physical=dict(passed=True,cases_bitwise_equal=True,arrays_bitwise_equal=arrays,
 head_zero_frame_audit=joint_audit,baseline_upper=upper_report(old),smoke_upper=upper_report(OUT/'checkpoint_reload'),
 baseline_cases=baseline['cases'],smoke_cases=reloaded['cases'],
 full_population_smoke_survival=True,new_posture_qualified=False,
 scope='192 robots,20-second same-actor replay and an eight-update64-env PPO/reload. No new posture or running qualification; physical coupling and normal PPO can still change locomotion.')
write(OUT/'physical_validation.json',physical)
full=dict(passed=True,full_load_proof=load,iterations=8,
 pre_update_KL=[x['ppo']['pre_update_kl'] for x in rows],
 command_difference=[x['behavior']['action_command_difference'] for x in rows],
 upper_refinement_batches=[x['upper_refinement']['batches'] for x in rows],
 source_checkpoint=pause['checkpoint'],source_sha256=pause['checkpoint_sha256'],
 smoke_actor_discarded_for_deployment=True,new_posture_qualified=False)
write(OUT/'full_state_smoke.json',full)
print(json.dumps(dict(passed=True,load=load,same_actor_arrays=len(arrays),
 baseline_head=[(x['command_m_s'],x['head_combined_rms_deg']) for x in physical['baseline_upper']],
 smoke_head=[(x['command_m_s'],x['head_combined_rms_deg']) for x in physical['smoke_upper']],
 baseline_arm_symmetry=[(x['command_m_s'],x['arm_symmetry_rms_deg']) for x in physical['baseline_upper']],
 smoke_arm_symmetry=[(x['command_m_s'],x['arm_symmetry_rms_deg']) for x in physical['smoke_upper']],
 baseline_gates=[(c['statistics']['command_m_s'],c['passed']) for c in baseline['cases']],
 smoke_gates=[(c['statistics']['command_m_s'],c['passed']) for c in reloaded['cases']]),indent=2))
