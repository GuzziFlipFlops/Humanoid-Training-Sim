import json,hashlib,shutil
from pathlib import Path
out=Path(__file__).resolve().parent;old=out.parent/'adaptive_check_20261006_1743';older=out.parent/'adaptive_check_20261006_1243'
run=Path(json.loads((out/'live_handoff.json').read_text())['active_run']['run'])
src=run/'source/physical_v29';dst=out/'control_source/physical_v29'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=False)
files=[]
for p in src.rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts:
  q=dst/p.relative_to(src);assert q.read_bytes()==p.read_bytes();files.append(str(p.relative_to(src)))
(out/'protected_source_copy.json').write_text(json.dumps(dict(protected_files_identical=files,source=str(src)),indent=2))
for name in ['pause_for_diagnosis.py','resume_after_diagnosis.py','capture_current_gradients.py','analyze_current_gradients.py']:
 shutil.copy2((older if 'gradients' in name else old)/name,out/name)
t=(out/'capture_current_gradients.py').read_text()
t=t.replace('original=LocomotionOnlyRetention._regularize;captured=0', '''from rsl_rl.storage.rollout_storage import RolloutStorage
original_generator=RolloutStorage.mini_batch_generator
current_batch=None
def passive_batches(self,*args,**kwargs):
 global current_batch
 for batch in original_generator(self,*args,**kwargs):
  current_batch=batch
  yield batch
 current_batch=None
RolloutStorage.mini_batch_generator=passive_batches
original=LocomotionOnlyRetention._regularize;captured=0''')
t=t.replace("number=captured;captured+=1", """assert current_batch is not None
 record['advantages']=current_batch.advantages.detach().cpu().clone()
 record['returns']=current_batch.returns.detach().cpu().clone()
 record['sampled_actions']=current_batch.actions.detach().cpu().clone()
 record['old_log_prob']=current_batch.old_actions_log_prob.detach().cpu().clone()
 record['old_distribution_params']=[p.detach().cpu().clone() for p in current_batch.old_distribution_params]
 number=captured;captured+=1""")
t=t.replace("record['incoming_gradient']=gradient.detach().cpu().clone()", """record['incoming_gradient']=gradient.detach().cpu().clone()
  record['current_distribution_params']=[p.detach().cpu().clone() for p in self.actor.output_distribution_params]""")
(out/'capture_current_gradients.py').write_text(t)
t=(old/'arm_response_probe.py').read_text()
start=t.index('    groups=[(');end=t.index('    per=32;',start)
t=t[:start]+"    groups=[(f'{speed}_stiffness{stiffness}',speed,0.,1.,None) for speed in [1.,3.5] for stiffness in [35.,50.,70.]]\n"+t[end:]
t=t.replace("raw.actuator.damping[:,shoulder_ids]=values[:,None]", """raw.actuator.damping[:,shoulder_ids]=values[:,None]
    stiffness_values=torch.tensor([float(g[0].split('stiffness')[1]) for g in groups],device=raw.device).repeat_interleave(per)
    assert float(raw.actuator.stiffness[:,shoulder_ids].min())==35.
    raw.actuator.stiffness[:,shoulder_ids]=stiffness_values[:,None]
    import xml.etree.ElementTree as ET
    xml=ET.parse(TRAIN/'assets/robot_training.urdf').getroot()
    head_name=next(j for j in xml.findall('joint') if j.attrib['name']=='head_pitch').find('child').attrib['link']
    head_ids,head_names=raw.robot.find_bodies(head_name)
    assert len(head_ids)==1 and head_names[0]==head_name
    head_id=head_ids[0]""")
t=t.replace("'heading_error','stiffness','damping']", "'heading_error','stiffness','damping','head_quaternion']")
t=t.replace("damping=raw.actuator.damping)", "damping=raw.actuator.damping,head_quaternion=raw.robot.data.body_quat_w[:,head_id])")
t=t.replace("shoulder_damping_Nms_rad=g[3],", "shoulder_damping_Nms_rad=g[3],shoulder_stiffness_Nm_rad=float(g[0].split('stiffness')[1]),")
t=t.replace("joint_names=np.array(raw.robot.joint_names),", "joint_names=np.array(raw.robot.joint_names),head_body_name=np.array(head_name),")
t=t.replace('diagnostic_shoulder_target_sines=True,diagnostic_shoulder_damping_changes=True', 'diagnostic_shoulder_target_sines=False,diagnostic_shoulder_damping_changes=False,diagnostic_shoulder_stiffness_changes=True')
t=t.replace('320 physical robots,20s: normal-limit shoulder damping1/2/4 at1/3.5 and0.04rad stopped shoulder sine tests0.75/1.5/2.66/3.25Hz.', '192 physical robots,20s: normal-limit shoulder stiffness35/50/70 with unchanged damping1 at1/3.5m/s; actual Isaac head-link orientation is recorded.')
(out/'arm_response_probe.py').write_text(t)
shutil.copy2(old/'analyze_arm_probe.py',out/'analyze_arm_probe.py')
print(json.dumps(dict(protected_files=len(files),gradient_wrapper_sha256=hashlib.sha256((out/'capture_current_gradients.py').read_bytes()).hexdigest(),arm_probe_sha256=hashlib.sha256(t.encode()).hexdigest(),production_changed=False)))
