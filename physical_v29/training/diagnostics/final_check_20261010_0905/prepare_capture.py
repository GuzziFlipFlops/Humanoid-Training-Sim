import json,hashlib,shutil,xml.etree.ElementTree as ET
from pathlib import Path
OUT=Path(__file__).resolve().parent;T=OUT.parents[1];R=T/'runs/locomotion_20261007_r40_goal_budget_full_state';TRAIN=R/'source/physical_v29/training'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sig(e):return e.tag,tuple(sorted(e.attrib.items())),(e.text or '').strip(),tuple(sig(c) for c in e)
original=TRAIN/'assets/robot_training.urdf';prepared=T/'diagnostics/video_body_review_20261004/playback_asset/robot_training_full_visuals.urdf'
a=ET.parse(original).getroot();b=ET.parse(prepared).getroot()
assert [sig(j) for j in a.findall('joint')]==[sig(j) for j in b.findall('joint')]
assert len(a.findall('link'))==len(b.findall('link'))==24
for old,new in zip(a.findall('link'),b.findall('link')):
 assert old.attrib==new.attrib and [sig(c) for c in old if c.tag!='visual']==[sig(c) for c in new if c.tag!='visual']
meshes=[]
for link in b.findall('link'):
 for v in link.findall('visual'):
  m=v.find('geometry/mesh');p=Path(m.get('filename'));assert p.is_file();meshes.append(dict(link=link.get('name'),visual=v.get('name'),file=str(p),sha256=sha(p)))
assert len(meshes)==140
asset=OUT/'playback_asset';asset.mkdir(exist_ok=True);dest=asset/'robot_training_full_visuals.urdf'
if not dest.exists():shutil.copy2(prepared,dest)
assert sha(dest)==sha(prepared)
cp=R/'1064_run_350_train_352/candidate.pt';clips=[]
for c in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5]:clips.append(dict(speed=c,checkpoint=str(cp),sha256=sha(cp),scope='Preserved single actor demonstration; complete 3.5 stage unqualified; 1.5 flight and slow path/slip retention failures labelled',gate_receipts=[str(R/'1065_run_350_gate_353_nominal'),str(R/'1066_run_350_gate_353_push')]))
config=dict(run=str(R),training=str(TRAIN),visual_urdf=str(dest),visual_sha256=sha(dest),training_urdf_sha256=sha(original),physics_joints_collisions_inertias_equal=True,mesh_count=140,meshes=meshes,clips=clips,destination=str(T/'videos/final_motor_capture_20261010_0905'),episode_initial_state_seed=7101,population_capture=str(OUT/'selected_diverse_nominal'),motor_mapping_helper=str(T/'automation/motor_shaft_metrics.py'))
(OUT/'capture_config.json').write_text(json.dumps(config,indent=2)+'\n')
old=T/'diagnostics/adaptive_check_20261004_2340/attempt02_instance_binding/record_full_body_gui.py'
text=old.read_text();start=text[:text.index('def main():')]
start=start.replace("pause=json.loads((OUT/'video_pause.json').read_text(encoding='utf-8-sig'))", "pause=json.loads((OUT/'stop_receipt.json').read_text(encoding='utf-8-sig'))")
start=start.replace('def verify_pause():','def verify_pause():\n    assert not psutil.pid_exists(pause[\'supervisor_pid\']), \'Owned supervisor must remain retired\'\n    assert json.loads((RUN/\'state.json\').read_text())[\'status\']==\'stopped_by_user_budget\'\n    return\n\ndef historical_unused_pause():')
start=start.replace("'paused_supervisor':pause['supervisor_pid']", "'retired_supervisor':pause['supervisor_pid']")
start=start.replace("['pid']", "['pid']")
start += (OUT/'capture_main.txt').read_text()
(OUT/'record_motor_gui.py').write_text(start)
print(json.dumps({k:v for k,v in config.items() if k not in ['meshes','clips']},indent=2))
