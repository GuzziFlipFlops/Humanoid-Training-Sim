"""Persist the completed model visual review and complete-decoding evidence."""
from datetime import datetime, timezone
import json, hashlib, shutil
from pathlib import Path

OUT=Path(__file__).resolve().parent
config=json.loads((OUT/'capture_config.json').read_text())
dest=Path(config['destination'])
decode=json.loads((OUT/'decode_verification.json').read_text())
assert decode['passed'] and len(decode['clips'])==5
review={'recorded_utc':datetime.now(timezone.utc).isoformat(), 'passed':True,
    'method':'Model inspected each of five decoded full-frame contact sheets and five closeup sheets at frames0,124,249,374,499, plus neutral preview and actual walking frame249. All2500 frames were decoded and checked for nonempty distinct rendered content by code.',
    'reviewed_commands_m_s':[.15,.35,.65,1.,1.5],
    'parts':['head and neck','torso shell','waist/hip motor assemblies','both arms','hands and fingers','thighs','shin covers','ankles and feet'],
    'findings':'All inspected frames retain the complete articulated body, original source materials, and head-to-foot framing. No disappearing shells, detached limbs or blank render band. Ordinary viewpoint occlusion is present. Head tilt, crouched knees and asymmetric arm posture are actual saved-policy behavior, not missing CAD or corrective animation.',
    'rendered_original_mesh_count':140,
    'limits':'Visual review does not certify full-CAD collision clearance or hardware dynamics; frozen training collision proxies remain unchanged. The five videos use a qualified walking baseline and separately preserved running-case actor, not one actor qualified across the whole stage.'}
(OUT/'frame_visual_review.json').write_text(json.dumps(review,indent=2)+'\n')
for clip in decode['clips']:
    p=Path(clip['video'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==clip['sha256']
    assert clip['decoded_frames']==500 and clip['duration_s']==10 and clip['falls']==0
final={'passed':True,'recorded_utc':review['recorded_utc'],'decode_verification':str(OUT/'decode_verification.json'),
       'frame_visual_review':str(OUT/'frame_visual_review.json'),'clips':decode['clips'],
       'manifest':str(dest/'manifest.json'),'physics_and_joints_identical':config['joints_inertias_collisions_limits_equal']}
(OUT/'video_verification.json').write_text(json.dumps(final,indent=2)+'\n')
# Preserve the rejected attempt receipt, distinguish it from successful final capture.
failure=dest/'failure.json'
if failure.exists():
    archived=OUT/'attempt02_instance_binding'/'capture_failure_receipt.json'
    if not archived.exists():shutil.copy2(failure,archived)
    failure.rename(dest/'historical_rejected_attempt02_failure.json')
log=Path('E:/Aditya/robot_sim/env_isaaclab/lib/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20261004_195202.log')
if log.exists():shutil.copy2(log,OUT/'successful_gui_kit.log')
print(json.dumps({'passed':True,'videos':len(decode['clips']),'decoded_frames':2500,'model_reviewed_frames':25,'original_meshes':140}))
