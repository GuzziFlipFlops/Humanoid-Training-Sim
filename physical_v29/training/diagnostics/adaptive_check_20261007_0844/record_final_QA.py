import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
p=OUT/'plot_qa.json';q=json.loads(p.read_text(encoding='utf-8-sig'))
for name in ['upper_joint_angles.png','upper_joint_velocities.png']:
    f=OUT/'final_upper'/name
    q['files'].append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'visually_inspected':True})
q['final_QA_utc']=datetime.now(timezone.utc).isoformat()
q['final_actor']='142_run_350_train_045'
q['final_findings']='All final142 angle and velocity panels inspected. At1m unequal shoulder/elbow motion; at2.5/3 slow shoulder-pitch bias reverses across seconds while cycle swing persists. At3.5 left shoulder-pitch oscillation is larger around negative bias,right remains aroundpositive bias with smaller excursion; roll/elbow/head remain gait-locked and asymmetric. Faint physical targets exceed actual fast pitch excursion. Representative50Hz velocities contain sharp periodic peaks;10Hz population cannot certify fast jitter. No natural-motion or optical-gaze certification.'
p.write_text(json.dumps(q,indent=2)+'\n')
v=json.loads((OUT/'final_learning_verification.json').read_text());r=json.loads((OUT/'diagnostic_resume.json').read_text())
resumed=[b for b in v['blocks'] if Path(b['folder']).name=='142_run_350_train_045']
assert len(resumed)==1 and resumed[0]['updates']==64 and resumed[0]['full_load_proof']['source_sha256']==r['checkpoint_sha256']
assert resumed[0]['all_preupdate_KL_and_command_mismatch_zero']
(OUT/'resumed_main_block.json').write_text(json.dumps(resumed[0],indent=2)+'\n')
print('Eight images have documented visual QA; resumed142 main64 updates load exact139 full state.')
