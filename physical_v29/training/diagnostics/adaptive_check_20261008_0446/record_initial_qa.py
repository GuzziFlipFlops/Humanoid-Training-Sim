import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plots=[OUT/'initial_upper'/n for n in ['upper_joint_angles.png','upper_joint_velocities.png','upper_bias_decomposition.png','upper_target_bias.png']]
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),plots=[dict(path=str(p),sha256=sha(p),visually_inspected=True) for p in plots],
    observations='Actual518 upper timelines show opposite held shoulder pitch at2.5/3/3.5,unequal pitch/roll/elbow amplitude,head gait oscillation. Slow held components dominate absolute posture;full span is not cycle amplitude. Physical target slow components carry the same opposite bias,with left response error larger than right. No human/Unitree/optical-gaze certification.',
    sampling='32 robots per command10Hz population versus16 total representatives50Hz,first-episode/reset segmentation;population10Hz cannot certify faster jitter.',
    reference_scope='Primary wrist-IMU human amplitude/asymmetry/smoothness features only;no mapped speed-matched official G1/H1 joint encoder curves obtained. Native head zero torso-relative;CAD opticalaxis uncalibrated.',
    references=['https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043','https://github.com/unitreerobotics/unitree_rl_mjlab'])
(OUT/'plot_qa.json').write_text(json.dumps(receipt,indent=2)+'\n')
source=OUT.parent/'adaptive_check_20261007_1845/gate_com_contact_audit.py'
s=source.read_text().replace("OUT/'final_gate_review.json'", "OUT/'initial_gate_review.json'")
(OUT/'gate_com_contact_audit.py').write_text(s)
print('Four initial upper plots inspected; synchronized COM/contact audit prepared.')
