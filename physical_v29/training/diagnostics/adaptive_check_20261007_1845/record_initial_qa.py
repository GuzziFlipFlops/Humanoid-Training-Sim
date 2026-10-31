import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'initial_gate_review.json').read_text());q=OUT/'initial_upper'
plots=['upper_joint_angles.png','upper_joint_velocities.png','upper_bias_decomposition.png']
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),checkpoint=r['latest']['checkpoint'],
    checkpoint_sha256=r['latest']['sha256'],visually_inspected=True,
    plots={n:hashlib.sha256((q/n).read_bytes()).hexdigest() for n in plots},
    observations=[
      'Actual head, shoulder pitch/roll and elbow angle/target timelines, plus separate measured velocities, were inspected at1,2.5,3,3.5m/s.',
      '1m/s representative shoulders have opposite but unequal bias and swing, with unequal elbow coordination.',
      '2.5m/s shoulders remain at opposite roughly30deg pitch biases;3m/s contains substantial slow bias drift.',
      '3.5m/s nominal population fast pitch residual ranges13.99/7.75deg versus slow mean biases-21.60/+13.37deg. Right pitch swing is smaller.',
      'High-rate representative velocities show gait-linked oscillation and spikes;10Hz population sampling does not certify faster jitter.',
      'Native head angles are torso-relative. Optical face/camera calibration remains unavailable; no world-gaze certification.',
      'No mapped speed-matched official Unitree encoder reference was obtained. Primary human comparison remains feature-based and morphology-limited, not exact curve likeness.'
    ],qualification_changed=False)
(OUT/'initial_plot_qa.json').write_text(json.dumps(record,indent=2)+'\n')
print('Initial angle, velocity and bias plot QA saved.')
