import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
v=json.loads((OUT/'initial_upper/upper_joint_time_audit.json').read_text());rows=[]
for r in v['records']:
 if r['test']!='nominal':continue
 rows.append(dict(actor=r['actor_tag'],command=r['command_m_s'],head_RMS_deg=r['head_rms_deg'],
  shoulder_means_deg=[r['joints'][x+'_shoulder_pitch']['mean_deg'] for x in ['left','right']],
  shoulder_pitch_ranges_deg=[r['joints'][x+'_shoulder_pitch']['median_per_robot_p95_p05_range_deg'] for x in ['left','right']],
  shoulder_roll_means_deg=[r['joints'][x+'_shoulder_roll']['mean_deg'] for x in ['left','right']],
  elbow_means_deg=[r['joints'][x+'_elbow_pitch']['mean_deg'] for x in ['left','right']],
  coordination=r['coordination'],representative_50hz=r.get('representative_50hz')))
(OUT/'upper_summary.json').write_text(json.dumps(rows,indent=2))
qa=[dict(path=str(OUT/f),sha256=hashlib.sha256((OUT/f).read_bytes()).hexdigest(),visually_inspected=True) for f in ['initial_upper/upper_joint_angles.png','initial_upper/upper_joint_velocities.png']]
(OUT/'plot_qa.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),files=qa,
 findings='At3.5 left pitch rhythmic swing exceeds right; gait-locked neck oscillation and shoulder roll remain. At2.5/3 posture bias drifts over seconds, so total range is not purely cycle amplitude. Physical targets lead/overshoot actual arms.50Hz representative velocities oscillate sharply; population10Hz is insufficient for high-frequency certification.',
 human_Unitree_likeness_certified=False,world_optical_gaze_calibrated=False),indent=2))
print(json.dumps([r for r in rows if r['actor']=='latest' and r['command'] in [1.,3.5]],indent=2))
