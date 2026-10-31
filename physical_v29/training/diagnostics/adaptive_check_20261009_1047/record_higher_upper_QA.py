from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parent/'high_speed_extrapolation'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
for label in ['nominal','pushed']:
    folder=ROOT/('upper_'+label)
    audit=read(folder/'upper_joint_time_audit.json')
    plots=[folder/'upper_joint_angles.png',folder/'upper_joint_velocities.png']
    receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),visually_inspected=True,
        plot_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in plots},
        reviewed_commands=[3.5,4,4.5,5],source_audit=str(folder/'upper_joint_time_audit.json'),
        observation=('3.5 has opposite held shoulder-pitch biases and unequal pitch/roll/elbow amplitudes. '
          'At 4, measured shoulder/elbow biases drift over the plotted 2-8s interval. At 4.5/5, shoulder pitch shifts positive '
          'on both sides, roll stays unequal, and native head pitch bias grows. Physical target traces accompany those changes. '
          'Velocity traces retain fast gait-linked asymmetry; only representative first-episode 50Hz traces are plotted. '
          'The full population is sampled at10Hz,32 robots per command; no natural-motion,Unitree or optical-gaze certification.'),
        representative_trace_limitations='One plotted robot per selected higher command; population distributions are saved separately. '
          'The plotted interval does not represent the whole20second trial or certify jitter beyond the sampling bandwidth.')
    (folder/'visual_QA.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print('Recorded both higher-speed upper angle/target/velocity visual inspections.')
