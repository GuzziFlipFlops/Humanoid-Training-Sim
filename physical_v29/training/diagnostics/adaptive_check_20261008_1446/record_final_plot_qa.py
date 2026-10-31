"""Record the fourteen final plots already visually inspected in this check."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
folder = OUT / 'final_upper'
review = json.loads((OUT / 'final_gate_review.json').read_text(encoding='utf-8'))['latest']
observations = {
    '': 'At 3.5 m/s, unequal small fast shoulder pitch motion surrounds held negative-left/positive-right offsets. Roll and elbow motion are asymmetric and the head oscillates with gait. At 2.5 and 3 m/s, slow physical target means and actual shoulder biases drift or reverse; these reversals are not pure cycle swing. Angles, targets, velocities and both slow-bias plots were inspected.',
    'slow': 'Stopped nominal positions are nearly static. At 0.15, 0.35 and 0.65 m/s, unequal shoulder pitch swings ride opposite held offsets, with head yaw variation and unequal roll/elbow motion. The 0.15 hip trace shows short shuffling excursions. Both angle/target and velocity timelines were inspected.',
    'middle': 'At 1 m/s, counter-pitch swing rides held opposite shoulder offsets and elbow amplitudes differ. At 1.5 m/s, pitch swing is smaller around large opposite offsets. At 2 and 2.5 m/s, slow target bias and actual pitch drift or reverse, alongside elbow posture drift. Both angle/target and velocity timelines were inspected.',
    'pushed_fast': 'The pushed 1, 2.5, 3 and 3.5 m/s timelines retain unequal shoulder swing and elbow/roll coordination. Slow reversals at 2.5/3 m/s differ from cycle motion. Fast 3.5 m/s pitch is smaller around opposite held biases. Both angle/target and velocity timelines were inspected.',
    'pushed_slow': 'Stopped push produces a brief transient that settles without reset. The 0.15, 0.35 and 0.65 m/s plots show unequal pitch swings, opposite offsets and asymmetric roll/elbows. Both angle/target and velocity timelines were inspected.',
    'pushed_middle': 'The pushed 1, 1.5, 2 and 2.5 m/s traces show opposite shoulder offsets, unequal swing, elbow posture differences and slow bias drift at 2/2.5. Both angle/target and velocity timelines were inspected.'
}
recorded = datetime.now(timezone.utc).isoformat()
for subfolder, observation in observations.items():
    dest = folder / subfolder
    names = ['upper_joint_angles.png', 'upper_joint_velocities.png']
    if not subfolder:
        names += ['upper_bias_decomposition.png', 'upper_target_bias.png']
    receipt = {
        'recorded_utc': recorded,
        'checkpoint': review['checkpoint'],
        'checkpoint_sha256': review['sha256'],
        'plots': [{'path': str(dest / name), 'sha256': hashlib.sha256((dest / name).read_bytes()).hexdigest(), 'visually_inspected': True} for name in names],
        'observations': observation,
        'sampling': '32 robots per command at 10 Hz; 16 selected representatives per gate at 50 Hz. Reset/terminal rows segmented. Population sampling does not certify fast jitter.',
        'natural_human_or_Unitree_motion_certified': False,
        'mapped_speed_matched_Unitree_encoder_curves_available': False,
        'CAD_optical_gaze_calibrated': False
    }
    if subfolder in ['slow', 'pushed_slow']:
        receipt['interpretation_uncertainty'] = 'Stopped position traces are almost constant despite small nonzero reported head/hip velocity residuals. Solver position corrections and sampling effects have not been attributed at 5 ms. This observation does not establish a runtime clock fault or a motor ceiling.'
    (dest / 'visual_inspection.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
audit_path = folder / 'audit_set.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
audit['plots_need_visual_QA'] = False
audit['all_plots_visually_inspected'] = True
audit['visual_inspection_recorded_utc'] = recorded
for supplement in audit['supplements']:
    supplement['needs_visual_QA'] = False
    supplement['visual_inspection'] = str(Path(supplement['folder']) / 'visual_inspection.json')
audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
print('Saved fourteen final plot inspection receipts bound to exact actor 716.')
