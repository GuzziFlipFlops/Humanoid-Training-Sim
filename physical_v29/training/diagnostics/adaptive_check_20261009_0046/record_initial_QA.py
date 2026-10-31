"""Receipts for the fourteen actor827 images visually inspected this check."""
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone
OUT = Path(__file__).resolve().parent
root = OUT / 'initial_upper'
notes = {
    '': 'Actual827 head is gait-locked;1m pitch swing/elbow motion are unequal.2.5/3m opposite shoulder bias is mostly held during these traces rather than a large slow reversal.3.5 physical target swing is much larger than measured pitch swing. Bias and target-bias plots confirm held offsets with smaller faster residuals. Roll/elbow asymmetry and velocity peaks remain. No natural-motion certification.',
    'slow': 'Stopped joints settle with small residual target error. Moving .15/.35/.65 traces show unequal opposite shoulder pitch biases, larger left pitch motion, unequal elbow cycles and gait-locked neck motion. Velocity peaks are bilateral and unequal.',
    'middle': 'At1m shoulder pitch and elbows have unequal amplitude/bias. At1.5/2/2.5m shoulder pitch stays at opposite held offsets, with smaller fast motion than physical targets. Neck/roll/elbow velocity peaks remain gait-linked. No human encoder match inferred.',
    'pushed_fast': 'Pushed1m retains unequal shoulder/elbow cycles. Pushed2.5/3/3.5m pitch is held around opposite biases with small fast swing versus larger targets. Roll/elbow and neck oscillation persist. These representative traces do not certify population jitter.',
    'pushed_slow': 'Push near5s visibly disturbs stopped shoulder/elbow/head/hip traces and transiently affects .15 gait. .35/.65 return to similar unequal cycles. Reset/first-episode segmentation is retained; no natural-motion inference.',
    'pushed_middle': 'Pushed1/1.5/2/2.5 maintain the unequal pitch/roll/elbow and gait-linked neck pattern. Small actual pitch motion at faster commands tracks held bias rather than all of the target cycle. Joint velocities are inspected independently of angles.'}
records = []
for folder, observations in notes.items():
    path = root / folder
    names = ['upper_joint_angles.png', 'upper_joint_velocities.png']
    if not folder: names += ['upper_bias_decomposition.png', 'upper_target_bias.png']
    record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), actor='827_run_350_train_273',
        plots=[dict(path=str(path / n), sha256=hashlib.sha256((path / n).read_bytes()).hexdigest(), visually_inspected=True) for n in names],
        observations=observations, sampling='32 per command at10Hz population;16 selected representatives per gate at50Hz; reset rows segmented.',
        natural_human_or_Unitree_motion_certified=False, optical_gaze_calibrated=False)
    (path / 'visual_inspection.json').write_text(json.dumps(record, indent=2) + '\n')
    records.append(record)
manifest = json.loads((root / 'audit_set.json').read_text())
manifest['plots_need_visual_QA'] = False
for entry in manifest['supplements']: entry['needs_visual_QA'] = False
manifest['visual_QA_receipts'] = [str(root / f / 'visual_inspection.json') for f in notes]
(root / 'audit_set.json').write_text(json.dumps(manifest, indent=2) + '\n')
(OUT / 'upper_reference_review.json').write_text(json.dumps(dict(
    human_source='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043',
    human_comparison='Primary human wrist/trunk IMU study provides amplitude,asymmetry and jerk features at preferred/fast walking; it does not provide mapped shoulder pitch/roll and elbow curves at this CAD morphology or exact speed.',
    official_Unitree_source='https://github.com/unitreerobotics/unitree_rl_mjlab',
    Unitree_comparison='Official RL runtime reference; exact speed-matched encoder curves mapped to this CAD were not obtained.',
    human_or_Unitree_curves_fabricated=False, likeness_certified=False, original_CAD_optical_axis_calibrated=False,
    reporting='Feature-based diagnostic comparison only; native neck zero is torso-relative. Isaac-head/FK parity does not calibrate optical gaze.'), indent=2) + '\n')
print('Saved fourteen inspected actor827 image receipts and reference limits.')
