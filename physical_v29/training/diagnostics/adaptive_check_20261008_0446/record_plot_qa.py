"""Record a human/model visual inspection already performed, never infer it."""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True)
parser.add_argument('--observations', required=True)
args = parser.parse_args()
folder = OUT / args.folder
names = ['upper_joint_angles.png', 'upper_joint_velocities.png',
         'upper_bias_decomposition.png', 'upper_target_bias.png']
record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
    plots=[dict(path=str(folder / name),
                sha256=hashlib.sha256((folder / name).read_bytes()).hexdigest(),
                visually_inspected=True) for name in names],
    observations=args.observations,
    sampling='32 robots per command at 10 Hz versus 16 total selected representatives at 50 Hz; resets segmented. Population sampling does not certify fast jitter.',
    natural_human_or_Unitree_motion_certified=False,
    mapped_speed_matched_Unitree_encoder_curves_available=False,
    CAD_optical_gaze_calibrated=False)
(folder / 'visual_inspection.json').write_text(json.dumps(record, indent=2) + '\n')
print('Saved inspection receipt for ' + args.folder)
