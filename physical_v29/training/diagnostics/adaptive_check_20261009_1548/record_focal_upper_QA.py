"""Record explicitly inspected focal plots; no automatic visual certification."""
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--folder', required=True)
p.add_argument('--gate-review', required=True)
p.add_argument('--notes', required=True)
p.add_argument('--two-plots', action='store_true')
a = p.parse_args()
root = OUT / a.folder
review = json.loads((OUT / a.gate_review).read_text())
pair = review['latest']
names = ['upper_joint_angles.png', 'upper_joint_velocities.png']
if not a.two_plots:
    names += ['upper_bias_decomposition.png', 'upper_target_bias.png']
record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
    checkpoint=pair['checkpoint'], checkpoint_sha256=pair['sha256'],
    exact_nominal=pair['nominal'], exact_pushed=pair['pushed'],
    observations=a.notes, population_sampling_Hz=10,
    representative_sampling_Hz=50, reset_segments_preserved=True,
    focal_traces_do_not_cover_every_population_robot=True,
    human_or_Unitree_encoder_certified=False, CAD_optical_gaze_calibrated=False,
    plots=[dict(path=str(root / n), sha256=hashlib.sha256((root / n).read_bytes()).hexdigest(),
                visually_inspected=True) for n in names])
(root / 'visual_inspection.json').write_text(json.dumps(record, indent=2) + '\n')
print(f'Saved {len(names)} explicitly inspected focal plot receipts for {Path(pair["checkpoint"]).parent.name}.')
