"""CPU-only full-population direction audit of existing independent rollouts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from heading_control import direction_statistics, direction_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--evaluation', action='append', required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in args.evaluation:
        path = run/folder/'telemetry.npz'
        report = json.loads((run/folder/'evaluation.json').read_text())
        with np.load(path, allow_pickle=False) as data:
            pose = data['root_pose']
            # Legacy nominal eval resets to the identity quaternion. Audit the
            # frozen reset/InitialStateCfg before using this legacy reference.
            dt = float(data['step_dt'])*5
            alive = data['metrics'][::5,:,0] > .5
            assert alive.shape == pose.shape[:2]
            labels, speeds = data['case_labels'], data['case_speeds']
            for case, speed in enumerate(speeds):
                mask = labels == case
                stats = direction_statistics(pose[:,mask,:2], pose[:,mask,3:7],
                                             alive[:,mask], 0., dt, float(speed))
                checks = direction_checks(float(speed), stats)
                original = report['cases'][case]
                rows.append(dict(folder=folder, speed=float(speed),
                    legacy_gate_passed=original['passed'], direction_checks=checks,
                    direction_passed=all(checks.values()), **stats))
    record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        run=str(run), rows=rows,
        verifier_sha256=hashlib.sha256(Path(__file__).with_name('heading_control.py').read_bytes()).hexdigest(),
        scope='All 128 robots at saved 10 Hz, first-episode live masks, reset identity heading. '
              'Offline diagnostic only, not fresh qualification or contact-pressure certification.')
    (out/'full_population_direction.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
