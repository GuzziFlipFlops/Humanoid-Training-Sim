"""Use observed Isaac articulation order; URDF metadata order is different."""
import json
from pathlib import Path

def runtime_joint_order(folder):
    rows = [json.loads(s.split('[V29] ENVIRONMENT_READY ', 1)[1])
            for s in (Path(folder) / 'stdout.log').read_text(errors='replace').splitlines()
            if '[V29] ENVIRONMENT_READY ' in s]
    assert len(rows) == 1, 'Require one observed runtime articulation record'
    names = rows[0]['joints']
    assert len(names) == 23 and len(set(names)) == 23
    metadata = json.loads((Path(folder) / 'run_config.json').read_text())['joints']
    assert set(names) == set(metadata), 'Joint identities must agree despite different order'
    return names
