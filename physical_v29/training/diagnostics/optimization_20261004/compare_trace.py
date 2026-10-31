"""Compare synchronized traces; report differences without hiding divergence."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('expected')
p.add_argument('actual')
p.add_argument('--output', required=True)
args = p.parse_args()
a, b = np.load(args.expected), np.load(args.actual)
assert set(a.files) == set(b.files)
rows = {}
for key in a.files:
    expected, actual = a[key], b[key]
    assert expected.shape == actual.shape, key
    exact = np.array_equal(expected, actual)
    delta = np.abs(expected.astype(np.float64)-actual.astype(np.float64))
    rows[key] = dict(shape=list(expected.shape), bitwise_equal=exact,
                     max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()))
report = dict(all_bitwise_equal=all(v['bitwise_equal'] for v in rows.values()),
              fields=rows, scope='Same seed, checkpoint, population and tick schedule; physical replay, not a locomotion gate')
Path(args.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
