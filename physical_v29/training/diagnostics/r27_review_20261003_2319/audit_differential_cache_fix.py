"""Compare the live-run implementation with the narrowly repaired cache."""
import gc
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'physical_v29/training/runs/locomotion_20261003_r27_heading_commands_checked/source/physical_v29/cad_differential.py'
NEW = ROOT / 'physical_v29/cad_differential.py'


def load(path, name):
    module = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location(name, path))
    spec.loader.exec_module(module)
    return module.CadDifferential


def footprint():
    gc.collect()
    m = psutil.Process().memory_info()
    return dict(private_mb=getattr(m, 'private', m.rss)/2**20, rss_mb=m.rss/2**20, gc_objects=len(gc.get_objects()))


def main():
    torch.set_num_threads(2)
    old, new = load(OLD, 'cad_old'), load(NEW, 'cad_new')
    specs = json.loads((ROOT / 'physical_v29/assets/differential_geometry.json').read_text())
    result = dict(old_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(), new_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(), cases=[], cache=[], completed=False)
    for name, spec in specs.items():
        for dtype in (torch.float32, torch.float64):
            generator = torch.Generator().manual_seed(901)
            q = (torch.rand((1024, 2), generator=generator, dtype=dtype)-.5)*2.4
            q[:4] = torch.tensor([[0., 0.], [.02, -.015], [-.015, .02], [3., 3.]], dtype=dtype)
            qd = (torch.rand((1024, 2), generator=generator, dtype=dtype)-.5)*120.
            desired = (torch.rand((1024, 2), generator=generator, dtype=dtype)-.5)*400.
            a, b = old(spec), new(spec)
            before, after = a.state(q), b.state(q)
            for x, y in zip(before, after):
                torch.testing.assert_close(x, y, rtol=0., atol=0., equal_nan=True)
            xo, xs = a.limit(q, qd, desired, 45., 42., .95)
            yo, ys = b.limit(q, qd, desired, 45., 42., .95)
            torch.testing.assert_close(xo, yo, rtol=0., atol=0., equal_nan=True)
            for key in xs:
                torch.testing.assert_close(xs[key], ys[key], rtol=0., atol=0., equal_nan=True)
            result['cases'].append(dict(pair=name, dtype=str(dtype), poses=1024, valid=int(after[2].sum()), state_and_limit_bitwise_equal=True))
            del a, b, before, after, xs, ys, xo, yo, x, y
        for label, cls in [('old', old), ('fixed', new)]:
            model = cls(spec)
            before = footprint()
            started = time.perf_counter()
            for tick in range(300):
                model.state(torch.full((72, 2), tick/10000., dtype=torch.float64))
            elapsed = time.perf_counter()-started
            after = footprint()
            result['cache'].append(dict(pair=name, implementation=label, iterations=300, batch_size=72, cache_entries=len(model._tensors), before=before, after=after, cpu_wall_seconds=elapsed))
            if label == 'fixed':
                assert len(model._tensors) <= 16
                assert all(all(isinstance(v, (int, float)) for v in key[2]) for key in model._tensors)
            del model
    result['completed'] = True
    (OUT / 'differential_cache_equivalence.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
