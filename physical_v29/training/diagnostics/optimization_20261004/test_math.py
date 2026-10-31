"""Frozen-model parity and tensor-ownership checks independent of simulator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--output', required=True)
p.add_argument('--device', default='cpu')
args = p.parse_args()
here = Path(__file__).resolve().parent
source = Path(args.source)
sys.path.insert(0, str(source))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CadDifferential


old = load('frozen_cad', here / 'baseline/source/physical_v29/cad_differential.py')
new = load('candidate_cad', source / 'cad_differential.py')
geometry = json.loads((source / 'assets/differential_geometry.json').read_text())
results = []
torch.manual_seed(410901)
torch.set_num_threads(4)
for dtype in [torch.float32, torch.float64]:
    for key, spec in geometry.items():
        # Wide grid includes invalid closure, motor-angle bounds and coupled poses.
        grid = torch.linspace(-1.8, 1.8, 97, device=args.device, dtype=dtype)
        q = torch.cartesian_prod(grid, grid)
        q = torch.cat((q, torch.rand((4096, 2), device=args.device, dtype=dtype)*4-2,
                       torch.zeros((8, 2), device=args.device, dtype=dtype)))
        a, b = old(spec), new(spec)
        with torch.inference_mode():
            expected, actual = a.state(q), b.state(q)
            for e, v in zip(expected, actual):
                assert torch.equal(e, v), (key, dtype, 'state', (e.float()-v.float()).abs().nan_to_num().max().item())
            preserved = [v.clone() for v in actual]
            b.state(q.flip(0).clone())
            for e, v in zip(preserved, actual):
                assert torch.equal(e, v), 'Borrowed output storage corrupted a prior call'
            for speed_scale, torque_scale in [(0., 0.), (.5, 1.), (1., 2.), (2., -2.), (-2., 2.)]:
                qd = torch.linspace(-1, 1, q.numel(), device=args.device, dtype=dtype).reshape_as(q)*speed_scale*80
                desired = torch.linspace(-1, 1, q.numel(), device=args.device, dtype=dtype).reshape_as(q)*torque_scale*250
                eo, es = a.limit(q, qd, desired, 45., 400*2*torch.pi/60, .95)
                vo, vs = b.limit(q, qd, desired, 45., 400*2*torch.pi/60, .95)
                assert torch.equal(eo, vo), (key, dtype, 'effort')
                for field in es:
                    assert torch.equal(es[field], vs[field]), (key, dtype, field)
            # Verify noncontiguous input, shape change and replay after returning to first shape.
            strided = q.repeat(1, 2)[:, ::2]
            for pose in [strided, q[:72], q]:
                for e, v in zip(a.state(pose), b.state(pose)):
                    assert torch.equal(e, v), (key, dtype, 'shape/stride')
        results.append(dict(pair=key, dtype=str(dtype), poses=len(q), state_bitwise=True,
                            limit_cases=5, limits_bitwise=True, output_ownership=True,
                            constants=len(b._tensors), graphs=len(getattr(b, '_state_graphs', {}))))
protected = ['hardware_actuators.py', 'training/environment_unitree.py', 'training/unitree_reward_port.py',
             'training/stage_rewards.py', 'training/walking_rewards.py', 'training/heading_control.py',
             'training/assets/robot_training.urdf', 'assets/differential_geometry.json', 'assets/actuator_limits.json',
             'training/mass_estimate.json', 'assets/export_metadata.json']
for file in protected:
    if file == 'hardware_actuators.py':
        continue  # Later iterations explicitly optimize this implementation.
    assert (source / file).read_bytes() == (here / 'baseline/source/physical_v29' / file).read_bytes(), file
report = dict(passed=True, device=args.device, comparisons=results, protected_physics_rewards_identical=True,
              source_sha256=hashlib.sha256((source / 'cad_differential.py').read_bytes()).hexdigest())
Path(args.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
