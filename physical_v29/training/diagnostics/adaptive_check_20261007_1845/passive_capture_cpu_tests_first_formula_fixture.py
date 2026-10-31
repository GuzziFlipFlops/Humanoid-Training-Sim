"""Verify hook ordering and passivity; no inference/training change."""
import torch,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;torch.set_num_threads(2)
tests=[]
for seed in range(5):
    torch.manual_seed(seed);base=torch.randn(11,23);extra=torch.randn(11,23)*.01
    x=base.clone().requires_grad_(True);before=[];after=[]
    x.register_hook(lambda g:before.append(g.detach().clone()) or g)
    x.register_hook(lambda g:g+extra)
    x.register_hook(lambda g:after.append(g.detach().clone()) or g)
    (x.square().mean()).backward();reference=2*base/base.numel()+extra
    assert torch.equal(x.grad,reference)
    assert torch.equal(before[0],2*base/base.numel())
    assert torch.equal(after[0],reference)
    tests.append(dict(seed=seed,gradient_unchanged=True,pure_PPO_captured_before_prior=True))
(OUT/'passive_capture_cpu_tests.json').write_text(json.dumps(dict(passed=True,tests=tests,
    scope='Five deterministic hook-order fixtures prove passive copies return the same gradients. Runtime real-PPO and full-state checks are separate.'),indent=2)+'\n')
print('5 passive gradient hook-order fixtures pass.')
