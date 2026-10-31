"""CPU exposure counterexamples and unchanged-bucket parity."""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT / 'control_source/physical_v29/training'
sys.path[:0] = [str(OUT), str(TRAIN), str(TRAIN.parent)]
import torch
from focused_prior_sampler import sample_focused, baseline

torch.set_num_threads(2)
checks = []
def check(name, condition):
    assert bool(condition), name
    checks.append(name)

generator = torch.Generator().manual_seed(7911)
n = 500000
ids = torch.arange(n)
r = torch.rand(n, generator=generator)
v = torch.rand(n, generator=generator)
r_before = r.clone()
v_before = v.clone()
old = baseline(3.5, ids, r, v)
new = sample_focused(3.5, ids, r, v)
prior_mask = (r >= .15) & (r < .45) & (ids % 5 != 0)
check('all nonprior values bitwise unchanged', torch.equal(old[~prior_mask], new[~prior_mask]))
check('all standing membership unchanged', torch.equal(old <= .1, new <= .1))
check('permanently stopped fifth unchanged', bool((new[ids % 5 == 0] == 0).all()))
check('no command outside original range', bool(((new >= 0) & (new <= 3.5)).all()))
for command in [.15, 1.5]:
    frequency = float(((new - command).abs() < 1e-6).float().mean())
    check(f'focused {command} exposure approximately 4.8 percent', abs(frequency - .048) < .002)
    check(f'focused {command} exposure increases', frequency > float(((old-command).abs()<1e-6).float().mean()))
for command in [.35, .65, 1., 2., 2.5, 3.]:
    frequency = float(((new - command).abs() < 1e-6).float().mean())
    check(f'other prior {command} exposure approximately 2.4 percent', abs(frequency - .024) < .002)
for frontier in [0., .35, .65, 1., 1.5, 2., 2.5, 3., 4., 4.5, 5.]:
    check(f'frontier {frontier} is bitwise unchanged', torch.equal(sample_focused(frontier, ids[:1000], r[:1000], v[:1000]), baseline(frontier, ids[:1000], r[:1000], v[:1000])))
boundary_r = torch.tensor([0., .149999, .15, .449999, .45, .749999, .75, 1.])
boundary_v = torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.])
boundary_ids = torch.tensor([1, 2, 3, 4, 6, 7, 8, 9])
check('closed endpoint draws remain finite and in range', bool(torch.isfinite(sample_focused(3.5, boundary_ids, boundary_r, boundary_v)).all()))
check('no draw mutation', torch.equal(r, r_before) and torch.equal(v, v_before))
result = dict(passed=True, checks=checks, count=len(checks), seed=7911,
    samples=n, scope='Exposure and bucket parity only; not PPO, behavioral progress or causality.')
(OUT / 'focused_prior_cpu_tests.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))
