"""Current reward counterfactuals and unchanged retention/PPO diagnostic gradients.

Historical byte comparisons in test_phase_feedback refer to pre-R28/R30 sources.
Reuse its behavioral and gradient assertions here; the separate pinned migration
test checks current protected bytes. Neither fixture suite qualifies locomotion.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import test_phase_feedback as phase
import torch

tree = ast.parse(Path(phase.__file__).read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
saved_line = next(n.lineno for n in node.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'saved' for t in n.targets))
raw_line = next(n.lineno for n in node.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'raw' for t in n.targets))
protected_line = next(n.lineno for n in node.body if isinstance(n, ast.For)
                      and isinstance(n.target, ast.Name) and n.target.id == 'rel')
node.body = [n for n in node.body if n.lineno < saved_line or raw_line <= n.lineno < protected_line]
node.body += ast.parse('return dict(checks=checks, cycles=cycles, first_step=dict(planted=stationary, unloaded=unload, lifted=lift))').body
namespace = vars(phase).copy()
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), phase.__file__, 'exec'), namespace)
result = namespace['run'](Path(__file__).parent / 'running_tests')

reward = phase.CurriculumRewards(phase.NAMES, phase.BODIES, phase.BODIES, phase.ROOT, phase.FOOT)
high_speed = []
for speed in [1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]:
    scores = {}
    for behavior in ['matched', 'in_place', 'backward', 'loaded_slide', 'same_foot_hop']:
        rows = []
        for tick in range(100):
            period = 1 / (.9 + .5 * speed)
            p = tick * .02 / period % 1
            flags = (p < .43, (p + .5) % 1 < .43)
            if behavior == 'loaded_slide': flags = (True, True)
            if behavior == 'same_foot_hop': flags = (p < .43, p < .43)
            e = phase.fixture(speed, tick)
            d = e.scene['robot'].data
            forward = 0. if behavior == 'in_place' else -speed if behavior == 'backward' else speed
            d.root_lin_vel_b[0, 0] = forward
            d.root_lin_vel_w[0, 0] = forward
            if behavior == 'loaded_slide': d.body_lin_vel_w[:, phase.FEET, 0] = speed
            metrics = phase.telemetry(e, flags)
            metrics['phase'] = torch.tensor([p])
            metrics['clearance'][0] = torch.tensor([0. if f else .065 for f in flags])
            e.cad_metrics = metrics
            total, _, _ = reward.evaluate(e)
            assert torch.isfinite(total).all()
            rows.append(float(total / .02))
        scores[behavior] = sum(rows) / len(rows)
    assert all(scores['matched'] > scores[b] for b in scores if b != 'matched'), (speed, scores)
    high_speed.append(dict(command_m_s=speed, cycle_scores=scores, passed=True))

result.update(passed=True, reward_version=phase.REWARD_VERSION, high_speed=high_speed,
              historical_assertion_scope='Behavioral/gradient assertions only; historical source comparisons replaced by current strict migration proof',
              phase_test_sha256=hashlib.sha256(Path(phase.__file__).read_bytes()).hexdigest(),
              reward_sha256=hashlib.sha256((HERE/'stage_rewards.py').read_bytes()).hexdigest(),
              scope='Controlled CPU telemetry; not physical feasibility, human gait or learned speed qualification')
path = Path(__file__).with_name('running_extension_tests.json')
path.write_text(json.dumps(result, indent=2))
print(json.dumps(dict(passed=True, behavioral_gradient_checks=len(result['checks']), running_speeds=len(high_speed))))
