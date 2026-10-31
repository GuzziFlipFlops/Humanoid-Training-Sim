"""Check actual physical histories, independent states, target map and resume."""
import json, hashlib
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
r = read(OUT / 'arm_target_response/result.json')
d = np.load(OUT / 'arm_target_response/telemetry_50hz.npz')
l = np.load(OUT / 'arm_target_response/teacher_targets_10hz.npz')
m = np.load(OUT / 'arm_target_response/baseline_motors_200hz.npz')
tests = []
def check(name, value):
    assert value, name
    tests.append(name)
check('Four actual native5ms actuator calls per control step', len(m['cap_ratio']) == 4000 and len(d['time']) == 1000)
check('Exact original action-to-physical-target reconstruction', r['maximum_physical_target_reconstruction_error_rad'] < 1e-6)
check('Observed and action commands identical', np.abs(d['observed_command'] - d['action_command']).max() == 0.)
check('Complete reward term sum reconstructs', r['maximum_reward_term_sum_error'] < 1e-3)
check('All synchronous COM,foot,joint,action,target data finite', all(np.isfinite(d[k]).all() for k in
    ['com', 'foot_position', 'foot_velocity', 'q', 'qd', 'tau', 'target', 'root', 'applied_raw_action', 'student_raw_action']))
per = int(d['robots_per_group']); groups = r['groups']
index = {(g['seed'], g['initial_command'], g['mode']): i for i, g in enumerate(groups)}
for i, g in enumerate(groups):
    ids = slice(i * per, (i + 1) * per)
    initial = np.concatenate([d['initial_q'][ids], d['initial_qd'][ids]], 1)
    check(g['name'] + ':32 individually seeded states', len({row.tobytes() for row in initial}) == 32)
    base = index[(g['seed'], g['initial_command'], 'control')]
    paired = slice(base * per, (base + 1) * per)
    check(g['name'] + ':initial states paired with command-matched control',
        np.array_equal(d['initial_q'][ids], d['initial_q'][paired]) and np.array_equal(d['initial_qd'][ids], d['initial_qd'][paired]))
    if g['transition']:
        at = int(np.argmin(abs(l['time'] - 6.)))
        frames = l['observed_history'][at, ids].reshape(per, 5, 80)
        check(g['name'] + ':newest command truthful at switch', np.all(frames[:, -1, 6] == g['command']))
        check(g['name'] + ':older command history retained truthfully', np.all(frames[:, :-1, 6] == g['initial_command']))
        check(g['name'] + ':original teacher context excluded at mixed-command switch', not l['context_eligible'][at, ids].any())
check('Teacher3.5 extrapolation excluded from certified label contract',
    not l['teacher_within_certified_command_range'][l['command'][:, :, 0] > r['teacher_maximum_certified_command_m_s'] + 1e-6].any())
check('No external diagnostic controller admitted to production', not r['diagnostic_controllers_deployed'] and not r['locomotion_qualified'])
resume = read(OUT / 'diagnostic_resume.json')
check('Exact owned supervisor safely resumed', resume['same_supervisor'] and resume['supervisor_pid'] == 39000)
check('Exact selected checkpoint retained through resume', resume['checkpoint_sha256'] == r['checkpoint_sha256']
    and hashlib.sha256(Path(resume['checkpoint']).read_bytes()).hexdigest() == r['checkpoint_sha256'])
check('No production source or learned diagnostic actor deployed', not resume['training_source_changed'] and not resume['diagnostic_actors_deployed'])
check('Real four-update full-state PPO capture validation passed', read(OUT / 'bounded_diagnosis/validation.json')['passed'])
(OUT / 'physical_validation.json').write_text(json.dumps(dict(passed=True, count=len(tests), cases=tests,
    checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'], scope='Bounded diagnostic physics and passive labels/gradients; not gate qualification or a deployed repair.'), indent=2) + '\n')
print(json.dumps(dict(passed=True, count=len(tests))))
