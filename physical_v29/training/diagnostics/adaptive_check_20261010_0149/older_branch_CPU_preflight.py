"""Read-only CPU preparation for a later complete-state branch; no simulator/load/deploy."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import torch

OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(4)
cert = read(OUT / 'neutral_teacher_certificate.json')
pause = read(OUT / 'diagnostic_pause.json')
old_path, current_path = Path(cert['checkpoint']), Path(pause['checkpoint'])
assert sha(old_path) == cert['checkpoint_sha256']
assert sha(current_path) == pause['checkpoint_sha256']
old = torch.load(old_path, map_location='cpu', weights_only=False)
current = torch.load(current_path, map_location='cpu', weights_only=False)
checks = {}
for name in ['actor_state_dict', 'critic_state_dict']:
    checks[name + '_key_and_shape_match'] = (set(old[name]) == set(current[name]) and
        all(old[name][k].shape == current[name][k].shape for k in old[name]))
    checks[name + '_old_all_finite'] = all(torch.isfinite(t).all().item()
                                          for t in old[name].values() if torch.is_tensor(t))
for field in ['contract', 'asset_sha256', 'motion_sha256', 'actuator_sha256',
              'reward_version', 'reward_signature', 'training_command_version',
              'upper_refinement_version']:
    checks[field + '_match'] = old['infos'].get(field) == current['infos'].get(field)
    checks[field + '_present'] = field in old['infos'] and field in current['infos']
optimizer = old['optimizer_state_dict']
checks['earlier_Adam_present'] = bool(optimizer['state']) and bool(optimizer['param_groups'])
checks['earlier_Adam_finite'] = all(torch.isfinite(t).all().item()
    for s in optimizer['state'].values() for t in s.values() if torch.is_tensor(t))
exploration = {k: v for k, v in old['actor_state_dict'].items() if 'log_std' in k}
checks['trained_exploration_present'] = bool(exploration)
old_args = read(old_path.parent / 'run_config.json')['arguments']
current_args = read(current_path.parent / 'run_config.json')['arguments']
teacher = {}
for key in ['standing_prior', 'locomotion_prior_certificate']:
    a, b = Path(old_args[key]), Path(current_args[key])
    if key.endswith('certificate'):
        oa, ca = read(a), read(b)
        oc, cc = Path(oa['checkpoint']), Path(ca['checkpoint'])
    else:
        oc, cc = a, b
    teacher[key] = dict(earlier=str(a), current=str(b),
        earlier_actor_sha256=sha(oc), current_actor_sha256=sha(cc),
        actor_bytes_match=sha(oc) == sha(cc))
    checks[key + '_actor_match'] = teacher[key]['actor_bytes_match']
source_run = old_path.parent.parent
context_paths = list(source_run.glob('**/teacher_head_context.json'))
context = OUT / 'control_source/physical_v29/training/teacher_head_context.json'
context_matches = [dict(path=str(p), sha256=sha(p), matches_current=sha(p) == sha(context))
                   for p in context_paths]
checks['earlier_frozen_context_found'] = bool(context_matches)
checks['earlier_frozen_context_equal'] = bool(context_matches) and all(p['matches_current'] for p in context_matches)
result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), earlier_checkpoint=str(old_path),
    earlier_sha256=sha(old_path), current_comparison_anchor=str(current_path),
    current_sha256=sha(current_path), checks=checks, all_CPU_preflight_checks_pass=all(checks.values()),
    teacher_provenance=teacher, context_provenance=context_matches,
    earlier_Adam_states=len(optimizer['state']), trained_exploration_keys=list(exploration),
    reused_nominal_pushed_certificate=str(OUT / 'neutral_teacher_certificate.json'),
    production_change=False, earlier_continuation_learning_tested=False,
    scope='CPU metadata/tensor preparation only. No worker, simulator, checkpoint runtime load or earlier branch training was started. Later work still requires real strict full-state/teacher/context recovery, smoke, current-state controls, two learning seeds, all gates and independent initial states. Preserve both complete learned states; no from-scratch or tilted-head initialization.')
(OUT / 'older_branch_CPU_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(passed=result['all_CPU_preflight_checks_pass'], checks=checks,
                      earlier_Adam_states=result['earlier_Adam_states'], context_paths=len(context_matches))))
