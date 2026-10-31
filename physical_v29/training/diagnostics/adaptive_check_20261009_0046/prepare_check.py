"""Prepare a new read-only diagnostic check, preserving previous artifacts."""
import shutil
from pathlib import Path
OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
OLD = OUT.parent / 'adaptive_check_20261008_1946'
assert not (OUT / 'initial_cpu').exists()
for name in ['review_gates.py', 'verify_recovery.py', 'upper_joint_time_audit.py', 'upper_bias_audit.py',
             'upper_target_bias_audit.py', 'run_upper_audit_set.py', 'record_plot_qa.py']:
    assert not (OUT / name).exists()
    shutil.copy2(OLD / name, OUT / name)
path = OUT / 'verify_recovery.py'
source = path.read_text(encoding='utf-8')
needle = ",OUT/'diagnostic_resume.json']"
assert source.count(needle) == 1
path.write_text(source.replace(needle, ",OUT.parent/'adaptive_check_20261008_1946/diagnostic_resume.json',OUT/'diagnostic_resume.json']"), encoding='utf-8')
shutil.copy2(TRAIN / 'automation/NEXT_CHECK.json', OUT / 'previous_NEXT_CHECK.json')
shutil.copy2(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml'), OUT / 'previous_automation.toml')
print('Prepared diagnostic tools and preserved earlier handoff. No production changes.')
