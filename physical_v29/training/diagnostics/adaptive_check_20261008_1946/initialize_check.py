"""Prepare a dated CPU-only check, preserving the previous focused handoff."""
import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
PREV = OUT.parent / 'adaptive_check_20261008_1446'
for name in ['refresh_evidence.py', 'review_gates.py', 'verify_recovery.py',
             'upper_joint_time_audit.py', 'upper_bias_audit.py', 'upper_target_bias_audit.py',
             'run_upper_audit_set.py', 'record_plot_qa.py', 'audit_current_motors.py']:
    shutil.copy2(PREV / name, OUT / name)
shutil.copy2(TRAIN / 'automation/NEXT_CHECK.json', OUT / 'previous_NEXT_CHECK.json')
shutil.copy2(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml'), OUT / 'previous_automation.toml')
path = OUT / 'verify_recovery.py'
text = path.read_text(encoding='utf-8')
old = "OUT/'attempt01_initial_mlp_failure/diagnostic_resume.json',OUT/'attempt02_diagnostic_graph_cycle/diagnostic_resume.json',OUT/'diagnostic_resume.json'"
new = "OUT.parent/'adaptive_check_20261008_1446/attempt01_initial_mlp_failure/diagnostic_resume.json',OUT.parent/'adaptive_check_20261008_1446/attempt02_diagnostic_graph_cycle/diagnostic_resume.json',OUT.parent/'adaptive_check_20261008_1446/diagnostic_resume.json',OUT/'diagnostic_resume.json'"
assert old in text
path.write_text(text.replace(old, new), encoding='utf-8')
(OUT / 'setup_attempt01.json').write_text(json.dumps({
    'failure': 'PowerShell quoting truncated the inline Python initialization expression before execution; following log redirection also found no destination directory.',
    'preserved_tool_error': 'SyntaxError: unexpected EOF while parsing; Out-File: Could not find a part of the path.',
    'repair': 'Saved standalone Python initialization source and execute sequentially after its successful preparation.',
    'simulator_started': False, 'training_paused': False, 'production_change': False
}, indent=2) + '\n', encoding='utf-8')
print('Prepared CPU-only audit scripts, previous handoff and preserved setup failure.')
