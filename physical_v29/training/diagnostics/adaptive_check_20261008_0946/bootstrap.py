"""CPU-only audit bootstrap; no training or simulator mutation."""
import json,shutil,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
PREV=OUT.parent/'adaptive_check_20261008_0446'
TRAINING=OUT.parents[1]
OUT.mkdir(parents=True,exist_ok=True)
for name in ['review_gates.py','verify_recovery.py','upper_joint_time_audit.py','upper_bias_audit.py','upper_target_bias_audit.py','refresh_evidence.py','record_plot_qa.py','gate_com_contact_audit.py']:
    shutil.copy2(PREV/name,OUT/name)
shutil.copy2(TRAINING/'automation/NEXT_CHECK.json',OUT/'previous_NEXT_CHECK.json')
shutil.copy2(Path.home()/'.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml',OUT/'previous_automation.toml')
p=OUT/'verify_recovery.py';s=p.read_text();s=s.replace("OUT/'diagnostic_resume.json'", "OUT.parent/'adaptive_check_20261008_0446/diagnostic_resume.json',OUT/'diagnostic_resume.json'");p.write_text(s)
subprocess.run([sys.executable,'-u','-X','utf8',str(OUT/'refresh_evidence.py'),'--tag','initial'],check=True,stdout=(OUT/'initial_summary.log').open('w'),stderr=subprocess.STDOUT)
print((OUT/'initial_summary.log').read_text())
