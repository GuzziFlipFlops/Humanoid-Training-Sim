"""Preserve last handoff and reuse CPU evidence tools outside frozen sources."""
import shutil
from pathlib import Path
out=Path(__file__).resolve().parent;train=out.parents[1]
old=out.parent/'adaptive_check_20261009_0547'
names=['review_gates.py','verify_recovery.py','refresh_evidence.py','upper_joint_time_audit.py',
 'upper_bias_audit.py','upper_target_bias_audit.py','run_upper_audit_set.py','summarize_upper.py',
 'record_upper_QA.py','action_projection.py','prepare_and_pause.py','resume_after_diagnosis.py',
 'capture_current_gradients.py','analyze_current_gradients.py','gate_com_contact_audit.py',
 'audit_current_motors.py','compact_pair.py','record_diagnostic_memory.py']
for name in names:
 assert not (out/name).exists()
 shutil.copy2(old/name,out/name)
p=out/'verify_recovery.py';s=p.read_text()
s=s.replace("OUT/'diagnostic_resume.json'", "OUT.parent/'adaptive_check_20261009_0547/diagnostic_resume.json',OUT/'diagnostic_resume.json'")
s=s.replace("OUT/'attempt01_signature_metadata/diagnostic_resume.json'", "OUT.parent/'adaptive_check_20261009_0547/attempt01_signature_metadata/diagnostic_resume.json'")
s=s.replace("OUT/'head_arm_factorial/diagnostic_resume.json'", "OUT.parent/'adaptive_check_20261009_0547/head_arm_factorial/diagnostic_resume.json'")
p.write_text(s)
p=out/'refresh_evidence.py';s=p.read_text().replace('adaptive_check_20261009_0046/final_gate_review.json','adaptive_check_20261009_0547/liveness_gate_review.json');p.write_text(s)
shutil.copy2(train/'automation/NEXT_CHECK.json',out/'previous_NEXT_CHECK.json')
shutil.copy2(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml'),out/'previous_automation.toml')
print('Prepared new external CPU audit; original frozen learner preserved.')
