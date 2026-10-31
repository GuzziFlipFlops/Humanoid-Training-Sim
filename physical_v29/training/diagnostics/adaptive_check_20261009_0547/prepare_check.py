"""Prepare external diagnostics and preserve the preceding check unchanged."""
import shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent
TRAIN=OUT.parents[1]
OLD=OUT.parent/'adaptive_check_20261009_0046'
names=['review_gates.py','verify_recovery.py','refresh_evidence.py','upper_joint_time_audit.py',
    'upper_bias_audit.py','upper_target_bias_audit.py','run_upper_audit_set.py','summarize_upper.py',
    'action_projection.py','prepare_and_pause.py','resume_after_diagnosis.py',
    'capture_current_gradients.py','analyze_current_gradients.py','gate_com_contact_audit.py',
    'audit_current_motors.py','compact_pair.py']
for name in names:
    assert not (OUT/name).exists()
    shutil.copy2(OLD/name,OUT/name)
p=OUT/'verify_recovery.py';s=p.read_text()
s=s.replace(",OUT/'diagnostic_resume.json']:",
    ",OUT.parent/'adaptive_check_20261009_0046/diagnostic_resume.json',OUT/'diagnostic_resume.json']:")
s=s.replace("[OUT/'attempt01/diagnostic_resume.json',OUT/'attempt02/diagnostic_resume.json',OUT/'replay_sham/diagnostic_resume.json']",
    "[OUT.parent/'adaptive_check_20261009_0046/attempt01/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_0046/attempt02/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_0046/replay_sham/diagnostic_resume.json']")
p.write_text(s)
p=OUT/'refresh_evidence.py';s=p.read_text().replace('adaptive_check_20261008_1946/final_owned_gate_review.json',
    'adaptive_check_20261009_0046/final_gate_review.json');p.write_text(s)
p=OUT/'prepare_and_pause.py';s=p.read_text().replace(
    'Bounded arm/base coupling and teacher effective-target attribution on truthful settled/transition histories; no diagnostic actor or controller deployed',
    'Bounded qualified-teacher body/current-upper compatibility on truthful settled/transition feedback; diagnostic hybrids never deployed')
p.write_text(s)
p=OUT/'resume_after_diagnosis.py';s=p.read_text().replace("'arm_target_probe.py'}", "'arm_target_probe.py','teacher_body_probe.py'}");p.write_text(s)
shutil.copy2(TRAIN/'automation/NEXT_CHECK.json',OUT/'previous_NEXT_CHECK.json')
shutil.copy2(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml'),OUT/'previous_automation.toml')
print('Prepared external evidence tools; live frozen learner unchanged.')
