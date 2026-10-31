"""CPU-only fresh evidence; preserve earlier check artifacts."""
import shutil,subprocess,sys,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
PREV=OUT.parent/'adaptive_check_20261007_2346'
for name in ['review_gates.py','verify_recovery.py','upper_joint_time_audit.py','upper_bias_audit.py','upper_target_bias_audit.py','refresh_evidence.py']:
    shutil.copy2(PREV/name,OUT/name)
p=OUT/'verify_recovery.py';s=p.read_text();s=s.replace("OUT/'diagnostic_resume.json'", "OUT.parent/'adaptive_check_20261007_2346/diagnostic_resume.json',OUT/'diagnostic_resume.json'");p.write_text(s)
shutil.copy2(PREV/'post_resume_gate_review.json',OUT/'initial_gate_review.json')
subprocess.run([sys.executable,'-u','-X','utf8',str(OUT/'refresh_evidence.py'),'--tag','initial'],check=True,stdout=(OUT/'initial_summary.log').open('w'),stderr=subprocess.STDOUT)
print((OUT/'initial_summary.log').read_text())
