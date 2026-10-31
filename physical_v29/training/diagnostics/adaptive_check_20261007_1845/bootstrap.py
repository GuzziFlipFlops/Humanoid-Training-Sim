"""Prepare dated CPU evidence tools and capture owned training without a simulator."""
import json,shutil,importlib.util
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1];PREVIOUS=OUT.parent/'adaptive_check_20261007_1344'
for name in ['review_gates.py','verify_recovery.py','upper_joint_time_audit.py','upper_bias_audit.py']:
    shutil.copy2(PREVIOUS/name,OUT/name)
p=OUT/'verify_recovery.py';text=p.read_text()
old="[OUT.parent/'adaptive_check_20261007_0844/diagnostic_resume.json',OUT/'diagnostic_resume.json']"
new="[OUT.parent/'adaptive_check_20261007_0844/diagnostic_resume.json',OUT.parent/'adaptive_check_20261007_1344/diagnostic_resume.json',OUT/'diagnostic_resume.json']"
assert old in text;p.write_text(text.replace(old,new))
spec=importlib.util.spec_from_file_location('owned_recorder',TRAINING/'record_owned_observation_handoff.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT/'initial_cpu';m.main()
