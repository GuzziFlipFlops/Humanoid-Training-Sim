import shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent
for name in ['prepare_and_pause.py','resume_after_diagnosis.py']:
    shutil.copy2(OUT.parent/'adaptive_check_20261008_0446'/name,OUT/name)
p=OUT/'prepare_and_pause.py';s=p.read_text().replace('Bounded phase-balanced shoulder-mean full-state64/128 comparison; same original supervisor resumes','Bounded passive truthful-state, individual-cycle and actual PPO gradient diagnosis; no learning candidate deployed');p.write_text(s)
p=OUT/'resume_after_diagnosis.py';s=p.read_text().replace("'shoulder_candidate.py','diverse_eval.py'", "'shoulder_candidate.py','diverse_eval.py','capture_policy_states.py'");p.write_text(s)
shutil.copy2(OUT.parent/'adaptive_check_20261007_1845/capture_current_gradients.py',OUT/'capture_current_gradients.py')
print('Prepared passive diagnostic wrappers; production sources unchanged.')
