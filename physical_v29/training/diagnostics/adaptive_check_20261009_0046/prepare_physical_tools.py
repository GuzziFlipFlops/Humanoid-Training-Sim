"""Preserve original tools; new scope stays entirely diagnostic."""
import shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261008_1946'
for name in ['prepare_and_pause.py','resume_after_diagnosis.py','capture_current_gradients.py','analyze_current_gradients.py','audit_current_motors.py']:
    assert not (OUT/name).exists()
    shutil.copy2(OLD/name,OUT/name)
p=OUT/'prepare_and_pause.py';source=p.read_text()
source=source.replace('Bounded truthful retention exposure and native motor attribution, exact-full-state controls; no diagnostic actor deployed',
    'Bounded arm/base coupling and teacher effective-target attribution on truthful settled/transition histories; no diagnostic actor or controller deployed')
p.write_text(source)
p=OUT/'resume_after_diagnosis.py';source=p.read_text()
source=source.replace("'motor_retention_probe.py'}", "'motor_retention_probe.py','arm_target_probe.py'}")
p.write_text(source)
print('Prepared owned pause/resume and passive diagnostics; no production source changes.')
