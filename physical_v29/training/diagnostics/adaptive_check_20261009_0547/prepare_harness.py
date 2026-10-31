"""Adapt the proved bounded harness; no training operation here."""
from pathlib import Path
OUT=Path(__file__).resolve().parent
source=(OUT.parent/'adaptive_check_20261009_0046/run_bounded_diagnosis.py').read_text()
source=source.replace("'arm_target_probe.py'", "'teacher_body_probe.py'")
source=source.replace("'arm_probe_failure.json'", "'teacher_probe_failure.json'")
source=source.replace("'arm_smoke_response'", "'teacher_smoke'")
source=source.replace("'arm_target_response'", "'teacher_compatibility'")
source=source.replace("'--seed', '179'", "'--seed', '181'")
source=source.replace('maximum_physical_target_reconstruction_error_rad', 'maximum_physical_target_reconstruction_error_rad')
assert source.count('teacher_body_probe.py')==4
assert not (OUT/'run_bounded_diagnosis.py').exists()
(OUT/'run_bounded_diagnosis.py').write_text(source)
print('Prepared bounded owned pause/smoke/physics/passive-PPO/resume harness.')
