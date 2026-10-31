from pathlib import Path
import shutil
OUT=Path(__file__).resolve().parent
dst=OUT/'attempt01_initial_import_failure';dst.mkdir(exist_ok=True)
for name in ['pause_receipt','resume_receipt','truthful_nominal','diagnostic_pause.json','diagnostic_resume.json','stopped_boundary.json','state_before_diagnosis.json']:
    source=OUT/name;target=dst/name
    assert source.resolve().is_relative_to(OUT.resolve()) and target.resolve().is_relative_to(OUT.resolve())
    if source.exists():assert not target.exists();shutil.move(str(source),str(target))
for name in ['run_diagnostics.py','capture_policy_states.py']:
    shutil.copy2(OUT/name,dst/name)
p=OUT/'verify_recovery.py';s=p.read_text().replace("OUT/'diagnostic_resume.json'", "OUT/'attempt01_initial_import_failure/diagnostic_resume.json',OUT/'diagnostic_resume.json'");p.write_text(s)
print('Preserved failed import attempt and exact resume receipt.')
