import shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent
for name in ['prepare_and_pause.py','resume_after_diagnosis.py']:
    shutil.copy2(OUT.parent/'adaptive_check_20261007_2346'/name,OUT/name)
p=OUT/'prepare_and_pause.py';s=p.read_text().replace('Bounded saved-actor command-response comparison and passive actual PPO audit; original supervisor resumes','Bounded phase-balanced shoulder-mean full-state64/128 comparison; same original supervisor resumes');p.write_text(s)
p=OUT/'resume_after_diagnosis.py';s=p.read_text().replace("'capture_current_gradients.py'", "'capture_current_gradients.py','shoulder_candidate.py','diverse_eval.py'");p.write_text(s)
source=OUT.parent/'adaptive_check_20261007_1344/diverse_heading_eval.py'
s=source.read_text();s=s.replace("candidate='--candidate-course' in sys.argv\nif candidate:sys.argv.remove('--candidate-course')",'candidate=False')
s=s.replace("  if candidate:\n   from heading_candidate import patch_environment\n   patch_environment(cls)\n",'')
(OUT/'diverse_eval.py').write_text(s)
print('Prepared immutable-source copy/pause and truthful independently seeded state protocol.')
