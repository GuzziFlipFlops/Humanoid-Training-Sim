import json,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];OLD=HERE/'diagnostics/adaptive_check_20261005_1746'
OUT.mkdir(exist_ok=True)
for name in ['initialize_check.py','pause_for_diagnosis.py','resume_after_diagnosis.py','trajectory_parameters.py']:
 if name=='initialize_check.py':continue
 text=(OLD/name).read_text()
 if name=='pause_for_diagnosis.py':text=text.replace('Bounded normal-limit full-return trajectory search; resume same supervisor','Bounded phase-advance physics and smoothed reward comparison; preserve full state')
 if name=='resume_after_diagnosis.py':text=text.replace("['trajectory_search.py','trajectory_validation.py','worker_unitree.py']","['timing_probe.py','worker_unitree.py']")
 (OUT/name).write_text(text)
text=(OLD/'initialize_check.py').read_text()
# Reuse the checkpoint-bound CPU gate parser, without recreating a historical pause script.
text=text.split("pause=(HERE/")[0]
(OUT/'initial_gate_evidence.py').write_text(text)
shutil.copy2(HERE/'diagnostics/walk_regression_20261003/live_handoff.json',OUT/'initial_owned_observation.json')
print(json.dumps({'folder':str(OUT),'run':json.loads((OUT/'initial_owned_observation.json').read_text())['active_run']['run']}))
