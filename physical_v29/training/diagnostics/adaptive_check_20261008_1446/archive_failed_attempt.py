"""Preserve the completed failed diagnostic and exact safe-resume receipt."""
import json,shutil,sys
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent.resolve();memory='--memory-guard' in sys.argv
dest=(OUT/('attempt02_diagnostic_graph_cycle' if memory else 'attempt01_initial_mlp_failure')).resolve()
assert dest.is_relative_to(OUT) and not dest.exists()
resume=json.loads((OUT/'diagnostic_resume.json').read_text())
assert resume['same_supervisor'] and resume['supervisor_pid']==39000
assert psutil.Process(39000).status()!=psutil.STATUS_STOPPED
dest.mkdir()
for name in ['matched_sequence','diagnostic_pause.json','diagnostic_resume.json','state_before_diagnosis.json','stopped_boundary.json',
    'pause.log','resume.log','protected_source_copy.json','physical_parity.json']:
    source=(OUT/name).resolve();target=(dest/name).resolve()
    assert source.is_relative_to(OUT) and target.is_relative_to(OUT)
    if source.exists():shutil.move(str(source),str(target))
for name in ['sequence_shoulder_refinement.py','sequence_candidate.py','test_sequence_mean.py','run_sequence_matched.py']:
    shutil.copy2(OUT/name,dest/name)
(dest/'attempt_summary.json').write_text(json.dumps(dict(reproduced_error=('Diagnostic hook self-reference retains GPU graphs' if memory else 'RSL-RL MLP subclass does not support slicing construction'),
    completed_diagnostic_updates_before_error=(12 if memory else 1),diagnostic_actors_discarded=True,production_unchanged=True,original_supervisor_resumed=True,
    pause_seconds=resume['pause_seconds']),indent=2))
p=OUT/'verify_recovery.py';s=p.read_text();old="OUT/'diagnostic_resume.json']";assert old in s
p.write_text(s.replace(old,"OUT/'"+dest.name+"/diagnostic_resume.json',OUT/'diagnostic_resume.json']"))
print('Failed attempt and actual resume interval archived; training is running.')
