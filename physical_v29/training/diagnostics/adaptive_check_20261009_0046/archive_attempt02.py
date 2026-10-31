import json
from pathlib import Path
import psutil
OUT = Path(__file__).resolve().parent.resolve(); dst = OUT / 'attempt02'
resume = json.loads((OUT / 'diagnostic_resume.json').read_text())
assert resume['same_supervisor'] and not resume['training_source_changed']
assert psutil.Process(resume['supervisor_pid']).status() != psutil.STATUS_STOPPED
assert dst.resolve().is_relative_to(OUT) and not dst.exists(); dst.mkdir()
for name in ['arm_target_response', 'bounded_diagnosis', 'bounded_diagnosis.log', 'pause.log', 'resume.log',
        'diagnostic_pause.json', 'diagnostic_resume.json', 'state_before_diagnosis.json', 'stopped_boundary.json',
        'arm_probe_failure.json', 'resume_status_attempt01.log']:
    source = OUT / name
    assert source.resolve().is_relative_to(OUT) and source.exists()
    source.rename(dst / name)
(dst / 'RECORD.json').write_text(json.dumps(dict(outcome='Failure before first rollout step; safely resumed exact supervisor.',
    reason='Initial raw environment observation is a Python dict,not the TensorDict returned by later wrapper steps. A dict has no clone method.',
    repair='Clone each tensor through the common mapping items API. Add both dict and TensorDict ownership checks and a small actual physical smoke before full probe.',
    resume_note='First immediate stopped-status assertion failed after owned shutdown. Fresh identity/status inspection verified39000 still stopped with no children; the same strict resume then succeeded. Failed log preserved.',
    checkpoint=resume['checkpoint'], checkpoint_sha256=resume['checkpoint_sha256'], pause_seconds=resume['pause_seconds'],
    production_fault=False, physical_results_collected=False, supervisor_resumed=True), indent=2))
print('Second failed diagnostic and successful strict resume preserved.')
