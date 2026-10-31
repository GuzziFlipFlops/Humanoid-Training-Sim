"""Preserve failed diagnostic evidence inside this check before retrying."""
import json, shutil, hashlib
from pathlib import Path
import psutil
OUT = Path(__file__).resolve().parent.resolve()
resume = json.loads((OUT / 'diagnostic_resume.json').read_text())
assert resume['same_supervisor'] and not resume['training_source_changed']
p = psutil.Process(resume['supervisor_pid']); assert p.status() != psutil.STATUS_STOPPED
dst = OUT / 'attempt01'
assert dst.resolve().is_relative_to(OUT) and not dst.exists()
dst.mkdir()
for name in ['arm_target_response', 'bounded_diagnosis', 'bounded_diagnosis.log', 'pause.log', 'resume.log',
        'diagnostic_pause.json', 'diagnostic_resume.json', 'state_before_diagnosis.json', 'stopped_boundary.json',
        'arm_probe_failure.json', 'failed_probe_shutdown_cleanup.json']:
    source = OUT / name
    assert source.resolve().is_relative_to(OUT) and source.exists()
    source.rename(dst / name)
source = (OUT / 'arm_target_probe.py').read_text()
source = source.replace('OnPolicyRunner(env, copy.deepcopy(config),', 'OnPolicyRunner(env, config,')
(dst / 'failed_probe_source_reconstruction.py').write_text(source)
(dst / 'RECORD.json').write_text(json.dumps(dict(
    outcome='Failed before physical data collection; exact owned supervisor safely resumed.',
    reason='OnPolicyRunner mutates nested algorithm configuration. Teacher initialization reused the consumed dictionary and lacked class_name.',
    repair='Pass independent deep-copied runner configurations.',
    source_record_scope='Reconstruction of the two failing call sites from preserved traceback and repaired source; not an asserted original-byte snapshot.',
    preserved_checkpoint=resume['checkpoint'], preserved_checkpoint_sha256=resume['checkpoint_sha256'],
    pause_seconds=resume['pause_seconds'], production_fault=False, production_sources_changed=False,
    physical_results_collected=False, supervisor_resumed=True, diagnostics_samples_counted_as_main=False), indent=2))
print('Failed attempt archived under checked current report directory.')
