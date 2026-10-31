"""Preserve the failed external metadata check and strict owned resume."""
import hashlib,json,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent.resolve()
dest=OUT/'attempt01_signature_metadata'
assert dest.resolve().is_relative_to(OUT) and not dest.exists()
dest.mkdir()
resume=json.loads((OUT/'diagnostic_resume.json').read_text())
assert resume['same_supervisor'] and resume['supervisor_pid']==39000
for name in ['teacher_smoke','bounded_diagnosis','bounded_diagnosis.log','pause.log','resume.log',
    'diagnostic_pause.json','diagnostic_resume.json','state_before_diagnosis.json','stopped_boundary.json',
    'protected_source_copy.json','teacher_probe_failure.json']:
    src=OUT/name;target=dest/name
    assert src.resolve().is_relative_to(OUT) and target.resolve().is_relative_to(dest)
    if src.exists():shutil.move(str(src),str(target))
for name in ['diagnostic_controllers.py','controller_cpu_tests.json','run_bounded_diagnosis.py']:
    shutil.copy2(OUT/name,dest/name)
record=dict(checkpoint=resume['checkpoint'],sha256=resume['checkpoint_sha256'],pause_seconds=resume['pause_seconds'],
    error='External diagnostic incorrectly required the original349 embedded reward_signature to equal its independently requalified post-travel-guard signature.',
    actual_rollout_steps=0,learned_actor_changed=False,production_fault_inferred=False,same_supervisor_resumed=True,
    preserved_executed_source=str(dest/'teacher_smoke/executed_source.py'),
    fix='Validate exact qualified349 SHA,asset/observation contract,current certificate and matching complete gates. Keep strict current-student signature check; no generic migration/signature bypass.',
    resume_evidence=str(dest/'diagnostic_resume.json'))
(dest/'RECORD.json').write_text(json.dumps(record,indent=2)+'\n')
p=OUT/'verify_recovery.py';s=p.read_text()
needle="OUT.parent/'adaptive_check_20261009_0046/replay_sham/diagnostic_resume.json']"
assert s.count(needle)==1
s=s.replace(needle,"OUT.parent/'adaptive_check_20261009_0046/replay_sham/diagnostic_resume.json',OUT/'attempt01_signature_metadata/diagnostic_resume.json']")
p.write_text(s)
print('Failed smoke and126.95s owned pause/resume preserved.')
