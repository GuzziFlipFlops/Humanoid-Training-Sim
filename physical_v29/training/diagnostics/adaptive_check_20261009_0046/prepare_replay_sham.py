"""Add the necessary zero-offset replay control without editing running code."""
import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;SHAM=OUT/'replay_sham'
assert json.loads((OUT/'diagnostic_resume.json').read_text())['same_supervisor']
SHAM.mkdir(exist_ok=False)
for name in ['action_projection.py','prepare_and_pause.py','resume_after_diagnosis.py']:
 text=(OUT/name).read_text()
 if name in ['prepare_and_pause.py','resume_after_diagnosis.py']:
  text=text.replace('HERE=OUT.parents[1]','HERE=OUT.parents[2]')
 (SHAM/name).write_text(text,encoding='utf-8')
text=(OUT/'arm_target_probe.py').read_text()
old="""for mode, sign in [('control', 0), ('body_replay_positive', 1),
                    ('body_replay_negative', -1), ('closed_loop_positive', 1), ('closed_loop_negative', -1)]:"""
new="""for mode, sign in [('control', 0), ('body_replay_zero', 0),
                    ('body_replay_positive', 1), ('body_replay_negative', -1)]:"""
assert text.count(old)==1;text=text.replace(old,new)
old="""        for speed in [.15, 1.5]:
            groups.append(dict(name=f'{seed}_3.5_to_{speed}', seed=seed, command=speed,
                initial_command=3.5, mode='transition', sign=0, transition=True))
"""
assert text.count(old)==1;text=text.replace(old,'')
(SHAM/'arm_target_probe.py').write_text(text,encoding='utf-8')
(SHAM/'source_preparation.json').write_text(json.dumps(dict(original_probe_sha256=hashlib.sha256((OUT/'arm_target_probe.py').read_bytes()).hexdigest(),
 diagnostic_only=True,changes=['Two seeds,three commands;control and replay zero/positive/negative,32 paired states/cohort',
 'No repeated transition cohorts','Identical original gains/caps/actor loader/native map'],
 source_frozen_before_execution=True),indent=2))
text=(OUT/'run_bounded_diagnosis.py').read_text()
start=text.index("    launch([str(OUT / 'arm_target_probe.py'), '--smoke'")
end=text.index('\nexcept BaseException:',start)
text=text[:start]+"""    launch([str(OUT / 'arm_target_probe.py'), '--headless', '--device', 'cuda:0'], OUT / 'arm_target_response', 780)
    result=read(OUT/'arm_target_response/result.json')
    assert any(g['mode']=='body_replay_zero' for g in result['groups'])
    assert result['maximum_physical_target_reconstruction_error_rad']<1e-6
    (ROOT/'validation.json').write_text(json.dumps(dict(passed=True,explicit_zero_offset_replay=True,
        diagnostic_controllers_not_deployed=True,work_seconds=time.monotonic()-WORK_START),indent=2))
"""+text[end:]
(SHAM/'run_bounded_diagnosis.py').write_text(text,encoding='utf-8')
print('Prepared independent zero-offset sham replay diagnosis.')
