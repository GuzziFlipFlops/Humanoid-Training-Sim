"""Stage an isolated current-full-state trial only after physical support."""
import hashlib,json,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRIAL=OUT/'coupled_training_trial'
physical=json.loads((OUT/'neutral_teacher_response_analysis.json').read_text())
assert all(physical['diverse_teacher_gates'][test]['passed'] for test in ['diverse_nominal','diverse_push'])
assert all(physical['physical_screens']['neutral_body_current_head'].values())
assert json.loads((OUT/'command_local_teacher_CPU_checks.json').read_text())['passed']
TRIAL.mkdir(exist_ok=False);records=[]
for name in ['prepare_and_pause.py','resume_after_diagnosis.py','neutral_teacher_probe.py','diagnostic_teacher_rows.py',
             'command_local_teacher_target.py','neutral_teacher_candidate.py','diverse_eval.py']:
    original=OUT/name;text=original.read_text()
    if name in ['prepare_and_pause.py','resume_after_diagnosis.py','neutral_teacher_probe.py']:
        text=text.replace('HERE=OUT.parents[1]',f'HERE=Path({str(OUT.parents[1])!r})')
    (TRIAL/name).write_text(text);records.append(dict(file=name,source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),staged_sha256=hashlib.sha256((TRIAL/name).read_bytes()).hexdigest()))
for name in ['neutral_teacher_certificate.json','teacher_diagnosis_preparation.json']:shutil.copy2(OUT/name,TRIAL/name)
# Diversity evaluates the learned actor with ordinary inference. Training-only
# hooks are absent, including for experimental checkpoints.
old=OUT.parent/'adaptive_check_20261009_1047/run_reward_matched.py'
header=old.read_text().split('ROOT.mkdir(exist_ok=False)')[0]
header=header.replace("ROOT=OUT/'matched_slow_reward'","ROOT=OUT/'matched_coupled_teacher'").replace('WORK_LIMIT=6600','WORK_LIMIT=7200')
header=header.replace("'slow_reward_candidate.py'","'neutral_teacher_candidate.py'")
header=header.replace("read(folder/'experimental_reward.json')['applied']","read(folder/'experimental_teacher_target.json')['applied']")
header=header.replace("'motor_retention_probe.py'","'neutral_teacher_probe.py'")
header=header.replace("assert time.monotonic()<deadline,'Bounded diagnostic worker exceeded operation budget'","assert time.monotonic()<deadline,'Bounded diagnostic worker exceeded operation budget'\n                assert psutil.virtual_memory().available/2**20>=2048,'Diagnostic host memory guard'")
body=(OUT/'run_coupled_trial_body.txt').read_text()
(TRIAL/'run_coupled_trial.py').write_text(header+body)
compile((TRIAL/'run_coupled_trial.py').read_text(),str(TRIAL/'run_coupled_trial.py'),'exec')
(TRIAL/'staging_receipt.json').write_text(json.dumps(dict(files=records,physical_support=str(OUT/'neutral_teacher_response_analysis.json'),
    diagnostics_only=True,no_production_change=True),indent=2)+'\n');print(str(TRIAL))
