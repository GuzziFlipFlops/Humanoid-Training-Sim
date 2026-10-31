"""Finite follow-through for this check; no recurring job or parallel CUDA."""
from pathlib import Path
import json,subprocess,sys,time
OUT=Path(__file__).resolve().parent
deadline=time.monotonic()+6900
while True:
    if (OUT/'matched_slow_reward/failure.json').exists():
        raise RuntimeError('Matched experiment failed; inspect the preserved failure before higher tests')
    if (OUT/'matched_slow_reward/completion.json').exists() and (OUT/'diagnostic_resume.json').exists():break
    assert time.monotonic()<deadline,'Finite higher-test queue timed out; no training process changed'
    time.sleep(5)
subprocess.run([sys.executable,'-u','-X','utf8',str(OUT/'run_high_speed.py')],check=True)
