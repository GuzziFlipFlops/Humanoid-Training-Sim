"""Record the user's requested 8192 count and retain the superseded run."""
from pathlib import Path
import json
import shutil
import time
from status_io import atomic_json

HERE = Path(__file__).resolve().parent
OLD = HERE/'runs/locomotion_20261001_r3'
RUN = HERE/'runs/locomotion_20261001_r4_8192'
state = json.loads((RUN/'state.json').read_text())
assert state['status'] == 'running' and state['environment_count'] == 8192
old = json.loads((OLD/'state.json').read_text())
old.update(status='superseded_by_user_requested_8192', child_pid=None,
           updated_unix=time.time(), superseded_by=str(RUN),
           preserved_resume_checkpoint=str(HERE/'checks/recovered_r3_model50/resume.pt'))
atomic_json(OLD/'state.json', old)
atomic_json(HERE/'ACTIVE_RUN.json', dict(run_directory=str(RUN), state_file=str(RUN/'state.json'),
    scope='Current 23-axis physical v29; user requested 8192 stand/walk/run chain',
    created_date='2026-10-01', planning_mass_kg=34.46334144435157, reserve_mib=300,
    initial_environment_count=8192, iterations_per_block=25, max_hours=16,
    resumed_checkpoint=str(HERE/'checks/recovered_r3_model50/resume.pt'),
    historical_attempt=str(OLD),
    reason='User explicitly requested returning to 8192 environments; short blocks bound accumulated memory and retain the 300 MiB guard',
    qualification='Inspect live progress and independent gates; standing remains unpassed at switch',
    local_only=True))
note = '''## Latest change: user requested 8,192 environments

The active run is `runs/locomotion_20261001_r4_8192`, starting at **8,192
environments** as requested. It resumed from r3's latest `model_50.pt`, with
verified provenance metadata restored; actor/critic tensors are unchanged and
the original periodic save is preserved. The r3 process tree was stopped only
after that checkpoint was recovered. Its evidence and superseded state remain.

Current training blocks are **25 PPO updates**. Shorter worker lifetimes reduce
the accumulated memory growth observed in the previous sustained 8,192 run
and check behavior more frequently. The 300 MiB guard, memory fallback,
hardware limits and independent stand/walk/run/retention gates remain active.
Neither switching the environment count nor recovering a save qualifies
standing. The current restart has a 16-hour ceiling. Use `ACTIVE_RUN.json` and
its live state for progress. Earlier count/block settings below are history.

'''
(HERE/'SWITCH_8192_20261001.md').write_text('# Requested environment-count change — October 1, 2026\n\n'+note)
for path in [HERE/'README.md', Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING.md')]:
    text = path.read_text()
    first_break = text.index('\n\n') + 2
    if '## Latest change: user requested 8,192 environments' not in text:
        text = text[:first_break] + note + text[first_break:]
    text = text.replace('The active r3 run uses 100 PPO updates per block.',
                        'The active r4 run uses 25 PPO updates per block.')
    path.write_text(text)
repo = Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot')
shutil.copy2(repo/'TRAINING.md', HERE.parents[1]/'docs/robot/TRAINING.md')
print('Active pointer identifies user-requested 8192 run; previous run preserved.')
