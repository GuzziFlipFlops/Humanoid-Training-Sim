"""Local status pointers for the verified r3 restart; no Git commands."""
from pathlib import Path
import json
import shutil
from status_io import atomic_json

HERE = Path(__file__).resolve().parent
RUN = HERE/'runs/locomotion_20261001_r3'
REPO = Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot')
state = json.loads((RUN/'state.json').read_text())
assert state['status'] == 'running' and state.get('progress', {}).get('iteration', -1) >= 0
atomic_json(HERE/'ACTIVE_RUN.json', dict(run_directory=str(RUN), state_file=str(RUN/'state.json'),
    scope='Current 23-axis physical v29; resumed stand/walk/run training', created_date='2026-10-01',
    planning_mass_kg=34.46334144435157, reserve_mib=300, initial_environment_count=4096,
    iterations_per_block=100, max_hours=16,
    qualification='Inspect live state and gates; standing remains unpassed at restart',
    resumed_checkpoint=str(HERE/'checks/recovered_r2_model50/resume.pt'),
    historical_attempt=str(HERE/'runs/locomotion_20261001_r2'),
    historical_attempt_note='150-update block completed; nominal standing failed; following block stopped at update 58 due to a Windows status-file rename lock. Original evidence preserved.',
    fix='Status writes retry transient Windows sharing violations; real held-reader tests passed', local_only=True))
note = '''## October 1 restart after the status check

The r2 chain stopped after a Windows reader temporarily blocked renaming
`progress.tmp` onto `progress.json`. It had completed a 150-update training
block, failed the independent nominal standing gate, and then advanced 58
updates into the next block. This is a logging failure, not a policy promotion.

The status writer now retries transient sharing violations for up to five
seconds. Three tests passed, including a real held Windows read handle and a
permanent-lock case that preserves the old file and reports failure. The
supervisor uses the same writer; the helper is included in each frozen source
snapshot. No dynamics, rewards, actuator limits or gates were changed.

The latest periodic `model_50.pt` was recovered from the interrupted block.
Its missing provenance metadata was restored using that worker's run config
and matching source/asset hashes. Actor/critic tensors were checked unchanged;
the original checkpoint remains untouched. No standing qualification was added.

The r3 chain resumes at **4,096 environments**, the count that completed the
previous sustained block. It uses **100-update blocks**, a **300 MiB reserve**
and a **16-hour ceiling**. The same stand/walk/run stages and independent
nominal/push/retention gates apply. Read `ACTIVE_RUN.json` and the indicated
live state for actual progress; this note is a dated restart record.

All files remain local. No Git commit/push or external experiment logger.

'''
(HERE/'RESUME_20261001.md').write_text('# Training restart — October 1, 2026\n\n'+note)
readme = HERE/'README.md'
text = readme.read_text()
if '## October 1 restart after the status check' not in text:
    text = text.replace('## Training and promotion', note+'## Training and promotion', 1)
text = text.replace('Each block is 150 PPO updates.', 'The active r3 run uses 100 PPO updates per block.')
text = text.replace('The chain has an 18-hour\nceiling;', 'The original r2 chain had an 18-hour ceiling; the active r3 restart has a\n16-hour ceiling;')
readme.write_text(text)
training = REPO/'TRAINING.md'
text = training.read_text()
if '## October 1 restart after the status check' not in text:
    text = text.replace('## Model and weights', note+'## Model and weights', 1)
text = text.replace('active run at handoff:', 'original run at handoff:')
training.write_text(text)
shutil.copy2(training, HERE.parents[1]/'docs/robot/TRAINING.md')
shutil.copy2(HERE/'RESUME_20261001.md', REPO/'TRAINING_RESTART_20261001.md')
shutil.copy2(HERE/'RESUME_20261001.md', HERE.parents[1]/'docs/robot/TRAINING_RESTART_20261001.md')
print('Active pointer and local restart documentation now identify r3.')
