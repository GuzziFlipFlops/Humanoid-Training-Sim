"""Record the local reward repair and authoritative run pointer; never use Git."""
import json, re
from pathlib import Path
from status_io import atomic_json

HERE = Path(__file__).parent
RUN = HERE / 'runs/locomotion_20261001_r6_measured_steps'
REPO = Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot')
DOCS = [REPO / 'docs/robot', HERE.parents[1] / 'docs/robot']
state = json.loads((RUN / 'state.json').read_text())
assert state['status'] == 'running', state['status']
config = json.loads((RUN / 'chain_config.json').read_text())
assert (RUN / 'source/physical_v29/training/walking_rewards.py').read_bytes() == (HERE / 'walking_rewards.py').read_bytes()
atomic_json(HERE / 'ACTIVE_RUN.json', dict(
    run_directory=str(RUN),state_file=str(RUN / 'state.json'),
    scope='Current 23-axis physical v29; measured-step reward repair and stand/walk/run chain',
    created_date='2026-10-01',planning_mass_kg=34.46334144435157,
    reserve_mib=300,initial_environment_count=8192,iterations_per_block=25,max_hours=12,
    resumed_checkpoint=str(HERE / 'checks/reward_repair_20261001/stand_normalization_repaired.pt'),
    standing_source_checkpoint=str(HERE / 'runs/locomotion_20261001_r4_8192/027_stand_train_013/candidate.pt'),
    historical_attempt=str(HERE / 'runs/locomotion_20261001_r4_8192'),
    reason='User requested fixing stationary walking reward loophole; preserve and revalidate trusted standing actor',
    reward_version='measured_steps_v2',qualification='Reward tests remove observed incentive; live behavioral gates alone qualify stages',local_only=True))

notice = '''## Current reward repair — October 1, 2026

The active chain is `runs/locomotion_20261001_r6_measured_steps`. It fixes the
observed stationary walking incentive and starts from r4's verified standing
actor, with a fresh critic/optimizer for the changed reward. It revalidates
standing, then trains 0.15/0.35/0.65/1.0 m/s walking and 1.5/2.0/2.5 m/s running.
Actual forward progress, unloaded swing and ordered forward lift/land events
replace planted-foot gait credit. Existing gate thresholds remain; all prior
commands must pass nominal and pushed retention tests.

28 reward/reference/normalization/capacity tests and the 14 gate/three logging tests passed. Replaying
the saved bobbing rollout removed 96.43% of its positive locomotion credit.
This does not establish learned walking. R5 completed physics/PPO/reload execution checks
and capacity benchmarks with identical frozen training code. R6 reuses this
recorded evidence and revalidates the standing policy. R5's 8192 benchmark
dipped to 243 MiB free; R6 correctly selects 6144 with the 300 MiB reserve.
Unused command/phase input scales inherited from standing are repaired in a
checkpoint copy; all physical-input statistics and network weights remain.
All 256 CPU standing-input trials produced exactly identical outputs.
The run requests 8192
environments, preserves the 300 MiB reserve and fallback, uses 25-update blocks
and a 12-hour ceiling. Read `ACTIVE_RUN.json` and the live state for progress.
Details/evidence: [reward repair](REWARD_REPAIR_20261001.md).

The following earlier restart entries are historical. All changes are local;
no Git commit/push or external logger.

'''
readme = HERE / 'README.md'
text = readme.read_text()
if '## Current reward repair' not in text:
    title, remainder = text.split('\n', 1)
    remainder = remainder.replace('## Latest change: user requested 8,192 environments', '## Historical r4 restart: user requested 8,192 environments', 1)
    readme.write_text(title + '\n\n' + notice + remainder.lstrip())
else:
    readme.write_text(re.sub(r'## Current reward repair.*?no Git commit/push or external logger\.\n\n',notice,text,count=1,flags=re.S))

repair = (HERE / 'REWARD_REPAIR_20261001.md').read_text()
for folder in DOCS:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'TRAINING_REWARD_REPAIR_20261001.md').write_text(repair)
    page = folder / 'TRAINING.md'
    if page.exists():
        text = page.read_text()
        if '## Current reward repair' not in text:
            title, remainder = text.split('\n', 1)
            remainder = remainder.replace('## Latest change: user requested 8,192 environments', '## Historical r4 restart: user requested 8,192 environments', 1)
            page.write_text(title + '\n\n' + notice.replace('REWARD_REPAIR_20261001.md','TRAINING_REWARD_REPAIR_20261001.md') + remainder.lstrip())
        else:
            page.write_text(re.sub(r'## Current reward repair.*?no Git commit/push or external logger\.\n\n',notice.replace('REWARD_REPAIR_20261001.md','TRAINING_REWARD_REPAIR_20261001.md'),text,count=1,flags=re.S))
    journal = folder / 'TRAINING_JOURNAL_20261001.md'
    if journal.exists() and '## Stationary walking reward repair' not in journal.read_text():
        with journal.open('a') as output:
            output.write('\n\n## Stationary walking reward repair\n\nR4 passed standing but not walking. Saved joint/contact telemetry showed\nstationary leg oscillation. User requested a fix. The ordered-step and\nsigned-progress reward repair passed 23 targeted CPU tests; the recorded\nplanted-foot rollout lost 96.43% of its positive motion credit. Existing\nbehavioral thresholds remain. A fresh r5 chain starts from the trusted stand\nactor, resetting critic/optimizer estimates when the reward changes. The new\nchain performs fresh Isaac checks and capacity benchmarks, revalidates stand,\nthen adds an easier 0.15 m/s walking stage. No walking success is claimed by\nthis entry. Full details: TRAINING_REWARD_REPAIR_20261001.md. Local only.\n')

index = REPO / 'Learning/HumanoidTraining/FILE_LOCATIONS.md'
text = index.read_text()
marker = '## Current walking reward repair — October 1, 2026'
if marker not in text:
    title, remainder = text.split('\n', 1)
    current = '''
## Current walking reward repair — October 1, 2026

Authoritative run pointer is `E:/Aditya/robot_sim/physical_v29/training/ACTIVE_RUN.json`.
The r6 measured-step chain replaces the stationary walking incentive and starts
from r4's passed standing actor. See `docs/robot/TRAINING_REWARD_REPAIR_20261001.md`.
Read live state/gates before claiming progress; old dated pointers below are
historical. No Git commit/push.

'''
    index.write_text(title + '\n' + current + remainder.lstrip())
else:
    index.write_text(text.replace('The r5 measured-step chain','The r6 measured-step chain',1))
print(json.dumps(dict(active_run=str(RUN),documentation_folders=[str(p) for p in DOCS],local_only=True),indent=2))
