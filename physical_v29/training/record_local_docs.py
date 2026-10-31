"""Record the October 1 physical-v29 handoff locally; never runs Git commands."""
from pathlib import Path
import csv
import json
import shutil

HERE = Path(__file__).resolve().parent
REPO = Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot')
DOCS = REPO / 'docs/robot'
SKILL = Path('C:/Users/rshan/.codex/skills/humanoid-robot-training')


def replace_once(path, before, after):
    text = path.read_text(encoding='utf-8')
    if after in text:
        return
    assert text.count(before) == 1, (str(path), before)
    path.write_text(text.replace(before, after, 1), encoding='utf-8')


def main():
    replace_once(HERE / 'README.md', 'execution evidence,+not',
                 'execution evidence,\nnot')
    replace_once(HERE / 'MASS_ESTIMATE.md', 'weigh under30 kg', 'weigh under 30 kg')
    replace_once(HERE / 'MASS_ESTIMATE.md', 'about7.5 kg', 'about 7.5 kg')
    shutil.copy2(HERE / 'MASS_ESTIMATE.md', DOCS / 'MASS_ESTIMATE.md')
    mass = json.loads((HERE / 'mass_estimate.json').read_text())
    with (DOCS / 'MASS_BASIS.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['link', 'part', 'body', 'category', 'mass_kg', 'basis'])
        writer.writeheader()
        for part in mass['parts']:
            writer.writerow({key: part[key] for key in writer.fieldnames})

    (DOCS / 'TRAINING.md').write_text('''# Current physical v29 locomotion — October 1, 2026

The current 23-axis robot has started a new free-base Isaac Lab stand/walk/run
chain. Standing PPO updates were observed advancing at **8,192 environments**.
No learned standing or locomotion stage is claimed passed at this handoff.

The authoritative entry point is
`E:/Aditya/robot_sim/physical_v29/training/README.md`. Read
`training/ACTIVE_RUN.json` and its indicated live `state.json` for progress;
this dated report is not a live status display. Active run at handoff:
`training/runs/locomotion_20261001_r2`.

## Model and weights

The source is the independently reopened wrist-corrected v29 F3D. Its 23 body
axes and saved CAD frames are retained. Current hardware assumptions include
95% added-transmission efficiency, 7 N·m/45 RPM elbows, 120 RPM shoulder pitch,
and pose-dependent shared motor-space ankle/waist constraints. Fingers and
wrists are static. No stereo camera is simulated.

[Mass estimate](MASS_ESTIMATE.md) and [part basis CSV](MASS_BASIS.csv) record
**34.46 kg including the battery**, or **33.98 kg** using AmazingHand's
published 400 g complete-hand value. Training retains the conservative estimate.
Each leg is about 8.7 kg and each arm with a hand is 2.3–2.6 kg. These are
component estimates with provisional gearset, print, hardware and electronics
allowances, not weighed assemblies. Bounding-box part inertias approximate the
native shapes. The older 38 kg model and the 24 kg movement fixture are separate.

## Method, gates and evidence

Primary public Unitree RL Lab, Humanoid-Gym, AMP, Cyclotron/Asimov and
BeyondMimic implementations/papers were reviewed together with old journal
GIFs, source videos, rewards and checkpoint evaluations. Findings and source
hashes are in `training/RESEARCH.md` and `training/research/sources.json`.
Earlier locomotion achieved forward speed but had slipping/stiff-gait
limitations; recovery rewards repeatedly rose without a strict standing pass.
Old 22-axis policy weights are incompatible with current v29.

The new policy uses CAD-IK gait references with residual PPO, four-frame
proprioceptive actor history, a privileged critic and dynamics variation. It
is not AMP or a proven human-motion controller. Train stand, then
0.35/0.65/1.0 m/s walk, then 1.5/2.0/2.5 m/s run. Separate nominal and pushed
20-second evaluations cover all earlier commands. Failure continues training;
earlier-skill regression rolls back. Promotion uses raw physics gates for
survival, posture, tracking, slip, loaded feet, support, bilateral steps,
cadence, clearance and running flight. Resets never erase failed trials.

Four-second contact preflight passed at 337.45 N support versus 338.09 N
expected. Shared differential tests and actual PPO/checkpoint-load smoke tests
passed their execution checks. The smoke checkpoint did **not** pass learned
standing. Physics tests do not certify balance, gait or full casing contact.

Benchmarks at 4,096/6,144/8,192 environments passed. 8,192 achieved about
10,019 environment steps/s, with 1,368 MiB minimum sampled free VRAM. The
unattended chain preserves a **300 MiB reserve** and falls back to 6,144 then
4,096 on memory failure. Its ceiling is 18 hours; an unpassed stage stays
unpassed. Source snapshots, checkpoints, raw gate telemetry, stdout/stderr,
live state and GPU CSV are saved in the run.

The first attempt's benchmarks are retained. Its evaluation checkpoint save
failed because RSL-RL 5's logger writer was uninitialized; the fix was tested
before the replacement run. A `.pt` file alone is not completion evidence.

The sustained 8,192 block reached the 300 MiB guard at update 42, saved a
memory checkpoint and resumed from it at 6,144. Advancing resumed updates were
observed with over 2 GiB free. This verifies the fallback after the short
benchmarks. Fourteen CPU-only gate tests passed, including rejection of
fall/rise, sliding, one-leg, support, bad-torso and invalid-pose exploits.
The robot-training skill passed its quick validator under Isaac Python.

## Remaining qualification

The contact asset uses simplified envelopes, with self-collision enabled.
It underapproximates actual housings/casing seams. Real masses, loaded motor
curves, thermal duty, reflected gear inertia, controller current and battery
sag still need measurement. Natural gait needs later visual review. No get-up,
backflip, payload or hardware-readiness claim follows from this chain.

All changes and logs remain local. No Git commit, push or external logger.
''', encoding='utf-8')

    report = DOCS / 'BUILD_REPORT.md'
    replace_once(report, '**v29 · September 30, 2026 · local documentation only.**',
                 '**v29 · CAD September 30 / training October 1, 2026 · local documentation only.**')
    replace_once(report,
        'Actual masses, measured thermal/loaded duty and contact dynamics remain unqualified. No training started; the older 22-axis model stays separate.',
        'A provisional component estimate is 34.46 kg including the battery, or 33.98 kg with the published complete-hand mass. A separate free-base contact asset passed physics preflight, and the 8,192-environment stand/walk/run chain started October 1. Learned standing and gait gates remain pending at this handoff. Actual masses, measured thermal/loaded duty and full native contact remain unqualified. The older 22-axis model stays separate. See [weight details](MASS_ESTIMATE.md) and [current training](TRAINING.md).')
    replace_once(report,
        '| [Simulation handoff](SIMULATION.md) | Current actuator configuration and original movement-fixture evidence |',
        '| [Simulation handoff](SIMULATION.md) | Current actuator configuration and original movement-fixture evidence |\n| [Mass estimate](MASS_ESTIMATE.md) / [part basis](MASS_BASIS.csv) | Provisional total, limb mass, materials and allowances |\n| [Current training](TRAINING.md) | Physical v29 stand/walk/run chain, primary research and live-status pointers |')

    sim = DOCS / 'SIMULATION.md'
    replace_once(sim, '## Run the fast GUI preview',
        '## Separate locomotion training\n\nOctober 1: a new free-base, gravity/contact training chain started at 8,192 environments. Read [current training](TRAINING.md) and [mass estimate](MASS_ESTIMATE.md); live status is `physical_v29/training/ACTIVE_RUN.json` then its indicated `state.json`. Training uses a separate lightweight contact asset and conservative 34.46 kg component estimate. The fixture and commands below retain their geometry-test scope. Avoid opening another GPU simulator alongside the running chain.\n\n## Run the fast GUI preview')
    replace_once(sim, 'No Git commit/push or locomotion training was performed.',
        'These original movement checks performed no locomotion training. The separate October 1 chain above has started. No Git commit/push was performed.')

    bom = DOCS / 'BOM.md'
    replace_once(bom, '## Actuation',
        '## Mass planning\n\n[Mass estimate](MASS_ESTIMATE.md) / [part basis CSV](MASS_BASIS.csv): conservative total 34.46 kg with battery; 33.98 kg using the official 400 g complete-hand value. Motor variants, printed fill, bearings and unmodeled gearset/electronics allowances are explicit estimates. The complete hand mass includes its servos; do not add the separately listed servo masses again. This mass report does not change BOM quantities or qualify a purchase.\n\n## Actuation')

    locations = REPO / 'Learning/HumanoidTraining/FILE_LOCATIONS.md'
    replace_once(locations, '## Current physical robot v29 — September 30, 2026',
                 '## Current physical robot v29 — CAD September 30 / training October 1, 2026')
    replace_once(locations, 'See current local report; no training started.',
                 'See current local report and `docs/robot/TRAINING.md`: a separate free-base stand/walk/run chain started October 1; no learned stage is claimed passed at this handoff.')
    replace_once(locations, '## Historical physical robot v27 — September 30, 2026',
        '## Current physical-v29 locomotion and mass estimate — October 1, 2026\n\nEntry point: `E:/Aditya/robot_sim/physical_v29/training/README.md`. Read `ACTIVE_RUN.json`, then the live state it indicates; active handoff run is `runs/locomotion_20261001_r2`. Frozen source, config hashes, 150-update training blocks, separate nominal/push gates, checkpoint rollback and 300 MiB GPU reserve are documented there. 8,192 environments was the measured fastest count. This dated pointer is not proof of current process liveness.\n\nWeight basis: `training/MASS_ESTIMATE.md`, `mass_estimate.json`, `mass_by_link.csv`; local report copies are `docs/robot/MASS_ESTIMATE.md` and `MASS_BASIS.csv`. Conservative 34.46 kg includes battery and 640 g/hand; official 400 g complete-hand value gives 33.98 kg. The 24 kg movement-fixture proxies and old 38 kg training robot are separate. Contact/mass asset is `training/assets/robot_training.urdf`, not the original full-visual movement asset. Review `training/RESEARCH.md`, `research/sources.json` and the preserved old GIF/video samples for the policy research and limitations.\n\n## Historical physical robot v27 — September 30, 2026')

    index_text = locations.read_text(encoding='utf-8')
    index_lines = index_text.splitlines(keepends=True)
    for i, line in enumerate(index_lines):
        if line.startswith('## Current ') and not line.startswith(('## Current physical robot v29', '## Current physical-v29')):
            index_lines[i] = line.replace('## Current ', '## Historical ', 1)
    locations.write_text(''.join(index_lines), encoding='utf-8')

    replace_once(SKILL / 'SKILL.md', '- Current recovery training is isolated',
                 '- Historical recovery training was isolated')
    replace_once(SKILL / 'SKILL.md', 'The active 8,192-environment run is `logs\\guarded_getup_v12_reverse_20260906_033513`; inspect its `state.json` before reporting status.',
                 'The historical 8,192-environment run is `logs\\guarded_getup_v12_reverse_20260906_033513`; inspect its evidence before quoting an outcome. Current physical-v29 training uses the October 1 reference above.')
    mirror = HERE.parents[1] / 'docs/robot'
    mirror.mkdir(parents=True, exist_ok=True)
    for name in ['BUILD_REPORT.md', 'SIMULATION.md', 'BOM.md', 'MASS_ESTIMATE.md', 'MASS_BASIS.csv', 'TRAINING.md']:
        shutil.copy2(DOCS / name, mirror / name)
    (DOCS / 'TRAINING_JOURNAL_20261001.md').write_text('''# October 1 physical-v29 training handoff

Reviewed older humanoid journals, source walking/running videos and dashboard
GIFs, guarded locomotion/recovery code and failure reports. The old 22-axis
model is preserved. Primary policy research and exact sources are indexed in
the new training package.

Built a separate component-mass/light-contact asset, CAD IK gait references,
history/asymmetric-critic residual PPO and independent stand/walk/run gates.
Fixed fictitious overlapping hip-marker collisions while keeping self-collision
enabled. Physics/contact/shared-transmission checks passed; PPO and checkpoint
reload executed. Their short rollout was not a learned standing pass.

All three environment-count benchmarks passed. The first attempt then failed
an RSL-RL evaluation logger/save call. Fixed that call, retested saving, kept
the failure evidence, and launched the replacement sustained stand block at
8,192 environments. Iterations were observed advancing with more than 300 MiB
free VRAM. Exact live progress and promotions belong in the run's state/gate
files, not this dated note. No stage is certified by this entry.

Mass estimate is 34.46 kg conservative or 33.98 kg with the published hand
mass; it is not weighed. Local build report, BOM mass basis, simulation
handoff, source index and robot-training skill were updated. No Git commit or
push and no external experiment service.
''', encoding='utf-8')
    shutil.copy2(DOCS / 'TRAINING_JOURNAL_20261001.md', mirror / 'TRAINING_JOURNAL_20261001.md')
    print('Local robot reports, BOM mass basis, source index and skill updated; no Git operations.')


if __name__ == '__main__':
    main()
