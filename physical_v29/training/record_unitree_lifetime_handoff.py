"""Refresh local current indexes while preserving all prior diagnostic history."""
from pathlib import Path
import json,re,shutil
HERE=Path(__file__).parent
REPO=Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot')
DOCS=REPO/'docs/robot'
DIAG=HERE/'diagnostics/unitree_termination_20261002'
MARK='<!-- unitree-nominal-20261002 -->'
BACKUP=DIAG/'documentation_before_nominal_repair'

def prepend(path,title,text):
    old=path.read_text(encoding='utf-8')
    if MARK in old:return
    BACKUP.mkdir(parents=True,exist_ok=True)
    key=str(path).replace(':','').replace('\\','_').replace('/','_')
    dest=BACKUP/(key+'.txt')
    if not dest.exists():shutil.copy2(path,dest)
    path.write_text(title+'\n\n'+MARK+'\n\n'+text+'\n\n## Historical record before the nominal-reference repair\n\n'+old,encoding='utf-8')

def main():
    active=json.loads((HERE/'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
    assert Path(active['run']).name=='locomotion_20261002_r22_unitree_nominal'
    status=json.loads((DIAG/'handoff_status.json').read_text())
    assert status['active']['run']==active['run']
    assert status['state']['operation']=='train' and status['state']['stage']!='preflight'
    current='''The current replacement is **r22_unitree_nominal**, reward
`unitree_g1_lifetime_v2`. Read `E:/Aditya/robot_sim/physical_v29/training/ACTIVE_RUN.json`,
its live state/progress, verified worker/supervisor processes and independent
gates for current status. R20 was drained and preserved after 68 full updates;
R21 was preserved after validation, before full standing training. Neither is
active or qualified. R20's replay first breached the waist differential
motor-angle limit at 0.08 seconds, then fell backward without alternating steps.
Its original aggregate reward rates were all negative, allowing early episodes
to avoid future costs.

R22 retains the pinned Unitree RL Lab G1 core bodies and weights, adding the
declared RL Gym nonnegative regular aggregate followed by MJLab H2's -200
physical-termination weight (-4 per 20 ms event). Timeouts are unpenalized.
Per-cause termination telemetry separates motor faults from ordinary falls.
This is an adapted public Unitree design, not unmodified complete G1/H2 or a
commercial pretrained policy. 5,593 core and 149 lifetime CPU checks passed.
Actual new physics/PPO/load/capacity checks are retained in R22's run directory.
Smoke, reward rankings and screenshots do not qualify learned walking.

The original adaptation's expanded reference aliased the nominal pose. A
randomized reset could change that shared origin and every other robot's
reference. The actual-source CPU regression reproduced 0.63665 rad drift after
512 resets; corrected private storage gives zero drift and leaves unreset rows
unchanged. This is a reproduction, not a measurement of old GPU drift. Every
real PPO update now logs/asserts zero nominal drift. Contract is
`physical_v29_history5_unitree_defaultoffset_v2`, still 400/415 actor/critic
dimensions. A fresh actor is used; no moving IK target or standing teacher.
Physical-v29 mass 34.46334 kg, CAD joint frames, motor limits, differential
geometry and collisions are unchanged. CAD v32 stays separate. Do not load
old history4/IK or another robot's policy tensors into this fresh actor.

The chain targets 8192, reserve 300 MiB, fallback, and a 180-second continuous
healthy-live-PPO ramp at checkpoint boundaries. Host RAM recycling preserves
actor/critic/Adam/LR; the allocation leak itself remains unresolved. Keep one
Isaac/CUDA worker only. No training stereo/render camera is instantiated.
Stages: stand; walk 0.35/0.65/1.0; run 1.5/2.0/2.5, with 0.15 retention. Promotion
requires independent current/prior nominal and pushed 20-second gates. The
running cadence/flight extension is declared, not commercial Unitree weights.
Every later learned candidate needs visual review before a human-gait claim.

Report: `E:/Aditya/robot_sim/physical_v29/training/UNITREE_NOMINAL_REPAIR_20261002.md`.
Source/weight comparison: `training/UNITREE_REWARD_PORT_20261002.md`.
Frame review: `training/diagnostics/unitree_port_20261002/frame-review.html`.
The review contains 280 scripted screenshots, 80 learned-r20 replay screenshots,
and 207 explicitly labeled CPU smoke reconstructions (567 frames across 15 cases), with synchronized angles,
contacts, failure causes and original H1/G1 rates; 47 official GIF frames were
inspected too. None of the failed candidates matches the reference gait.
Local-only documentation/source changes; no Git commit or push.
'''
    p=status['progress'];s=status['state']
    current += f'''\nDated handoff {status['checked_at_utc']}: verified live full
{s['stage']} PPO at {p['num_envs']} environments, with {s['total_training_iterations']}
saved full updates and current worker iteration {p['iteration']}. Free VRAM
{p['gpu']['free_mb']} MiB; available host RAM {p['gpu']['ram_available_mb']} MiB.
Frozen worker/environment/reward files match the tested workspace. This is
runtime evidence, not a behavioral qualification. Inspect fresh active status.
Current nominal-pose drift: {p['policy']['nominal_pose_max_abs_drift_rad']} rad.
The first host-RAM recycle saved 17 full updates and resumed at 8192. Actual
continuation Adam counters advanced420 to440 after the next PPO update;
actor/contract/reward/asset provenance is retained in
`training/diagnostics/unitree_termination_20261002/optimizer_resume_r22.json`.
'''
    prepend(HERE/'README.md','# Physical-v29 corrected Unitree adaptation',current+'\n[Repair report](UNITREE_NOMINAL_REPAIR_20261002.md).\nUse `monitor_training.ps1 -Watch` without starting a second simulator.\n')
    prepend(HERE/'RESEARCH.md','# Unitree source and lifetime evidence',current)
    prepend(Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),'# Current physical-v29 corrected Unitree adaptation — October 2, 2026',current)
    prepend(REPO/'Learning/HumanoidTraining/FILE_LOCATIONS.md','# Critical file locations — current training r22',current)
    prepend(DOCS/'TRAINING.md','# Physical-v29 corrected Unitree adaptation — October 2, 2026',current+'\n[Full repair report](TRAINING_UNITREE_NOMINAL_20261002.md).\n')
    report=(HERE/'UNITREE_NOMINAL_REPAIR_20261002.md').read_text(encoding='utf-8')
    def absolute_link(match):
        target=match.group(1)
        if target.startswith(('https://','http://','#')):return match.group(0)
        path=HERE/target
        return ']('+path.resolve().as_posix()+')' if path.exists() else match.group(0)
    (DOCS/'TRAINING_UNITREE_NOMINAL_20261002.md').write_text(re.sub(r'\]\(([^)]+)\)',absolute_link,report),encoding='utf-8')
    journal=DOCS/'TRAINING_JOURNAL_20261001.md'
    old=journal.read_text(encoding='utf-8')
    if MARK not in old:
        key=str(journal).replace(':','').replace('\\','_').replace('/','_')
        shutil.copy2(journal,BACKUP/(key+'.txt'))
        journal.write_text(old+'\n\n## October 2: Unitree nominal-reference and lifetime repairs\n\n'+MARK+'\n\n'+current,encoding='utf-8')
    print('Current local indexes, skill reference and journal updated; historical text/backups preserved; no Git commands.')

if __name__=='__main__':main()
